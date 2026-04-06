"""
Bias model classes for synthetic observation generation.

Each class implements the BiasModel ABC, providing an ``apply()`` method that
returns per-observation (Δra*cos(dec), Δdec) offsets in arcseconds.  Biases
are composable via ``CompoundBias``.

All optional input arrays (mag, zenith_angle, etc.) may be ``None``.  When a
bias requires an input that is ``None`` it logs a warning and returns zeros.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for clarity
_ArrayOrNone = Optional[np.ndarray]
_BiasResult = Tuple[np.ndarray, np.ndarray]  # (Δra*cos(dec), Δdec) in arcsec


def _zeros_like(ra: np.ndarray) -> _BiasResult:
    """Return zero-offset arrays matching the shape of *ra*."""
    return np.zeros_like(ra), np.zeros_like(ra)


class BiasModel(ABC):
    """Abstract base class for all astrometric bias models."""

    @abstractmethod
    def apply(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        obstime: np.ndarray,
        mag: _ArrayOrNone = None,
        zenith_angle: _ArrayOrNone = None,
        parallactic_angle: _ArrayOrNone = None,
        object_rate: _ArrayOrNone = None,
        field_ra: _ArrayOrNone = None,
        field_dec: _ArrayOrNone = None,
    velocity_ra_unit: _ArrayOrNone = None,
    velocity_dec_unit: _ArrayOrNone = None,
    ) -> _BiasResult:
        """
        Compute astrometric bias offsets.

        Parameters
        ----------
        ra : np.ndarray
            Topocentric RA in degrees (N,).
        dec : np.ndarray
            Topocentric Dec in degrees (N,).
        obstime : np.ndarray
            Observation times in MJD (N,).
        mag : np.ndarray or None
            Apparent magnitudes (N,). May be None.
        zenith_angle : np.ndarray or None
            Zenith angle in degrees (N,). May be None.
        parallactic_angle : np.ndarray or None
            Parallactic angle in degrees (N,). May be None.
        object_rate : np.ndarray or None
            Object angular rate in arcsec/hour (N,). May be None.
        field_ra : np.ndarray or None
            Field centre RA in degrees (N,). May be None.
        field_dec : np.ndarray or None
            Field centre Dec in degrees (N,). May be None.
        velocity_ra_unit : np.ndarray or None
            Unit vector component of sky-plane motion in the RA·cos(dec)
            direction (dimensionless, N,). Together with ``velocity_dec_unit``
            defines the instantaneous direction of motion. May be None when
            geodetic site coordinates are unavailable.
        velocity_dec_unit : np.ndarray or None
            Unit vector component of sky-plane motion in the Dec direction
            (dimensionless, N,). May be None.

        Returns
        -------
        (delta_ra_cosdec_arcsec, delta_dec_arcsec) : tuple of np.ndarray
            Signed offsets in arcseconds.  ``delta_ra_cosdec_arcsec`` is
            Δα·cos(δ), matching ADES convention.
        """

    @property
    def bias_type(self) -> str:
        """Short class name used in truth_biases.csv."""
        return type(self).__name__

    def params_dict(self) -> dict:
        """Return a JSON-serialisable dict of bias parameters."""
        return {}


# ---------------------------------------------------------------------------
# Compound
# ---------------------------------------------------------------------------


class CompoundBias(BiasModel):
    """Sum of a list of BiasModel contributions."""

    def __init__(self, biases: List[BiasModel]) -> None:
        self.biases = biases

    def apply(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        obstime: np.ndarray,
        mag: _ArrayOrNone = None,
        zenith_angle: _ArrayOrNone = None,
        parallactic_angle: _ArrayOrNone = None,
        object_rate: _ArrayOrNone = None,
        field_ra: _ArrayOrNone = None,
        field_dec: _ArrayOrNone = None,
    velocity_ra_unit: _ArrayOrNone = None,
    velocity_dec_unit: _ArrayOrNone = None,
    ) -> _BiasResult:
        total_ra = np.zeros_like(ra)
        total_dec = np.zeros_like(dec)
        for b in self.biases:
            d_ra, d_dec = b.apply(
                ra, dec, obstime, mag, zenith_angle, parallactic_angle,
                object_rate, field_ra, field_dec, velocity_ra_unit, velocity_dec_unit,
            )
            total_ra += d_ra
            total_dec += d_dec
        return total_ra, total_dec

    def params_dict(self) -> dict:
        return {"biases": [{"type": b.bias_type, "params": b.params_dict()} for b in self.biases]}


# ---------------------------------------------------------------------------
# I. Astrometric positional
# ---------------------------------------------------------------------------


class ConstantBias(BiasModel):
    """Uniform RA/Dec offset, independent of all covariates."""

    def __init__(self, delta_ra: float, delta_dec: float) -> None:
        self.delta_ra = float(delta_ra)
        self.delta_dec = float(delta_dec)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        return (
            np.full_like(ra, self.delta_ra),
            np.full_like(dec, self.delta_dec),
        )

    def params_dict(self) -> dict:
        return {"delta_ra": self.delta_ra, "delta_dec": self.delta_dec}


class FieldRotationBias(BiasModel):
    """
    Small WCS rotation error.

    The centroid offset is perpendicular to the radial direction from the
    field centre, with magnitude proportional to angular distance from centre.
    ``rotation_arcsec`` is the rotation expressed as the offset at 1 degree
    from centre (i.e., rotation angle in arcsec/deg × 1 deg, but kept as a
    single scalar).
    """

    def __init__(self, rotation_arcsec: float) -> None:
        self.rotation_arcsec = float(rotation_arcsec)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if field_ra is None or field_dec is None:
            logger.warning("FieldRotationBias requires field_ra/field_dec; returning zero bias.")
            return _zeros_like(ra)

        # Angular separation from field centre (degrees)
        dra = (ra - field_ra) * np.cos(np.deg2rad(field_dec))
        ddec = dec - field_dec
        ang_dist = np.sqrt(dra**2 + ddec**2)  # degrees

        # Position angle from field centre (radians)
        pa = np.arctan2(dra, ddec)

        # Perpendicular direction (rotation by +90°)
        perp_ra = np.cos(pa)   # +RA direction component of perpendicular
        perp_dec = -np.sin(pa) # +Dec direction component of perpendicular

        amplitude = self.rotation_arcsec * ang_dist  # arcsec, scales with distance

        return amplitude * perp_ra, amplitude * perp_dec

    def params_dict(self) -> dict:
        return {"rotation_arcsec": self.rotation_arcsec}


class PlateScaleBias(BiasModel):
    """
    Radial offset proportional to angular distance from field centre in ppm
    of that distance.
    """

    def __init__(self, scale_error_ppm: float) -> None:
        self.scale_error_ppm = float(scale_error_ppm)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if field_ra is None or field_dec is None:
            logger.warning("PlateScaleBias requires field_ra/field_dec; returning zero bias.")
            return _zeros_like(ra)

        dra = (ra - field_ra) * np.cos(np.deg2rad(field_dec))
        ddec = dec - field_dec
        ang_dist_deg = np.sqrt(dra**2 + ddec**2)
        ang_dist_arcsec = ang_dist_deg * 3600.0

        # Radial direction unit vector
        with np.errstate(invalid="ignore", divide="ignore"):
            unit_ra = np.where(ang_dist_deg > 0, dra / ang_dist_deg, 0.0)
            unit_dec = np.where(ang_dist_deg > 0, ddec / ang_dist_deg, 0.0)

        amplitude = self.scale_error_ppm * 1e-6 * ang_dist_arcsec  # arcsec

        return amplitude * unit_ra, amplitude * unit_dec

    def params_dict(self) -> dict:
        return {"scale_error_ppm": self.scale_error_ppm}


# ---------------------------------------------------------------------------
# II. Catalog-induced
# ---------------------------------------------------------------------------


class CatalogEpochBias(BiasModel):
    """
    Reference stars at wrong epoch → proper-motion-induced offset.

    Offset = pm * epoch_error, applied uniformly to all observations
    (the median proper motion represents the aggregate frame shift).
    """

    def __init__(
        self,
        epoch_error_years: float,
        pm_ra_median_masyr: float = 50.0,
        pm_dec_median_masyr: float = 30.0,
    ) -> None:
        self.epoch_error_years = float(epoch_error_years)
        self.pm_ra_median_masyr = float(pm_ra_median_masyr)
        self.pm_dec_median_masyr = float(pm_dec_median_masyr)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        # mas/yr * yr → mas → arcsec
        d_ra = self.pm_ra_median_masyr * self.epoch_error_years / 1000.0
        d_dec = self.pm_dec_median_masyr * self.epoch_error_years / 1000.0
        return np.full_like(ra, d_ra), np.full_like(dec, d_dec)

    def params_dict(self) -> dict:
        return {
            "epoch_error_years": self.epoch_error_years,
            "pm_ra_median_masyr": self.pm_ra_median_masyr,
            "pm_dec_median_masyr": self.pm_dec_median_masyr,
        }


# ---------------------------------------------------------------------------
# III. Timing
# ---------------------------------------------------------------------------


class TimingBias(BiasModel):
    """
    Constant clock offset → along-track position shift.

    shift = delta_t_sec * rate / 3600.0  (arcsec), projected along the
    object's actual sky-plane motion direction via velocity_ra_unit /
    velocity_dec_unit.  When the velocity direction is unavailable the shift
    is applied in the +RA direction as a fallback.
    """

    def __init__(self, delta_t_sec: float) -> None:
        self.delta_t_sec = float(delta_t_sec)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if object_rate is None:
            logger.warning(
                "TimingBias: object_rate is None; applying timing shift in RA only "
                "assuming rate = 0.5 arcsec/hour."
            )
            rate = np.full_like(ra, 0.5)
        else:
            rate = np.asarray(object_rate, dtype=float)

        shift = self.delta_t_sec / 3600.0 * rate  # arcsec, scalar magnitude
        if velocity_ra_unit is not None and velocity_dec_unit is not None:
            return shift * np.asarray(velocity_ra_unit), shift * np.asarray(velocity_dec_unit)
        # Fallback: RA-only (old behaviour)
        return shift, np.zeros_like(dec)

    def params_dict(self) -> dict:
        return {"delta_t_sec": self.delta_t_sec}


class ClockDrift(BiasModel):
    """Secular timing error that grows linearly with time from ref_mjd."""

    def __init__(self, drift_sec_per_year: float, ref_mjd: float) -> None:
        self.drift_sec_per_year = float(drift_sec_per_year)
        self.ref_mjd = float(ref_mjd)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        years_elapsed = (obstime - self.ref_mjd) / 365.25
        delta_t = self.drift_sec_per_year * years_elapsed  # seconds

        if object_rate is None:
            rate = np.full_like(ra, 0.5)
        else:
            rate = np.asarray(object_rate, dtype=float)

        shift = delta_t / 3600.0 * rate  # arcsec, scalar magnitude
        if velocity_ra_unit is not None and velocity_dec_unit is not None:
            return shift * np.asarray(velocity_ra_unit), shift * np.asarray(velocity_dec_unit)
        return shift, np.zeros_like(dec)

    def params_dict(self) -> dict:
        return {"drift_sec_per_year": self.drift_sec_per_year, "ref_mjd": self.ref_mjd}


# ---------------------------------------------------------------------------
# IV. Atmospheric / environmental
# ---------------------------------------------------------------------------


class DCRBias(BiasModel):
    """
    Differential chromatic refraction bias.

    Uses the approximate formula:
        Δ = -K * (1/λ² - 1/λ_ref²) * tan(z)

    where K = 2.44e7 arcsec·nm² and z is the zenith angle.  The offset is
    applied in the parallactic direction (towards zenith).

    Requires zenith_angle and parallactic_angle.
    """

    # K = 206265 * (n₀ - 1) * C_Cauchy  where (n₀-1) ≈ 2.91e-4 (standard atm.)
    # and C_Cauchy ≈ 7520 nm² (Filippenko 1982, converted from μm² to nm²).
    # K ≈ 206265 × 2.91e-4 × 7520 ≈ 4.5e5 arcsec·nm²
    K = 4.5e5  # arcsec·nm²

    def __init__(
        self,
        bandpass_nm: float = 200.0,
        ref_wavelength_nm: float = 550.0,
    ) -> None:
        self.bandpass_nm = float(bandpass_nm)
        self.ref_wavelength_nm = float(ref_wavelength_nm)
        # Effective wavelength (simple mid-band approximation)
        self._lam = ref_wavelength_nm - bandpass_nm / 4.0
        self._dcr_factor = self.K * (1.0 / self._lam**2 - 1.0 / ref_wavelength_nm**2)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if zenith_angle is None or parallactic_angle is None:
            logger.warning(
                "DCRBias requires zenith_angle and parallactic_angle; returning zero bias."
            )
            return _zeros_like(ra)

        z_rad = np.deg2rad(zenith_angle)
        pa_rad = np.deg2rad(parallactic_angle)

        with np.errstate(divide="ignore", invalid="ignore"):
            tan_z = np.tan(np.clip(z_rad, 0, np.deg2rad(85)))

        delta_para = -self._dcr_factor * tan_z  # arcsec in parallactic direction

        # Parallactic direction: towards zenith = (−sin(pa), cos(pa)) in (RA, Dec)
        d_ra = delta_para * (-np.sin(pa_rad))
        d_dec = delta_para * np.cos(pa_rad)

        return d_ra, d_dec

    def params_dict(self) -> dict:
        return {
            "bandpass_nm": self.bandpass_nm,
            "ref_wavelength_nm": self.ref_wavelength_nm,
        }


class RefractionModelError(BiasModel):
    """
    Error in the refraction correction due to wrong atmospheric model.

    The dominant term scales as tan(zenith_angle).  Pressure and temperature
    deviations produce an offset along the zenith direction.
    """

    # Standard refraction constant A (arcsec) scaled by pressure (hPa) and temp (K)
    # ΔR ≈ (dA/dP * dP + dA/dT * dT) * tan(z)
    # dA/dP ≈ 0.00116 arcsec/hPa, dA/dT ≈ -0.0039 arcsec/K (approximate)
    _dA_dP = 0.00116  # arcsec/hPa
    _dA_dT = -0.0039  # arcsec/K

    def __init__(
        self,
        delta_pressure_hPa: float = 0.0,
        delta_temp_K: float = 0.0,
    ) -> None:
        self.delta_pressure_hPa = float(delta_pressure_hPa)
        self.delta_temp_K = float(delta_temp_K)
        self._amplitude = (
            self._dA_dP * self.delta_pressure_hPa
            + self._dA_dT * self.delta_temp_K
        )

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if zenith_angle is None:
            logger.warning(
                "RefractionModelError requires zenith_angle; returning zero bias."
            )
            return _zeros_like(ra)

        z_rad = np.deg2rad(zenith_angle)
        with np.errstate(divide="ignore", invalid="ignore"):
            tan_z = np.tan(np.clip(z_rad, 0, np.deg2rad(85)))

        delta_z = self._amplitude * tan_z  # arcsec, in zenith direction

        if parallactic_angle is None:
            # Apply entirely to Dec as a fallback
            return np.zeros_like(ra), delta_z

        pa_rad = np.deg2rad(parallactic_angle)
        d_ra = delta_z * (-np.sin(pa_rad))
        d_dec = delta_z * np.cos(pa_rad)
        return d_ra, d_dec

    def params_dict(self) -> dict:
        return {
            "delta_pressure_hPa": self.delta_pressure_hPa,
            "delta_temp_K": self.delta_temp_K,
        }


# ---------------------------------------------------------------------------
# V. Instrumental / detector
# ---------------------------------------------------------------------------


class CTEBias(BiasModel):
    """
    Charge transfer efficiency (CTE) trailing in a fixed sky direction.

    The amplitude is constant in that direction; the readout_direction_deg
    is measured East of North (standard PA convention).
    """

    def __init__(self, amplitude_arcsec: float, readout_direction_deg: float) -> None:
        self.amplitude_arcsec = float(amplitude_arcsec)
        self.readout_direction_deg = float(readout_direction_deg)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        pa_rad = np.deg2rad(self.readout_direction_deg)
        # PA = East of North → (sin(PA), cos(PA)) = (RA, Dec) unit components
        d_ra = np.full_like(ra, self.amplitude_arcsec * np.sin(pa_rad))
        d_dec = np.full_like(dec, self.amplitude_arcsec * np.cos(pa_rad))
        return d_ra, d_dec

    def params_dict(self) -> dict:
        return {
            "amplitude_arcsec": self.amplitude_arcsec,
            "readout_direction_deg": self.readout_direction_deg,
        }


# ---------------------------------------------------------------------------
# VI. Object-rate dependent
# ---------------------------------------------------------------------------


class TrailingBias(BiasModel):
    """
    Centroid pulled in direction of motion.

    Δ = trailing_factor × object_rate (arcsec/hour), projected along the
    object's actual sky-plane motion direction via velocity_ra_unit /
    velocity_dec_unit.  When the velocity direction is unavailable the shift
    falls back to the +RA direction.
    """

    def __init__(self, trailing_factor: float) -> None:
        self.trailing_factor = float(trailing_factor)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if object_rate is None:
            logger.warning(
                "TrailingBias requires object_rate; returning zero bias."
            )
            return _zeros_like(ra)
        shift = self.trailing_factor * np.asarray(object_rate, dtype=float)
        if velocity_ra_unit is not None and velocity_dec_unit is not None:
            return shift * np.asarray(velocity_ra_unit), shift * np.asarray(velocity_dec_unit)
        # Fallback: RA-only (used when geodetic site coords unavailable)
        return shift, np.zeros_like(dec)

    def params_dict(self) -> dict:
        return {"trailing_factor": self.trailing_factor}


class MagnitudeDependentBias(BiasModel):
    """Linear centroid error in magnitude offset from ref_mag."""

    def __init__(
        self,
        slope_ra: float,
        slope_dec: float,
        ref_mag: float = 18.0,
    ) -> None:
        self.slope_ra = float(slope_ra)
        self.slope_dec = float(slope_dec)
        self.ref_mag = float(ref_mag)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if mag is None:
            logger.warning(
                "MagnitudeDependentBias requires mag; returning zero bias."
            )
            return _zeros_like(ra)
        dm = np.asarray(mag, dtype=float) - self.ref_mag
        return self.slope_ra * dm, self.slope_dec * dm

    def params_dict(self) -> dict:
        return {
            "slope_ra": self.slope_ra,
            "slope_dec": self.slope_dec,
            "ref_mag": self.ref_mag,
        }


class ColorDependentBias(BiasModel):
    """
    Catalog color term error, linear in a relative color proxy.

    Uses apparent magnitude as a *relative* proxy for object color.  Only the
    differential matters (brighter/fainter relative to the median), so the bias
    is computed as::

        dc = mag - median(mag)
        d_ra  = slope_ra  * dc
        d_dec = slope_dec * dc

    This avoids the physically unsound ``mag/20`` proxy previously used.
    Brighter minor planets (lower mag) tend to be larger; the sign of the slope
    encodes the direction of the color term for the specific catalog and filter.

    When ``mag`` is ``None`` or all-NaN, a warning is logged and zero bias is
    returned.  The ``ref_color`` parameter is retained for API compatibility but
    is no longer used in the calculation.
    """

    def __init__(
        self,
        slope_ra: float,
        slope_dec: float,
        ref_color: float = 0.6,
    ) -> None:
        self.slope_ra = float(slope_ra)
        self.slope_dec = float(slope_dec)
        self.ref_color = float(ref_color)  # kept for API compatibility

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        if mag is None:
            logger.warning(
                "ColorDependentBias requires mag as color proxy; returning zero."
            )
            return _zeros_like(ra)
        mag_arr = np.asarray(mag, dtype=float)
        valid = np.isfinite(mag_arr)
        if not valid.any():
            logger.warning(
                "ColorDependentBias: all mag values are NaN; returning zero."
            )
            return _zeros_like(ra)
        # Differential color proxy: relative to the median magnitude of this
        # observation set.  Only the *difference* from median carries physical
        # meaning as a color stand-in.
        ref_mag = np.nanmedian(mag_arr)
        dc = mag_arr - ref_mag
        return self.slope_ra * dc, self.slope_dec * dc

    def params_dict(self) -> dict:
        return {
            "slope_ra": self.slope_ra,
            "slope_dec": self.slope_dec,
            "ref_color": self.ref_color,
        }


# ---------------------------------------------------------------------------
# VII. Time-variable
# ---------------------------------------------------------------------------


class SeasonalBias(BiasModel):
    """Sinusoidal bias with a 365.25-day period."""

    _PERIOD = 365.25  # days

    def __init__(
        self,
        amplitude_ra: float,
        amplitude_dec: float,
        phase_days: float = 0.0,
    ) -> None:
        self.amplitude_ra = float(amplitude_ra)
        self.amplitude_dec = float(amplitude_dec)
        self.phase_days = float(phase_days)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        phase = 2 * np.pi * (obstime - self.phase_days) / self._PERIOD
        return self.amplitude_ra * np.sin(phase), self.amplitude_dec * np.sin(phase)

    def params_dict(self) -> dict:
        return {
            "amplitude_ra": self.amplitude_ra,
            "amplitude_dec": self.amplitude_dec,
            "phase_days": self.phase_days,
        }


class StepChangeBias(BiasModel):
    """Step-function bias at change_mjd (e.g., detector swap, pipeline update)."""

    def __init__(
        self,
        delta_ra_before: float,
        delta_dec_before: float,
        delta_ra_after: float,
        delta_dec_after: float,
        change_mjd: float,
    ) -> None:
        self.delta_ra_before = float(delta_ra_before)
        self.delta_dec_before = float(delta_dec_before)
        self.delta_ra_after = float(delta_ra_after)
        self.delta_dec_after = float(delta_dec_after)
        self.change_mjd = float(change_mjd)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        before = obstime < self.change_mjd
        d_ra = np.where(before, self.delta_ra_before, self.delta_ra_after)
        d_dec = np.where(before, self.delta_dec_before, self.delta_dec_after)
        return d_ra.astype(float), d_dec.astype(float)

    def params_dict(self) -> dict:
        return {
            "delta_ra_before": self.delta_ra_before,
            "delta_dec_before": self.delta_dec_before,
            "delta_ra_after": self.delta_ra_after,
            "delta_dec_after": self.delta_dec_after,
            "change_mjd": self.change_mjd,
        }


class NightlyDrift(BiasModel):
    """
    Linear drift within each UTC night, centred on the middle of astronomical
    night.

    Hours are computed from UTC midnight of each observation, then shifted so
    that observations at the centre of a typical astronomical night (01:00 UTC,
    i.e. roughly midway between evening twilight ~19:00 and morning twilight
    ~07:00) produce zero offset.  This means a station with a non-zero slope
    sees near-zero *mean* bias across a night (the LOOO will not detect a
    spurious constant offset), but has increasing scatter with observation time
    within each night.

    The centering formula is::

        hours_from_center = (hours_utc - 1.0 + 12.0) % 24.0 - 12.0

    which maps 01:00 UTC → 0.0, 19:00 UTC → -6.0, and 07:00 UTC → +6.0.
    """

    def __init__(self, slope_ra_per_hour: float, slope_dec_per_hour: float) -> None:
        self.slope_ra_per_hour = float(slope_ra_per_hour)
        self.slope_dec_per_hour = float(slope_dec_per_hour)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        # Hours since midnight UTC (MJD fractional day × 24)
        hours_utc = (obstime % 1.0) * 24.0
        # Astronomical night is roughly 19–07 UTC; centre at 01:00 UTC.
        # Shift so that centre-of-night (01:00) maps to 0, keeping the range
        # symmetric within ±12 hours.
        hours_from_center = (hours_utc - 1.0 + 12.0) % 24.0 - 12.0
        d_ra = self.slope_ra_per_hour * hours_from_center
        d_dec = self.slope_dec_per_hour * hours_from_center
        return d_ra, d_dec

    def params_dict(self) -> dict:
        return {
            "slope_ra_per_hour": self.slope_ra_per_hour,
            "slope_dec_per_hour": self.slope_dec_per_hour,
        }


# ---------------------------------------------------------------------------
# VIII. Pipeline / reporting
# ---------------------------------------------------------------------------


class ReportingTruncation(BiasModel):
    """
    Quantise RA/Dec to a fixed precision (simulating MPC reporting rounding).

    ``precision_arcsec`` is the rounding step in arcseconds.  The bias is the
    difference between the rounded and true position; its expectation is zero
    but it adds a discrete noise component.

    Implementation note: this class rounds the *true* position rather than the
    final noise+bias-perturbed position.  Because the rounding step (~0.01–0.1
    arcsec) is much smaller than typical Gaussian noise (~0.1–0.5 arcsec), the
    quantisation effect on the total residual is statistically equivalent either
    way.  The implementation is therefore left as-is.
    """

    def __init__(self, precision_arcsec: float) -> None:
        self.precision_arcsec = float(precision_arcsec)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        prec_deg = self.precision_arcsec / 3600.0
        ra_rounded = np.round(ra / prec_deg) * prec_deg
        dec_rounded = np.round(dec / prec_deg) * prec_deg
        cos_dec = np.cos(np.deg2rad(dec))
        d_ra_cosdec = (ra_rounded - ra) * cos_dec * 3600.0  # arcsec
        d_dec = (dec_rounded - dec) * 3600.0  # arcsec
        return d_ra_cosdec, d_dec

    def params_dict(self) -> dict:
        return {"precision_arcsec": self.precision_arcsec}


class WrongSiteBias(BiasModel):
    """
    Site coordinates wrong by (coord_error_east_m, coord_error_north_m,
    coord_error_up_m) metres.

    Converts the position error to an RA/Dec offset using the parallax
    factor at a nominal object distance of 2.5 AU (if not provided).

    Δα = (ΔE * cos(LST) - ΔN * sin(lat)*sin(LST)) / (distance * AU_to_m)
    converted to arcsec.  This is a simplified formula that captures the
    dominant effect.
    """

    _AU_M = 1.495978707e11  # metres per AU

    def __init__(
        self,
        coord_error_east_m: float,
        coord_error_north_m: float,
        coord_error_up_m: float = 0.0,
        nominal_distance_au: float = 2.5,
    ) -> None:
        self.coord_error_east_m = float(coord_error_east_m)
        self.coord_error_north_m = float(coord_error_north_m)
        self.coord_error_up_m = float(coord_error_up_m)
        self.nominal_distance_au = float(nominal_distance_au)

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None) -> _BiasResult:
        dist_m = self.nominal_distance_au * self._AU_M
        # Parallax shift in radians ≈ baseline_m / distance_m
        # East error → RA shift; North error → Dec shift (approximate)
        d_ra_rad = self.coord_error_east_m / dist_m
        d_dec_rad = self.coord_error_north_m / dist_m
        d_ra_arcsec = np.degrees(d_ra_rad) * 3600.0
        d_dec_arcsec = np.degrees(d_dec_rad) * 3600.0
        return (
            np.full_like(ra, d_ra_arcsec),
            np.full_like(dec, d_dec_arcsec),
        )

    def params_dict(self) -> dict:
        return {
            "coord_error_east_m": self.coord_error_east_m,
            "coord_error_north_m": self.coord_error_north_m,
            "coord_error_up_m": self.coord_error_up_m,
            "nominal_distance_au": self.nominal_distance_au,
        }
