import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from mpc_obscodes import mpc_obscodes
from mpcq import MPCObservations


def get_spacebased_stns() -> List[str]:
    """Return all known STN codes without fixed Earth coordinates"""
    with open(mpc_obscodes) as mpc_file:
        obscodes = json.load(mpc_file)
        return [k for k, v in obscodes.items() if "Longitude" not in v]


def mpc_to_od_observations(
    obs_set: MPCObservations,
    prevent_nans: bool = True,
    diag_nan: float = 1.0e-9,
    bias_table: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[OrbitDeterminationObservations]:
    """
    Convert MPC observations into OD observations.

    Parameters:
    -----------
    obs_set: MPCObservations (size N)
      set of MPC observations to convert
    prevent_nans: bool, default True
      should NaN/null values on covariance matrix be replaced with finite numbers.
      diagonal NaNs are replaced with diag_nan, off-diagonal with zeros
    diag_nan: float, default 1.0e-9
      the value to replace NaNs in covariance diagonals if prevent_nans is True
    bias_table: dict[str, tuple[float, float]] or None, default None
      Optional per-station bias correction keyed by MPC obs_code. Each value is
      ``(bias_ra_arcsec, bias_dec_arcsec)`` where ``bias_ra_arcsec`` is in the
      cos(dec)-corrected tangent-plane frame (matching the LOOO residual
      convention in adam_orbit_det_eval). Bias sign is observed - predicted, so
      the correction subtracts the bias from the observation:
        corrected_dec = obs_dec - bias_dec_arcsec / 3600
        corrected_ra  = obs_ra  - (bias_ra_arcsec / 3600) / cos(dec)
      Stations absent from the table are passed through unchanged.

    Returns:
    --------
    Set of observations for orbit determination, size N, or None if the input set is
    malformed, for example, it has NULLs in the STN codes.
    """
    obs_time = obs_set.obstime
    codes = obs_set.stn
    if not np.all(codes):
        print(
            f"STN codes for {obs_set.requested_provid.unique().to_pylist()} include nulls"
        )
        return None

    # `mpcq`'s `MPCObservations` includes uncertainty columns:
    # - rmsra, rmsdec, rmscorr (and rmsmag)
    #
    # These RMS values come from the MPC database and are in arcseconds. By ADES/MPC convention,
    # `rmsra` is RA uncertainty *cos(dec). We convert into degrees and back out RA sigma
    # (so that downstream `Residuals` can apply its own cos(latitude) scaling consistently).
    dec_deg = obs_set.dec.to_numpy(zero_copy_only=False)
    cos_dec = np.cos(np.deg2rad(dec_deg))

    sigma_ra_cosdec_deg = obs_set.rmsra.to_numpy(zero_copy_only=False) / 3600.0
    sigma_dec_deg = obs_set.rmsdec.to_numpy(zero_copy_only=False) / 3600.0
    sigma_ra_deg = np.where(
        np.isfinite(cos_dec) & (cos_dec != 0.0),
        sigma_ra_cosdec_deg / cos_dec,
        np.nan,
    )

    # Include RA/Dec correlation if present; treat missing correlation as 0 (uncorrelated).
    corr = obs_set.rmscorr.to_numpy(zero_copy_only=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    cov = np.full((len(obs_set), 6, 6), np.nan, dtype=np.float64)
    # Prevent 'Covariance matrix has NaNs on the diagonal' and 'Singular matrix
    cov[:, 1, 1] = sigma_ra_deg**2
    cov[:, 2, 2] = sigma_dec_deg**2
    if prevent_nans:
        cov[:, 1, 1] = np.nan_to_num(cov[:, 1, 1], nan=diag_nan)
        cov[:, 2, 2] = np.nan_to_num(cov[:, 2, 2], nan=diag_nan)
    cov[:, 1, 2] = corr * sigma_ra_deg * sigma_dec_deg
    cov[:, 2, 1] = cov[:, 1, 2]
    # Prevents 'UserWarning: Covariance matrix has NaNs on the off-diagonal (these will be assumed to be 0.0).'
    if prevent_nans:
        cov = np.nan_to_num(cov)

    lon = obs_set.ra.to_numpy(zero_copy_only=False).astype(np.float64, copy=True)
    lat = obs_set.dec.to_numpy(zero_copy_only=False).astype(np.float64, copy=True)

    if bias_table is not None:
        stn_codes = codes.to_pylist()
        bias_ra_arcsec = np.zeros(len(obs_set), dtype=np.float64)
        bias_dec_arcsec = np.zeros(len(obs_set), dtype=np.float64)
        for i, code in enumerate(stn_codes):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bias_ra_arcsec[i] = entry[0]
            bias_dec_arcsec[i] = entry[1]
        # cos(dec) here uses the (uncorrected) observed declination — the bias
        # is small enough that evaluating cos at the corrected dec changes the
        # result by O(bias^2) which is negligible at sub-arcsec scales.
        cos_dec_for_bias = np.cos(np.deg2rad(lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            ra_correction_deg = np.where(
                np.isfinite(cos_dec_for_bias) & (cos_dec_for_bias != 0.0),
                (bias_ra_arcsec / 3600.0) / cos_dec_for_bias,
                0.0,
            )
        lon = lon - ra_correction_deg
        lat = lat - bias_dec_arcsec / 3600.0

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obs_time,
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
        covariance=CoordinateCovariances.from_matrix(cov),
    )

    observers = Observers.from_codes(codes=codes, times=obs_time)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=obs_set.mag,
        rmsmag=obs_set.rmsmag,
        band=obs_set.band,
    )

    od_observations = OrbitDeterminationObservations.from_kwargs(
        id=obs_set.obsid.to_numpy(zero_copy_only=False),
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return od_observations
