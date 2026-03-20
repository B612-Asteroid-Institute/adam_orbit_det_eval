"""
Core synthetic observation generation pipeline.

This module provides two public functions:

- ``generate_truth_ephemeris``: propagate a truth orbit to observation times at
  each real site and cache the result.
- ``generate_synthetic_observations``: full pipeline from truth orbit to
  synthetic ``MPCObservations`` table with injected noise and bias.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# MJD epoch offset (J2000 → MJD is 51544.5; not used here directly)
_DAYS_PER_YEAR = 365.25

# ---------------------------------------------------------------------------
# Hardcoded geodetic coordinates (lat_deg, lon_deg) for Phase 1 stations.
# Longitude is East-positive; latitude is geodetic.
# Used as fallback when no MPC ObsCodes catalog is available.
# ---------------------------------------------------------------------------
_MPC_GEODETIC: Dict[str, Tuple[float, float]] = {
    "F51": (20.7073, -156.2573),   # Pan-STARRS 1, Haleakala
    "F52": (20.7073, -156.2573),   # Pan-STARRS 2, Haleakala (same site)
    "G96": (32.4428, -110.7885),   # Mt. Lemmon Survey
    "703": (32.4428, -110.7885),   # Catalina Sky Survey (same mountain)
    "691": (31.6814, -110.8779),   # Spacewatch, Kitt Peak
    "W84": (-24.6157, -70.1920),   # Cerro Tololo, Chile
    "568": (33.3564, -116.8650),   # Mauna Kea (Palomar)
    "T09": (-31.2722, -70.7378),   # ATLAS Chile, El Sauce
    "V00": (-24.6157, -70.1920),   # Cerro Tololo-DECam
    "X05": (-30.2407, -70.7366),   # Cerro Pachon, Rubin/LSST
    "T05": (19.8333, -155.4750),   # ATLAS Mauna Loa
    "I41": (-31.2722, -70.7378),   # El Sauce
    "W68": (-29.0056, -70.7389),   # Las Cumbres 1m
    "O18": (-22.5340, -68.1767),   # Atacama
}


def _get_site_geodetic(real_code: str) -> Optional[Tuple[float, float]]:
    """
    Return geodetic (lat_deg, lon_deg) for a real MPC station code.

    Looks up the hardcoded Phase 1 station table.  Returns ``None`` if the
    code is not found.

    Parameters
    ----------
    real_code : str
        MPC observatory code (e.g. ``"F51"``).

    Returns
    -------
    (lat_deg, lon_deg) or None
        Geodetic latitude and East-positive longitude in degrees, or ``None``
        if the code is unknown.
    """
    return _MPC_GEODETIC.get(real_code, None)


def _compute_obs_geometry(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    obstime_mjd: np.ndarray,
    site_lat_deg: float,
    site_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute zenith angle, parallactic angle, and approximate object rate.

    Parameters
    ----------
    ra_deg, dec_deg : ndarray
        Topocentric RA/Dec of the object in degrees.
    obstime_mjd : ndarray
        Observation times in MJD (UTC).
    site_lat_deg, site_lon_deg : float
        Observer geodetic latitude and East-positive longitude in degrees.

    Returns
    -------
    (zenith_angle_deg, parallactic_angle_deg, object_rate_arcsec_per_hour)
        All three are 1-D arrays of the same length as *ra_deg*.
    """
    lat_rad = np.deg2rad(site_lat_deg)
    dec_rad = np.deg2rad(dec_deg)
    ra_rad = np.deg2rad(ra_deg)

    # GMST in radians (IAU 1982 approximation, accurate to ~0.1" over decades)
    # JD_UTC ≈ MJD + 2400000.5
    jd_utc = obstime_mjd + 2400000.5
    T = (jd_utc - 2451545.0) / 36525.0  # Julian centuries from J2000
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_utc - 2451545.0)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
    gmst_rad = np.deg2rad(gmst_deg % 360.0)

    # Local Sidereal Time
    lst_rad = (gmst_rad + np.deg2rad(site_lon_deg)) % (2 * np.pi)

    # Hour angle
    ha_rad = (lst_rad - ra_rad) % (2 * np.pi)

    # Zenith angle: cos(z) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(H)
    cos_z = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zenith_angle_deg = np.degrees(np.arccos(cos_z))

    # Parallactic angle: q = atan2(sin(H)*cos(φ), sin(φ)*cos(δ) - cos(φ)*sin(δ)*cos(H))
    sin_q = np.sin(ha_rad) * np.cos(lat_rad)
    cos_q = (
        np.sin(lat_rad) * np.cos(dec_rad)
        - np.cos(lat_rad) * np.sin(dec_rad) * np.cos(ha_rad)
    )
    parallactic_angle_deg = np.degrees(np.arctan2(sin_q, cos_q))

    # Object rate: approximate from RA/Dec change between adjacent observations.
    # Use finite differences; endpoints use one-sided differences.
    if len(ra_deg) >= 2:
        dt_hours = np.gradient(obstime_mjd) * 24.0
        dra = np.gradient(ra_deg) * np.cos(dec_rad) * 3600.0  # arcsec
        ddec = np.gradient(dec_deg) * 3600.0  # arcsec
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.sqrt(dra**2 + ddec**2) / np.where(
                np.abs(dt_hours) > 1e-10, np.abs(dt_hours), np.nan
            )
        rate = np.where(np.isfinite(rate), rate, 0.0)
    else:
        rate = np.zeros_like(ra_deg)

    return zenith_angle_deg, parallactic_angle_deg, rate


# ---------------------------------------------------------------------------
# Truth ephemeris generation
# ---------------------------------------------------------------------------


def generate_truth_ephemeris(
    truth_orbit: "Orbits",  # noqa: F821
    obs_template: "MPCObservations",  # noqa: F821
    real_codes: List[str],
    propagator: "Propagator",  # noqa: F821
    cache_dir: Optional[Path] = None,
    object_id: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Propagate *truth_orbit* to the observation times of each real observatory.

    For each real_code that appears in *obs_template*, propagates the orbit
    to all observation times at that site and returns topocentric RA/Dec in
    degrees.

    Parameters
    ----------
    truth_orbit : Orbits
        Single-object truth orbit (len == 1).
    obs_template : MPCObservations
        Template observations providing times and station codes.
    real_codes : list of str
        Real MPC station codes to generate ephemerides for.
    propagator : Propagator
        Propagator instance (e.g. TwoBodyPropagator or ASSISTPropagator).
    cache_dir : Path, optional
        If provided, reads/writes per-object parquet cache at
        ``{cache_dir}/{object_id}_truth_ephemeris.parquet``.
    object_id : str, optional
        Object identifier for cache file naming.

    Returns
    -------
    dict : real_code -> (ra_deg, dec_deg, obstime_mjd)
        Arrays of true topocentric positions at each observation time.
    """
    from adam_core.observers.observers import Observers
    from adam_core.time import Timestamp

    # ------------------------------------------------------------------
    # Cache load
    # ------------------------------------------------------------------
    cache_path: Optional[Path] = None
    if cache_dir is not None and object_id is not None:
        cache_path = cache_dir / f"{object_id.replace('/', '_')}_truth_ephemeris.parquet"
        if cache_path.exists():
            logger.debug(f"Loading truth ephemeris from cache: {cache_path}")
            cached = pq.read_table(cache_path).to_pydict()
            result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            cached_codes = np.array(cached["real_code"])
            for rc in real_codes:
                mask = cached_codes == rc
                if mask.any():
                    result[rc] = (
                        np.array(cached["ra_true_deg"])[mask],
                        np.array(cached["dec_true_deg"])[mask],
                        np.array(cached["obstime_mjd"])[mask],
                    )
            if result:
                return result

    # ------------------------------------------------------------------
    # Build per-code ephemerides
    # ------------------------------------------------------------------
    result = {}
    template_stns = obs_template.stn.to_numpy(zero_copy_only=False).astype(str)
    template_times_mjd = _extract_mjd(obs_template.obstime)

    all_obsids = obs_template.obsid.to_numpy(zero_copy_only=False).astype(str)

    cache_rows: Dict[str, list] = {
        "obsid": [],
        "real_code": [],
        "ra_true_deg": [],
        "dec_true_deg": [],
        "obstime_mjd": [],
    }

    for real_code in real_codes:
        mask = template_stns == real_code
        if not mask.any():
            logger.debug(f"No observations for real code {real_code!r} in template.")
            continue

        mjds = template_times_mjd[mask]
        obsids = all_obsids[mask]

        # Build Timestamp for these observations
        times = Timestamp.from_mjd(mjds, scale="utc")

        codes_arr = pa.array(
            [real_code] * len(mjds), type=pa.large_utf8()
        )
        observers = Observers.from_codes(codes=codes_arr, times=times)

        try:
            ephem = propagator.generate_ephemeris(truth_orbit, observers)
        except Exception as exc:
            logger.error(
                f"generate_ephemeris failed for {real_code!r}: {exc}"
            )
            continue

        # Extract RA/Dec from SphericalCoordinates
        ra_deg = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(float)
        dec_deg = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(float)

        result[real_code] = (ra_deg, dec_deg, mjds)

        # Accumulate for cache
        cache_rows["obsid"].extend(obsids.tolist())
        cache_rows["real_code"].extend([real_code] * len(mjds))
        cache_rows["ra_true_deg"].extend(ra_deg.tolist())
        cache_rows["dec_true_deg"].extend(dec_deg.tolist())
        cache_rows["obstime_mjd"].extend(mjds.tolist())

    # ------------------------------------------------------------------
    # Cache write
    # ------------------------------------------------------------------
    if cache_path is not None and cache_rows["obsid"]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tbl = pa.table(
            {
                "obsid": pa.array(cache_rows["obsid"], type=pa.large_utf8()),
                "real_code": pa.array(cache_rows["real_code"], type=pa.large_utf8()),
                "ra_true_deg": pa.array(cache_rows["ra_true_deg"], type=pa.float64()),
                "dec_true_deg": pa.array(cache_rows["dec_true_deg"], type=pa.float64()),
                "obstime_mjd": pa.array(cache_rows["obstime_mjd"], type=pa.float64()),
            }
        )
        pq.write_table(tbl, cache_path)
        logger.debug(f"Truth ephemeris cached to {cache_path}")

    return result


# ---------------------------------------------------------------------------
# Full synthetic observation pipeline
# ---------------------------------------------------------------------------


def generate_synthetic_observations(
    truth_orbit: "Orbits",  # noqa: F821
    obs_template: "MPCObservations",  # noqa: F821
    observatory_map: "ObservatoryMap",  # noqa: F821
    propagator: "Propagator",  # noqa: F821
    noise_seed: int = 42,
    cache_dir: Optional[Path] = None,
    object_id: Optional[str] = None,
) -> "MPCObservations":  # noqa: F821
    """
    Generate synthetic ``MPCObservations`` from a truth orbit and template.

    Pipeline
    --------
    1. For each observation in *obs_template*, look up its real station code
       in *observatory_map*; skip observations whose code has no fake assignment.
    2. Propagate the truth orbit to all observation times (cached per object).
    3. Apply Gaussian noise using each fake observatory's noise sigmas.
    4. Apply systematic bias models.
    5. Build an ``MPCObservations`` table with the same schema as real data.

    Parameters
    ----------
    truth_orbit : Orbits
        Single-object truth orbit (len == 1).
    obs_template : MPCObservations
        Template observations (provide times, stations, magnitudes, etc.).
    observatory_map : ObservatoryMap
        Mapping from real → fake station codes with noise/bias parameters.
    propagator : Propagator
        Propagator instance for ephemeris generation.
    noise_seed : int
        RNG seed for reproducible Gaussian noise draws.
    cache_dir : Path, optional
        Directory for truth ephemeris parquet cache.
    object_id : str, optional
        Object identifier used for cache file naming and obs ID construction.

    Returns
    -------
    MPCObservations
        Synthetic observations table with fake station codes, injected noise
        and bias, and explicit rmsra/rmsdec uncertainties.
    """
    from mpcq.observations import MPCObservations

    rng = np.random.default_rng(noise_seed)

    # ------------------------------------------------------------------
    # 1. Filter template to mapped stations only
    # ------------------------------------------------------------------
    template_stns = obs_template.stn.to_numpy(zero_copy_only=False).astype(str)
    mapped_real = set(observatory_map.all_real_codes())

    keep_mask = np.array([s in mapped_real for s in template_stns])
    if not keep_mask.any():
        logger.warning("No template observations matched any mapped real code.")
        return MPCObservations.empty()

    # Work with filtered slice
    filtered = obs_template.apply_mask(pa.array(keep_mask))
    filtered_stns = template_stns[keep_mask]

    # ------------------------------------------------------------------
    # 2. Generate truth ephemeris for each real code
    # ------------------------------------------------------------------
    real_codes_needed = list(set(filtered_stns.tolist()))
    truth_cache = generate_truth_ephemeris(
        truth_orbit=truth_orbit,
        obs_template=filtered,
        real_codes=real_codes_needed,
        propagator=propagator,
        cache_dir=cache_dir,
        object_id=object_id,
    )

    # ------------------------------------------------------------------
    # 3. Build per-observation arrays
    # ------------------------------------------------------------------
    n_filtered = len(filtered)
    filtered_mjds = _extract_mjd(filtered.obstime)
    filtered_mags = filtered.mag.to_numpy(zero_copy_only=False).astype(float)

    out_ra = np.full(n_filtered, np.nan)
    out_dec = np.full(n_filtered, np.nan)
    out_rmsra = np.full(n_filtered, np.nan)
    out_rmsdec = np.full(n_filtered, np.nan)
    out_fake_stn = np.empty(n_filtered, dtype=object)
    out_astcat = np.empty(n_filtered, dtype=object)
    valid = np.zeros(n_filtered, dtype=bool)

    # Map each observation to its truth position
    for rc in real_codes_needed:
        if rc not in truth_cache:
            continue
        fobs = observatory_map.fake_for_real(rc)
        if fobs is None:
            continue

        rc_mask = filtered_stns == rc
        rc_indices = np.where(rc_mask)[0]

        ra_true, dec_true, ephem_mjds = truth_cache[rc]

        # Verify alignment: ephem_mjds should match filtered_mjds[rc_mask]
        expected_mjds = filtered_mjds[rc_mask]
        if len(ra_true) != len(expected_mjds):
            logger.error(
                f"Ephemeris length mismatch for {rc!r}: "
                f"{len(ra_true)} vs {len(expected_mjds)} template obs."
            )
            continue

        # Noise draw
        sigma_ra_arr = np.full(len(rc_indices), fobs.noise_sigma_ra)
        sigma_dec_arr = np.full(len(rc_indices), fobs.noise_sigma_dec)
        noise_ra, noise_dec = rng.normal(0.0, sigma_ra_arr), rng.normal(0.0, sigma_dec_arr)

        # Bias apply
        obs_mag = filtered_mags[rc_mask]
        bias_model = fobs.compound_bias

        # Compute observing geometry for this site so zenith-angle-dependent
        # bias models (DCRBias, RefractionModelError, TrailingBias) receive
        # real values instead of None.
        site_geodetic = _get_site_geodetic(rc)
        if site_geodetic is not None:
            lat_deg, lon_deg = site_geodetic
            zenith_ang, parallactic_ang, obj_rate = _compute_obs_geometry(
                ra_true, dec_true, ephem_mjds, lat_deg, lon_deg
            )
        else:
            zenith_ang = parallactic_ang = obj_rate = None

        d_ra_cosdec, d_dec = bias_model.apply(
            ra=ra_true,
            dec=dec_true,
            obstime=ephem_mjds,
            mag=obs_mag if np.any(np.isfinite(obs_mag)) else None,
            zenith_angle=zenith_ang,
            parallactic_angle=parallactic_ang,
            object_rate=obj_rate,
        )

        # Total offset in arcsec (noise + bias), both in cos(dec)-corrected frame
        total_ra_offset_arcsec = noise_ra + d_ra_cosdec  # Δα·cos(δ) in arcsec
        total_dec_offset_arcsec = noise_dec + d_dec       # Δδ in arcsec

        # Convert back to degrees
        dec_rad = np.deg2rad(dec_true)
        cos_dec = np.cos(dec_rad)
        # Guard against dec ≈ ±90°
        safe_cos = np.where(np.abs(cos_dec) > 1e-10, cos_dec, 1e-10)

        syn_ra = ra_true + (total_ra_offset_arcsec / safe_cos) / 3600.0
        syn_dec = dec_true + total_dec_offset_arcsec / 3600.0

        # Wrap RA to [0, 360)
        syn_ra = syn_ra % 360.0

        out_ra[rc_indices] = syn_ra
        out_dec[rc_indices] = syn_dec
        out_rmsra[rc_indices] = sigma_ra_arr  # sigma_ra is cos-dec corrected
        out_rmsdec[rc_indices] = sigma_dec_arr
        out_fake_stn[rc_indices] = fobs.fake_code
        out_astcat[rc_indices] = fobs.astcat
        valid[rc_indices] = True

    # Drop observations for which we have no ephemeris
    if not valid.all():
        n_dropped = (~valid).sum()
        logger.warning(
            f"Dropping {n_dropped} template observations with no truth ephemeris."
        )
        filtered = filtered.apply_mask(pa.array(valid))
        out_ra = out_ra[valid]
        out_dec = out_dec[valid]
        out_rmsra = out_rmsra[valid]
        out_rmsdec = out_rmsdec[valid]
        out_fake_stn = out_fake_stn[valid]
        out_astcat = out_astcat[valid]

    if len(out_ra) == 0:
        return MPCObservations.empty()

    # ------------------------------------------------------------------
    # 4. Build MPCObservations output table
    # ------------------------------------------------------------------
    n_out = len(out_ra)

    # Generate unique obs IDs
    obj_safe = (object_id or "obj").replace("/", "_").replace(" ", "_")
    obsids = pa.array(
        [f"{out_fake_stn[i]}_{obj_safe}_{i}" for i in range(n_out)],
        type=pa.large_utf8(),
    )

    # Copy passthrough columns from the filtered template
    def _col(name: str) -> pa.Array:
        return filtered.table.column(name)

    def _str_col(name: str) -> pa.Array:
        arr = _col(name)
        # Ensure large_utf8 type
        if arr.type != pa.large_utf8():
            arr = arr.cast(pa.large_utf8())
        return arr

    # Build the timestamp struct for updated_at / created_at using current time
    now_mjd = _utc_now_mjd()
    now_days = int(now_mjd)
    now_nanos = int((now_mjd - now_days) * 86400.0 * 1e9)
    now_ts_struct = pa.array(
        [{"days": now_days, "nanos": now_nanos}] * n_out,
        type=pa.struct([("days", pa.int64()), ("nanos", pa.int64())]),
    )

    try:
        table = pa.table(
            {
                "requested_provid": _str_col("requested_provid"),
                "primary_designation": _str_col("primary_designation"),
                "obsid": obsids,
                "trksub": _str_col("trksub"),
                "provid": _str_col("provid"),
                "permid": _str_col("permid"),
                "submission_id": _str_col("submission_id"),
                "obssubid": _str_col("obssubid"),
                "obstime": _col("obstime"),
                "ra": pa.array(out_ra, type=pa.float64()),
                "dec": pa.array(out_dec, type=pa.float64()),
                "rmsra": pa.array(out_rmsra, type=pa.float64()),
                "rmsdec": pa.array(out_rmsdec, type=pa.float64()),
                "rmscorr": pa.array(np.zeros(n_out), type=pa.float64()),
                "mag": _col("mag"),
                "rmsmag": _col("rmsmag"),
                "band": _str_col("band"),
                "stn": pa.array(out_fake_stn.tolist(), type=pa.large_utf8()),
                "updated_at": now_ts_struct,
                "created_at": now_ts_struct,
                "status": pa.array(["P"] * n_out, type=pa.large_utf8()),
                "astcat": pa.array(out_astcat.tolist(), type=pa.large_utf8()),
                "mode": pa.array(["CCD"] * n_out, type=pa.large_utf8()),
            }
        )
    except Exception as exc:
        logger.error(f"Failed to build output MPCObservations table: {exc}")
        raise

    return MPCObservations(table)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_mjd(obstime_column) -> np.ndarray:
    """
    Extract MJD floats from an MPCObservations obstime column.

    The obstime column in mpcq is a struct<days:int64, nanos:int64>
    representing a Timestamp.  We convert to MJD (float64).
    """
    # Try the Timestamp quivr approach (adam_core)
    try:
        # If it is already a Timestamp-like object with .mjd()
        return obstime_column.mjd().to_numpy(zero_copy_only=False).astype(float)
    except AttributeError:
        pass

    # Try treating it as a struct array {days, nanos}
    try:
        tbl = obstime_column
        if hasattr(tbl, "to_pandas"):
            df = tbl.to_pandas()
            days = df["days"].values.astype(float)
            nanos = df["nanos"].values.astype(float)
            # MJD = days + nanos/1e9/86400
            return days + nanos / 1e9 / 86400.0
        # pyarrow struct array
        days = pc.struct_field(tbl, "days").to_numpy(zero_copy_only=False).astype(float)
        nanos = pc.struct_field(tbl, "nanos").to_numpy(zero_copy_only=False).astype(float)
        return days + nanos / 1e9 / 86400.0
    except Exception:
        pass

    raise TypeError(
        f"Cannot extract MJD from obstime column of type {type(obstime_column)}"
    )


def _utc_now_mjd() -> float:
    """Return current UTC time as MJD float."""
    import time
    # Unix epoch to MJD: MJD = (unix_sec / 86400) + 40587.0
    return time.time() / 86400.0 + 40587.0
