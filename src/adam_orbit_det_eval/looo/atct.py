"""
Along-Track / Cross-Track (AT/CT) decomposition for real-data LOOO results.

Computes sky-plane velocity vectors from original MPC catalog orbits via
generate_ephemeris(), then projects RA/Dec residuals into along-track and
cross-track components.

This is a POST-PROCESSING step on existing LOOO results.  It does not
modify the core LOOO pipeline or LOOOResult schema.

Along-track  (AT) = residual component in the direction of apparent motion.
Cross-track  (CT) = residual component perpendicular to motion (90 deg CCW).

Velocity source: the ORIGINAL MPC catalog orbit (not the refitted orbit),
computed analytically via generate_ephemeris (vlon/vlat fields).

Unit note
---------
Ephemeris vlon is d(lon)/dt in deg/day — NOT multiplied by cos(dec).
For AT/CT decomposition we need:
    v_ra_cosdec = vlon * cos(dec)
    v_dec       = vlat
The rotate function expects v_ra in the cos(dec)-corrected frame, matching
residual_ra_arcsec which is RA*cos(dec).
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.propagator.propagator import Propagator
from adam_core.time import Timestamp

logger = logging.getLogger(__name__)

# Minimum sky-plane speed (deg/day) below which AT/CT is undefined.
# Objects moving slower than this are near-stationary and the velocity
# direction is noise-dominated.  1e-6 deg/day ~ 0.0036 arcsec/day.
MIN_SPEED_DEG_PER_DAY = 1e-6


def rotate_to_atct(
    residual_ra: np.ndarray,
    residual_dec: np.ndarray,
    v_ra_unit: np.ndarray,
    v_dec_unit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate RA/Dec residuals into along-track / cross-track components.

    Parameters
    ----------
    residual_ra : ndarray
        Residuals in RA*cos(dec) direction (arcsec).
    residual_dec : ndarray
        Residuals in Dec direction (arcsec).
    v_ra_unit : ndarray
        Unit vector component along RA*cos(dec) of sky-plane velocity.
    v_dec_unit : ndarray
        Unit vector component along Dec of sky-plane velocity.

    Returns
    -------
    (at, ct) : tuple of ndarray
        Along-track and cross-track residuals in arcsec.
        NaN where the velocity unit vector is zero (near-stationary).
    """
    at = residual_ra * v_ra_unit + residual_dec * v_dec_unit
    ct = -residual_ra * v_dec_unit + residual_dec * v_ra_unit
    return at, ct


def compute_velocity_vectors(
    orbits: Orbits,
    obs_ids: np.ndarray,
    obs_object_ids: np.ndarray,
    obs_times: Timestamp,
    obs_codes: np.ndarray,
    propagator: Propagator,
) -> pd.DataFrame:
    """
    Compute sky-plane velocity unit vectors for a set of observations.

    For each unique object, calls generate_ephemeris on the original orbit
    to get analytically-computed vlon/vlat, then converts to the cos(dec)-
    corrected frame and normalizes.

    Parameters
    ----------
    orbits : Orbits
        Original MPC catalog orbits (one per object).
    obs_ids : ndarray of str
        Observation IDs to compute velocities for.
    obs_object_ids : ndarray of str
        Object ID for each observation (parallel to obs_ids).
    obs_times : Timestamp
        Observation times (parallel to obs_ids).
    obs_codes : ndarray of str
        MPC observatory codes (parallel to obs_ids).
    propagator : Propagator
        Propagator instance for ephemeris generation.

    Returns
    -------
    DataFrame
        Columns: obs_id, v_ra_unit, v_dec_unit, speed_deg_per_day.
        One row per input observation.  Near-stationary rows have
        v_ra_unit=NaN, v_dec_unit=NaN.
    """
    unique_objects = np.unique(obs_object_ids)

    # Build orbit lookup: object_id -> index in orbits table
    orbit_object_ids = orbits.object_id.to_numpy(zero_copy_only=False)

    # Pre-allocate output arrays
    v_ra_unit_out = np.full(len(obs_ids), np.nan)
    v_dec_unit_out = np.full(len(obs_ids), np.nan)
    speed_out = np.full(len(obs_ids), np.nan)

    n_missing_orbits = 0

    for obj_id in unique_objects:
        # Find the orbit for this object
        orbit_mask = orbit_object_ids == obj_id
        if not np.any(orbit_mask):
            n_missing_orbits += 1
            logger.warning(f"No orbit found for object {obj_id}, skipping")
            continue

        orbit_idx = np.where(orbit_mask)[0][0]
        orbit = orbits.take(pa.array([orbit_idx]))

        # Find all observations for this object
        obs_mask = obs_object_ids == obj_id
        obs_indices = np.where(obs_mask)[0]

        if len(obs_indices) == 0:
            continue

        # Build observers for this object's observation times
        obj_times = obs_times.take(pa.array(obs_indices))
        obj_codes = obs_codes[obs_indices]

        try:
            observers = Observers.from_codes(
                codes=pa.array(obj_codes, type=pa.large_string()),
                times=obj_times,
            )

            ephemeris = propagator.generate_ephemeris(
                orbits=orbit,
                observers=observers,
                max_processes=1,
            )
        except Exception as e:
            logger.warning(
                f"generate_ephemeris failed for {obj_id} "
                f"({len(obs_indices)} obs): {e}"
            )
            continue

        # Extract velocity components from ephemeris
        # vlon is d(lon)/dt in deg/day (NOT multiplied by cos(dec))
        # vlat is d(lat)/dt in deg/day
        vlon = ephemeris.coordinates.vlon.to_numpy(zero_copy_only=False)
        vlat = ephemeris.coordinates.vlat.to_numpy(zero_copy_only=False)
        dec_deg = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
        dec_rad = np.deg2rad(dec_deg)

        # Convert to cos(dec)-corrected frame
        v_ra_cosdec = vlon * np.cos(dec_rad)  # deg/day
        v_dec = vlat  # deg/day

        # Compute speed and unit vector
        speed = np.sqrt(v_ra_cosdec**2 + v_dec**2)  # deg/day

        # Guard against near-stationary objects
        stationary = speed < MIN_SPEED_DEG_PER_DAY
        v_ra_u = np.where(stationary, np.nan, v_ra_cosdec / speed)
        v_dec_u = np.where(stationary, np.nan, v_dec / speed)
        speed_safe = np.where(stationary, np.nan, speed)

        if np.any(stationary):
            n_stationary = int(np.sum(stationary))
            logger.debug(
                f"{obj_id}: {n_stationary}/{len(obs_indices)} near-stationary obs"
            )

        # Store results — ephemeris rows should correspond 1:1 with obs_indices
        # since we built observers in the same order
        v_ra_unit_out[obs_indices] = v_ra_u
        v_dec_unit_out[obs_indices] = v_dec_u
        speed_out[obs_indices] = speed_safe

    if n_missing_orbits > 0:
        logger.warning(
            f"{n_missing_orbits} objects not found in orbits file, skipped"
        )

    return pd.DataFrame({
        "obs_id": obs_ids,
        "v_ra_unit": v_ra_unit_out,
        "v_dec_unit": v_dec_unit_out,
        "speed_deg_per_day": speed_out,
    })


def augment_looo_results_with_atct(
    looo_df: pd.DataFrame,
    orbits: Orbits,
    obs_times: Timestamp,
    obs_ids: np.ndarray,
    obs_object_ids: np.ndarray,
    obs_codes: np.ndarray,
    propagator: Propagator,
) -> pd.DataFrame:
    """
    End-to-end AT/CT augmentation of LOOO results.

    Takes a DataFrame of LOOO results (with residual_ra_arcsec,
    residual_dec_arcsec, obs_id, object_id), computes velocity vectors
    from the original orbits, rotates residuals, and returns the
    augmented DataFrame with additional columns.

    Parameters
    ----------
    looo_df : DataFrame
        LOOO results with columns: obs_id, object_id, residual_ra_arcsec,
        residual_dec_arcsec.
    orbits : Orbits
        Original MPC catalog orbits.
    obs_times : Timestamp
        Observation times for ALL observations (not just LOOO results).
    obs_ids : ndarray
        Observation IDs parallel to obs_times.
    obs_object_ids : ndarray
        Object IDs parallel to obs_times.
    obs_codes : ndarray
        Observatory codes parallel to obs_times.
    propagator : Propagator
        Propagator instance.

    Returns
    -------
    DataFrame
        Input DataFrame with added columns: residual_at_arcsec,
        residual_ct_arcsec, speed_deg_per_day, v_ra_unit, v_dec_unit.
    """
    # Build lookup from obs_id -> index in the observation arrays
    obs_id_to_idx = {oid: i for i, oid in enumerate(obs_ids)}

    # Find the indices in the observation arrays that correspond to LOOO result obs_ids
    looo_obs_ids = looo_df["obs_id"].values
    matched_indices = []
    unmatched = 0
    for oid in looo_obs_ids:
        idx = obs_id_to_idx.get(oid)
        if idx is not None:
            matched_indices.append(idx)
        else:
            matched_indices.append(-1)
            unmatched += 1

    if unmatched > 0:
        logger.warning(
            f"{unmatched}/{len(looo_obs_ids)} LOOO obs_ids not found in observations"
        )

    matched_indices = np.array(matched_indices)
    valid_mask = matched_indices >= 0

    # Extract times/codes for matched observations
    valid_indices = matched_indices[valid_mask]
    valid_obs_ids_arr = looo_obs_ids[valid_mask]
    valid_object_ids = obs_object_ids[valid_indices]
    valid_times = obs_times.take(pa.array(valid_indices))
    valid_codes = obs_codes[valid_indices]

    # Compute velocity vectors
    vel_df = compute_velocity_vectors(
        orbits=orbits,
        obs_ids=valid_obs_ids_arr,
        obs_object_ids=valid_object_ids,
        obs_times=valid_times,
        obs_codes=valid_codes,
        propagator=propagator,
    )

    # Merge velocity vectors back into LOOO results
    vel_lookup = vel_df.set_index("obs_id")

    v_ra_unit = np.full(len(looo_df), np.nan)
    v_dec_unit = np.full(len(looo_df), np.nan)
    speed = np.full(len(looo_df), np.nan)

    for i, oid in enumerate(looo_obs_ids):
        if oid in vel_lookup.index:
            row = vel_lookup.loc[oid]
            v_ra_unit[i] = row["v_ra_unit"]
            v_dec_unit[i] = row["v_dec_unit"]
            speed[i] = row["speed_deg_per_day"]

    # Rotate residuals
    res_ra = looo_df["residual_ra_arcsec"].values.astype(float)
    res_dec = looo_df["residual_dec_arcsec"].values.astype(float)

    at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

    # Build output
    result = looo_df.copy()
    result["residual_at_arcsec"] = at
    result["residual_ct_arcsec"] = ct
    result["speed_deg_per_day"] = speed
    result["v_ra_unit"] = v_ra_unit
    result["v_dec_unit"] = v_dec_unit

    n_valid_atct = int(np.isfinite(at).sum())
    logger.info(
        f"AT/CT computed for {n_valid_atct}/{len(looo_df)} observations "
        f"({100 * n_valid_atct / max(len(looo_df), 1):.1f}%)"
    )

    return result
