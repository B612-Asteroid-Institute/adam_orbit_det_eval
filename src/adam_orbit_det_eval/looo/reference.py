"""
Reference-orbit evaluation for orbit determination quality assessment.

This module implements the inverse of LOOO: instead of leaving one observatory
out and fitting with the rest, we fit the orbit using ONLY a designated
reference station's observations, then evaluate how well that reference orbit
predicts observations from every other station.

Motivation
----------
LOOO asks "how well can the rest of the network predict this station?"
Reference-orbit evaluation asks "how well does a trusted, dense anchor station
predict the broader network?"  For LSST (X05), which will produce tens to
hundreds of observations per object, this is the natural calibration direction:
use the X05 orbit as ground truth and measure systematic errors at other
stations relative to it.

Output schema
-------------
The output is identical to ``LOOOResult`` so the existing ``03_analyze.py``
notebook works unchanged.  Column semantics are re-mapped as follows:

  stn                      — the non-reference station being evaluated
  n_obs_remaining          — number of reference-station observations used
                             for the fit (equivalent to "hold-in" count)
  arc_length_remaining_days — arc length of reference-station observations
  held_out_fraction        — n_stn_obs / (n_ref_obs + n_stn_obs)
                             (fraction of the combined set that is "held out")
  hold_in_reduced_chi2     — reduced chi2 of the reference-station-only fit
                             (same for all evaluated stations on a given object)
  n_obs_held_out           — number of observations at the evaluated station
  delta_q_au / delta_e / delta_i_deg
                           — orbital element difference between the full MPC
                             reference orbit and the reference-station-only fit

Eligibility
-----------
A (object, station) pair is evaluated when ALL of the following hold:

  - The reference station has >= min_ref_obs observations for this object
    (ensures the reference orbit is well-determined before we trust it)
  - The evaluated station has >= min_obs_held_out observations
    (avoids single-observation noise dominance in statistics)

The arc-length and held-out-fraction filters from LOOOConfig are NOT applied
here because the reference-orbit fit quality is controlled by min_ref_obs alone.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv

from adam_core.orbit_determination.differential_correction import fit_least_squares
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbit_determination.fitted_orbits import FittedOrbits
from adam_core.orbits.orbits import Orbits
from adam_core.propagator.propagator import Propagator

from .core import LOOOConfig, LOOOResult, _arc_length_days, _orbit_delta
from .pipeline import _checkpoint_path, _load_completed_ids, _propagator_fqn, merge_checkpoints
from ..utils import get_spacebased_stns, mpc_to_od_observations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-object reference-orbit function
# ---------------------------------------------------------------------------


def run_reference_orbit_for_object(
    object_id: str,
    observations: OrbitDeterminationObservations,
    reference_orbit: Orbits,
    propagator: Propagator,
    config: Optional[LOOOConfig] = None,
    astcats: Optional[List[Optional[str]]] = None,
    reference_stn: str = "X05",
    min_ref_obs: int = 20,
) -> LOOOResult:
    """
    Fit an orbit using only the reference station's observations, then evaluate
    all other stations against that orbit.

    One DC fit is performed per object (using only ``reference_stn`` obs).
    The resulting orbit is then used to predict at every non-reference station's
    observation times, and residuals are collected.

    Parameters
    ----------
    object_id : str
        Object identifier used for labeling in output rows.
    observations : OrbitDeterminationObservations
        Full observation set for the object (all stations combined).
    reference_orbit : Orbits
        Starting orbit for differential correction (e.g., the MPC nominal orbit).
        Should be a single orbit (len == 1).
    propagator : Propagator
        Propagator instance used for ephemeris generation and DC.
    config : LOOOConfig, optional
        Configuration object.  Only ``ls_kwargs`` and ``min_obs_held_out`` are
        used here (arc/fraction filters are replaced by ``min_ref_obs``).
        Defaults to LOOOConfig().
    astcats : list of str or None, optional
        Astrometric catalog codes parallel to observations rows.  If provided,
        stored in the output for per-catalog downstream analysis.
    reference_stn : str
        MPC station code for the anchor / reference station.  Default ``"X05"``
        (Rubin Observatory / LSST).
    min_ref_obs : int
        Minimum number of reference-station observations required to perform
        the fit.  If the reference station has fewer observations than this
        threshold, the function returns an empty table.

    Returns
    -------
    LOOOResult
        One row per observation at a non-reference station that passed
        eligibility filters.  Empty table if the reference station does not
        have enough observations or the DC fit fails.
    """
    if config is None:
        config = LOOOConfig()

    all_stns = observations.coordinates.origin.code.to_numpy(zero_copy_only=False)
    all_obs_ids = observations.id.to_numpy(zero_copy_only=False)
    n_obs_total = len(observations)

    # --- Identify reference station observations ---
    ref_mask = all_stns == reference_stn
    n_ref_obs = int(ref_mask.sum())

    if n_ref_obs < min_ref_obs:
        logger.debug(
            f"{object_id}: reference station {reference_stn} has only {n_ref_obs} obs "
            f"(need >= {min_ref_obs}), skipping"
        )
        return LOOOResult.empty()

    ref_obs = observations.apply_mask(pa.array(ref_mask))
    arc_ref_days = _arc_length_days(ref_obs)

    # --- Fit orbit using only reference station observations ---
    try:
        ref_fit_orbit, ref_fit_members = fit_least_squares(
            reference_orbit,
            ref_obs,
            propagator,
            **config.ls_kwargs,
        )
    except Exception as e:
        logger.warning(f"{object_id}: DC on {reference_stn} obs failed: {e}")
        return LOOOResult.empty()

    if len(ref_fit_orbit) == 0:
        logger.debug(f"{object_id}: DC on {reference_stn} returned no orbit, skipping")
        return LOOOResult.empty()

    # Cache the reference-orbit chi2 — it is the same for all evaluated stations
    if ref_fit_orbit.reduced_chi2[0].is_valid:
        ref_reduced_chi2 = float(ref_fit_orbit.reduced_chi2[0].as_py())
    else:
        ref_reduced_chi2 = np.nan

    ref_fit_success = (
        ref_fit_orbit.success[0].as_py()
        if ref_fit_orbit.success[0].is_valid
        else None
    )

    # Orbital element deltas (reference MPC orbit → reference-station-only fit)
    dq, de, di = _orbit_delta(reference_orbit, ref_fit_orbit)

    # --- Evaluate every non-reference station ---
    unique_stns = np.unique(all_stns)
    results: List[LOOOResult] = []

    for stn in unique_stns:
        if stn == reference_stn:
            continue

        stn_mask = all_stns == stn
        n_stn_obs = int(stn_mask.sum())

        if n_stn_obs < config.min_obs_held_out:
            logger.debug(
                f"{object_id} / {stn}: only {n_stn_obs} obs (need >= "
                f"{config.min_obs_held_out}), skipping"
            )
            continue

        stn_obs = observations.apply_mask(pa.array(stn_mask))
        held_out_fraction = n_stn_obs / (n_ref_obs + n_stn_obs)

        # --- Predict at the evaluated station's observation times ---
        try:
            _, stn_eval_members = evaluate_orbits(
                ref_fit_orbit.to_orbits(),
                stn_obs,
                propagator,
                parameters=6,
            )
        except Exception as e:
            logger.warning(
                f"{object_id} / {stn}: evaluate_orbits failed: {e}"
            )
            continue

        # --- Collect residuals ---
        # residuals.values layout: [rho, ra, dec, vrho, vra, vdec]
        # indices 1 and 2 are RA and Dec residuals in degrees
        residual_array = stn_eval_members.residuals.to_array()  # (N, 6)
        res_ra_arcsec = residual_array[:, 1] * 3600.0   # cos-dec correction applied by Residuals
        res_dec_arcsec = residual_array[:, 2] * 3600.0

        # --- Per-observation sigma from the evaluated station observations ---
        cov_matrix = stn_obs.coordinates.covariance.to_matrix()  # (N, 6, 6)
        sigma_ra_deg = np.sqrt(np.abs(cov_matrix[:, 1, 1]))
        sigma_dec_deg = np.sqrt(np.abs(cov_matrix[:, 2, 2]))
        dec_deg = stn_obs.coordinates.lat.to_numpy(zero_copy_only=False)
        cos_dec = np.cos(np.deg2rad(dec_deg))
        sigma_ra_cosdec_arcsec = sigma_ra_deg * cos_dec * 3600.0
        sigma_dec_arcsec_vals = sigma_dec_deg * 3600.0

        chi2_vals = stn_eval_members.residuals.chi2.to_numpy(zero_copy_only=False)

        # --- astcat values for evaluated station observations (if provided) ---
        if astcats is not None:
            stn_astcats = [astcats[i] for i, m in enumerate(stn_mask) if m]
        else:
            stn_astcats = [None] * n_stn_obs

        stn_obs_ids = all_obs_ids[stn_mask]

        result = LOOOResult.from_kwargs(
            object_id=np.full(n_stn_obs, object_id, dtype=object),
            obs_id=stn_obs_ids,
            stn=np.full(n_stn_obs, stn, dtype=object),
            residual_ra_arcsec=res_ra_arcsec,
            residual_dec_arcsec=res_dec_arcsec,
            sigma_ra_cosdec_arcsec=np.where(
                np.isfinite(sigma_ra_cosdec_arcsec), sigma_ra_cosdec_arcsec, None
            ),
            sigma_dec_arcsec=np.where(
                np.isfinite(sigma_dec_arcsec_vals), sigma_dec_arcsec_vals, None
            ),
            chi2=np.where(np.isfinite(chi2_vals), chi2_vals, None),
            astcat=pa.array(stn_astcats, type=pa.large_utf8()),
            # n_obs_held_out = observations at the evaluated station
            n_obs_held_out=np.full(n_stn_obs, n_stn_obs, dtype=np.int64),
            # n_obs_remaining = reference-station observations used for fit
            n_obs_remaining=np.full(n_stn_obs, n_ref_obs, dtype=np.int64),
            # arc_length_remaining = arc of reference-station observations
            arc_length_remaining_days=np.full(n_stn_obs, arc_ref_days),
            # held_out_fraction = stn_obs / (ref_obs + stn_obs)
            held_out_fraction=np.full(n_stn_obs, held_out_fraction),
            n_obs_total=np.full(n_stn_obs, n_obs_total, dtype=np.int64),
            delta_q_au=np.full(n_stn_obs, dq if dq is not None else np.nan),
            delta_e=np.full(n_stn_obs, de if de is not None else np.nan),
            delta_i_deg=np.full(n_stn_obs, di if di is not None else np.nan),
            # hold_in_reduced_chi2 = chi2 of the reference-station-only fit
            # (same constant for all evaluated stations on this object)
            hold_in_reduced_chi2=np.full(n_stn_obs, ref_reduced_chi2),
            hold_in_fit_success=np.full(n_stn_obs, ref_fit_success),
        )
        results.append(result)
        logger.info(
            f"{object_id} / {stn}: {n_stn_obs} evaluated obs, "
            f"chi2/obs={float(np.nanmean(chi2_vals)):.2f}"
        )

    if not results:
        return LOOOResult.empty()

    combined = qv.concatenate(results)
    if combined.fragmented():
        combined = qv.defragment(combined)
    return combined


# ---------------------------------------------------------------------------
# Per-object worker — module-level so it is picklable by ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _ref_worker(
    object_id: str,
    obs_parquet: str,
    orbits_parquet: str,
    propagator_class_fqn: str,
    config: LOOOConfig,
    checkpoint_dir: str,
    sigma_model: str = "veres2017",
    reference_stn: str = "X05",
    min_ref_obs: int = 20,
) -> Tuple[str, int]:
    """
    Worker function executed in a subprocess for the reference-orbit pipeline.

    Loads its own data slice from disk so large tables do not need to be
    pickled across process boundaries.  Writes a per-object checkpoint file
    and returns ``(object_id, n_rows)``.

    Parameters
    ----------
    object_id : str
        Object to process.
    obs_parquet : str
        Path to the full MPCObservations Parquet input file.
    orbits_parquet : str
        Path to the full MPCOrbits Parquet input file.
    propagator_class_fqn : str
        Fully-qualified class name of the propagator, e.g.
        ``"adam_assist.ASSISTPropagator"``.
    config : LOOOConfig
        Eligibility filter configuration (``min_obs_held_out`` and
        ``ls_kwargs`` are used; arc/fraction filters are replaced by
        ``min_ref_obs``).
    checkpoint_dir : str
        Directory in which to write the per-object ``.parquet`` checkpoint.
    sigma_model : str
        How to fill missing rmsra/rmsdec — ``"veres2017"`` or ``"const"``.
    reference_stn : str
        MPC code of the reference / anchor station.
    min_ref_obs : int
        Minimum reference-station observations required to perform the fit.

    Returns
    -------
    tuple of (str, int)
        ``(object_id, number_of_result_rows)``
    """
    import importlib
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _log = _logging.getLogger(__name__)

    checkpoint_dir_path = Path(checkpoint_dir)
    ckpt = _checkpoint_path(checkpoint_dir_path, object_id)

    # --- Load data slice for this object ---
    try:
        from mpcq.observations import MPCObservations
        from mpcq.orbits import MPCOrbits

        all_obs = MPCObservations(pq.read_table(obs_parquet))
        all_orb = MPCOrbits(pq.read_table(orbits_parquet))
    except Exception as e:
        _log.error(f"{object_id}: failed to load data: {e}")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    obs_mask = pc.equal(all_obs.requested_provid, object_id)
    obj_mpc_obs = all_obs.apply_mask(obs_mask)
    orbit_mask = pc.equal(all_orb.requested_provid, object_id)
    obj_mpc_orbit = all_orb.apply_mask(orbit_mask)

    if len(obj_mpc_obs) == 0 or len(obj_mpc_orbit) == 0:
        _log.warning(f"{object_id}: missing observations or orbit, skipping")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Drop space-based observatories (but keep the reference station) ---
    spacebased = set(get_spacebased_stns())
    # Never drop the reference station even if it appears in the spacebased list
    spacebased.discard(reference_stn)
    ground_mask = pc.invert(
        pc.is_in(
            obj_mpc_obs.stn,
            value_set=pa.array(list(spacebased), type=pa.large_utf8()),
        )
    )
    obj_mpc_obs = obj_mpc_obs.apply_mask(ground_mask)
    if len(obj_mpc_obs) == 0:
        _log.warning(f"{object_id}: no ground-based observations, skipping")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Convert to OD observations ---
    try:
        od_obs = mpc_to_od_observations(
            obj_mpc_obs, prevent_nans=True, sigma_model=sigma_model
        )
    except Exception as e:
        _log.warning(f"{object_id}: mpc_to_od_observations failed: {e}")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0
    if od_obs is None:
        _log.warning(f"{object_id}: could not convert observations, skipping")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Get reference orbit ---
    try:
        reference_orbit = obj_mpc_orbit.orbits()
    except Exception as e:
        _log.warning(f"{object_id}: could not get reference orbit: {e}")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Instantiate propagator ---
    try:
        module_name, class_name = propagator_class_fqn.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        propagator_class = getattr(mod, class_name)
        propagator = propagator_class()
    except Exception as e:
        _log.error(
            f"{object_id}: could not instantiate propagator "
            f"{propagator_class_fqn}: {e}"
        )
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Run reference-orbit evaluation ---
    astcats = obj_mpc_obs.astcat.to_pylist()
    try:
        result = run_reference_orbit_for_object(
            object_id=object_id,
            observations=od_obs,
            reference_orbit=reference_orbit,
            propagator=propagator,
            config=config,
            astcats=astcats,
            reference_stn=reference_stn,
            min_ref_obs=min_ref_obs,
        )
    except Exception as e:
        _log.error(
            f"{object_id}: run_reference_orbit_for_object failed: {e}",
            exc_info=True,
        )
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Write checkpoint ---
    if len(result) > 0:
        result.to_parquet(ckpt)
    else:
        pq.write_table(LOOOResult.empty().table, ckpt)

    _log.info(f"{object_id}: done — {len(result)} rows written to {ckpt.name}")
    return object_id, len(result)


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------


def run_reference_orbit_pipeline(
    mpc_observations,
    mpc_orbits,
    propagator_class: Type[Propagator],
    output_path: Path,
    config: Optional[LOOOConfig] = None,
    object_ids: Optional[List[str]] = None,
    max_processes: Optional[int] = None,
    sigma_model: str = "veres2017",
    reference_stn: str = "X05",
    min_ref_obs: int = 20,
) -> LOOOResult:
    """
    Run reference-orbit evaluation for all (or a subset of) objects, in parallel.

    For each object, one DC fit is performed on the reference station's
    observations.  All other stations are then evaluated against that orbit.
    Results are checkpointed per-object immediately after completion, so an
    interrupted run can be resumed automatically from where it left off.

    Parameters
    ----------
    mpc_observations : MPCObservations
        All observations (grouped by ``requested_provid`` internally).
    mpc_orbits : MPCOrbits
        Reference orbits used as DC starting points.
    propagator_class : Type[Propagator]
        Class (not instance) of the propagator to use.
    output_path : Path
        Final merged Parquet output path.
    config : LOOOConfig, optional
        Eligibility filter configuration.  The ``min_obs_held_out`` field is
        used to filter evaluated stations; arc-length and held-out-fraction
        filters are replaced by ``min_ref_obs``.
    object_ids : list of str, optional
        Restrict processing to these object IDs.  Defaults to all objects
        found in ``mpc_observations``.
    max_processes : int, optional
        Number of parallel worker processes.  Defaults to CPU count.
    sigma_model : str
        How to fill missing rmsra/rmsdec — ``"veres2017"`` (default) or
        ``"const"``.
    reference_stn : str
        MPC code of the reference / anchor station.  Default ``"X05"``
        (Rubin Observatory / LSST).
    min_ref_obs : int
        Minimum number of reference-station observations required per object
        before the reference-orbit fit is trusted.  Objects below this
        threshold are skipped.

    Returns
    -------
    LOOOResult
        Full combined result table (same schema as the LOOO pipeline output).
    """
    if config is None:
        config = LOOOConfig()
    if max_processes is None:
        max_processes = mp.cpu_count()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Workers load from disk — write temporary Parquet files if not already there.
    obs_parquet = str(output_path.parent / "_obs_input.parquet")
    orbits_parquet = str(output_path.parent / "_orbits_input.parquet")
    if not Path(obs_parquet).exists():
        mpc_observations.to_parquet(obs_parquet)
    if not Path(orbits_parquet).exists():
        mpc_orbits.to_parquet(orbits_parquet)

    # Determine which objects to process
    if object_ids is None:
        object_ids = mpc_observations.requested_provid.unique().to_pylist()
    object_ids = sorted(set(object_ids))

    # Skip already-checkpointed objects (resume support)
    completed = _load_completed_ids(checkpoint_dir)
    remaining = [oid for oid in object_ids if oid not in completed]
    if completed:
        logger.info(
            f"Resuming: {len(completed)} objects already done, "
            f"{len(remaining)} remaining"
        )
    logger.info(
        f"Processing {len(remaining)} objects with {max_processes} workers "
        f"(reference_stn={reference_stn}, min_ref_obs={min_ref_obs})"
    )

    fqn = _propagator_fqn(propagator_class)

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {
            executor.submit(
                _ref_worker,
                oid,
                obs_parquet,
                orbits_parquet,
                fqn,
                config,
                str(checkpoint_dir),
                sigma_model,
                reference_stn,
                min_ref_obs,
            ): oid
            for oid in remaining
        }
        n_done = len(completed)
        for future in as_completed(futures):
            oid = futures[future]
            try:
                _, n_rows = future.result()
                n_done += 1
                logger.info(
                    f"[{n_done}/{len(object_ids)}] {oid} — {n_rows} rows"
                )
            except Exception as e:
                logger.error(
                    f"{oid}: worker raised exception: {e}", exc_info=True
                )

    logger.info("Merging checkpoints into final output...")
    return merge_checkpoints(checkpoint_dir, output_path)
