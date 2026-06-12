"""
Parallel LOOO pipeline over a collection of objects.

Uses ProcessPoolExecutor so each worker runs in its own process with its own
propagator instance.  Results are checkpointed per-object so interrupted runs
can be resumed without reprocessing completed objects.

Usage
-----
from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline

results = run_looo_pipeline(
    mpc_observations=mpc_obs,   # MPCObservations
    mpc_orbits=mpc_orbits,       # MPCOrbits
    propagator_class=ASSISTPropagator,
    output_path=Path("results/looo_results.parquet"),
    max_processes=6,
)
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import quivr as qv

from adam_core.time import Timestamp

from adam_core.orbit_determination.orbit_fitter import OrbitFitter
from adam_core.propagator.propagator import Propagator

from mpcq.observations import MPCObservations
from mpcq.orbits import MPCOrbits

from ..utils import get_spacebased_stns, mpc_to_od_observations
from .core import LOOOConfig, LOOOResult, run_looo_for_object
from .gcs_checkpoint import GCSCheckpointStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-LOOO observation-time filter (bead tcu)
# ---------------------------------------------------------------------------

def _iso_to_mjd_utc(iso_date: str) -> float:
    """Convert an ISO-8601 date/time string to MJD (UTC scale)."""
    return float(Timestamp.from_iso8601([iso_date], scale="utc").mjd()[0].as_py())


def filter_observations_by_obstime(
    mpc_observations: MPCObservations,
    obstime_min: Optional[str] = None,
    obstime_max: Optional[str] = None,
) -> MPCObservations:
    """
    Filter observations to the (obstime_min, obstime_max] window.

    This is the pre-LOOO step of the v2 parametric time filter (bead tcu,
    docs/v2-scope.md): it runs BEFORE any refit so the hold-in orbit only
    sees in-window data. Window semantics follow the v2 pre/post-2017
    partition — an observation is kept when ``obstime > obstime_min`` AND
    ``obstime <= obstime_max`` (min exclusive, max inclusive), so
    ``obstime_max=2017-01-01`` and ``obstime_min=2017-01-01`` split the
    archive with no overlap and no gap.

    Parameters
    ----------
    mpc_observations : MPCObservations
        Input observations.
    obstime_min, obstime_max : str, optional
        ISO-8601 bounds (UTC), e.g. ``"2017-01-01"``. None = unbounded.
        When either bound is set, observations with a null obstime are
        dropped (their window membership cannot be established).

    Returns
    -------
    MPCObservations
        The in-window subset (the input table, unchanged, if no bound is set).
    """
    if obstime_min is None and obstime_max is None:
        return mpc_observations

    mjd = mpc_observations.obstime.mjd().to_numpy(zero_copy_only=False)
    mask = np.isfinite(mjd)
    if obstime_min is not None:
        mask &= mjd > _iso_to_mjd_utc(obstime_min)
    if obstime_max is not None:
        mask &= mjd <= _iso_to_mjd_utc(obstime_max)

    filtered = mpc_observations.apply_mask(pa.array(mask))
    logger.info(
        f"obstime filter ({obstime_min or '-inf'}, {obstime_max or '+inf'}]: "
        f"{len(filtered)}/{len(mpc_observations)} observations kept"
    )
    return filtered


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir: Path, object_id: str) -> Path:
    """Return the per-object checkpoint file path."""
    safe_id = object_id.replace("/", "_").replace(" ", "_")
    return checkpoint_dir / f"{safe_id}.parquet"


def _load_completed_ids(checkpoint_dir: Path) -> set:
    """Return the set of object IDs that already have a checkpoint file."""
    if not checkpoint_dir.exists():
        return set()
    completed = set()
    for f in checkpoint_dir.glob("*.parquet"):
        try:
            tbl = pq.read_table(f, columns=["object_id"])
            completed.update(tbl.column("object_id").unique().to_pylist())
        except Exception:
            pass  # corrupt checkpoint — will be re-run
    return completed


def merge_checkpoints(checkpoint_dir: Path, output_path: Path) -> LOOOResult:
    """Merge all per-object checkpoint files into a single Parquet result file."""
    checkpoint_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not checkpoint_files:
        return LOOOResult.empty()

    writer = None
    total_rows = 0
    try:
        for f in checkpoint_files:
            tbl = pq.read_table(f)
            if len(tbl) == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(output_path, tbl.schema)
            writer.write_table(tbl)
            total_rows += len(tbl)
    finally:
        if writer is not None:
            writer.close()

    logger.info(
        f"Merged {len(checkpoint_files)} checkpoints → {output_path} ({total_rows} rows)"
    )
    if output_path.exists():
        return LOOOResult(pq.read_table(output_path))
    return LOOOResult.empty()


# ---------------------------------------------------------------------------
# Per-object worker — module-level so it is picklable
# ---------------------------------------------------------------------------

def _worker(
    object_id: str,
    obs_parquet: str,
    orbits_parquet: str,
    propagator_class_fqn: str,
    config: LOOOConfig,
    checkpoint_dir: str,
    sigma_model: str = "veres2017",
    orbit_fitter: Optional[OrbitFitter] = None,
    gcs_prefix: Optional[str] = None,
    group_by: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """
    Worker function executed in a subprocess.

    Loads its own data slice from disk so that large tables don't need to be
    pickled across process boundaries.  Writes a checkpoint and returns
    (object_id, n_rows).
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

    # --- Drop space-based observatories ---
    spacebased = set(get_spacebased_stns())
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
        od_obs = mpc_to_od_observations(obj_mpc_obs, prevent_nans=True, sigma_model=sigma_model)
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
        _log.error(f"{object_id}: could not instantiate propagator {propagator_class_fqn}: {e}")
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Run LOOO ---
    # Per-observation group-key source columns, pulled directly from the MPC
    # observations (parallel arrays — mpc_to_od_observations is size-preserving,
    # so indices stay aligned with od_obs). band is plumbed the same way as
    # astcat / prog rather than being routed through adam_core.Observations
    # (bead wl0 pre-flight).
    astcats = obj_mpc_obs.astcat.to_pylist()
    program_codes_col = getattr(obj_mpc_obs, 'prog', None)
    if program_codes_col is not None:
        program_codes = program_codes_col.to_pylist()
    else:
        program_codes = [None] * len(obj_mpc_obs)
    bands_col = getattr(obj_mpc_obs, 'band', None)
    if bands_col is not None:
        bands = bands_col.to_pylist()
    else:
        bands = [None] * len(obj_mpc_obs)
    try:
        result = run_looo_for_object(
            object_id=object_id,
            observations=od_obs,
            reference_orbit=reference_orbit,
            propagator=propagator,
            config=config,
            astcats=astcats,
            program_codes=program_codes,
            bands=bands,
            group_by=group_by,
            orbit_fitter=orbit_fitter,
        )
    except Exception as e:
        _log.error(f"{object_id}: run_looo_for_object failed: {e}", exc_info=True)
        pq.write_table(LOOOResult.empty().table, ckpt)
        return object_id, 0

    # --- Write checkpoint ---
    if len(result) > 0:
        result.to_parquet(ckpt)
    else:
        pq.write_table(LOOOResult.empty().table, ckpt)

    _log.info(f"{object_id}: done — {len(result)} rows written to {ckpt.name}")

    # --- Optional: upload checkpoint to GCS ---
    if gcs_prefix is not None:
        try:
            store = GCSCheckpointStore(
                local_dir=checkpoint_dir_path, gcs_prefix=gcs_prefix,
            )
            store.upload_checkpoint(object_id)
        except Exception as e:  # best-effort — local checkpoint is authoritative
            _log.warning(f"{object_id}: GCS upload failed: {e}")

    return object_id, len(result)


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def _propagator_fqn(propagator_class: Type[Propagator]) -> str:
    """Return the fully-qualified class name for pickling across processes."""
    return f"{propagator_class.__module__}.{propagator_class.__qualname__}"


def run_looo_pipeline(
    mpc_observations: MPCObservations,
    mpc_orbits: MPCOrbits,
    propagator_class: Type[Propagator],
    output_path: Path,
    config: Optional[LOOOConfig] = None,
    object_ids: Optional[List[str]] = None,
    max_processes: Optional[int] = None,
    propagator_kwargs: Optional[dict] = None,
    write_interval: int = 50,  # kept for API compatibility, no longer used
    sigma_model: str = "veres2017",
    orbit_fitter: Optional[OrbitFitter] = None,
    gcs_checkpoint_store: Optional[GCSCheckpointStore] = None,
    group_by: Optional[List[str]] = None,
    obstime_min: Optional[str] = None,
    obstime_max: Optional[str] = None,
) -> LOOOResult:
    """
    Run LOOO cross-validation for all (or a subset of) objects, in parallel.

    Each object is processed in its own subprocess.  Results are checkpointed
    per-object immediately after completion, so a killed run resumes
    automatically from where it left off.

    Parameters
    ----------
    mpc_observations : MPCObservations
        All observations (grouped by `requested_provid` internally).
    mpc_orbits : MPCOrbits
        Reference orbits used as DC starting points.
    propagator_class : Type[Propagator]
        Class (not instance) of the propagator to use.
    output_path : Path
        Final merged Parquet output path.
    config : LOOOConfig, optional
        Eligibility filter configuration.
    object_ids : list of str, optional
        Restrict to these object IDs.  Defaults to all objects in
        mpc_observations.
    max_processes : int, optional
        Number of parallel worker processes.  Defaults to CPU count.
    propagator_kwargs : dict, optional
        Currently unused (workers instantiate propagators with no args).
    write_interval : int
        Ignored (kept for API compatibility).
    orbit_fitter : OrbitFitter, optional
        If provided, use this fitter's `initial_fit` for hold-in fits instead
        of the scipy-based `fit_least_squares`. Must be picklable for
        ProcessPoolExecutor.
    gcs_checkpoint_store : GCSCheckpointStore, optional
        If provided, mirror checkpoints to GCS. At startup existing GCS
        checkpoints are downloaded into the local checkpoint dir (resume).
        During execution each per-object checkpoint is uploaded after write.
        A SIGTERM handler is registered to flush state on spot preemption.
    group_by : list of str, optional
        Subset of {stn, prog, band, astcat} defining the LOOO hold-out unit
        (bead wl0). Defaults to ``[stn, prog, band]`` (the v2_full profile).
    obstime_min, obstime_max : str, optional
        ISO-8601 bounds of the pre-LOOO observation-time window (bead tcu):
        kept observations satisfy ``obstime_min < obstime <= obstime_max``.
        The filter runs before any refit, so hold-in fits only see in-window
        data, and the per-pair eligibility criteria (min obs remaining, min
        arc length, held-out fraction) are evaluated on the windowed
        observation set — an object that passes over its full arc may be
        excluded within a window.

    Returns
    -------
    LOOOResult
        Full combined result table.
    """
    if config is None:
        config = LOOOConfig()
    if max_processes is None:
        max_processes = mp.cpu_count()

    # Pre-LOOO observation-time window (bead tcu). Applied before the worker
    # input parquet is written so every downstream step (object selection,
    # eligibility, hold-in fits, held-out predictions) sees only in-window data.
    mpc_observations = filter_observations_by_obstime(
        mpc_observations, obstime_min=obstime_min, obstime_max=obstime_max
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional: pull checkpoints from GCS for resume ---
    gcs_prefix: Optional[str] = None
    if gcs_checkpoint_store is not None:
        # Point the store at our checkpoint dir in case caller didn't.
        gcs_checkpoint_store.local_dir = checkpoint_dir
        gcs_prefix = gcs_checkpoint_store.gcs_prefix
        try:
            gcs_checkpoint_store.sync_from_gcs()
        except Exception as e:
            logger.warning(f"sync_from_gcs failed at startup: {e}")
        try:
            gcs_checkpoint_store.register_sigterm_handler()
        except Exception as e:
            logger.warning(f"register_sigterm_handler failed: {e}")

    # Workers load from disk — write temporary parquet files if inputs aren't
    # already on disk.  We reuse the existing output_dir for temp files.
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
        f"Processing {len(remaining)} objects with {max_processes} workers"
    )

    fqn = _propagator_fqn(propagator_class)

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {
            executor.submit(
                _worker,
                oid,
                obs_parquet,
                orbits_parquet,
                fqn,
                config,
                str(checkpoint_dir),
                sigma_model,
                orbit_fitter,
                gcs_prefix,
                group_by,
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
                logger.error(f"{oid}: worker raised exception: {e}", exc_info=True)

    logger.info("Merging checkpoints into final output...")
    return merge_checkpoints(checkpoint_dir, output_path)
