"""
Ray-parallel LOOO pipeline over a collection of objects.

Usage
-----
from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline

results = run_looo_pipeline(
    mpc_observations=mpc_obs,   # MPCObservations
    mpc_orbits=mpc_orbits,       # MPCOrbits
    propagator_class=TwoBodyPropagator,
    output_path=Path("results/looo_results.parquet"),
)
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Type

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import quivr as qv

from adam_core.propagator.propagator import Propagator

from mpcq.observations import MPCObservations
from mpcq.orbits import MPCOrbits

from ..utils import get_spacebased_stns, mpc_to_od_observations
from .core import LOOOConfig, LOOOResult, run_looo_for_object

# Cached set of station codes that lack fixed Earth coordinates
_SPACEBASED_STNS: set = set()

logger = logging.getLogger(__name__)


def _process_one_object(
    object_id: str,
    mpc_observations: MPCObservations,
    mpc_orbits: MPCOrbits,
    propagator: Propagator,
    config: LOOOConfig,
) -> LOOOResult:
    """Process a single object. Called by worker functions."""
    # Filter observations and orbit for this object
    obs_mask = pc.equal(mpc_observations.requested_provid, object_id)
    obj_mpc_obs = mpc_observations.apply_mask(obs_mask)

    orbit_mask = pc.equal(mpc_orbits.requested_provid, object_id)
    obj_mpc_orbit = mpc_orbits.apply_mask(orbit_mask)

    if len(obj_mpc_obs) == 0 or len(obj_mpc_orbit) == 0:
        logger.warning(f"{object_id}: missing observations or orbit, skipping")
        return LOOOResult.empty()

    # Filter out space-based observatory observations (no fixed Earth coordinates)
    global _SPACEBASED_STNS
    if not _SPACEBASED_STNS:
        _SPACEBASED_STNS = set(get_spacebased_stns())
    ground_mask = pc.invert(
        pc.is_in(
            obj_mpc_obs.stn,
            value_set=pa.array(list(_SPACEBASED_STNS), type=pa.large_utf8()),
        )
    )
    n_before = len(obj_mpc_obs)
    obj_mpc_obs = obj_mpc_obs.apply_mask(ground_mask)
    if len(obj_mpc_obs) < n_before:
        logger.debug(
            f"{object_id}: dropped {n_before - len(obj_mpc_obs)} space-based obs"
        )
    if len(obj_mpc_obs) == 0:
        logger.warning(f"{object_id}: no ground-based observations remaining, skipping")
        return LOOOResult.empty()

    # Convert MPC observations to OD format
    try:
        od_obs = mpc_to_od_observations(obj_mpc_obs, prevent_nans=True)
    except Exception as e:
        logger.warning(f"{object_id}: mpc_to_od_observations failed: {e}")
        return LOOOResult.empty()
    if od_obs is None:
        logger.warning(f"{object_id}: could not convert observations, skipping")
        return LOOOResult.empty()

    # Get the reference orbit (MPC nominal orbit as starting point for DC)
    try:
        reference_orbit = obj_mpc_orbit.orbits()
    except Exception as e:
        logger.warning(f"{object_id}: could not get reference orbit: {e}")
        return LOOOResult.empty()

    # Pass through astcat column for per-catalog analysis
    astcats = obj_mpc_obs.astcat.to_pylist()

    result = run_looo_for_object(
        object_id=object_id,
        observations=od_obs,
        reference_orbit=reference_orbit,
        propagator=propagator,
        config=config,
        astcats=astcats,
    )
    return result


def _checkpoint_path(checkpoint_dir: Path, object_id: str) -> Path:
    """Return the per-object checkpoint file path, sanitizing the object ID for use as filename."""
    safe_id = object_id.replace("/", "_").replace(" ", "_")
    return checkpoint_dir / f"{safe_id}.parquet"


def _load_completed_ids(checkpoint_dir: Path) -> set:
    """Return the set of object IDs that have already been checkpointed."""
    if not checkpoint_dir.exists():
        return set()
    completed = set()
    for f in checkpoint_dir.glob("*.parquet"):
        # Read just the object_id column to recover the original ID
        try:
            tbl = pq.read_table(f, columns=["object_id"])
            ids = tbl.column("object_id").unique().to_pylist()
            completed.update(ids)
        except Exception:
            pass  # corrupt checkpoint — will be re-run
    return completed


def merge_checkpoints(checkpoint_dir: Path, output_path: Path) -> LOOOResult:
    """Merge all per-object checkpoint files into a single Parquet result file."""
    checkpoint_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not checkpoint_files:
        return LOOOResult.empty()

    schema = None
    writer = None
    total_rows = 0
    try:
        for f in checkpoint_files:
            tbl = pq.read_table(f)
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(tbl)
            total_rows += len(tbl)
    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Merged {len(checkpoint_files)} checkpoints → {output_path} ({total_rows} rows)")
    return LOOOResult(pq.read_table(output_path))


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
) -> LOOOResult:
    """
    Run LOOO cross-validation for all (or a specified subset of) objects in
    mpc_observations.

    Results are written as one Parquet file per object in a ``checkpoints/``
    subdirectory next to ``output_path``.  Each checkpoint is written and
    closed atomically after the object finishes, so a killed run can be
    resumed: already-checkpointed objects are skipped automatically.

    After all objects are done the checkpoints are merged into ``output_path``.

    Parameters
    ----------
    mpc_observations : MPCObservations
        All observations. The pipeline groups by `requested_provid`.
    mpc_orbits : MPCOrbits
        Reference orbits for each object (used as DC starting points).
    propagator_class : Type[Propagator]
        Class (not instance) of the propagator to use.
    output_path : Path
        Where to write the final merged Parquet result file.
    config : LOOOConfig, optional
        Eligibility filter configuration.
    object_ids : list of str, optional
        Restrict to these object IDs. If None, all objects in
        mpc_observations are processed.
    max_processes : int, optional
        Number of parallel workers. Defaults to CPU count.
    propagator_kwargs : dict, optional
        Kwargs passed to propagator_class constructor.
    write_interval : int
        Ignored (kept for API compatibility).

    Returns
    -------
    LOOOResult
        Full combined result table.
    """
    if config is None:
        config = LOOOConfig()
    if propagator_kwargs is None:
        propagator_kwargs = {}
    if max_processes is None:
        max_processes = mp.cpu_count()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine which objects to process
    if object_ids is None:
        object_ids = mpc_observations.requested_provid.unique().to_pylist()
    object_ids = sorted(set(object_ids))  # deterministic order

    # Skip objects that are already checkpointed (resume support)
    completed = _load_completed_ids(checkpoint_dir)
    remaining = [oid for oid in object_ids if oid not in completed]
    if completed:
        logger.info(
            f"Resuming: {len(completed)} objects already done, "
            f"{len(remaining)} remaining"
        )
    logger.info(f"Processing {len(remaining)} objects with up to {max_processes} workers")

    propagator = propagator_class(**propagator_kwargs)

    for i, object_id in enumerate(remaining):
        logger.info(f"[{len(completed)+i+1}/{len(object_ids)}] {object_id}")
        result = _process_one_object(
            object_id=object_id,
            mpc_observations=mpc_observations,
            mpc_orbits=mpc_orbits,
            propagator=propagator,
            config=config,
        )
        # Write checkpoint immediately after each object (atomic: write then close)
        ckpt = _checkpoint_path(checkpoint_dir, object_id)
        if len(result) > 0:
            result.to_parquet(ckpt)
        else:
            # Write an empty file so we know this object was attempted
            pq.write_table(LOOOResult.empty().table, ckpt)
        logger.debug(f"  Checkpointed {object_id} → {ckpt.name}")

    # Merge all checkpoints into the final output file
    logger.info("Merging checkpoints into final output...")
    return merge_checkpoints(checkpoint_dir, output_path)
