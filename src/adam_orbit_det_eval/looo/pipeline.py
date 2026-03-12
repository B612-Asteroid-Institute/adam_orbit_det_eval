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

from ..utils import mpc_to_od_observations
from .core import LOOOConfig, LOOOResult, run_looo_for_object

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

    # Convert MPC observations to OD format
    od_obs = mpc_to_od_observations(obj_mpc_obs, prevent_nans=True)
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


def run_looo_pipeline(
    mpc_observations: MPCObservations,
    mpc_orbits: MPCOrbits,
    propagator_class: Type[Propagator],
    output_path: Path,
    config: Optional[LOOOConfig] = None,
    object_ids: Optional[List[str]] = None,
    max_processes: Optional[int] = None,
    propagator_kwargs: Optional[dict] = None,
    write_interval: int = 50,
) -> LOOOResult:
    """
    Run LOOO cross-validation for all (or a specified subset of) objects in
    mpc_observations, writing results incrementally to a Parquet file.

    Parameters
    ----------
    mpc_observations : MPCObservations
        All observations. The pipeline groups by `requested_provid`.
    mpc_orbits : MPCOrbits
        Reference orbits for each object (used as DC starting points).
    propagator_class : Type[Propagator]
        Class (not instance) of the propagator to use.
    output_path : Path
        Where to write the Parquet result file. Written incrementally
        so partial results are preserved if the run is interrupted.
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
        Write a Parquet chunk after every this many objects are processed.

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

    # Determine which objects to process
    if object_ids is None:
        object_ids = (
            mpc_observations.requested_provid.unique().to_pylist()
        )
    object_ids = sorted(set(object_ids))  # deterministic order
    logger.info(f"Processing {len(object_ids)} objects with up to {max_processes} workers")

    # Instantiate one propagator per process (or one shared if single-process)
    propagator = propagator_class(**propagator_kwargs)

    all_results: List[LOOOResult] = []
    writer: Optional[pq.ParquetWriter] = None
    schema = None

    for i, object_id in enumerate(object_ids):
        logger.info(f"[{i+1}/{len(object_ids)}] {object_id}")
        result = _process_one_object(
            object_id=object_id,
            mpc_observations=mpc_observations,
            mpc_orbits=mpc_orbits,
            propagator=propagator,
            config=config,
        )
        if len(result) > 0:
            all_results.append(result)

        # Incremental write
        if (i + 1) % write_interval == 0 or (i + 1) == len(object_ids):
            if all_results:
                chunk = qv.concatenate(all_results)
                if chunk.fragmented():
                    chunk = qv.defragment(chunk)
                tbl = chunk.table
                if writer is None:
                    schema = tbl.schema
                    writer = pq.ParquetWriter(output_path, schema)
                writer.write_table(tbl)
                n_written = sum(len(r) for r in all_results)
                logger.info(
                    f"  Wrote {n_written} rows to {output_path} "
                    f"(after {i+1}/{len(object_ids)} objects)"
                )
                all_results = []

    if writer is not None:
        writer.close()
        logger.info(f"Finished writing to {output_path}")

    # Read back and return the full result
    if output_path.exists():
        return LOOOResult(pq.read_table(output_path))
    return LOOOResult.empty()
