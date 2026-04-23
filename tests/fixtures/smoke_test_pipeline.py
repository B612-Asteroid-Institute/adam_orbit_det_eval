#!/usr/bin/env python3
"""
Build-time smoke test for the cloud LOOO pipeline image.

Exercises the same code path as 12_run_looo_cloud_shard.py without GCS:
  load fixture -> configure propagator/fitter -> run_looo_pipeline -> analysis

Exits non-zero if the pipeline produces zero result rows or analysis crashes.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path


def main() -> None:
    mp.set_start_method("spawn")

    fixture_dir = Path("/app/fixtures")
    output_dir = Path("/tmp/smoke_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load fixture --
    from mpcq.observations import MPCObservations
    from mpcq.orbits import MPCOrbits

    mpc_obs = MPCObservations.from_parquet(fixture_dir / "smoke_obs.parquet")
    mpc_orbits = MPCOrbits.from_parquet(fixture_dir / "smoke_orbits.parquet")
    n_objects = len(mpc_obs.requested_provid.unique())
    print(f"Loaded {n_objects} objects, {len(mpc_obs)} observations")

    # -- Configure propagator and orbit fitter --
    from adam_assist import ASSISTPropagator
    from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter

    fo_dir = "/tmp/fo_smoke"
    os.makedirs(fo_dir, exist_ok=True)
    orbit_fitter = FindOrbOrbitFitter(fo_result_dir=fo_dir)

    # -- Run LOOO pipeline --
    from adam_orbit_det_eval.looo.core import LOOOConfig
    from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline

    results_path = output_dir / "looo_results.parquet"
    results = run_looo_pipeline(
        mpc_observations=mpc_obs,
        mpc_orbits=mpc_orbits,
        propagator_class=ASSISTPropagator,
        output_path=results_path,
        config=LOOOConfig(),
        orbit_fitter=orbit_fitter,
        max_processes=1,
    )
    n_rows = len(results)
    print(f"Pipeline returned {n_rows} result rows")

    if n_rows == 0:
        print("FAIL: pipeline produced zero result rows")
        sys.exit(1)

    # -- Run analysis (same path as 12_run_looo_cloud_shard.py) --
    from adam_orbit_det_eval.looo.analysis import (
        compute_observatory_stats,
        compute_program_code_stats,
    )

    obs_stats = compute_observatory_stats(results)
    print(f"Observatory stats: {len(obs_stats)} stations")

    prog_stats = compute_program_code_stats(results)
    print(f"Program code stats: {len(prog_stats)} groups")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
