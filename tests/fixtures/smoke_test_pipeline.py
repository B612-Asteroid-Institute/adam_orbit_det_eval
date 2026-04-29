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
    # Pass the same propagator type used downstream so hold-in chi2 is
    # consistent with held-out evaluation.
    orbit_fitter = FindOrbOrbitFitter(
        fo_result_dir=fo_dir,
        propagator=ASSISTPropagator(),
    )

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

    # -- Column-population gates (regression for hw1) --
    # Pilot v10 silently produced all-null hold_in_reduced_chi2 / hold_in_fit_success.
    # Without these gates the smoke test passes on broken data.
    import pyarrow.compute as pc
    import numpy as np

    nn_chi2 = pc.sum(
        pc.cast(pc.is_valid(results.hold_in_reduced_chi2), "int64")
    ).as_py() or 0
    nn_succ = pc.sum(
        pc.cast(pc.is_valid(results.hold_in_fit_success), "int64")
    ).as_py() or 0
    chi2_frac = nn_chi2 / n_rows
    succ_frac = nn_succ / n_rows
    print(
        f"Non-null hold_in_reduced_chi2: {nn_chi2}/{n_rows} ({chi2_frac:.1%})"
    )
    print(
        f"Non-null hold_in_fit_success:  {nn_succ}/{n_rows} ({succ_frac:.1%})"
    )

    if chi2_frac < 0.5:
        print(
            f"FAIL: hold_in_reduced_chi2 non-null fraction {chi2_frac:.1%} < 50% "
            f"(regression: hold-in fit metrics not flowing through)"
        )
        sys.exit(1)
    if succ_frac < 0.5:
        print(
            f"FAIL: hold_in_fit_success non-null fraction {succ_frac:.1%} < 50% "
            f"(regression: hold-in fit metrics not flowing through)"
        )
        sys.exit(1)

    # Residual sanity: catastrophic divergence would yield arcseconds of
    # tens of thousands. Cap at 60 arcsec on this fixture.
    max_abs_ra = float(
        np.nanmax(np.abs(results.residual_ra_arcsec.to_numpy(zero_copy_only=False)))
    )
    print(f"max(|residual_ra_arcsec|): {max_abs_ra:.2f} arcsec")
    if not np.isfinite(max_abs_ra) or max_abs_ra >= 60.0:
        print(
            f"FAIL: max(|residual_ra_arcsec|) = {max_abs_ra} arcsec "
            f"(fits diverging — chi2 column would be junk at scale)"
        )
        sys.exit(1)

    # -- Run analysis (same path as 12_run_looo_cloud_shard.py) --
    from adam_orbit_det_eval.looo.analysis import (
        compute_observatory_stats,
        compute_program_code_stats,
    )

    obs_stats = compute_observatory_stats(results)
    print(f"Observatory stats: {len(obs_stats)} stations")
    if len(obs_stats) < 1:
        print(
            "FAIL: compute_observatory_stats returned 0 rows "
            "(chi2 filter dropping every row — column likely all-null)"
        )
        sys.exit(1)

    prog_stats = compute_program_code_stats(results)
    print(f"Program code stats: {len(prog_stats)} groups")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
