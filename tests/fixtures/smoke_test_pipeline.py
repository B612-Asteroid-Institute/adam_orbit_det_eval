#!/usr/bin/env python3
"""
Build-time smoke test for the cloud LOOO pipeline image.

Exercises the same code path as 12_run_looo_cloud_shard.py without GCS:
  load fixture -> configure propagator/fitter -> run_looo_pipeline -> analysis

Exits non-zero if the pipeline produces zero result rows or analysis crashes.

Witness gate (bead dez): the fixture pairs two cooperative pilot-v11 winners
that should still pass under tighter LOOO filters. 2014 EG12 (270 obs, 9
stations, 22-yr arc, max station 37%) covers the long-arc dense case; 2013
UF5 (48 obs, 7 stations, 61-day arc, max station 23%) covers the short-arc
multi-station case. Both had hold_in_reduced_chi2 < 1, max|residual| < 1.1"
in pilot v11 — so any regression in warm-start plumbing or DC convergence
will surface as residual blow-up here. The 1981 QE2 / 2020 ML22 pair was
retired after they exposed catastrophic behavior the -v hint cannot rescue.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

WITNESS_OBJECT_IDS = ["2014 EG12", "2013 UF5"]


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

    import numpy as np
    import pyarrow.compute as pc

    # -- Global regression: hold_in_reduced_chi2 must never be null (bead hw1) --
    # Pilot v10 silently produced all-null hold_in_reduced_chi2; the existing
    # non-null-fraction check let half-broken results through. Tighten to "any
    # null fails" so the hold-in metrics column is fully populated.
    object_id_col = results.object_id.to_pylist()
    chi2_valid = pc.is_valid(results.hold_in_reduced_chi2).to_pylist()
    null_chi2_rows = [
        oid for oid, valid in zip(object_id_col, chi2_valid) if not valid
    ]
    if null_chi2_rows:
        print(
            f"FAIL: hold_in_reduced_chi2 is null on {len(null_chi2_rows)} rows "
            f"(regression on bead hw1 — hold-in fit metrics not flowing through). "
            f"Affected object_ids: {sorted(set(null_chi2_rows))[:5]}"
        )
        sys.exit(1)

    # -- Per-witness gate (bead dez seed-warm-start regression check) --
    # If reference_orbit isn't reaching FindOrb at fit time, hold-in fits
    # diverge from the MPC seed. On 1981 QE2 (3,291 obs, 53 stations) this
    # produced median chi2 ~ 4.7e9 and arcsecond residuals in the 10^5 range
    # in pilot v11. With the warm-start fix, fits refine around the seed and
    # both residuals and chi2 are sane.
    chi2 = results.hold_in_reduced_chi2.to_numpy(zero_copy_only=False)
    ra_arcsec = results.residual_ra_arcsec.to_numpy(zero_copy_only=False)
    dec_arcsec = results.residual_dec_arcsec.to_numpy(zero_copy_only=False)
    success = results.hold_in_fit_success.to_pylist()
    success_valid = pc.is_valid(results.hold_in_fit_success).to_pylist()

    # -- Global column-population gates (54t v2-full smoke spec) --
    # The null check above already guarantees hold_in_reduced_chi2 is fully
    # populated; these add the value-distribution and companion-column gates
    # the v2 brief requires, so a build that regresses to the pilot-v10 NaN
    # regime OR the 2tj chi2-inflation regime (median ~10^7) fails at build.
    global_failures: list[str] = []

    finite_chi2 = chi2[np.isfinite(chi2)]
    median_chi2 = float(np.median(finite_chi2)) if len(finite_chi2) else float("nan")
    print(
        f"Global: hold_in_reduced_chi2 median={median_chi2:.4g} "
        f"over {len(finite_chi2)}/{len(chi2)} finite rows"
    )
    if not (0.01 <= median_chi2 <= 100.0):
        global_failures.append(
            f"global hold_in_reduced_chi2 median={median_chi2} outside [0.01, 100] "
            f"(chi2-inflation regression on bead 2tj or all-NaN regression)"
        )

    success_nonnull_frac = (
        sum(1 for v in success_valid if v) / len(success_valid)
        if success_valid
        else 0.0
    )
    print(f"Global: hold_in_fit_success non-null on {success_nonnull_frac:.0%} of rows")
    if success_nonnull_frac <= 0.5:
        global_failures.append(
            f"hold_in_fit_success non-null on only {success_nonnull_frac:.0%} of rows "
            f"(<=50%; fit-success column not flowing through)"
        )

    finite_resid = int(
        np.sum(np.isfinite(ra_arcsec) & np.isfinite(dec_arcsec))
    )
    print(f"Global: {finite_resid}/{len(ra_arcsec)} rows have finite RA+Dec residuals")
    if finite_resid < 1:
        global_failures.append(
            "no row has a finite (non-NaN) RA/Dec residual (residual computation "
            "produced all-NaN — propagation or observer-state regression)"
        )

    if global_failures:
        print("FAIL: global column-population gate(s) tripped:")
        for f in global_failures:
            print(f"  - {f}")
        sys.exit(1)

    failures: list[str] = []
    for witness in WITNESS_OBJECT_IDS:
        idx = [i for i, oid in enumerate(object_id_col) if oid == witness]
        if not idx:
            failures.append(
                f"{witness}: 0 result rows (witness not exercised — fixture or "
                f"eligibility filter regression)"
            )
            continue

        w_chi2 = chi2[idx]
        w_ra = ra_arcsec[idx]
        w_success = [success[i] for i in idx]
        w_success_valid = [success_valid[i] for i in idx]

        max_abs_ra = float(np.nanmax(np.abs(w_ra))) if len(w_ra) else float("nan")
        catastrophic_chi2 = int(np.sum(w_chi2 > 100.0))
        catastrophic_frac = catastrophic_chi2 / len(w_chi2)
        bad_success = sum(
            1
            for v, ok in zip(w_success_valid, w_success)
            if (not v) or (ok is False)
        )

        print(
            f"{witness}: {len(idx)} rows, max|residual_ra|={max_abs_ra:.2f}\", "
            f"chi2>100 on {catastrophic_chi2}/{len(w_chi2)} "
            f"({catastrophic_frac:.0%}), bad_success={bad_success}"
        )

        if not np.isfinite(max_abs_ra) or max_abs_ra > 60.0:
            failures.append(
                f"{witness}: max|residual_ra_arcsec|={max_abs_ra} > 60 "
                f"(seed warm-start regression — fits diverging from MPC seed)"
            )
        if catastrophic_frac > 0.5:
            failures.append(
                f"{witness}: hold_in_reduced_chi2 > 100 on "
                f"{catastrophic_frac:.0%} of rows (catastrophic regime)"
            )
        if bad_success:
            failures.append(
                f"{witness}: hold_in_fit_success null-or-False on {bad_success} "
                f"rows (FindOrb fit failures)"
            )

    if failures:
        print("FAIL: witness gate(s) tripped:")
        for f in failures:
            print(f"  - {f}")
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
