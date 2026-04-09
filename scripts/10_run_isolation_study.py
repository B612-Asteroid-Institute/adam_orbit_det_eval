#!/usr/bin/env python3
"""
10_run_isolation_study.py
=========================
Run the isolation sensitivity study: for each observatory in the Phase 1 map,
test each bias type individually while all other observatories are clean.

This lets us measure:
  - Pure detection power for each bias type at each station (no cross-contamination)
  - Baseline for comparing against the full multi-bias run

Scenario naming convention:  {fake_code}_{bias_short_name}
  e.g.  AA01_clean, AA01_constant, AA01_timing, AA02_clean, AA02_constant, ...

Outputs are written to:
  data/sim_products/isolation_study/{scenario}/datasets/default/
  data/sim_products/isolation_study/{scenario}/looo_results/default/
  data/sim_products/isolation_study/{scenario}/analysis/default/
  data/sim_products/isolation_study/{scenario}/recovery/default/

A combined summary CSV is written to:
  data/sim_products/isolation_study/combined_recovery.csv

Usage
-----
    # Run all 81 scenarios (9 stations × 9 bias types):
    python scripts/10_run_isolation_study.py --input-dir data/sim_sample

    # Run only one station:
    python scripts/10_run_isolation_study.py --input-dir data/sim_sample --station AA01

    # Run only a specific bias across all stations:
    python scripts/10_run_isolation_study.py --input-dir data/sim_sample --bias constant

    # Run scenarios in parallel (--scenario-parallelism N × --max-processes M = total cores):
    python scripts/10_run_isolation_study.py --input-dir data/looo_sample_3500 \\
        --scenario-parallelism 16 --max-processes 6

    # Dry-run: print scenario names without executing:
    python scripts/10_run_isolation_study.py --dry-run
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from adam_orbit_det_eval.isolation_study import (
    BIAS_NAMES,
    list_scenarios,
    run_scenario,
    scenario_id,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir", type=Path, default=Path("data/sim_sample"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/sim_products/isolation_study"),
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Truth ephemeris cache (default: data/sim_products/cache)",
    )
    p.add_argument(
        "--propagator", choices=["twobody", "assist"], default="twobody",
    )
    p.add_argument("--max-processes", type=int, default=6)
    p.add_argument(
        "--scenario-parallelism", type=int, default=1,
        help=(
            "Number of scenarios to run concurrently (default: 1 = sequential). "
            "Total worker count = scenario_parallelism × max_processes. "
            "On a 96-core VM, try --scenario-parallelism 16 --max-processes 6."
        ),
    )
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Recovery threshold arcsec (default: 0.1)")
    p.add_argument(
        "--station", nargs="*", metavar="FAKE_CODE",
        help="Restrict to these fake codes (e.g. AA01 AA02)",
    )
    p.add_argument(
        "--bias", nargs="*", metavar="BIAS_NAME", choices=BIAS_NAMES,
        help="Restrict to these bias names",
    )
    p.add_argument("--object-ids", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print scenario names without running.")
    return p.parse_args()


def main():
    args = parse_args()
    scenarios = list_scenarios(args.station, args.bias)

    if args.dry_run:
        print(f"{len(scenarios)} scenarios:")
        for fc, bn in scenarios:
            print(f"  {scenario_id(fc, bn)}")
        return

    if not scenarios:
        logger.error("No scenarios selected.")
        sys.exit(1)

    logger.info(f"Running {len(scenarios)} isolation scenarios.")

    # Load input data once
    obs_path    = args.input_dir / "mpc_observations.parquet"
    orbits_path = args.input_dir / "mpc_orbits.parquet"
    for p, label in [(obs_path, "observations"), (orbits_path, "orbits")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    logger.info("Loading input data...")
    from mpcq.observations import MPCObservations
    from mpcq.orbits import MPCOrbits
    obs_template = MPCObservations.from_parquet(obs_path)
    truth_orbits = MPCOrbits.from_parquet(orbits_path)

    object_ids = args.object_ids or obs_template.requested_provid.unique().to_pylist()
    logger.info(f"  {len(object_ids)} objects")

    # Propagator
    if args.propagator == "twobody":
        from adam_orbit_det_eval.propagators import TwoBodyPropagator
        propagator_class = TwoBodyPropagator
    else:
        from adam_assist import ASSISTPropagator
        propagator_class = ASSISTPropagator

    cache_dir = args.cache_dir or Path("data/sim_products/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shared kwargs for every run_scenario call
    scenario_kwargs = dict(
        obs_template=obs_template,
        truth_orbits=truth_orbits,
        object_ids=object_ids,
        propagator_class=propagator_class,
        output_base=args.output_dir,
        cache_dir=cache_dir,
        max_processes=args.max_processes,
        threshold_arcsec=args.threshold,
        force=args.force,
    )

    all_rows: list = []
    failed:   list = []

    if args.scenario_parallelism <= 1:
        # Sequential (original behaviour)
        for i, (fc, bn) in enumerate(scenarios):
            sname = scenario_id(fc, bn)
            logger.info(f"[{i+1}/{len(scenarios)}] {sname}")
            try:
                rows = run_scenario(fc, bn, **scenario_kwargs)
                (all_rows.extend(rows) if rows else failed.append(sname))
            except Exception as exc:
                logger.error(f"Scenario {sname} failed: {exc}", exc_info=True)
                failed.append(sname)
    else:
        # Parallel scenarios — ThreadPoolExecutor so that the CPU work stays in
        # the inner ProcessPoolExecutors spawned by run_scenario → _run_looo.
        logger.info(
            f"Running {len(scenarios)} scenarios with scenario_parallelism="
            f"{args.scenario_parallelism} "
            f"(total workers ≈ {args.scenario_parallelism * args.max_processes})"
        )
        with ThreadPoolExecutor(max_workers=args.scenario_parallelism) as pool:
            future_to_sname = {
                pool.submit(run_scenario, fc, bn, **scenario_kwargs): scenario_id(fc, bn)
                for fc, bn in scenarios
            }
            for future in as_completed(future_to_sname):
                sname = future_to_sname[future]
                try:
                    rows = future.result()
                    (all_rows.extend(rows) if rows else failed.append(sname))
                except Exception as exc:
                    logger.error(f"Scenario {sname} failed: {exc}", exc_info=True)
                    failed.append(sname)

    # Write combined CSV
    if all_rows:
        import pandas as pd
        combined = pd.DataFrame(all_rows)
        combined_path = args.output_dir / "combined_recovery.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"Combined recovery → {combined_path}")
        _print_summary(combined)

    if failed:
        logger.warning(f"Failed ({len(failed)}): {failed}")
    logger.info("Done.")


def _print_summary(df):
    """Print a compact station × bias detection matrix."""
    import numpy as np

    target = df[df["is_target"]].copy()
    if target.empty:
        return

    print()
    print("=" * 90)
    print("  ISOLATION STUDY — TARGET STATION RECOVERY")
    print("=" * 90)
    hdr = (f"{'Scenario':<22}  {'BiasType':<20}  "
           f"{'Inj_RA':>7}  {'Rec_RA':>7}  {'Err_RA':>7}  "
           f"{'SNR_RA':>7}  {'N_obs':>6}  {'DetRA':>5}  {'DetDec':>6}")
    print(hdr)
    print("-" * len(hdr))

    for _, row in target.sort_values(["target_station", "applied_bias"]).iterrows():
        def _f(v):
            return f"{v:+7.3f}" if isinstance(v, float) and np.isfinite(v) else "   N/A "
        print(
            f"{row['scenario']:<22}  "
            f"{row['applied_bias']:<20}  "
            f"{_f(row['injected_ra_arcsec'])}  "
            f"{_f(row['recovered_mean_ra_arcsec'])}  "
            f"{_f(row['recovery_error_ra'])}  "
            f"{_f(row['detection_snr_ra'])}  "
            f"{int(row['n_obs']):>6}  "
            f"{'Y' if row['detected_ra'] else 'N':>5}  "
            f"{'Y' if row['detected_dec'] else 'N':>6}"
        )

    print()
    print("Detection rate (RA) by bias type at target station:")
    for bn in BIAS_NAMES:
        sub = target[target["applied_bias"] == bn]
        if sub.empty:
            continue
        det = sub["detected_ra"].sum()
        tot = len(sub)
        print(f"  {bn:<14}: {det}/{tot} ({100*det//max(tot,1)}%)")
    print()


if __name__ == "__main__":
    main()
