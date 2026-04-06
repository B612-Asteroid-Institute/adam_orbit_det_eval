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

    # Dry-run: print scenario names without executing:
    python scripts/10_run_isolation_study.py --dry-run
"""

import argparse
import io
import contextlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bias catalogue — canonical parameters used across all stations
# ---------------------------------------------------------------------------

STEP_MJD = 59945.0  # 2023-01-01


def _get_bias_catalogue():
    """Build the bias catalogue dict (deferred import so module loads fast)."""
    import adam_orbit_det_eval.simulation.bias_models as bm
    return {
        "clean":    [],
        "constant": [bm.ConstantBias(delta_ra=0.5, delta_dec=0.0)],
        "timing":   [bm.TimingBias(delta_t_sec=1.0)],
        "mag_dep":  [bm.MagnitudeDependentBias(slope_ra=0.05, slope_dec=0.02)],
        "epoch":    [bm.CatalogEpochBias(epoch_error_years=5.0)],
        "seasonal": [bm.SeasonalBias(amplitude_ra=0.3, amplitude_dec=0.1)],
        "step":     [bm.StepChangeBias(
            delta_ra_before=0.0, delta_dec_before=0.0,
            delta_ra_after=0.4, delta_dec_after=-0.2,
            change_mjd=STEP_MJD,
        )],
        "dcr":      [bm.DCRBias(bandpass_nm=200.0, ref_wavelength_nm=550.0)],
        "trailing": [bm.TrailingBias(trailing_factor=0.1)],
    }


BIAS_NAMES = ["clean", "constant", "timing", "mag_dep", "epoch",
              "seasonal", "step", "dcr", "trailing"]


# ---------------------------------------------------------------------------
# Phase 1 station definitions
# ---------------------------------------------------------------------------

PHASE1_STATIONS = [
    dict(fake_code="AA00", real_code="F51",  noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA01", real_code="G96",  noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA02", real_code="F52",  noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA03", real_code="703",  noise_ra=0.35, noise_dec=0.35, astcat="Gaia2"),
    dict(fake_code="AA04", real_code="691",  noise_ra=0.35, noise_dec=0.35, astcat="UCAC4"),
    dict(fake_code="AA05", real_code="W84",  noise_ra=0.20, noise_dec=0.20, astcat="Gaia3E"),
    dict(fake_code="AA06", real_code="W68",  noise_ra=0.25, noise_dec=0.25, astcat="Gaia3E"),
    dict(fake_code="AA07", real_code="T09",  noise_ra=0.25, noise_dec=0.25, astcat="Gaia3E"),
    dict(fake_code="AA08", real_code="V00",  noise_ra=0.35, noise_dec=0.35, astcat="Gaia2"),
]


def build_isolation_map(target_fake_code: str, bias_name: str):
    """
    Return an ObservatoryMap where *target_fake_code* has *bias_name* applied
    and all other stations are clean (noise only, no systematic bias).
    """
    from adam_orbit_det_eval.simulation import FakeObservatory, ObservatoryMap

    catalogue = _get_bias_catalogue()
    assignments = []
    for stn in PHASE1_STATIONS:
        if stn["fake_code"] == target_fake_code and bias_name != "clean":
            biases = catalogue[bias_name]
        else:
            biases = []

        assignments.append(
            FakeObservatory(
                fake_code=stn["fake_code"],
                real_code=stn["real_code"],
                noise_sigma_ra=stn["noise_ra"],
                noise_sigma_dec=stn["noise_dec"],
                biases=biases,
                astcat=stn["astcat"],
            )
        )
    return ObservatoryMap(assignments)


def scenario_id(fake_code: str, bias_name: str) -> str:
    return f"{fake_code}_{bias_name}"


def list_scenarios(
    stations: list | None,
    biases: list | None,
) -> list[tuple[str, str]]:
    all_fake = [s["fake_code"] for s in PHASE1_STATIONS]
    target_stations = stations if stations else all_fake
    target_biases = biases if biases else BIAS_NAMES
    return [
        (fc, bn)
        for fc in all_fake
        for bn in BIAS_NAMES
        if fc in target_stations and bn in target_biases
    ]


# ---------------------------------------------------------------------------
# Per-scenario pipeline
# ---------------------------------------------------------------------------

def run_scenario(
    fake_code: str,
    bias_name: str,
    obs_template,
    truth_orbits,
    object_ids: list,
    propagator_class,
    output_base: Path,
    cache_dir: Path,
    max_processes: int,
    threshold_arcsec: float,
    force: bool,
) -> list | None:
    """
    Run full pipeline for one isolation scenario.

    Returns a list of recovery row dicts (one per station), tagged with
    scenario/target metadata, or None on failure.
    """
    from adam_orbit_det_eval.simulation import SimulationConfig, SimulationDataset
    from adam_orbit_det_eval.looo.core import LOOOConfig

    sname = scenario_id(fake_code, bias_name)
    scenario_dir = output_base / sname
    dataset_dir = scenario_dir / "datasets" / "default"
    looo_dir = scenario_dir / "looo_results" / "default"
    analysis_dir = scenario_dir / "analysis" / "default"
    recovery_dir = scenario_dir / "recovery" / "default"

    obs_out = dataset_dir / "mpc_observations.parquet"
    looo_out = looo_dir / "looo_results.parquet"
    obs_stats_path = analysis_dir / "observatory_stats.parquet"

    # --- Dataset generation ---
    if obs_out.exists() and not force:
        logger.info(f"  [{sname}] Dataset exists, skipping generation.")
    else:
        obs_map = build_isolation_map(fake_code, bias_name)
        config = SimulationConfig(
            run_id=sname,
            objects=object_ids,
            observatory_map=obs_map,
            propagator_class=propagator_class,
            noise_seed=42,
            looo_config=LOOOConfig(),
        )
        SimulationDataset(config).generate(
            obs_template=obs_template,
            truth_orbits=truth_orbits,
            output_dir=dataset_dir,
            cache_dir=cache_dir,
            force=force,
        )

    # --- LOOO pipeline ---
    if looo_out.exists() and not force:
        logger.info(f"  [{sname}] LOOO results exist, skipping pipeline.")
    else:
        looo_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_looo(
                dataset_dir=dataset_dir,
                looo_out=looo_out,
                propagator_class=propagator_class,
                max_processes=max_processes,
                force=force,
            )
        except Exception as exc:
            logger.error(f"  [{sname}] LOOO failed: {exc}", exc_info=True)
            return None

    # --- Analysis step ---
    if obs_stats_path.exists() and not force:
        logger.info(f"  [{sname}] Analysis exists, skipping.")
    else:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_analysis(looo_out=looo_out, analysis_dir=analysis_dir)
        except Exception as exc:
            logger.error(f"  [{sname}] Analysis failed: {exc}", exc_info=True)
            return None

    # --- Recovery evaluation ---
    truth_csv = dataset_dir / "truth_biases.csv"
    if not obs_stats_path.exists() or not truth_csv.exists():
        logger.warning(f"  [{sname}] Missing inputs for evaluation, skipping.")
        return None

    try:
        from adam_orbit_det_eval.simulation import evaluate_recovery, print_recovery_summary
        recovery_df = evaluate_recovery(
            observatory_stats_parquet=obs_stats_path,
            truth_biases_csv=truth_csv,
            threshold_arcsec=threshold_arcsec,
        )
        recovery_df["scenario"] = sname
        recovery_df["target_station"] = fake_code
        recovery_df["applied_bias"] = bias_name
        recovery_df["is_target"] = recovery_df["fake_code"] == fake_code

        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_df.to_csv(recovery_dir / "recovery_report.csv", index=False)

        summary_txt = recovery_dir / "recovery_summary.txt"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_recovery_summary(recovery_df)
        summary_txt.write_text(buf.getvalue())

        return recovery_df.to_dict("records")

    except Exception as exc:
        logger.error(f"  [{sname}] Evaluation failed: {exc}", exc_info=True)
        return None


def _run_looo(dataset_dir, looo_out, propagator_class, max_processes, force=False):
    import shutil
    from mpcq.observations import MPCObservations
    from mpcq.orbits import MPCOrbits
    from adam_orbit_det_eval.looo import run_looo_pipeline
    from adam_orbit_det_eval.looo.core import LOOOConfig

    # Clean up stale input snapshots and checkpoints so run_looo_pipeline always
    # writes fresh copies consistent with the current dataset and propagator.
    if force:
        looo_dir = looo_out.parent
        for stale in ["_obs_input.parquet", "_orbits_input.parquet"]:
            p = looo_dir / stale
            if p.exists():
                p.unlink()
                logger.info(f"    Removed stale {stale}")
        ckpt_dir = looo_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info("    Removed stale checkpoints/")

    obs = MPCObservations.from_parquet(dataset_dir / "mpc_observations.parquet")
    orbits = MPCOrbits.from_parquet(dataset_dir / "mpc_orbits.parquet")
    config = LOOOConfig(
        min_obs_held_out=1,
        min_obs_remaining=6,
        min_arc_length_days=7.0,
        max_held_out_fraction=0.8,
    )
    run_looo_pipeline(
        mpc_observations=obs,
        mpc_orbits=orbits,
        propagator_class=propagator_class,
        output_path=looo_out,
        config=config,
        object_ids=None,
        max_processes=max_processes,
        write_interval=50,
        sigma_model="veres2017",
    )


def _run_analysis(looo_out, analysis_dir):
    import pyarrow.parquet as pq
    from adam_orbit_det_eval.looo.core import LOOOResult
    from adam_orbit_det_eval.looo.analysis import compute_observatory_stats

    results = LOOOResult(pq.read_table(looo_out))
    obs_stats = compute_observatory_stats(results, min_obs_per_stn=5)
    obs_stats.to_parquet(analysis_dir / "observatory_stats.parquet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    obs_path = args.input_dir / "mpc_observations.parquet"
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

    all_rows = []
    failed = []

    scenario_parallelism = getattr(args, "scenario_parallelism", 1)

    def _run_one(fc, bn, idx, total):
        sname = scenario_id(fc, bn)
        logger.info(f"[{idx}/{total}] {sname}")
        return run_scenario(
            fake_code=fc,
            bias_name=bn,
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

    if scenario_parallelism <= 1:
        for i, (fake_code, bias_name) in enumerate(scenarios):
            sname = scenario_id(fake_code, bias_name)
            try:
                rows = _run_one(fake_code, bias_name, i + 1, len(scenarios))
                if rows:
                    all_rows.extend(rows)
                else:
                    failed.append(sname)
            except Exception as exc:
                logger.error(f"Scenario {sname} failed: {exc}", exc_info=True)
                failed.append(sname)
    else:
        logger.info(
            f"Running {len(scenarios)} scenarios with scenario_parallelism={scenario_parallelism} "
            f"(total workers ≈ {scenario_parallelism * args.max_processes})"
        )
        # ThreadPoolExecutor: scenarios dispatch work; CPU parallelism comes from
        # the ProcessPoolExecutor inside each run_scenario → _run_looo call.
        with ThreadPoolExecutor(max_workers=scenario_parallelism) as pool:
            futures = {
                pool.submit(_run_one, fc, bn, i + 1, len(scenarios)): scenario_id(fc, bn)
                for i, (fc, bn) in enumerate(scenarios)
            }
            for fut in as_completed(futures):
                sname = futures[fut]
                try:
                    rows = fut.result()
                    if rows:
                        all_rows.extend(rows)
                    else:
                        failed.append(sname)
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
    import pandas as pd

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

    # Detection rate by bias type
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
