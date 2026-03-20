#!/usr/bin/env python3
"""
09_evaluate_sim_recovery.py
===========================
Compare injected simulation biases against biases recovered by the LOOO
pipeline and write a recovery report.

This is the final step in the simulation validation pipeline:
  06_fetch_sim_sample.py         →  data/sim_sample/
  07_generate_sim_dataset.py     →  data/sim_products/<run>/datasets/default/
  08_run_sim_pipeline.py         →  data/sim_products/<run>/analysis/default/
  09_evaluate_sim_recovery.py    →  data/sim_products/<run>/recovery/default/

Outputs (written to ``--output-dir/``):
  recovery_report.csv      — one row per fake station, all metrics
  recovery_summary.txt     — human-readable table

Usage
-----
    python scripts/09_evaluate_sim_recovery.py \\
        --analysis-dir data/sim_products/phase1_constant_bias/analysis/default \\
        --truth-biases data/sim_products/phase1_constant_bias/datasets/default/truth_biases.csv \\
        --output-dir data/sim_products/phase1_constant_bias/recovery/default \\
        --threshold 0.05
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing observatory_stats.parquet "
             "(output of 08_run_sim_pipeline.py or 03_analyze.py).",
    )
    p.add_argument(
        "--truth-biases",
        type=Path,
        required=True,
        help="truth_biases.csv produced by 07_generate_sim_dataset.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write recovery_report.csv and recovery_summary.txt.  "
             "Defaults to a 'recovery' subdirectory next to --analysis-dir.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Recovery error threshold (arcsec) for bias detection (default: 0.05).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    analysis_dir = Path(args.analysis_dir).resolve()
    truth_biases = Path(args.truth_biases).resolve()

    if args.output_dir is None:
        output_dir = analysis_dir.parent / "recovery" / analysis_dir.name
    else:
        output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate inputs ---
    obs_stats_path = analysis_dir / "observatory_stats.parquet"
    if not obs_stats_path.exists():
        logger.error(f"observatory_stats.parquet not found: {obs_stats_path}")
        logger.error("Run 08_run_sim_pipeline.py first.")
        sys.exit(1)

    if not truth_biases.exists():
        logger.error(f"truth_biases.csv not found: {truth_biases}")
        logger.error(
            "Run 07_generate_sim_dataset.py first to generate the synthetic dataset."
        )
        sys.exit(1)

    # --- Load and evaluate ---
    try:
        from adam_orbit_det_eval.simulation import evaluate_recovery, print_recovery_summary
    except ImportError as e:
        logger.error(f"Could not import simulation module: {e}")
        sys.exit(1)

    logger.info(f"Observatory stats: {obs_stats_path}")
    logger.info(f"Truth biases: {truth_biases}")
    logger.info(f"Detection threshold: {args.threshold} arcsec")

    recovery_df = evaluate_recovery(
        observatory_stats_parquet=obs_stats_path,
        truth_biases_csv=truth_biases,
        threshold_arcsec=args.threshold,
    )

    if recovery_df.empty:
        logger.warning("Recovery DataFrame is empty — no stations matched between stats and truth.")
    else:
        logger.info(f"Recovery evaluation complete: {len(recovery_df)} fake stations.")

    # --- Print summary ---
    from adam_orbit_det_eval.simulation.evaluate import print_recovery_summary
    print_recovery_summary(recovery_df)

    # --- Write outputs ---
    import io, contextlib

    report_path = output_dir / "recovery_report.csv"
    # Convert bias_params dict column to JSON string for CSV output
    df_out = recovery_df.copy()
    import json
    if "bias_params" in df_out.columns:
        df_out["bias_params_json"] = df_out["bias_params"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )
        df_out = df_out.drop(columns=["bias_params"])
    df_out.to_csv(report_path, index=False)
    logger.info(f"Recovery report → {report_path}")

    summary_path = output_dir / "recovery_summary.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_recovery_summary(recovery_df)
    summary_path.write_text(buf.getvalue())
    logger.info(f"Recovery summary → {summary_path}")

    # Print overall detection stats
    if not recovery_df.empty:
        n = len(recovery_df)
        n_det_ra = int(recovery_df["detected_ra"].sum())
        n_det_dec = int(recovery_df["detected_dec"].sum())
        logger.info(
            f"Detection rate: RA {n_det_ra}/{n} "
            f"({100 * n_det_ra / max(n, 1):.0f}%)  "
            f"Dec {n_det_dec}/{n} "
            f"({100 * n_det_dec / max(n, 1):.0f}%)"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
