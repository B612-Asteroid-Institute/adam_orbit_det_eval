#!/usr/bin/env python3
"""
03_analyze.py
=============
Load LOOO results and compute per-observatory and per-(observatory, catalog)
summary statistics.

This is the final step in the three-script pipeline:
  01_fetch_mpc_sample.py  →  data/looo_sample/
  02_run_looo.py          →  data/looo_results/looo_results.parquet
  03_analyze.py           →  data/looo_analysis/

Outputs
-------
observatory_stats.parquet
    Per-observatory summary statistics (bias, scatter, chi2 calibration,
    orbit sensitivity).
catalog_stats.parquet
    Per-(observatory, astrometric-catalog) statistics.
analysis_config.json
    Parameters used for this analysis run (for reproducibility).
observatory_summary.txt
    Human-readable table of the top observatories by observation count.

Stratification Filters
----------------------
The analysis can be restricted to rows that meet additional quality cuts,
independently from the cuts applied during the LOOO run itself.  These let
you explore how statistics change as a function of orbit quality:

  --min-obs-remaining   Only include rows where >= N hold-in observations
  --min-arc-length      Only include rows where hold-in arc >= D days
  --max-held-out-frac   Only include rows where held-out fraction <= F
  --max-chi2            Exclude rows where hold-in reduced-chi2 > threshold
  --min-obs-per-stn     Minimum held-out observations per observatory to
                        report (avoids noisy statistics for rare stations)

Usage
-----
    python scripts/03_analyze.py \\
        --input-dir data/looo_results \\
        --output-dir data/looo_analysis \\
        --min-obs-remaining 6 \\
        --min-arc-length 7.0 \\
        --max-chi2 100.0 \\
        --top-n 50
"""

import argparse
import json
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
        "--input-dir",
        type=Path,
        default=Path("data/looo_results"),
        help="Directory containing looo_results.parquet (default: data/looo_results)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_analysis"),
        help="Directory to write output files (default: data/looo_analysis)",
    )
    p.add_argument(
        "--min-obs-remaining",
        type=int,
        default=None,
        help="Only analyse rows with >= N hold-in observations",
    )
    p.add_argument(
        "--min-arc-length",
        type=float,
        default=None,
        help="Only analyse rows with hold-in arc length >= D days",
    )
    p.add_argument(
        "--max-held-out-frac",
        type=float,
        default=None,
        help="Only analyse rows with held-out fraction <= F",
    )
    p.add_argument(
        "--max-chi2",
        type=float,
        default=100.0,
        help="Exclude rows with hold-in reduced-chi2 > this threshold (default: 100.0)",
    )
    p.add_argument(
        "--min-obs-per-stn",
        type=int,
        default=10,
        help="Min held-out observations per observatory to report (default: 10)",
    )
    p.add_argument(
        "--min-obs-per-catalog-group",
        type=int,
        default=10,
        help="Min observations per (stn, catalog) group to report (default: 10)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top observatories to include in the printed summary (default: 50)",
    )
    p.add_argument(
        "--no-catalog-stats",
        action="store_true",
        default=False,
        help="Skip computation of per-(stn, catalog) statistics",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load LOOO results ---
    results_path = args.input_dir / "looo_results.parquet"
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run 02_run_looo.py first.")
        sys.exit(1)

    logger.info(f"Loading LOOO results from {results_path}")
    try:
        from adam_orbit_det_eval.looo.core import LOOOResult
    except ImportError:
        logger.error(
            "adam_orbit_det_eval not installed. "
            "Install with: pip install -e adam_orbit_det_eval/"
        )
        sys.exit(1)

    import pyarrow.parquet as pq
    results = LOOOResult(pq.read_table(results_path))

    import pyarrow.compute as pc
    n_obs = len(results)
    n_objects = len(pc.unique(results.object_id))
    n_stns = len(pc.unique(results.stn))
    logger.info(
        f"Loaded {n_obs} held-out observations across "
        f"{n_objects} objects and {n_stns} observatories"
    )

    # --- Write analysis configuration for reproducibility ---
    analysis_config = {
        "input_results": str(results_path),
        "n_input_rows": n_obs,
        "n_input_objects": int(n_objects),
        "n_input_stns": int(n_stns),
        "min_obs_remaining": args.min_obs_remaining,
        "min_arc_length_days": args.min_arc_length,
        "max_held_out_fraction": args.max_held_out_frac,
        "max_hold_in_reduced_chi2": args.max_chi2,
        "min_obs_per_stn": args.min_obs_per_stn,
        "min_obs_per_catalog_group": args.min_obs_per_catalog_group,
    }
    config_path = args.output_dir / "analysis_config.json"
    config_path.write_text(json.dumps(analysis_config, indent=2))
    logger.info(f"Analysis configuration written to {config_path}")

    # --- Compute per-observatory statistics ---
    from adam_orbit_det_eval.looo.analysis import (
        compute_observatory_stats,
        compute_catalog_stats,
        print_observatory_summary,
    )

    logger.info("Computing per-observatory statistics...")
    obs_stats = compute_observatory_stats(
        results,
        min_obs_remaining=args.min_obs_remaining,
        min_arc_length_days=args.min_arc_length,
        max_held_out_fraction=args.max_held_out_frac,
        max_hold_in_reduced_chi2=args.max_chi2,
        min_obs_per_stn=args.min_obs_per_stn,
    )

    if len(obs_stats) == 0:
        logger.warning("No observatories passed the filters — observatory_stats will be empty.")
    else:
        logger.info(f"Observatory statistics computed for {len(obs_stats)} observatories")

    obs_stats_path = args.output_dir / "observatory_stats.parquet"
    obs_stats.to_parquet(obs_stats_path)
    logger.info(f"Observatory statistics written to {obs_stats_path}")

    # --- Print human-readable summary ---
    summary_path = args.output_dir / "observatory_summary.txt"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_observatory_summary(obs_stats, top_n=args.top_n)
    summary_text = buf.getvalue()
    print(summary_text)
    summary_path.write_text(summary_text)
    logger.info(f"Observatory summary written to {summary_path}")

    # --- Compute per-(stn, catalog) statistics ---
    if not args.no_catalog_stats:
        logger.info("Computing per-(observatory, catalog) statistics...")
        cat_stats = compute_catalog_stats(
            results,
            min_obs_remaining=args.min_obs_remaining,
            min_arc_length_days=args.min_arc_length,
            max_hold_in_reduced_chi2=args.max_chi2,
            min_obs_per_group=args.min_obs_per_catalog_group,
        )

        if len(cat_stats) == 0:
            logger.warning("No (stn, catalog) groups passed the filters.")
        else:
            logger.info(
                f"Catalog statistics computed for {len(cat_stats)} (stn, catalog) groups"
            )

        cat_stats_path = args.output_dir / "catalog_stats.parquet"
        cat_stats.to_parquet(cat_stats_path)
        logger.info(f"Catalog statistics written to {cat_stats_path}")

        # Also print a brief catalog summary
        if len(cat_stats) > 0:
            _print_catalog_summary(cat_stats, top_n=args.top_n)

    logger.info("Analysis complete.")
    logger.info(f"Outputs written to {args.output_dir}")


def _print_catalog_summary(cat_stats, top_n: int = 30) -> None:
    """Print a human-readable summary of (observatory, catalog) statistics."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping catalog summary print")
        return

    df = (
        cat_stats.table.to_pandas()
        .sort_values("n_obs", ascending=False)
        .head(top_n)
    )

    print(f"\n{'STN':<6} {'CATALOG':<10} {'N_obs':>7} "
          f"{'bias_RA\"':>9} {'bias_Dec\"':>10} "
          f"{'RMS_RA\"':>8} {'RMS_Dec\"':>9} {'chi2/obs':>9}")
    print("-" * 80)
    for _, row in df.iterrows():
        cat_label = str(row.astcat) if row.astcat and str(row.astcat) != "None" else "—"
        print(
            f"{row.stn:<6} {cat_label:<10} {int(row.n_obs):>7} "
            f"{row.mean_ra_arcsec:>+9.3f} {row.mean_dec_arcsec:>+10.3f} "
            f"{row.rms_ra_arcsec:>8.3f} {row.rms_dec_arcsec:>9.3f} "
            f"{row.mean_chi2_per_obs:>9.2f}"
        )


if __name__ == "__main__":
    main()
