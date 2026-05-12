#!/usr/bin/env python3
"""
17_generate_bias_table.py
=========================
Build the per-observatory bias catalogue with bootstrap 95% CIs.

Reads a LOOO results parquet (output of 02_run_looo.py, optionally augmented
with AT/CT via 16_atct_real_data.py) and emits:

    <output-dir>/bias_table.parquet
    <output-dir>/bias_table.csv
    <output-dir>/bias_table_config.json
    <output-dir>/validation_report.txt   (when --validate-anchors is set)

The same script is rerun for bead `ird` against the full MPC-cloud output.
No knobs should need changing beyond `--looo-results` and `--observations`.

Methodology
-----------
- Object-weighted aggregation: per-object mean residuals are computed first,
  then averaged with equal weight across objects in each group.
- Bootstrap CIs: objects (not observations) are resampled with replacement
  for each (obs_code, program_code) group, with a single shared set of
  indices producing 95% CIs for mean, median, and RMS in one pass.
  Default 2,000 resamples (>= 1000 required), seed=42, fully reproducible.
- Per-station outputs include point estimates AND 95% CIs for:
    - mean bias RA/Dec/AT/CT
    - median bias RA/Dec/AT/CT
    - RMS scatter RA/Dec/AT/CT
  plus a `bias_significant` boolean (True when the mean-bias CI excludes
  zero in RA or Dec — i.e. a statistically resolvable bias on at least
  one axis).
- The LOOO catalog intentionally does NOT consume reported rmsra/rmsdec
  for weighting; the bootstrap is over the empirical residual distribution
  per station.
- Groups produced:
    - one row per observatory (program_code null)
    - one row per (observatory, program_code) pair where program_code is
      non-null (the 3,500-obj baseline parquet has no program_code column
      and therefore produces only observatory-level rows)
- AT/CT columns are populated when the input parquet has `residual_at_arcsec`
  / `residual_ct_arcsec`; otherwise those output columns are NaN.

Usage
-----
    python scripts/17_generate_bias_table.py \\
        --looo-results data/looo_results/20260316T190152Z/looo_results.parquet \\
        --observations data/looo_sample_3500/mpc_observations.parquet \\
        --output-dir data/bias_catalog/3500obj \\
        --validate-anchors
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bias_table")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--looo-results",
        type=Path,
        required=True,
        help="Path to LOOO results parquet (from 02_run_looo.py, optionally "
             "augmented by 16_atct_real_data.py).",
    )
    p.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="Path to the source MPC observations parquet (from "
             "01_fetch_mpc_sample.py).  When provided, sigma_model_source and "
             "obs_epoch_start/end columns are populated.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write bias_table.parquet + .csv + config + report.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000).",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap (default: 42).",
    )
    p.add_argument(
        "--min-obs-per-group",
        type=int,
        default=10,
        help="Drop groups with fewer held-out observations (default: 10).",
    )
    p.add_argument(
        "--min-objects-per-group",
        type=int,
        default=3,
        help="Drop groups with fewer distinct objects (default: 3). "
             "Bootstrap CIs are degenerate below ~3 objects.",
    )
    p.add_argument(
        "--max-chi2",
        type=float,
        default=100.0,
        help="Drop rows with hold_in_reduced_chi2 above this (default: 100).",
    )
    p.add_argument(
        "--max-object-chi2",
        type=float,
        default=50.0,
        help="Drop entire objects whose mean held-out chi2 across all stations "
             "exceeds this threshold (default: 50). Filters non-gravitational-"
             "force targets whose inflation is object-level, not station-level. "
             "Matches the existing analysis.py baseline.",
    )
    p.add_argument(
        "--validate-anchors",
        action="store_true",
        help="Compare observatory-level rows against the known-biased / "
             "well-calibrated anchor stations from the 3,500-obj run and "
             "write validation_report.txt.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Observations preparation
# ---------------------------------------------------------------------------


def load_observations_with_mjd(path: Path) -> pd.DataFrame:
    """
    Load the source MPC observations parquet and convert the `obstime`
    struct column (days, nanos) into a scalar `obstime_mjd` float64.

    Only the columns needed downstream are materialised to keep memory
    low on the full-cloud run.
    """
    schema = pq.read_schema(path)
    columns = ["obsid"]
    if "rmsra" in schema.names:
        columns.append("rmsra")
    # Pull obstime only if present; the conversion is below.
    has_obstime = "obstime" in schema.names
    if has_obstime:
        columns.append("obstime")

    table = pq.read_table(path, columns=columns)
    if has_obstime:
        days = pc.struct_field(table.column("obstime"), "days")
        nanos = pc.struct_field(table.column("obstime"), "nanos")
        mjd = pc.add(
            pc.cast(days, pa.float64()),
            pc.divide(pc.cast(nanos, pa.float64()), pa.scalar(86400.0e9)),
        )
        table = table.append_column("obstime_mjd", mjd).drop(["obstime"])

    df = table.to_pandas()
    logger.info(
        "Loaded %d observations with columns %s",
        len(df), list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Anchor-station validation
# ---------------------------------------------------------------------------


def default_anchor_stations():
    """
    Anchor stations with expected values from the object-weighted baseline
    analysis (`data/looo_analysis/veres2017_3500obj_objweighted/
    observatory_stats.parquet`), cross-referenced against
    `runs/run_3500obj/README.md`.

    Note: some README values (e.g. 809=+0.411" RA) come from the
    observation-weighted analysis variant and do NOT match the
    object-weighted numbers — the coordinator confirmed the object-weighted
    variant is the primary validation target, so the anchors below use
    those values.

    "Well-calibrated" = |bias| below ~0.05" in both RA and Dec, NOT
    "CI spans zero" — at N=3,500 objects even sub-arcsec biases are
    statistically resolvable because SEMs drop to ~0.01".
    """
    from adam_orbit_det_eval.looo.bias_table import AnchorStation
    return [
        # Known-biased stations: CI must exclude zero
        AnchorStation("704", expected_bias_ra_arcsec=0.212,
                      expected_bias_dec_arcsec=0.434,
                      ci_excludes_zero=True, tolerance_arcsec=0.05),
        AnchorStation("699", expected_bias_ra_arcsec=0.154,
                      expected_bias_dec_arcsec=0.355,
                      ci_excludes_zero=True, tolerance_arcsec=0.05),
        AnchorStation("703", expected_bias_ra_arcsec=-0.114,
                      expected_bias_dec_arcsec=0.145,
                      ci_excludes_zero=True, tolerance_arcsec=0.05),
        AnchorStation("W84", expected_bias_ra_arcsec=0.076,
                      expected_bias_dec_arcsec=0.070,
                      ci_excludes_zero=True, tolerance_arcsec=0.05),
        AnchorStation("809", expected_bias_ra_arcsec=0.257,
                      expected_bias_dec_arcsec=0.112,
                      ci_excludes_zero=False,  # only ~75 objects, wide CI
                      tolerance_arcsec=0.05),
        # Well-calibrated (|bias| < 0.05")
        AnchorStation("F52", well_calibrated=True,
                      well_calibrated_max_bias_arcsec=0.05),
        AnchorStation("M22", well_calibrated=True,
                      well_calibrated_max_bias_arcsec=0.05),
        AnchorStation("R17", well_calibrated=True,
                      well_calibrated_max_bias_arcsec=0.05),
        AnchorStation("T09", well_calibrated=True,
                      well_calibrated_max_bias_arcsec=0.10),
        AnchorStation("W68", well_calibrated=True,
                      well_calibrated_max_bias_arcsec=0.05),
        # Additional reference points (no pass/fail, shown for context)
        AnchorStation("691", expected_bias_ra_arcsec=-0.134,
                      expected_bias_dec_arcsec=0.177,
                      tolerance_arcsec=0.05),
        AnchorStation("644", expected_bias_ra_arcsec=0.101,
                      expected_bias_dec_arcsec=0.343,
                      tolerance_arcsec=0.05),
        AnchorStation("705", expected_bias_ra_arcsec=0.130,
                      expected_bias_dec_arcsec=0.118,
                      tolerance_arcsec=0.05),
    ]


def format_validation_report(
    validation_df: pd.DataFrame,
    anchors,
) -> str:
    lines = []
    lines.append("Anchor-station validation")
    lines.append("=" * 72)
    lines.append(
        f"{'stn':<6} {'present':>8} {'bias_RA':>9} {'bias_Dec':>9} "
        f"{'CI_RA':>20} {'CI_Dec':>20}  checks"
    )
    lines.append("-" * 100)
    anchor_map = {a.obs_code: a for a in anchors}
    n_pass = 0
    n_total = 0
    for _, row in validation_df.iterrows():
        anchor = anchor_map[row["obs_code"]]
        present = row["present"]
        if not present:
            lines.append(f"{row['obs_code']:<6} {'MISSING':>8}")
            n_total += 1
            continue
        bias_ra = row["bias_ra_arcsec"]
        bias_dec = row["bias_dec_arcsec"]
        ci_ra = row["ci_ra"]
        ci_dec = row["ci_dec"]

        checks = []
        if anchor.expected_bias_ra_arcsec is not None:
            checks.append(f"RA_mag={'ok' if row['bias_ra_ok'] else 'FAIL'}")
        if anchor.expected_bias_dec_arcsec is not None:
            checks.append(f"Dec_mag={'ok' if row['bias_dec_ok'] else 'FAIL'}")
        if anchor.ci_excludes_zero:
            checks.append(f"CI_RA!=0 {'ok' if row['ci_ra_ok'] else 'FAIL'}")
            checks.append(f"CI_Dec!=0 {'ok' if row['ci_dec_ok'] else 'FAIL'}")
        elif anchor.well_calibrated:
            checks.append(f"|bias_RA|<{anchor.well_calibrated_max_bias_arcsec} "
                          f"{'ok' if row['ci_ra_ok'] else 'FAIL'}")
            checks.append(f"|bias_Dec|<{anchor.well_calibrated_max_bias_arcsec} "
                          f"{'ok' if row['ci_dec_ok'] else 'FAIL'}")

        row_pass = all([
            row.get("bias_ra_ok", True),
            row.get("bias_dec_ok", True),
            row.get("ci_ra_ok", True),
            row.get("ci_dec_ok", True),
        ])
        if row_pass:
            n_pass += 1
        n_total += 1

        lines.append(
            f"{row['obs_code']:<6} {'yes':>8} "
            f"{bias_ra:>+9.3f} {bias_dec:>+9.3f} "
            f"[{ci_ra[0]:+.3f},{ci_ra[1]:+.3f}]   "
            f"[{ci_dec[0]:+.3f},{ci_dec[1]:+.3f}]   "
            + ", ".join(checks)
        )

    lines.append("")
    lines.append(f"Pass: {n_pass}/{n_total}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.looo_results.exists():
        logger.error("LOOO results not found: %s", args.looo_results)
        return 1
    if args.observations is not None and not args.observations.exists():
        logger.error("Observations parquet not found: %s", args.observations)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load LOOO results as a pandas DataFrame.  We use pandas here (not
    # quivr) because the downstream aggregation is groupby-heavy and
    # pandas handles that far more ergonomically than pyarrow compute.
    # -----------------------------------------------------------------
    logger.info("Loading LOOO results from %s", args.looo_results)
    looo_table = pq.read_table(args.looo_results)
    schema_names = looo_table.schema.names
    logger.info("LOOO schema: %s", schema_names)
    looo_df = looo_table.to_pandas()
    logger.info(
        "Loaded %d held-out observations across %d objects and %d stations",
        len(looo_df),
        looo_df["object_id"].nunique(),
        looo_df["stn"].nunique(),
    )

    # -----------------------------------------------------------------
    # Optional observations join for provenance columns
    # -----------------------------------------------------------------
    obs_df = None
    if args.observations is not None:
        obs_df = load_observations_with_mjd(args.observations)

    # -----------------------------------------------------------------
    # Compute bias table
    # -----------------------------------------------------------------
    from adam_orbit_det_eval.looo.bias_table import (
        BootstrapConfig,
        compute_bias_table,
        validate_against_anchors,
    )

    bootstrap_cfg = BootstrapConfig(
        n_resamples=args.n_bootstrap,
        random_seed=args.random_seed,
    )

    logger.info(
        "Running bootstrap: n_resamples=%d seed=%d",
        bootstrap_cfg.n_resamples, bootstrap_cfg.random_seed,
    )
    bias_table = compute_bias_table(
        looo_df=looo_df,
        observations_df=obs_df,
        min_obs_per_group=args.min_obs_per_group,
        min_objects_per_group=args.min_objects_per_group,
        max_hold_in_reduced_chi2=args.max_chi2,
        max_object_mean_chi2=args.max_object_chi2,
        bootstrap=bootstrap_cfg,
    )

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    parquet_path = args.output_dir / "bias_table.parquet"
    csv_path = args.output_dir / "bias_table.csv"
    config_path = args.output_dir / "bias_table_config.json"

    bias_table.to_parquet(parquet_path, index=False)
    bias_table.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", parquet_path, len(bias_table))
    logger.info("Wrote %s", csv_path)

    config_dict = {
        "looo_results": str(args.looo_results),
        "observations": str(args.observations) if args.observations else None,
        "output_dir": str(args.output_dir),
        "min_obs_per_group": args.min_obs_per_group,
        "min_objects_per_group": args.min_objects_per_group,
        "max_hold_in_reduced_chi2": args.max_chi2,
        "max_object_mean_chi2": args.max_object_chi2,
        "bootstrap": asdict(bootstrap_cfg),
        "n_rows": int(len(bias_table)),
        "n_observatory_rows": int((bias_table["program_code"].isna()).sum()),
        "n_program_rows": int((bias_table["program_code"].notna()).sum()),
        "dims_populated": [
            d for d in ("ra", "dec", "at", "ct")
            if not bias_table[f"bias_{d}_arcsec"].isna().all()
        ],
    }
    config_path.write_text(json.dumps(config_dict, indent=2))
    logger.info("Wrote %s", config_path)

    n_obs_rows = config_dict["n_observatory_rows"]
    n_prog_rows = config_dict["n_program_rows"]
    logger.info(
        "Bias table: %d observatory rows + %d (obs, program_code) rows",
        n_obs_rows, n_prog_rows,
    )

    # -----------------------------------------------------------------
    # Anchor-station validation
    # -----------------------------------------------------------------
    if args.validate_anchors:
        anchors = default_anchor_stations()
        validation_df = validate_against_anchors(bias_table, anchors)
        report_text = format_validation_report(validation_df, anchors)
        report_path = args.output_dir / "validation_report.txt"
        report_path.write_text(report_text)
        print("\n" + report_text + "\n")
        logger.info("Wrote %s", report_path)

        # Per-anchor CI widths (mean/median/RMS) — used to confirm that CI
        # widths shrink as n_obs grows and that the bootstrap point estimate
        # tracks the simple object-weighted mean.
        anchor_codes = [a.obs_code for a in anchors]
        obs_only = bias_table[bias_table["program_code"].isna()].set_index("obs_code")
        anchor_rows = obs_only.reindex(anchor_codes)
        widths_lines = []
        widths_lines.append("Anchor-station CI widths (95%, arcsec)")
        widths_lines.append("=" * 96)
        widths_lines.append(
            f"{'stn':<5} {'n_obj':>6} {'n_obs':>8} "
            f"{'bias_ra':>9} {'w(RA)':>7} {'w(med RA)':>10} {'w(RMS RA)':>10} "
            f"{'bias_dec':>9} {'w(Dec)':>7} {'w(med Dec)':>11} {'w(RMS Dec)':>11} "
            f"{'sig':>4}"
        )
        widths_lines.append("-" * 96)
        for code in anchor_codes:
            if code not in anchor_rows.index or pd.isna(anchor_rows.loc[code]["n_obs"]):
                widths_lines.append(f"{code:<5} MISSING")
                continue
            r = anchor_rows.loc[code]
            w_ra = r["bias_ra_ci_high"] - r["bias_ra_ci_low"]
            w_ra_med = r["bias_ra_median_ci_high"] - r["bias_ra_median_ci_low"]
            w_ra_rms = r["rms_ra_ci_high"] - r["rms_ra_ci_low"]
            w_dec = r["bias_dec_ci_high"] - r["bias_dec_ci_low"]
            w_dec_med = r["bias_dec_median_ci_high"] - r["bias_dec_median_ci_low"]
            w_dec_rms = r["rms_dec_ci_high"] - r["rms_dec_ci_low"]
            sig = "Y" if bool(r["bias_significant"]) else "N"
            widths_lines.append(
                f"{code:<5} {int(r['n_objects']):>6d} {int(r['n_obs']):>8d} "
                f"{r['bias_ra_arcsec']:>+9.4f} {w_ra:>7.4f} "
                f"{w_ra_med:>10.4f} {w_ra_rms:>10.4f} "
                f"{r['bias_dec_arcsec']:>+9.4f} {w_dec:>7.4f} "
                f"{w_dec_med:>11.4f} {w_dec_rms:>11.4f} {sig:>4}"
            )
        widths_text = "\n".join(widths_lines)
        widths_path = args.output_dir / "anchor_ci_widths.txt"
        widths_path.write_text(widths_text)
        print("\n" + widths_text + "\n")
        logger.info("Wrote %s", widths_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
