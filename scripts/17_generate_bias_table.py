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
        --output-dir data/bias_catalog/3500obj
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
        default=None,
        help="Drop entire objects whose mean held-out chi2 across all stations "
             "exceeds this threshold (default: None — filter disabled). "
             "Historically defaulted to 50, but the chi2 implementation uses "
             "reported per-obs sigmas and is contaminated by stations with "
             "pathologically small sigmas (see bead gnm / N86). Pass an "
             "explicit value to re-enable.",
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
