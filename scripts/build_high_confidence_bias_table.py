#!/usr/bin/env python3
"""
build_high_confidence_bias_table.py
===================================
Emit the share-ready per-station bias catalog as a stand-alone file trio
(CSV + parquet + provenance JSON).

The source `bias_table.parquet` mixes per-station rollups (`program_code` IS
NULL) with per-(station x program_code) drill-down rows in one file; this
generator extracts only the per-station rollups and applies the publication
small-sample cutoff (n_obs >= 100 AND n_objects >= 20) so collaborators get a
single, well-labeled file that IS the publication-ready catalog. Bootstrap
95% CIs on RA/Dec mean, median, and RMS are preserved verbatim from the
source.

Column names match `bias_table.parquet` 1:1 — no renames. Downstream consumers
can join on `obs_code` against any other v12 catalog artifact without
translation.

AT/CT columns are dropped by default because they are all-null in v12
(bead d5b will populate them in a follow-up catalog). Pass `--include-atct`
to keep them once that lands; the flag is a no-op today but the option is
preserved so the script doesn't need to change later.

Outputs (under --out-dir):
    high_confidence_bias_table.csv      — share-with-team file
    high_confidence_bias_table.parquet  — machine-readable
    high_confidence_bias_table.json     — provenance (source sha256, row
                                          count, filters, generator commit,
                                          generated_at_utc, atct_included)

Idempotent — overwrites cleanly on rerun.

Usage
-----
    python scripts/build_high_confidence_bias_table.py
    python scripts/build_high_confidence_bias_table.py \\
        --source-table data/mpc_scale_results_20260510/bias_catalog_published/bias_table.parquet \\
        --out-dir data/mpc_scale_results_20260510/bias_catalog_published
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_high_confidence_bias_table")

DEFAULT_SOURCE = Path(
    "data/mpc_scale_results_20260510/bias_catalog_published/bias_table.parquet"
)
DEFAULT_OUT_DIR = Path("data/mpc_scale_results_20260510/bias_catalog_published")

MIN_OBS = 100
MIN_OBJECTS = 20

ATCT_COLUMNS = (
    "bias_at_arcsec", "bias_at_ci_low", "bias_at_ci_high",
    "bias_ct_arcsec", "bias_ct_ci_low", "bias_ct_ci_high",
    "bias_at_median_arcsec", "bias_at_median_ci_low", "bias_at_median_ci_high",
    "bias_ct_median_arcsec", "bias_ct_median_ci_low", "bias_ct_median_ci_high",
    "rms_at_arcsec", "rms_at_ci_low", "rms_at_ci_high",
    "rms_ct_arcsec", "rms_ct_ci_low", "rms_ct_ci_high",
    "sem_at_arcsec", "sem_ct_arcsec",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source-table",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to the source bias_table.parquet (default: {DEFAULT_SOURCE}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--include-atct",
        action="store_true",
        help="Keep the AT/CT columns. No-op for v12 (they are all-null); "
             "kept as a forward-compatible flag for the AT/CT-augmented "
             "catalog produced by bead d5b.",
    )
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_for(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.parent if path.is_file() else path,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.source_table.exists():
        logger.error("Source bias_table not found: %s", args.source_table)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", args.source_table)
    df = pq.read_table(args.source_table).to_pandas()
    logger.info("Source rows: %d (columns: %d)", len(df), len(df.columns))

    # Per-station rollups only. compute_bias_table marks them by writing
    # pd.NA into program_code; any non-null value (including the literal
    # empty string "", which appears 100 times in v12) belongs to the
    # per-(station x program_code) drill-down pass.
    if "program_code" not in df.columns:
        logger.error(
            "Source table missing `program_code` column; cannot distinguish "
            "per-station rollups from per-program drill-downs."
        )
        return 1
    is_rollup = df["program_code"].isna()
    rollup_df = df[is_rollup].copy()
    logger.info("Per-station rollups: %d", len(rollup_df))

    # Publication-ready cutoff.
    pub_df = rollup_df[
        (rollup_df["n_obs"] >= MIN_OBS)
        & (rollup_df["n_objects"] >= MIN_OBJECTS)
    ].copy()
    logger.info(
        "After n_obs >= %d AND n_objects >= %d: %d stations",
        MIN_OBS, MIN_OBJECTS, len(pub_df),
    )

    # AT/CT handling. Drop unless --include-atct is set; keep the column
    # `program_code` out of the output entirely (it's null on every row by
    # construction here).
    drop_cols = ["program_code"]
    atct_present = any(c in pub_df.columns for c in ATCT_COLUMNS)
    atct_populated = False
    if atct_present:
        # Are they actually populated, or all-null v12 placeholders?
        for c in ATCT_COLUMNS:
            if c in pub_df.columns and pub_df[c].notna().any():
                atct_populated = True
                break

    if not args.include_atct:
        drop_cols.extend(c for c in ATCT_COLUMNS if c in pub_df.columns)
        logger.info(
            "Dropping AT/CT columns (--include-atct not set). "
            "Source AT/CT populated: %s",
            atct_populated,
        )
    else:
        logger.info("Keeping AT/CT columns (--include-atct set). Populated: %s",
                    atct_populated)

    pub_df = pub_df.drop(columns=[c for c in drop_cols if c in pub_df.columns])

    # Stable, deterministic ordering.
    pub_df = pub_df.sort_values("obs_code").reset_index(drop=True)

    out_parquet = args.out_dir / "high_confidence_bias_table.parquet"
    out_csv = args.out_dir / "high_confidence_bias_table.csv"
    out_json = args.out_dir / "high_confidence_bias_table.json"

    pub_df.to_parquet(out_parquet, index=False)
    pub_df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_parquet, len(pub_df))
    logger.info("Wrote %s", out_csv)

    provenance = {
        "source_table": str(args.source_table),
        "source_sha256": sha256_file(args.source_table),
        "row_count": int(len(pub_df)),
        "columns": list(pub_df.columns),
        "filters_applied": {
            "program_code_null": True,
            "min_n_obs": MIN_OBS,
            "min_n_objects": MIN_OBJECTS,
        },
        "atct_included": bool(args.include_atct),
        "atct_populated_in_source": bool(atct_populated),
        "generator_script": __file__,
        "generator_git_commit": git_commit_for(Path(__file__).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    logger.info("Wrote %s", out_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
