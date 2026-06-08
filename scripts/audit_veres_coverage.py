#!/usr/bin/env python3
"""
audit_veres_coverage.py
=======================
One-shot, read-only audit of Veres 2017 sigma-model coverage for the
NULL-rmsra/rmsdec subset of an MPC-scale fetch (bead ie2, audit item G).

Why
---
Observations whose reported astrometric uncertainties (``rmsra`` /
``rmsdec``) are missing get their sigma filled in by the Veres et al. 2017
model (``adam_orbit_det_eval.utils.get_veres2017_sigma``). That model
resolves a sigma from the observation's ``(stn, astcat)`` pair in three
tiers:

    1. per-(station, catalog) override   (VERES2017_STN_CATALOG_OVERRIDES)
    2. per-catalog default               (VERES2017_CATALOG_DEFAULTS)
    3. global fallback (0.75")           (VERES2017_FALLBACK_SIGMA)

The bead's concern is tier 3: rows that hit the global fallback get a
blunt, one-size-fits-all sigma "that may be wrong". A row served by a
catalog default (tier 2) is NOT a gap — that is the model working as
designed. Coverage is therefore a property of the ``(stn, astcat)`` pair,
not of the station code alone. This audit flags the (stn, astcat)
combinations in the NULL-uncertainty subset that fall through to the
global fallback, then rolls them up by station code (the unit the bead
asks about) and by catalog (the actionable lever for extending the
Veres table).

This script is READ-ONLY: it loads fetch parquets, classifies rows, and
writes a coverage report. It never mutates the fetch.

Usage
-----
    # Audit the v1 reference dataset (default)
    python scripts/audit_veres_coverage.py

    # Audit an arbitrary fetch parquet directory
    python scripts/audit_veres_coverage.py \\
        --fetch-dir data/mpc_scale_pilot \\
        --report data/audit/veres_coverage_report.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

from adam_orbit_det_eval.utils import (
    VERES2017_CATALOG_DEFAULTS,
    VERES2017_STN_CATALOG_OVERRIDES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_veres_coverage")

DEFAULT_FETCH_DIR = Path("data/mpc_scale_results_20260510")
DEFAULT_REPORT = Path("data/audit/veres_coverage_report.csv")

# Columns we actually need — keep the projection narrow; the v12 merged
# parquet is ~7.4M rows.
NEEDED_COLUMNS = ["stn", "astcat", "rmsra", "rmsdec"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--fetch-dir",
        type=Path,
        default=DEFAULT_FETCH_DIR,
        help=f"Directory holding fetch parquets (default: {DEFAULT_FETCH_DIR})",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"CSV report output path (default: {DEFAULT_REPORT})",
    )
    return p.parse_args(argv)


def discover_obs_parquets(fetch_dir: Path) -> list[Path]:
    """
    Find observation parquet(s) under a fetch directory.

    Prefers a single merged parquet; falls back to per-shard parquets, then
    to any ``mpc_observations.parquet`` anywhere beneath the directory.
    """
    merged = fetch_dir / "mpc_observations_merged.parquet"
    if merged.exists():
        return [merged]
    shard_obs = sorted(fetch_dir.glob("_shards/shard_*/mpc_observations.parquet"))
    if shard_obs:
        return shard_obs
    any_obs = sorted(fetch_dir.rglob("mpc_observations.parquet"))
    if any_obs:
        return any_obs
    raise FileNotFoundError(
        f"No observation parquets found under {fetch_dir} "
        "(looked for mpc_observations_merged.parquet, "
        "_shards/shard_*/mpc_observations.parquet, and **/mpc_observations.parquet)"
    )


def _missing_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """
    True where an uncertainty value is effectively missing: NULL, NaN, or
    non-positive. This matches the production fill-in trigger in
    ``mpc_to_od_observations`` (``not isfinite(x) or x <= 0``), so the audit
    classifies exactly the rows the Veres model actually fills.
    """
    is_null = pc.is_null(col)
    is_nan = pc.is_nan(col)
    is_nonpos = pc.less_equal(col, 0.0)
    # NaN/non-positive comparisons yield null on null entries; coalesce so
    # the OR over the three conditions is well-defined.
    return pc.or_(
        is_null, pc.or_(pc.fill_null(is_nan, False), pc.fill_null(is_nonpos, False))
    )


def load_missing_subset(obs_files: list[Path]) -> tuple[pa.Table, int]:
    """
    Load the (stn, astcat) of rows with missing rmsra OR missing rmsdec.

    Returns (subset_table[stn, astcat], total_rows_scanned).
    """
    dataset = pds.dataset([str(p) for p in obs_files], format="parquet")
    total = dataset.count_rows()
    table = dataset.to_table(columns=NEEDED_COLUMNS)
    missing = pc.or_(_missing_mask(table["rmsra"]), _missing_mask(table["rmsdec"]))
    subset = table.filter(missing).select(["stn", "astcat"])
    return subset, total


def coverage_tier(stn: str | None, astcat: str | None) -> str:
    """
    Classify a (stn, astcat) pair by which Veres tier serves it.

    Mirrors get_veres2017_sigma's lookup order.
    """
    if stn and astcat and (stn, astcat) in VERES2017_STN_CATALOG_OVERRIDES:
        return "station_override"
    if astcat and astcat in VERES2017_CATALOG_DEFAULTS:
        return "catalog_default"
    return "global_fallback"


def build_report(subset: pa.Table) -> list[dict]:
    """
    Group the missing-uncertainty subset by (stn, astcat), classify each
    group's Veres tier, and return per-group rows (sorted: gaps first, then
    by descending row count).
    """
    grouped = subset.group_by(["stn", "astcat"]).aggregate([([], "count_all")])
    stns = grouped["stn"].to_pylist()
    astcats = grouped["astcat"].to_pylist()
    counts = grouped["count_all"].to_pylist()

    rows = []
    for stn, astcat, count in zip(stns, astcats, counts):
        tier = coverage_tier(stn, astcat)
        rows.append(
            {
                "station_code": stn if stn else "",
                "catalog": astcat if astcat else "",
                "null_uncertainty_rows": count,
                "veres_tier": tier,
                "is_gap": tier == "global_fallback",
            }
        )
    rows.sort(key=lambda r: (not r["is_gap"], -r["null_uncertainty_rows"]))
    return rows


def write_report(rows: list[dict], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "station_code",
                "catalog",
                "null_uncertainty_rows",
                "veres_tier",
                "is_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    obs_files = discover_obs_parquets(args.fetch_dir)
    logger.info(
        "Auditing %d observation parquet(s) under %s",
        len(obs_files),
        args.fetch_dir,
    )

    subset, total = load_missing_subset(obs_files)
    n_missing = subset.num_rows
    logger.info(
        "Scanned %d rows; %d (%.1f%%) have NULL/NaN/<=0 rmsra or rmsdec",
        total,
        n_missing,
        100.0 * n_missing / total if total else 0.0,
    )

    rows = build_report(subset)
    write_report(rows, args.report)

    gap_rows = [r for r in rows if r["is_gap"]]
    gap_stations = sorted({r["station_code"] for r in gap_rows})
    gap_row_count = sum(r["null_uncertainty_rows"] for r in gap_rows)
    gap_catalogs = sorted({r["catalog"] for r in gap_rows})

    # --- stdout report ----------------------------------------------------
    print()
    print("Veres 2017 fallback coverage audit")
    print("=" * 60)
    print(f"fetch dir                 : {args.fetch_dir}")
    print(f"rows scanned              : {total:,}")
    print(f"NULL-uncertainty rows     : {n_missing:,}")
    print(f"distinct (stn, catalog)   : {len(rows)}")
    print(f"report written            : {args.report}")
    print("-" * 60)
    print(
        "Coverage tiers (a row is a GAP only if it hits the global "
        '0.75" fallback;\ncatalog-default rows are served as designed and '
        "are NOT gaps):"
    )
    for tier in ("station_override", "catalog_default", "global_fallback"):
        tier_rows = [r for r in rows if r["veres_tier"] == tier]
        tier_count = sum(r["null_uncertainty_rows"] for r in tier_rows)
        print(
            f"  {tier:18s}: {tier_count:>12,} rows  ({len(tier_rows)} stn/cat groups)"
        )
    print("-" * 60)

    if not gap_rows:
        print("RESULT: no gaps")
        print(
            "Every NULL-uncertainty (stn, catalog) pair resolves to a Veres "
            "station override or catalog default."
        )
        return 0

    print(
        f"RESULT: {len(gap_stations)} station code(s) with "
        f"{gap_row_count:,} NULL-uncertainty rows hit the global fallback"
    )
    print()
    print('Uncovered (stn, catalog) groups — these hit the 0.75" fallback:')
    print(f"  {'station':<10}{'catalog':<14}{'null_rows':>12}")
    for r in gap_rows:
        print(
            f"  {r['station_code']:<10}{r['catalog'] or '<none>':<14}"
            f"{r['null_uncertainty_rows']:>12,}"
        )
    print()
    print(f"Distinct uncovered station codes : {', '.join(gap_stations)}")
    print(
        "Catalogs driving the gap (extend "
        "VERES2017_CATALOG_DEFAULTS/OVERRIDES) : "
        f"{', '.join(c or '<none>' for c in gap_catalogs)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
