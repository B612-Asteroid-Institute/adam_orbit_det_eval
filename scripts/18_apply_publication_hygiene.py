#!/usr/bin/env python3
"""
18_apply_publication_hygiene.py
==============================
Apply publication hygiene to a v12 LOOO catalog and emit `_published` siblings.

This script is the reproducible companion to bead `fqv`: it takes a merged
LOOO catalog (output of `13_collect_cloud_results.py`) and produces the
publication-ready siblings used by the bias-table and downstream FCCT14
comparison (bead 5z2):

    <results-dir>/merged_looo_results_published.parquet
    <results-dir>/observatory_stats_published.parquet
    <results-dir>/publication_hygiene_audit.json

Hygiene applied (see `src/adam_orbit_det_eval/looo/publication_hygiene.py`
for the rationale):

1. Drop rows whose obs_id has mode='OCC' in the source MPC observation
   shards. The LOOO pipeline already drops space-based stations (which is
   where every OCC obs in the v12 fetch lives), so this filter is usually
   a no-op; it is a hard guard against future occultation-timing rows.

2. Drop rows whose `stn` is not recognised by the installed adam_core
   observatory table. These rows should not normally exist (the OD prep
   step raises on unknown codes) — when they do it means the cloud image
   was built with a different mpc_obscodes than what we can verify against
   locally. The disposition is documented in
   `docs/mpc-bias-catalog-interpretation.md`.

3. Re-aggregate per-station stats on the cleaned merge using the same
   call the collector uses (`compute_observatory_stats` with
   `max_hold_in_reduced_chi2=None`), then apply the small-sample cutoff
   `n_obs >= 100 AND n_objects >= 20`.

Usage
-----
    python scripts/18_apply_publication_hygiene.py \
        --results-dir data/mpc_scale_results_20260510 \
        --source-obs-dir data/mpc_scale_full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("publication_hygiene")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Catalog directory containing merged_looo_results_filtered.parquet",
    )
    p.add_argument(
        "--source-obs-dir",
        type=Path,
        default=None,
        help="Directory of source MPC observation shards (shard_NNN/mpc_observations.parquet) "
             "used to identify mode='OCC' obs_ids. If omitted, the OCC drop is skipped.",
    )
    p.add_argument(
        "--input-merged",
        default="merged_looo_results_filtered.parquet",
        help="Filename of the input merged catalog under --results-dir "
             "(default: merged_looo_results_filtered.parquet).",
    )
    p.add_argument(
        "--output-merged",
        default="merged_looo_results_published.parquet",
        help="Output filename for the cleaned merged catalog (default: "
             "merged_looo_results_published.parquet).",
    )
    p.add_argument(
        "--output-stats",
        default="observatory_stats_published.parquet",
        help="Output filename for the cleaned per-station stats (default: "
             "observatory_stats_published.parquet).",
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=100,
        help="Small-sample cutoff: minimum n_obs per station (default: 100).",
    )
    p.add_argument(
        "--min-objects",
        type=int,
        default=20,
        help="Small-sample cutoff: minimum n_objects per station (default: 20).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir: Path = args.results_dir
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return 1

    merged_path = results_dir / args.input_merged
    if not merged_path.exists():
        logger.error("Input merged parquet not found: %s", merged_path)
        return 1

    # Imports are local so this script can be invoked without the whole
    # package being importable (e.g. in a stripped-down environment).
    from adam_orbit_det_eval.looo.analysis import compute_observatory_stats
    from adam_orbit_det_eval.looo.core import LOOOResult
    from adam_orbit_det_eval.looo.publication_hygiene import (
        apply_publication_hygiene,
        filter_observatory_stats_small_sample,
        format_hygiene_audit,
    )

    logger.info("Reading merged catalog: %s", merged_path)
    merged_results = LOOOResult(pq.read_table(merged_path))
    logger.info("Input rows: %d", len(merged_results))

    cleaned, hygiene_stats = apply_publication_hygiene(
        merged_results, source_obs_dir=args.source_obs_dir
    )
    logger.info(
        "After hygiene: %d rows in -> %d after OCC drop (%d dropped) "
        "-> %d after unknown-code drop (%d dropped, codes=%s)",
        hygiene_stats.rows_in,
        hygiene_stats.rows_after_occ_drop,
        hygiene_stats.occ_rows_dropped,
        hygiene_stats.rows_after_unknown_stn_drop,
        hygiene_stats.unknown_stn_rows_dropped,
        ", ".join(hygiene_stats.unknown_stn_codes) or "[none]",
    )

    out_merged = results_dir / args.output_merged
    pq.write_table(cleaned.table, out_merged)
    logger.info("Wrote published merged catalog: %s (%d rows)", out_merged, len(cleaned))

    # Re-aggregate per-station stats. Match the collector's call signature:
    # max_hold_in_reduced_chi2=None because bias_filter has already been applied
    # to the input merged_looo_results_filtered.parquet.
    logger.info("Re-aggregating per-station stats on cleaned merge...")
    obs_stats = compute_observatory_stats(cleaned, max_hold_in_reduced_chi2=None)
    logger.info("Aggregated stats: %d stations (before small-sample cutoff)", len(obs_stats))

    # Apply small-sample cutoff.
    published_stats, stats_audit = filter_observatory_stats_small_sample(
        obs_stats, min_obs=args.min_obs, min_objects=args.min_objects
    )
    logger.info(
        "Small-sample cutoff (n_obs >= %d AND n_objects >= %d): %d -> %d stations "
        "(%d dropped)",
        stats_audit["min_obs"],
        stats_audit["min_objects"],
        stats_audit["stations_in"],
        stats_audit["stations_out"],
        stats_audit["stations_dropped"],
    )

    out_stats = results_dir / args.output_stats
    published_stats.to_parquet(out_stats)
    logger.info("Wrote published observatory stats: %s", out_stats)

    audit_payload = {
        "input_merged": str(merged_path),
        "output_merged": str(out_merged),
        "output_stats": str(out_stats),
        "source_obs_dir": str(args.source_obs_dir) if args.source_obs_dir else None,
        "hygiene": hygiene_stats.summary(),
        "small_sample_cutoff": {
            "min_obs": stats_audit["min_obs"],
            "min_objects": stats_audit["min_objects"],
            "stations_in": stats_audit["stations_in"],
            "stations_out": stats_audit["stations_out"],
            "stations_dropped": stats_audit["stations_dropped"],
            "dropped_stations": stats_audit["dropped_stations"],
        },
    }
    audit_path = results_dir / "publication_hygiene_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2))
    logger.info("Wrote audit: %s", audit_path)

    print()
    print(format_hygiene_audit(hygiene_stats, stats_audit))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
