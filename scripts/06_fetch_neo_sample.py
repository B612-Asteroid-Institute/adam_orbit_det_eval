#!/usr/bin/env python3
"""
06_fetch_neo_sample.py
======================
Fetch a sample of well-observed Near-Earth Objects for timing-bias evaluation.

This script targets the highest-observation-count NEOs in the MPC database,
explicitly including Apophis (2004 MN4), Eros (A898 PA), and similar well-observed
objects. Unlike 06_fetch_sim_sample.py it does not rely on arc_length_total being
populated (some prominent NEOs have NULL there despite long arcs).

Outputs (written to --output-dir/):
  mpc_observations.parquet
  mpc_orbits.parquet
  sample_metadata.json

Usage
-----
    python scripts/06_fetch_neo_sample.py \\
        --output-dir data/sim_sample_neo \\
        --n-objects 30 \\
        --min-obs 500
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET_ID = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET_ID = "mpc_sbn_aurora_views"

# Well-known NEOs to always include (provid format used in MPC BQ table).
# These are listed in priority order; duplicates with the BQ sample are deduped.
PRIORITY_PROVIDS = [
    "2004 MN4",   # Apophis (99942) — has NULL arc_length_total in BQ, 9500+ obs
    "A898 PA",    # Eros (433)
    "1929 SH",    # Amor
    "A924 UB",    # Hermes class
    "1998 OH",    # Apollo
    "1985 DO2",   # Amor
    "1994 LY",    # Apollo
    "1972 XA",    # Apollo
    "1981 ET3",   # Apollo (Toutatis)
    "2001 MZ7",   # Amor
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET_ID)
    p.add_argument("--output-dir", type=Path, default=Path("data/sim_sample_neo"))
    p.add_argument("--n-objects", type=int, default=30,
                   help="Total target object count (default: 30)")
    p.add_argument("--min-obs", type=int, default=500,
                   help="Min observations per object when backfilling from BQ (default: 500)")
    p.add_argument("--min-stations", type=int, default=5,
                   help="Min distinct stations per object (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir.exists():
        logger.error(f"Output directory already exists: {args.output_dir}. Aborting.")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from google.cloud import bigquery as bq_lib
        from mpcq.client import BigQueryMPCClient
        import quivr as qv
        import pyarrow as pa
        import pyarrow.compute as pc
        import pandas as pd
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    bq = bq_lib.Client(project=args.project)
    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
        project=args.project,
    )

    # ── 1. Query BQ for top asteroid NEOs by observation count
    # Exclude comets (C/ and P/ designations) — they have non-gravitational forces
    # and their dynamics aren't appropriate for this simulation.
    # Also exclude objects with NULL arc_length_total only if nobs is low;
    # high-nobs objects (e.g. Apophis) are handled via PRIORITY_PROVIDS above.
    backfill_sql = f"""
SELECT unpacked_primary_provisional_designation AS provid, nobs_total
FROM `{args.project}.{args.dataset_id}.public_mpc_orbits`
WHERE q < 1.3
  AND nobs_total >= {args.min_obs}
  AND arc_length_total IS NOT NULL
  AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
  AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
ORDER BY nobs_total DESC
LIMIT {args.n_objects + 20}
""".strip()
    logger.info("Querying BQ for top NEOs by observation count...")
    bq_provids = [r.provid for r in bq.query(backfill_sql).result() if r.provid]
    logger.info(f"  BQ returned {len(bq_provids)} candidates")

    # ── 2. Merge priority list with BQ list, dedup, trim to n_objects
    seen = set()
    ordered = []
    for p in PRIORITY_PROVIDS + bq_provids:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
        if len(ordered) >= args.n_objects + 10:  # extra headroom for station filter
            break
    logger.info(f"Merged candidate list: {len(ordered)} objects")

    # ── 3. Fetch observations in batches
    # BATCH bumped 10→2000 on 2026-06-24 after BQ cost incident.
    BATCH = 2000
    obs_chunks = []
    for i in range(0, len(ordered), BATCH):
        batch = ordered[i: i + BATCH]
        logger.info(f"  Fetching obs batch {i // BATCH + 1}: {batch}")
        try:
            obs_chunks.append(client.query_observations(batch))
        except Exception as e:
            logger.warning(f"  Batch {i // BATCH + 1} failed: {e}")

    if not obs_chunks:
        logger.error("No observations fetched.")
        sys.exit(1)

    all_obs = qv.concatenate(obs_chunks)
    logger.info(f"Total observations fetched: {len(all_obs)}")

    # ── 4. Filter: min_stations
    obs_df = all_obs.table.select(["requested_provid", "stn"]).to_pandas()
    stn_counts = (
        obs_df.groupby("requested_provid")["stn"]
        .nunique()
        .reset_index()
        .rename(columns={"stn": "n_stations"})
    )
    eligible = stn_counts.loc[
        stn_counts["n_stations"] >= args.min_stations, "requested_provid"
    ].tolist()
    logger.info(f"Objects with >= {args.min_stations} distinct stations: {len(eligible)}")

    # Preserve priority ordering in final selection
    priority_set = set(PRIORITY_PROVIDS)
    final_provids = []
    for p in ordered:
        if p in eligible:
            final_provids.append(p)
        if len(final_provids) >= args.n_objects:
            break

    logger.info(f"Final object count: {len(final_provids)}")
    if len(final_provids) < args.n_objects:
        logger.warning(f"Only {len(final_provids)} objects met all criteria "
                       f"(requested {args.n_objects})")

    # Filter observations to final set
    final_arr = pa.array(final_provids, type=all_obs.requested_provid.type)
    filtered_obs = all_obs.apply_mask(
        pc.is_in(all_obs.requested_provid, value_set=final_arr)
    )

    # ── 5. Fetch orbits
    orbit_chunks = []
    for i in range(0, len(final_provids), BATCH):
        batch = final_provids[i: i + BATCH]
        logger.info(f"  Fetching orbits batch {i // BATCH + 1}: {batch}")
        try:
            orbit_chunks.append(client.query_orbits(batch))
        except Exception as e:
            logger.warning(f"  Orbit batch {i // BATCH + 1} failed: {e}")

    all_orbits = qv.concatenate(orbit_chunks)
    logger.info(f"Total orbits fetched: {len(all_orbits)}")

    # ── 6. Write outputs
    obs_path = args.output_dir / "mpc_observations.parquet"
    orbits_path = args.output_dir / "mpc_orbits.parquet"
    filtered_obs.to_parquet(obs_path)
    all_orbits.to_parquet(orbits_path)

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcp_project": args.project,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "sample_type": "neos_high_obs",
        "n_objects_requested": args.n_objects,
        "n_objects_fetched": len(final_provids),
        "n_observations": len(filtered_obs),
        "min_obs_filter": args.min_obs,
        "min_stations_filter": args.min_stations,
        "priority_provids": PRIORITY_PROVIDS,
        "provids": final_provids,
    }
    (args.output_dir / "sample_metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"Wrote {len(filtered_obs)} observations → {obs_path}")
    logger.info(f"Wrote {len(all_orbits)} orbits         → {orbits_path}")
    logger.info(f"Provids included: {final_provids}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
