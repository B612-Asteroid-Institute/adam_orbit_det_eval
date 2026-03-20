#!/usr/bin/env python3
"""
06_fetch_sim_sample.py
======================
Pull a sample of well-observed MPC objects for use as simulation templates.

Compared to ``01_fetch_mpc_sample.py`` this script applies stricter filters:
  - Long observational arcs (≥ ``--min-arc-years`` years)
  - Many total observations (≥ ``--min-total-obs``)
  - Observations from multiple distinct stations (≥ ``--min-stations``)

These criteria ensure that the synthetic observations derived from the sample
will have realistic multi-station cadence and enough temporal baseline for the
LOOO pipeline to run with generous eligibility margins.

The same BQ deterministic sampling approach (FARM_FINGERPRINT) is used so the
fetch is fully reproducible with the same arguments.

Usage
-----
    python scripts/06_fetch_sim_sample.py \\
        --project moeyens-thor-dev \\
        --dataset-id mpc_sbn_aurora \\
        --views-dataset-id mpc_sbn_aurora_views \\
        --n-objects 100 \\
        --min-arc-years 3.0 \\
        --min-total-obs 200 \\
        --min-stations 5 \\
        --output-dir data/sim_sample \\
        --seed 42

Sample types
------------
numbered    : Any numbered asteroid (default).
neos        : Near-Earth Objects (q < 1.3 AU).
mba         : Main Belt Asteroids (2.0 < a < 3.3 AU, e < 0.3).
all         : No orbital class filter.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"GCP project ID (default: {DEFAULT_PROJECT})",
    )
    p.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"BigQuery main dataset ID (default: {DEFAULT_DATASET_ID})",
    )
    p.add_argument(
        "--views-dataset-id",
        default=DEFAULT_VIEWS_DATASET_ID,
        help=f"BigQuery views dataset ID (default: {DEFAULT_VIEWS_DATASET_ID})",
    )
    p.add_argument(
        "--sample-type",
        choices=["numbered", "neos", "mba", "all"],
        default="numbered",
        help="Which orbit class to sample from (default: numbered)",
    )
    p.add_argument(
        "--n-objects",
        type=int,
        default=100,
        help="Number of objects to sample (default: 100)",
    )
    p.add_argument(
        "--min-arc-years",
        type=float,
        default=3.0,
        help="Minimum observational arc length in years (default: 3.0)",
    )
    p.add_argument(
        "--min-total-obs",
        type=int,
        default=200,
        help="Minimum total MPC observations per object (default: 200)",
    )
    p.add_argument(
        "--min-stations",
        type=int,
        default=5,
        help="Minimum distinct observing stations per object (default: 5)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sim_sample"),
        help="Directory to write output parquet files (default: data/sim_sample). "
             "Will not overwrite an existing directory unless --overwrite is set.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Integer seed for deterministic sampling in BigQuery (default: 42). "
             "Uses FARM_FINGERPRINT for stable ordering across runs.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting an existing output directory.",
    )
    return p.parse_args()


def build_sample_query(
    project: str,
    dataset_id: str,
    sample_type: str,
    n_objects: int,
    min_total_obs: int,
    min_arc_years: float,
    seed: int,
) -> str:
    """
    Build the BigQuery SQL that samples candidate provids.

    Filters are applied at the orbits table level (arc length and obs count).
    The station diversity filter is applied after fetching observations.

    Uses FARM_FINGERPRINT(CONCAT(provid, CAST(<seed> AS STRING))) for
    deterministic ordering stable across query runs.
    """
    type_filter = ""
    if sample_type == "neos":
        type_filter = "AND q < 1.3"
    elif sample_type == "mba":
        type_filter = "AND q / (1 - e) BETWEEN 2.0 AND 3.3 AND e < 0.3"

    # The BQ MPC schema has arc_length_total in days directly.
    min_arc_days = int(min_arc_years * 365.25)

    return f"""
SELECT unpacked_primary_provisional_designation AS provid
FROM `{project}.{dataset_id}.public_mpc_orbits`
WHERE nobs_total >= {min_total_obs}
  AND arc_length_total >= {min_arc_days}
  {type_filter}
ORDER BY FARM_FINGERPRINT(CONCAT(unpacked_primary_provisional_designation, CAST({seed} AS STRING)))
LIMIT {n_objects + 50}
""".strip()
# Fetch extra headroom to allow for station-diversity filtering


def main():
    args = parse_args()

    if args.output_dir.exists() and not args.overwrite:
        logger.error(
            f"Output directory already exists: {args.output_dir}\n"
            "Use --overwrite to allow overwriting, or choose a different --output-dir."
        )
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from google.cloud import bigquery as bq_lib
        from mpcq.client import BigQueryMPCClient
        import quivr as qv
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    bq = bq_lib.Client(project=args.project)
    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
        project=args.project,
    )

    # --- Sample candidate provids from BQ ---
    sample_sql = build_sample_query(
        args.project,
        args.dataset_id,
        args.sample_type,
        args.n_objects,
        args.min_total_obs,
        args.min_arc_years,
        args.seed,
    )
    logger.info(f"Sampling up to {args.n_objects} objects from BigQuery...")
    logger.info(f"Query:\n{sample_sql}")

    result = bq.query(sample_sql).result()
    provids = [row.provid for row in result if row.provid]
    logger.info(f"Got {len(provids)} candidate provids from BQ")

    # --- Fetch observations in batches ---
    BATCH_SIZE = 25
    obs_chunks = []
    for i in range(0, len(provids), BATCH_SIZE):
        batch = provids[i : i + BATCH_SIZE]
        logger.info(
            f"  Observations batch {i // BATCH_SIZE + 1}/"
            f"{(len(provids) - 1) // BATCH_SIZE + 1} ({len(batch)} objects)..."
        )
        obs_chunks.append(client.query_observations(batch))

    if not obs_chunks:
        logger.error("No observations fetched.")
        sys.exit(1)

    all_obs = qv.concatenate(obs_chunks)
    logger.info(f"Fetched {len(all_obs)} observations")

    # --- Filter: min_total_obs ---
    obs_counts = all_obs.table.group_by("requested_provid").aggregate(
        [("obsid", "count")]
    )
    eligible_obs = (
        obs_counts
        .filter(pc.greater_equal(obs_counts.column("obsid_count"), args.min_total_obs))
        .column("requested_provid")
        .to_pylist()
    )
    eligible_set = pa.array(eligible_obs, type=all_obs.requested_provid.type)
    filtered_obs = all_obs.apply_mask(
        pc.is_in(all_obs.requested_provid, value_set=eligible_set)
    )
    logger.info(
        f"Objects with >= {args.min_total_obs} obs: {len(eligible_obs)}"
    )

    # --- Filter: min_stations ---
    # Compute distinct station count per provid
    import pandas as pd
    obs_df = filtered_obs.table.select(["requested_provid", "stn"]).to_pandas()
    stn_counts = (
        obs_df.groupby("requested_provid")["stn"]
        .nunique()
        .reset_index()
        .rename(columns={"stn": "n_stations"})
    )
    eligible_stns = stn_counts.loc[
        stn_counts["n_stations"] >= args.min_stations, "requested_provid"
    ].tolist()
    eligible_stns_arr = pa.array(eligible_stns, type=filtered_obs.requested_provid.type)
    filtered_obs = filtered_obs.apply_mask(
        pc.is_in(filtered_obs.requested_provid, value_set=eligible_stns_arr)
    )
    logger.info(
        f"Objects with >= {args.min_stations} distinct stations: {len(eligible_stns)}"
    )

    # Trim to requested N objects
    final_provids = eligible_stns[:args.n_objects]
    final_arr = pa.array(final_provids, type=filtered_obs.requested_provid.type)
    filtered_obs = filtered_obs.apply_mask(
        pc.is_in(filtered_obs.requested_provid, value_set=final_arr)
    )
    logger.info(f"Final sample: {len(final_provids)} objects, {len(filtered_obs)} observations")

    # --- Fetch orbits ---
    orbit_chunks = []
    for i in range(0, len(final_provids), BATCH_SIZE):
        batch = final_provids[i : i + BATCH_SIZE]
        logger.info(
            f"  Orbits batch {i // BATCH_SIZE + 1}/"
            f"{(len(final_provids) - 1) // BATCH_SIZE + 1}..."
        )
        orbit_chunks.append(client.query_orbits(batch))

    all_orbits = qv.concatenate(orbit_chunks)
    logger.info(f"Fetched {len(all_orbits)} orbits")

    # --- Write outputs ---
    obs_path = args.output_dir / "mpc_observations.parquet"
    orbits_path = args.output_dir / "mpc_orbits.parquet"
    filtered_obs.to_parquet(obs_path)
    all_orbits.to_parquet(orbits_path)

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcp_project": args.project,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "sample_type": args.sample_type,
        "n_objects_requested": args.n_objects,
        "n_objects_fetched": len(final_provids),
        "n_observations": len(filtered_obs),
        "min_total_obs_filter": args.min_total_obs,
        "min_arc_years_filter": args.min_arc_years,
        "min_stations_filter": args.min_stations,
        "seed": args.seed,
        "sample_sql": sample_sql,
        "provids": final_provids,
    }
    meta_path = args.output_dir / "sample_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Wrote {len(filtered_obs)} observations  → {obs_path}")
    logger.info(f"Wrote {len(all_orbits)} orbits          → {orbits_path}")
    logger.info(f"Wrote metadata                         → {meta_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
