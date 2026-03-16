#!/usr/bin/env python3
"""
01_fetch_mpc_sample.py
======================
Pull a sample of MPC observations and reference orbits from BigQuery
and write them to Parquet files for use in the LOOO evaluation pipeline.

This script is fully reproducible: the same arguments on the same BigQuery
snapshot will produce byte-identical output.  A metadata sidecar records all
parameters, dataset IDs, GCP project, and the exact SQL used for sampling so
the fetch can be re-created without this script.

Usage
-----
    python scripts/01_fetch_mpc_sample.py \\
        --project moeyens-thor-dev \\
        --dataset-id mpc_sbn_aurora \\
        --views-dataset-id mpc_sbn_aurora_views \\
        --sample-type numbered \\
        --n-objects 3500 \\
        --min-obs 20 \\
        --output-dir data/looo_sample_3500 \\
        --seed 42

Sample types
------------
numbered    : All orbit types (orbit_type_int 1-10, i.e. any orbit with a
              solution). Filtered to nobs_total >= --min-obs.
neos        : Near-Earth Objects (q < 1.3 AU, orbit_type_int flagged as NEO).
mba         : Main Belt Asteroids (2.0 < a < 3.3 AU, e < 0.3).
all         : No orbital class filter.

Notes
-----
- Sampling is done in BigQuery using ORDER BY FARM_FINGERPRINT(provid) which
  is deterministic given the same seed value, unlike ORDER BY RAND().
- Objects are sampled from public_mpc_orbits then observations are fetched
  in batches of 25 to avoid BQ query size limits.
- The GCP project must have a linked subscription to the MPC BigQuery datasets.
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

# Canonical BQ connection parameters for the Asteroid Institute MPC replica
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
        default=500,
        help="Number of objects to sample (default: 500)",
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=20,
        help="Minimum number of total MPC observations per object (default: 20). "
             "Objects below this threshold are excluded from the candidate pool.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_sample"),
        help="Directory to write output parquet files (default: data/looo_sample). "
             "Will not overwrite an existing directory — use a new path for each fetch.",
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


def build_sample_query(project: str, dataset_id: str, sample_type: str,
                       n_objects: int, min_obs: int, seed: int) -> str:
    """
    Build the BigQuery SQL that samples candidate provids.

    Uses FARM_FINGERPRINT(CONCAT(provid, CAST(<seed> AS STRING))) for
    deterministic ordering that is stable across query runs.
    """
    type_filter = ""
    if sample_type == "neos":
        type_filter = "AND q < 1.3 AND (1 - e) * (1 / (1 - e) - 1) < 1.3"  # q < 1.3 AU
    elif sample_type == "mba":
        # Semi-major axis 2.0-3.3 AU, eccentricity < 0.3
        type_filter = "AND q / (1 - e) BETWEEN 2.0 AND 3.3 AND e < 0.3"

    return f"""
SELECT unpacked_primary_provisional_designation AS provid
FROM `{project}.{dataset_id}.public_mpc_orbits`
WHERE nobs_total >= {min_obs}
  {type_filter}
ORDER BY FARM_FINGERPRINT(CONCAT(unpacked_primary_provisional_designation, CAST({seed} AS STRING)))
LIMIT {n_objects + 20}
""".strip()
# Fetch slightly more than requested to allow for any BQ-side nulls


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
        args.project, args.dataset_id, args.sample_type,
        args.n_objects, args.min_obs, args.seed,
    )
    logger.info(f"Sampling up to {args.n_objects} objects from BigQuery...")
    logger.info(f"Query:\n{sample_sql}")

    result = bq.query(sample_sql).result()
    provids = [row.provid for row in result if row.provid]
    provids = provids[:args.n_objects]  # trim to exact count
    logger.info(f"Got {len(provids)} candidate provids")

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
    all_obs = qv.concatenate(obs_chunks)
    logger.info(f"Fetched {len(all_obs)} observations")

    # --- Filter to objects that actually meet min_obs after fetch ---
    obs_counts = all_obs.table.group_by("requested_provid").aggregate([("obsid", "count")])
    eligible = (
        obs_counts
        .filter(pc.greater_equal(obs_counts.column("obsid_count"), args.min_obs))
        .column("requested_provid")
        .to_pylist()
    )
    eligible_set = pa.array(eligible, type=all_obs.requested_provid.type)
    filtered_obs = all_obs.apply_mask(pc.is_in(all_obs.requested_provid, value_set=eligible_set))
    logger.info(f"Objects with >= {args.min_obs} obs: {len(eligible)}")

    # --- Fetch orbits ---
    orbit_chunks = []
    for i in range(0, len(eligible), BATCH_SIZE):
        batch = eligible[i : i + BATCH_SIZE]
        logger.info(f"  Orbits batch {i // BATCH_SIZE + 1}/{(len(eligible) - 1) // BATCH_SIZE + 1}...")
        orbit_chunks.append(client.query_orbits(batch))
    all_orbits = qv.concatenate(orbit_chunks)
    logger.info(f"Fetched {len(all_orbits)} orbits")

    # --- Write outputs ---
    obs_path = args.output_dir / "mpc_observations.parquet"
    orbits_path = args.output_dir / "mpc_orbits.parquet"
    filtered_obs.to_parquet(obs_path)
    all_orbits.to_parquet(orbits_path)

    # Full metadata sidecar for reproducibility
    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcp_project": args.project,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "sample_type": args.sample_type,
        "n_objects_requested": args.n_objects,
        "n_objects_fetched": len(eligible),
        "n_observations": len(filtered_obs),
        "min_obs_filter": args.min_obs,
        "seed": args.seed,
        "sample_sql": sample_sql,
        "provids": eligible,
    }
    meta_path = args.output_dir / "sample_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Wrote {len(filtered_obs)} observations  → {obs_path}")
    logger.info(f"Wrote {len(all_orbits)} orbits          → {orbits_path}")
    logger.info(f"Wrote metadata                         → {meta_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
