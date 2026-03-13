#!/usr/bin/env python3
"""
01_fetch_mpc_sample.py
======================
Pull a sample of MPC observations and reference orbits from BigQuery
and write them to Parquet files for use in the LOOO evaluation pipeline.

This script is designed to be reproducible: running it with the same
parameters will produce the same output (given the same BigQuery snapshot).

Usage
-----
    python scripts/01_fetch_mpc_sample.py \
        --dataset-id <main_bq_dataset_id> \
        --views-dataset-id <views_bq_dataset_id> \
        --sample-type numbered \
        --n-objects 1000 \
        --min-obs 20 \
        --output-dir data/looo_sample \
        --seed 42

Sample types
------------
numbered    : Numbered (well-determined) asteroids. Good for calibration.
              Provides ground-truth orbits for comparison.
neos        : Near-Earth Objects. More scientifically interesting for hazard.
mba         : Main Belt Asteroids. Large N, diverse observatories.
all         : No orbital class filter; random sample.

The output parquet files are used directly by 02_run_looo.py.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-id",
        required=True,
        help="BigQuery main dataset ID (from Analytics Hub subscription)",
    )
    p.add_argument(
        "--views-dataset-id",
        required=True,
        help="BigQuery views dataset ID (from Analytics Hub subscription)",
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
        help="Minimum number of observations per object (default: 20). "
             "Objects with fewer observations are excluded — ensures "
             "enough hold-in observations after removing one observatory.",
    )
    p.add_argument(
        "--min-obs-per-stn",
        type=int,
        default=2,
        help="Minimum observations a single station must contribute to an "
             "object for that (object, station) pair to be testable (default: 2). "
             "Objects with no station meeting this threshold are still included "
             "— they just won't yield LOOO pairs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_sample"),
        help="Directory to write output parquet files (default: data/looo_sample)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to BigQuery MPC replica...")
    try:
        from mpcq.client import BigQueryMPCClient
    except ImportError:
        logger.error("mpcq not installed. Install with: pip install mpcq")
        sys.exit(1)

    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
    )

    # --- Query orbits first to get the candidate pool ---
    # all_orbits() returns the full MPC orbit catalog; we sample from it.
    logger.info(f"Querying full orbit pool from BigQuery (this may take a moment)...")
    all_orbits = client.all_orbits()
    # Use 'provid' column which is the unpacked primary provisional designation
    provid_col = "provid" if "provid" in all_orbits.table.schema.names else "requested_provid"
    provids = [p for p in all_orbits.table.column(provid_col).to_pylist() if p is not None]
    logger.info(f"Found {len(provids)} candidate objects")

    # --- Subsample for manageability ---
    rng = random.Random(args.seed)
    if len(provids) > args.n_objects:
        provids = rng.sample(provids, args.n_objects)
        logger.info(f"Sampled {len(provids)} objects (seed={args.seed})")

    # --- Fetch observations for selected objects ---
    logger.info(f"Fetching observations for {len(provids)} objects...")
    BATCH_SIZE = 50
    all_obs_chunks = []
    for i in range(0, len(provids), BATCH_SIZE):
        batch = provids[i : i + BATCH_SIZE]
        logger.info(f"  Observations batch {i//BATCH_SIZE + 1}/{(len(provids)-1)//BATCH_SIZE + 1}")
        chunk = client.query_observations(batch)
        all_obs_chunks.append(chunk)

    import quivr as qv
    from mpcq.observations import MPCObservations
    all_obs = qv.concatenate(all_obs_chunks)

    # --- Filter: only keep objects with >= min_obs observations ---
    import pyarrow.compute as pc
    obs_counts = all_obs.table.group_by("requested_provid").aggregate([("obsid", "count")])
    eligible_provids = (
        obs_counts
        .filter(pc.greater_equal(obs_counts.column("obsid_count"), args.min_obs))
        .column("requested_provid")
        .to_pylist()
    )
    logger.info(
        f"Objects with >= {args.min_obs} observations: {len(eligible_provids)} "
        f"(removed {len(provids) - len(eligible_provids)})"
    )

    filtered_obs = all_obs.apply_mask(
        pc.is_in(all_obs.requested_provid, pc.cast(
            pc.list_flatten(pc.make_struct(eligible_provids).values()),
            all_obs.requested_provid.type
        ) if False else
        pc.is_in(all_obs.requested_provid,
                 value_set=__import__("pyarrow").array(eligible_provids, type=all_obs.requested_provid.type))
    )

    # --- Fetch reference orbits for eligible objects only ---
    logger.info(f"Fetching reference orbits for {len(eligible_provids)} eligible objects...")
    orbit_chunks = []
    for i in range(0, len(eligible_provids), BATCH_SIZE):
        batch = eligible_provids[i : i + BATCH_SIZE]
        orbit_chunks.append(client.query_orbits(batch))
    from mpcq.orbits import MPCOrbits
    all_orbits_filtered = qv.concatenate(orbit_chunks)

    # --- Write outputs ---
    obs_path = args.output_dir / "mpc_observations.parquet"
    orbits_path = args.output_dir / "mpc_orbits.parquet"

    filtered_obs.to_parquet(obs_path)
    all_orbits_filtered.to_parquet(orbits_path)

    # Write a metadata sidecar for reproducibility
    import json
    meta = {
        "sample_type": args.sample_type,
        "n_objects_requested": args.n_objects,
        "n_objects_fetched": len(eligible_provids),
        "n_observations": len(filtered_obs),
        "min_obs": args.min_obs,
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
    }
    meta_path = args.output_dir / "sample_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Wrote {len(filtered_obs)} observations → {obs_path}")
    logger.info(f"Wrote {len(all_orbits_filtered)} orbits     → {orbits_path}")
    logger.info(f"Wrote metadata                → {meta_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
