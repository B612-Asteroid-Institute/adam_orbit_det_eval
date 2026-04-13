#!/usr/bin/env python3
"""
15_fetch_mpc_scale.py
=====================
Fetch ALL eligible numbered objects from BigQuery with pre-filtering
and write sharded output for cloud consumption.

Pre-filters applied in BigQuery:
  1. Exclude comets (orbit_type_int NOT BETWEEN 1 AND 10)
  2. Require min N distinct observatories per object (default: 3)
  3. Require min total observations per object (default: 20)
  4. Optional max-objects cap for pilot runs

Observations are fetched via mpcq.BigQueryMPCClient. The returned
MPCObservations schema includes submitter/tracking info (trksub,
submission_id, obssubid, permid) alongside the usual ra/dec/stn/obstime.

Output is sharded into directories of ~1000 objects each (configurable),
designed for cloud consumption where each pod processes one shard:

    {output_dir}/
        fetch_metadata.json
        shard_000/
            mpc_observations.parquet
            mpc_orbits.parquet
        shard_001/
            ...

Supports resumability: if shards already exist (and --overwrite is not set),
previously-fetched objects are skipped.

Usage
-----
    # Pilot run (100 objects, 25 per shard)
    python scripts/15_fetch_mpc_scale.py \\
        --max-objects 100 --shard-size 25 \\
        --output-dir data/mpc_scale_pilot

    # Full scale (all eligible numbered objects)
    python scripts/15_fetch_mpc_scale.py \\
        --output-dir data/mpc_scale_full
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

# MPC orbit_type_int values for asteroid populations (excludes comets):
#   1=Atira, 2=Aten, 3=Apollo, 4=Amor, 5=Object with q<1.665 AU,
#   6=Hungaria, 7=MBA, 8=Hilda, 9=Jupiter Trojan, 10=Distant Object
ASTEROID_ORBIT_TYPES = (1, 10)  # inclusive range


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"GCP project ID (default: {DEFAULT_PROJECT})",
    )
    p.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"BigQuery dataset ID (default: {DEFAULT_DATASET_ID})",
    )
    p.add_argument(
        "--views-dataset-id",
        default=DEFAULT_VIEWS_DATASET_ID,
        help=f"BigQuery views dataset ID (default: {DEFAULT_VIEWS_DATASET_ID})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mpc_scale"),
        help="Root output directory for sharded parquet files (default: data/mpc_scale)",
    )
    p.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum number of objects to fetch (default: all eligible)",
    )
    p.add_argument(
        "--min-observatories",
        type=int,
        default=3,
        help="Minimum distinct observatories per object (default: 3)",
    )
    p.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Minimum total observations per object (default: 20)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of objects per output shard (default: 1000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic ordering via FARM_FINGERPRINT (default: 42)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing shards instead of resuming",
    )
    return p.parse_args()


def build_eligible_provids_query(
    project: str,
    dataset_id: str,
    min_observations: int,
    min_observatories: int,
    seed: int,
    max_objects: int | None,
) -> str:
    """
    Build a BigQuery SQL that returns eligible provids with pre-filtering.

    The query:
    1. Selects asteroid orbits (orbit_type_int 1-10) with enough total obs
    2. Joins to observations to count distinct observatories per object
    3. Filters to objects with >= min_observatories distinct stations
    4. Orders deterministically via FARM_FINGERPRINT for reproducibility
    """
    limit_clause = f"LIMIT {max_objects}" if max_objects else ""

    return f"""
WITH candidates AS (
    SELECT unpacked_primary_provisional_designation AS provid
    FROM `{project}.{dataset_id}.public_mpc_orbits`
    WHERE nobs_total >= {min_observations}
      AND orbit_type_int BETWEEN {ASTEROID_ORBIT_TYPES[0]} AND {ASTEROID_ORBIT_TYPES[1]}
),
obs_stats AS (
    SELECT
        obs.provid,
        COUNT(DISTINCT obs.stn) AS n_distinct_stns
    FROM `{project}.{dataset_id}.public_obs_sbn` AS obs
    INNER JOIN candidates AS c ON obs.provid = c.provid
    GROUP BY obs.provid
    HAVING COUNT(DISTINCT obs.stn) >= {min_observatories}
)
SELECT c.provid
FROM candidates AS c
INNER JOIN obs_stats AS o ON c.provid = o.provid
ORDER BY FARM_FINGERPRINT(CONCAT(c.provid, CAST({seed} AS STRING)))
{limit_clause}
""".strip()


def load_existing_metadata(output_dir: Path) -> dict | None:
    """Load fetch_metadata.json if it exists, for resumability."""
    meta_path = output_dir / "fetch_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def get_completed_provids(output_dir: Path, metadata: dict | None) -> set[str]:
    """Return the set of provids already fetched in existing shards."""
    if metadata is None:
        return set()
    completed = set()
    for shard_info in metadata.get("shards", []):
        completed.update(shard_info.get("provids", []))
    return completed


def write_shard(
    shard_idx: int,
    shard_provids: list[str],
    client,
    output_dir: Path,
    batch_size: int = 25,
):
    """
    Fetch observations and orbits for a shard's provids and write parquet files.

    Returns (n_observations, n_orbits) or raises on failure.
    """
    import quivr as qv

    shard_dir = output_dir / f"shard_{shard_idx:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Fetch observations in batches
    obs_chunks = []
    for i in range(0, len(shard_provids), batch_size):
        batch = shard_provids[i : i + batch_size]
        logger.info(
            f"  Shard {shard_idx:03d} obs batch "
            f"{i // batch_size + 1}/{(len(shard_provids) - 1) // batch_size + 1} "
            f"({len(batch)} objects)"
        )
        obs_chunks.append(client.query_observations(batch))
    all_obs = qv.concatenate(obs_chunks)

    # Fetch orbits in batches
    orbit_chunks = []
    for i in range(0, len(shard_provids), batch_size):
        batch = shard_provids[i : i + batch_size]
        logger.info(
            f"  Shard {shard_idx:03d} orbits batch "
            f"{i // batch_size + 1}/{(len(shard_provids) - 1) // batch_size + 1}"
        )
        orbit_chunks.append(client.query_orbits(batch))
    all_orbits = qv.concatenate(orbit_chunks)

    # Write parquet
    all_obs.to_parquet(shard_dir / "mpc_observations.parquet")
    all_orbits.to_parquet(shard_dir / "mpc_orbits.parquet")

    return len(all_obs), len(all_orbits)


def main():
    args = parse_args()

    # --- Import dependencies with clear error on missing ---
    try:
        from google.cloud import bigquery as bq_lib
    except ImportError:
        logger.error(
            "google-cloud-bigquery is not installed. "
            "Install it with: pdm add google-cloud-bigquery"
        )
        sys.exit(1)

    try:
        from mpcq.client import BigQueryMPCClient
    except ImportError:
        logger.error("mpcq is not installed. Install it with: pdm add mpcq")
        sys.exit(1)

    # --- Check BigQuery credentials ---
    try:
        bq = bq_lib.Client(project=args.project)
        # Quick connectivity check
        bq.query("SELECT 1").result()
    except Exception as e:
        logger.error(
            f"Cannot connect to BigQuery (project={args.project}). "
            f"Ensure credentials are configured (e.g., gcloud auth application-default login).\n"
            f"Error: {e}"
        )
        sys.exit(1)

    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
        project=args.project,
    )

    # --- Handle resumability ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_meta = None
    completed_provids: set[str] = set()
    if not args.overwrite:
        existing_meta = load_existing_metadata(args.output_dir)
        completed_provids = get_completed_provids(args.output_dir, existing_meta)
        if completed_provids:
            logger.info(
                f"Resuming: {len(completed_provids)} objects already fetched in "
                f"{len(existing_meta.get('shards', []))} shards"
            )

    # --- Query eligible provids ---
    eligible_sql = build_eligible_provids_query(
        args.project,
        args.dataset_id,
        args.min_observations,
        args.min_observatories,
        args.seed,
        args.max_objects,
    )
    logger.info("Querying BigQuery for eligible objects...")
    logger.info(f"Filters: min_obs={args.min_observations}, "
                f"min_observatories={args.min_observatories}, "
                f"orbit_type_int {ASTEROID_ORBIT_TYPES[0]}-{ASTEROID_ORBIT_TYPES[1]}")
    logger.info(f"Query:\n{eligible_sql}")

    result = bq.query(eligible_sql).result()
    all_provids = [row.provid for row in result if row.provid]
    logger.info(f"Found {len(all_provids)} eligible objects")

    # Filter out already-completed provids for resumability
    provids_to_fetch = [p for p in all_provids if p not in completed_provids]
    if len(provids_to_fetch) < len(all_provids):
        logger.info(
            f"After excluding completed: {len(provids_to_fetch)} objects remaining"
        )

    if not provids_to_fetch:
        logger.info("All eligible objects already fetched. Nothing to do.")
        sys.exit(0)

    # --- Shard and fetch ---
    # Determine starting shard index (for resumability)
    start_shard_idx = 0
    shard_metadata = []
    if existing_meta and not args.overwrite:
        start_shard_idx = len(existing_meta.get("shards", []))
        shard_metadata = list(existing_meta.get("shards", []))

    total_obs = sum(s.get("n_observations", 0) for s in shard_metadata)
    total_orbits = sum(s.get("n_orbits", 0) for s in shard_metadata)

    n_new_shards = (len(provids_to_fetch) + args.shard_size - 1) // args.shard_size
    logger.info(
        f"Writing {n_new_shards} new shards "
        f"(shard_size={args.shard_size}, starting at shard_{start_shard_idx:03d})"
    )

    for shard_offset in range(n_new_shards):
        shard_idx = start_shard_idx + shard_offset
        start = shard_offset * args.shard_size
        end = min(start + args.shard_size, len(provids_to_fetch))
        shard_provids = provids_to_fetch[start:end]

        logger.info(
            f"Shard {shard_idx:03d}: {len(shard_provids)} objects "
            f"({start + 1}-{end} of {len(provids_to_fetch)})"
        )

        n_obs, n_orb = write_shard(
            shard_idx, shard_provids, client, args.output_dir
        )
        total_obs += n_obs
        total_orbits += n_orb

        shard_metadata.append({
            "shard_idx": shard_idx,
            "n_objects": len(shard_provids),
            "n_observations": n_obs,
            "n_orbits": n_orb,
            "provids": shard_provids,
        })

        # Update metadata after each shard for crash-resumability
        _write_metadata(
            args, all_provids, shard_metadata, total_obs, total_orbits, eligible_sql
        )

        logger.info(
            f"  Shard {shard_idx:03d} complete: {n_obs} observations, {n_orb} orbits"
        )

    logger.info(
        f"Done. {len(all_provids)} objects total across "
        f"{len(shard_metadata)} shards, "
        f"{total_obs} observations, {total_orbits} orbits"
    )
    logger.info(f"Output: {args.output_dir}")


def _write_metadata(args, all_provids, shard_metadata, total_obs, total_orbits, sql):
    """Write/update the fetch_metadata.json file."""
    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcp_project": args.project,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "filters": {
            "min_observations": args.min_observations,
            "min_observatories": args.min_observatories,
            "orbit_type_int_range": list(ASTEROID_ORBIT_TYPES),
            "max_objects": args.max_objects,
        },
        "seed": args.seed,
        "shard_size": args.shard_size,
        "total_eligible_objects": len(all_provids),
        "total_objects_fetched": sum(s["n_objects"] for s in shard_metadata),
        "total_observations": total_obs,
        "total_orbits": total_orbits,
        "n_shards": len(shard_metadata),
        "eligible_sql": sql,
        "shards": shard_metadata,
    }
    meta_path = args.output_dir / "fetch_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
