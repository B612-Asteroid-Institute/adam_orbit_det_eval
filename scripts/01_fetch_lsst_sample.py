#!/usr/bin/env python3
"""
01_fetch_lsst_sample.py
=======================
Fetch objects with the most LSST (X05 / Rubin Observatory) observations
from BigQuery, along with all other observatories' observations for those
objects (needed to evaluate other stations against the LSST-derived orbit).

This is a companion to 01_fetch_mpc_sample.py, optimised for selecting
objects that are well-observed by a specific reference station rather than
a random cross-section of the MPC catalog.

Usage
-----
    python scripts/01_fetch_lsst_sample.py \\
        --reference-stn X05 \\
        --n-objects 100 \\
        --min-ref-obs 20 \\
        --output-dir data/looo_sample_lsst100

The output is identical in format to 01_fetch_mpc_sample.py so all
downstream scripts work unchanged.
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
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET_ID)
    p.add_argument(
        "--reference-stn",
        default="X05",
        help="Observatory code to select objects by (default: X05 = Rubin/LSST)",
    )
    p.add_argument(
        "--n-objects",
        type=int,
        default=100,
        help="Number of objects to fetch (default: 100)",
    )
    p.add_argument(
        "--min-ref-obs",
        type=int,
        default=20,
        help="Minimum observations from the reference station per object (default: 20)",
    )
    p.add_argument(
        "--min-other-obs",
        type=int,
        default=10,
        help="Minimum observations from non-reference stations per object (default: 10). "
             "Objects with fewer cross-observatory observations are less useful for calibration.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_sample_lsst100"),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
    )
    return p.parse_args()


def build_top_ref_stn_query(project: str, dataset_id: str, reference_stn: str,
                             n_objects: int, min_ref_obs: int) -> str:
    """
    Find the objects with the most observations from the reference station.

    Joins public_obs_sbn → public_current_identifications → public_mpc_orbits
    to resolve canonical unpacked_primary_provisional_designation for each object.
    Only objects that have an orbit solution in public_mpc_orbits are returned.
    """
    return f"""
SELECT
    orb.unpacked_primary_provisional_designation AS provid,
    COUNT(obs.obsid) AS n_ref_obs
FROM `{project}.{dataset_id}.public_obs_sbn` AS obs
JOIN `{project}.{dataset_id}.public_current_identifications` AS ci
    ON obs.provid = ci.unpacked_primary_provisional_designation
    OR obs.provid = ci.unpacked_secondary_provisional_designation
JOIN `{project}.{dataset_id}.public_mpc_orbits` AS orb
    ON ci.unpacked_primary_provisional_designation = orb.unpacked_primary_provisional_designation
WHERE obs.stn = '{reference_stn}'
  AND obs.status = 'P'
GROUP BY provid
HAVING n_ref_obs >= {min_ref_obs}
ORDER BY n_ref_obs DESC
LIMIT {n_objects + 30}
""".strip()


def main():
    args = parse_args()

    if args.output_dir.exists() and not args.overwrite:
        logger.error(
            f"Output directory already exists: {args.output_dir}\n"
            "Use --overwrite to allow overwriting."
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

    # --- Find top-N objects by reference-station observation count ---
    sample_sql = build_top_ref_stn_query(
        args.project, args.dataset_id, args.reference_stn,
        args.n_objects, args.min_ref_obs,
    )
    logger.info(f"Finding objects with most {args.reference_stn} observations...")
    logger.info(f"Query:\n{sample_sql}")

    result = bq.query(sample_sql).result()
    rows = [(row.provid, row.n_ref_obs) for row in result if row.provid]
    logger.info(f"Found {len(rows)} candidate objects with >= {args.min_ref_obs} {args.reference_stn} obs")
    if not rows:
        logger.error("No objects found — check reference station code and min-ref-obs threshold")
        sys.exit(1)

    # Take top N
    rows = rows[:args.n_objects]
    provids = [r[0] for r in rows]
    ref_counts = {r[0]: r[1] for r in rows}
    logger.info(f"Selected {len(provids)} objects; {args.reference_stn} obs range: "
                f"{min(ref_counts.values())}–{max(ref_counts.values())}")

    # --- Fetch ALL observations (reference + other stations) in batches ---
    BATCH_SIZE = 25
    obs_chunks = []
    for i in range(0, len(provids), BATCH_SIZE):
        batch = provids[i: i + BATCH_SIZE]
        logger.info(
            f"  Observations batch {i // BATCH_SIZE + 1}/"
            f"{(len(provids) - 1) // BATCH_SIZE + 1} ({len(batch)} objects)..."
        )
        obs_chunks.append(client.query_observations(batch))
    all_obs = qv.concatenate(obs_chunks)
    logger.info(f"Fetched {len(all_obs)} total observations")

    # --- Filter: require both min reference obs AND min other-station obs ---
    eligible = []
    for provid in provids:
        obj_mask = pc.equal(all_obs.requested_provid, provid)
        obj_obs = all_obs.apply_mask(obj_mask)
        stn_col = obj_obs.stn.to_pylist()
        n_ref = sum(1 for s in stn_col if s == args.reference_stn)
        n_other = sum(1 for s in stn_col if s != args.reference_stn)
        if n_ref >= args.min_ref_obs and n_other >= args.min_other_obs:
            eligible.append(provid)
        else:
            logger.debug(
                f"  {provid}: {n_ref} {args.reference_stn} obs, "
                f"{n_other} other obs — skipping"
            )

    logger.info(
        f"Objects passing filters (>= {args.min_ref_obs} {args.reference_stn} obs, "
        f">= {args.min_other_obs} other obs): {len(eligible)}"
    )
    if len(eligible) == 0:
        logger.error("No objects passed filters")
        sys.exit(1)

    eligible_set = pa.array(eligible, type=all_obs.requested_provid.type)
    filtered_obs = all_obs.apply_mask(pc.is_in(all_obs.requested_provid, value_set=eligible_set))

    # --- Fetch orbits ---
    orbit_chunks = []
    for i in range(0, len(eligible), BATCH_SIZE):
        batch = eligible[i: i + BATCH_SIZE]
        logger.info(f"  Orbits batch {i // BATCH_SIZE + 1}/{(len(eligible) - 1) // BATCH_SIZE + 1}...")
        orbit_chunks.append(client.query_orbits(batch))
    all_orbits = qv.concatenate(orbit_chunks)
    logger.info(f"Fetched {len(all_orbits)} orbits")

    # --- Write outputs ---
    obs_path = args.output_dir / "mpc_observations.parquet"
    orbits_path = args.output_dir / "mpc_orbits.parquet"
    filtered_obs.to_parquet(obs_path)
    all_orbits.to_parquet(orbits_path)

    ref_obs_by_provid = {}
    for provid in eligible:
        obj_mask = pc.equal(filtered_obs.requested_provid, provid)
        stn_col = filtered_obs.apply_mask(obj_mask).stn.to_pylist()
        ref_obs_by_provid[provid] = sum(1 for s in stn_col if s == args.reference_stn)

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcp_project": args.project,
        "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "reference_stn": args.reference_stn,
        "n_objects_requested": args.n_objects,
        "n_objects_fetched": len(eligible),
        "n_observations": len(filtered_obs),
        "min_ref_obs": args.min_ref_obs,
        "min_other_obs": args.min_other_obs,
        "sample_sql": sample_sql,
        "provids": eligible,
        "n_ref_obs_per_object": ref_obs_by_provid,
    }
    meta_path = args.output_dir / "sample_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Wrote {len(filtered_obs)} observations → {obs_path}")
    logger.info(f"Wrote {len(all_orbits)} orbits         → {orbits_path}")
    logger.info(f"Wrote metadata                        → {meta_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
