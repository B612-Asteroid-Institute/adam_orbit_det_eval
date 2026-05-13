#!/usr/bin/env python3
"""
concat_v12_shards.py
====================
One-shot helper for bead d5b.

Concatenates the 19 per-shard MPC observations + orbits parquets emitted by
`scripts/15_fetch_mpc_scale.py` into a single observations parquet and a
single orbits parquet, so they can be passed to `scripts/16_atct_real_data.py`
(which expects one file each, not a directory of shards).

Outputs are intermediates and are not committed to git
(data/mpc_scale_*/ is gitignored). Re-runs overwrite.

Verifies the totals against `fetch_metadata.json` and fails loudly on
mismatch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("concat_v12_shards")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shards-dir",
        type=Path,
        default=Path("data/mpc_scale_full"),
        help="Directory containing shard_XXX/ subdirectories and fetch_metadata.json.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mpc_scale_results_20260510"),
        help="Directory to write merged parquets into.",
    )
    return p.parse_args()


def _concat_to_parquet(
    shard_paths: list[Path],
    out_path: Path,
    label: str,
) -> int:
    tables: list[pa.Table] = []
    expected_schema: pa.Schema | None = None
    for sp in shard_paths:
        t = pq.read_table(sp)
        if expected_schema is None:
            expected_schema = t.schema
        elif not t.schema.equals(expected_schema):
            logger.error(
                "%s schema mismatch in %s vs first shard", label, sp
            )
            logger.error("expected: %s", expected_schema)
            logger.error("got:      %s", t.schema)
            sys.exit(2)
        tables.append(t)
    merged = pa.concat_tables(tables)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, out_path)
    logger.info("Wrote %s (%d rows)", out_path, merged.num_rows)
    return merged.num_rows


def main() -> int:
    args = parse_args()

    metadata_path = args.shards_dir / "fetch_metadata.json"
    if not metadata_path.exists():
        logger.error("fetch_metadata.json not found at %s", metadata_path)
        return 1
    metadata = json.loads(metadata_path.read_text())
    expected_n_shards = int(metadata["n_shards"])
    expected_obs = int(metadata["total_observations"])
    expected_orbits = int(metadata["total_orbits"])
    logger.info(
        "metadata: %d shards, %d observations, %d orbits",
        expected_n_shards, expected_obs, expected_orbits,
    )

    shard_dirs = sorted(args.shards_dir.glob("shard_*"))
    if len(shard_dirs) != expected_n_shards:
        logger.error(
            "Found %d shard_* dirs but metadata says %d.",
            len(shard_dirs), expected_n_shards,
        )
        return 1

    obs_paths = [d / "mpc_observations.parquet" for d in shard_dirs]
    orb_paths = [d / "mpc_orbits.parquet" for d in shard_dirs]
    for p in obs_paths + orb_paths:
        if not p.exists():
            logger.error("Missing shard file: %s", p)
            return 1

    obs_out = args.output_dir / "mpc_observations_merged.parquet"
    orb_out = args.output_dir / "mpc_orbits_merged.parquet"

    n_obs = _concat_to_parquet(obs_paths, obs_out, "observations")
    if n_obs != expected_obs:
        logger.error(
            "observation row-count mismatch: got %d, expected %d",
            n_obs, expected_obs,
        )
        return 2

    n_orb = _concat_to_parquet(orb_paths, orb_out, "orbits")
    if n_orb != expected_orbits:
        logger.error(
            "orbit row-count mismatch: got %d, expected %d",
            n_orb, expected_orbits,
        )
        return 2

    logger.info("OK: observations=%d orbits=%d", n_obs, n_orb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
