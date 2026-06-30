#!/usr/bin/env python3
"""
sample_mba_population.py
========================

Pull a wider main-belt-asteroid sample for the JPL-free held-out test, biased
toward LONG-ARC objects (which carry decades of historical, old-star-catalog
astrometry — the regime where EFCC18 debiasing has material to work on, and
where the n=9 MBA signal appeared). Deterministic sampling via FARM_FINGERPRINT.

Outputs a manifest (``provid``, ``stratum='main_belt'``, + orbit summary) for
``run_heldout_prediction_v2.py --population-manifest``.

Usage
-----
    pdm run python scripts/sample_mba_population.py [--n 300] \\
        [--output data/heldout_mba_wider/mba_manifest.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery as bq_lib

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=300, help="target sample size")
    p.add_argument("--output", type=Path,
                   default=Path("data/heldout_mba_wider/mba_manifest.parquet"))
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--min-arc-days", type=float, default=3000.0,
                   help="min total arc length (long arc => old-catalog material)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bq = bq_lib.Client(project=args.project)
    # Inner-to-mid main belt via perihelion (matches the proven main_belt stratum),
    # long arc for historical astrometry, moderate-to-rich obs.
    sql = f"""
SELECT unpacked_primary_provisional_designation AS provid,
       nobs_total, arc_length_total, q, e
FROM `{args.project}.{args.dataset_id}.public_mpc_orbits`
WHERE q >= 1.7 AND q < 3.3
  AND e < 0.4
  AND arc_length_total > {args.min_arc_days}
  AND nobs_total BETWEEN 80 AND 5000
  AND arc_length_total IS NOT NULL
  AND unpacked_primary_provisional_designation IS NOT NULL
  AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
  AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
ORDER BY MOD(ABS(FARM_FINGERPRINT(
    CONCAT(unpacked_primary_provisional_designation, '_mbawider'))), 1000003)
LIMIT {args.n}
""".strip()
    rows = list(bq.query(sql).result())
    recs = [{
        "provid": r.provid, "stratum": "main_belt",
        "n_obs_orbit": int(r.nobs_total) if r.nobs_total is not None else 0,
        "arc_days_orbit": float(r.arc_length_total) if r.arc_length_total is not None else 0.0,
        "q_au": float(r.q) if r.q is not None else float("nan"),
        "e_orbit": float(r.e) if r.e is not None else float("nan"),
    } for r in rows if r.provid is not None]
    df = pd.DataFrame(recs).drop_duplicates("provid").reset_index(drop=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} MBAs -> {args.output}")
    print(f"  arc_days: median={df.arc_days_orbit.median():.0f}, "
          f"min={df.arc_days_orbit.min():.0f}, max={df.arc_days_orbit.max():.0f}")
    print(f"  n_obs:    median={df.n_obs_orbit.median():.0f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
