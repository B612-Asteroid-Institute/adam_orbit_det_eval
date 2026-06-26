#!/usr/bin/env python3
"""
refresh_program_code.py
=======================
Repair the `program_code` column of the v1 published catalog.

Background
----------
The cloud LOOO pipeline that produced
`data/mpc_scale_results_20260510/merged_looo_results_published.parquet`
read its `program_code` annotation from `mpcq.MPCObservations.trksub`
(a per-submission tracklet ID, ~14k distinct values catalog-wide)
rather than from the canonical MPC `prog` field (≤98 distinct values
catalog-wide). The orbit fits and residuals themselves are unaffected
— only the annotation column is mislabeled.

This script:

1. Re-fetches source observations for the v1 object_ids via mpcq,
   selecting `obsid` + `prog` (mpcq's MPCObservations schema already
   carries `prog`; the pipeline's read site was updated in bead 43z).
   Writes a *new* parquet file
   `data/mpc_scale_results_20260510/mpc_observations_with_prog.parquet`.

2. Builds a sidecar residual parquet by LEFT-joining `prog` onto an
   in-memory copy of `merged_looo_results_published.parquet` keyed by
   `obs_id == obsid`. The sidecar carries both columns:
     - `program_code` (now populated from canonical MPC `prog`)
     - `trksub`       (the original column's contents, preserved for
                       tracklet-level forensics)
   The original residual parquet is NEVER overwritten — it remains
   the byte-identical canonical cloud-pipeline output. Verified by
   SHA-256 before and after.

3. Re-aggregates `compute_bias_table` and `compute_program_code_stats`
   programmatically from the sidecar (avoids fighting with the
   hardcoded paths in scripts/17_generate_bias_table.py and
   scripts/18_apply_publication_hygiene.py). Phase 1 (bead gnm) has
   already switched the default `max_object_mean_chi2` to None, so no
   extra knob is needed here.

4. Overwrites the published aggregated assets in place:
     - data/.../bias_table.parquet, bias_table.csv (top-level)
     - data/.../bias_catalog_published/bias_table.{parquet,csv}
     - data/.../bias_catalog_published_atct/bias_table.{parquet,csv}
     - data/.../bias_catalog_published_atct/program_code_stats.parquet

   (The high_confidence_bias_table.{csv,parquet,json} and showcase
   regeneration is left to scripts/build_high_confidence_bias_table.py
   and scripts/build_v1_showcase.py — invoke them separately.)

Idempotent — rerun-safe.

Usage
-----
    pdm run python scripts/refresh_program_code.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("refresh_program_code")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "mpc_scale_results_20260510"

ORIGINAL_RESIDUALS = DATA_DIR / "merged_looo_results_published.parquet"
ORIGINAL_RESIDUALS_ATCT = DATA_DIR / "merged_looo_results_published_atct.parquet"
ORIGINAL_OBS_MERGED = DATA_DIR / "mpc_observations_merged.parquet"

OBS_WITH_PROG = DATA_DIR / "mpc_observations_with_prog.parquet"
SIDECAR_RESIDUALS = DATA_DIR / "merged_looo_results_published_progfixed.parquet"
SIDECAR_RESIDUALS_ATCT = (
    DATA_DIR / "merged_looo_results_published_atct_progfixed.parquet"
)

BIAS_PUB_DIR = DATA_DIR / "bias_catalog_published"
BIAS_PUB_ATCT_DIR = DATA_DIR / "bias_catalog_published_atct"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_observations_with_mjd(path: Path):
    """Match scripts/17_generate_bias_table.py:load_observations_with_mjd."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    columns = ["obsid"]
    if "rmsra" in schema.names:
        columns.append("rmsra")
    has_obstime = "obstime" in schema.names
    if has_obstime:
        columns.append("obstime")
    table = pq.read_table(path, columns=columns)
    if has_obstime:
        days = pc.struct_field(table.column("obstime"), "days")
        nanos = pc.struct_field(table.column("obstime"), "nanos")
        mjd = pc.add(
            pc.cast(days, pa.float64()),
            pc.divide(pc.cast(nanos, pa.float64()), pa.scalar(86400.0e9)),
        )
        table = table.append_column("obstime_mjd", mjd).drop(["obstime"])
    df = table.to_pandas()
    logger.info(
        "Loaded %d observations with columns %s", len(df), list(df.columns)
    )
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="provids per mpcq query_observations call (default: 200).",
    )
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the BQ fetch step and use the existing "
             "mpc_observations_with_prog.parquet (for iterating on "
             "the join/aggregate steps).",
    )
    p.add_argument(
        "--project",
        default="moeyens-thor-dev",
        help="GCP project for mpcq (default: moeyens-thor-dev).",
    )
    p.add_argument(
        "--dataset-id",
        default="mpc_sbn_aurora",
        help="BQ dataset id for mpcq (default: mpc_sbn_aurora).",
    )
    return p.parse_args()


def fetch_obs_with_prog(batch_size: int, project: str, dataset_id: str) -> None:
    """Re-fetch source observations with the canonical `prog` column."""
    import pyarrow.parquet as pq
    import quivr as qv

    from mpcq.client import BigQueryMPCClient

    residuals = pq.read_table(ORIGINAL_RESIDUALS, columns=["object_id"]).to_pandas()
    provids = sorted(residuals["object_id"].dropna().unique().tolist())
    logger.info("Distinct provids to fetch: %d", len(provids))

    client = BigQueryMPCClient(
        dataset_id=dataset_id,
        project=project,
    )

    chunks = []
    n_batches = (len(provids) + batch_size - 1) // batch_size
    for i in range(0, len(provids), batch_size):
        batch = provids[i : i + batch_size]
        b_idx = i // batch_size + 1
        logger.info("  Batch %d/%d (%d provids)", b_idx, n_batches, len(batch))
        chunks.append(
            client.query_observations(
                batch,
                columns=["obsid", "prog", "provid", "stn"],
                # dedupe=False: SELECT DISTINCT collides with the default
                # ORDER BY obstime when obstime is not in the column set;
                # we dedupe by obsid in pandas after the fetch.
                dedupe=False,
            )
        )
    all_obs = qv.concatenate(chunks)
    logger.info("Fetched %d observation rows", len(all_obs))

    OBS_WITH_PROG.parent.mkdir(parents=True, exist_ok=True)
    all_obs.to_parquet(str(OBS_WITH_PROG))
    logger.info("Wrote %s", OBS_WITH_PROG)


def build_sidecar(source_residuals: Path, sidecar_path: Path) -> None:
    """Left-join prog onto a residual parquet to produce the sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pre_sha = sha256_of_file(source_residuals)
    logger.info("Source residual SHA-256: %s  (%s)", pre_sha, source_residuals.name)

    resid_tbl = pq.read_table(source_residuals)
    obs_tbl = pq.read_table(OBS_WITH_PROG, columns=["obsid", "prog"])
    logger.info(
        "Source residuals: %d rows; observations: %d rows",
        len(resid_tbl), len(obs_tbl),
    )

    # Dedupe the obs table on obsid: an obsid is a primary key in obs_sbn,
    # but mpcq's SELECT DISTINCT + the union-all candidate_matches join
    # path can produce duplicate rows if a provid is referenced through
    # multiple identification paths. Keep first non-null prog per obsid.
    obs_df = obs_tbl.to_pandas()
    obs_df = obs_df.sort_values("prog", na_position="last").drop_duplicates(
        subset="obsid", keep="first"
    )
    n_distinct_prog = obs_df["prog"].dropna().nunique()
    logger.info(
        "Distinct prog values catalog-wide: %d (null fraction: %.3f)",
        n_distinct_prog,
        float(obs_df["prog"].isna().mean()),
    )
    if n_distinct_prog > 100:
        raise RuntimeError(
            f"Distinct prog values ({n_distinct_prog}) exceeds expected ceiling "
            "(≤100). Likely fetched wrong column; aborting before clobbering "
            "downstream assets."
        )

    resid_df = resid_tbl.to_pandas()
    # The existing program_code column holds trksub values from the
    # cloud-pipeline run — preserve under the trksub name and replace
    # program_code with the canonical prog values.
    resid_df = resid_df.rename(columns={"program_code": "trksub"})
    merged = resid_df.merge(
        obs_df[["obsid", "prog"]].rename(
            columns={"obsid": "obs_id", "prog": "program_code"}
        ),
        on="obs_id",
        how="left",
    )

    if len(merged) != len(resid_df):
        raise RuntimeError(
            f"Sidecar row count drift ({len(merged)} vs {len(resid_df)}); "
            "left-join produced duplicate matches — investigate the dedupe."
        )

    # Sanity: program_code distinct count after join
    n_pc = merged["program_code"].dropna().nunique()
    logger.info(
        "Sidecar program_code: %d distinct values (%.3f null)",
        n_pc, float(merged["program_code"].isna().mean()),
    )

    # Restore column order: original column order with trksub immediately
    # after program_code so a casual reader sees them side by side.
    orig_cols = list(resid_tbl.column_names)
    new_cols = []
    for c in orig_cols:
        if c == "program_code":
            new_cols.append("program_code")
            new_cols.append("trksub")
        else:
            new_cols.append(c)
    sidecar_table = pa.Table.from_pandas(merged[new_cols], preserve_index=False)
    pq.write_table(sidecar_table, sidecar_path)
    logger.info("Wrote sidecar %s (%d rows)", sidecar_path, len(sidecar_table))

    post_sha = sha256_of_file(source_residuals)
    if post_sha != pre_sha:
        raise RuntimeError(
            f"Source residual SHA-256 drifted from {pre_sha} to {post_sha}! "
            "Aborting — canonical cloud output has been corrupted."
        )
    logger.info("Source residual SHA-256 unchanged after sidecar build ✓")


def aggregate_from_sidecar(
    sidecar_path: Path,
    out_dir: Path,
    *,
    observations_for_provenance: Path | None,
) -> None:
    """Re-aggregate bias_table from a sidecar residual parquet."""
    import json
    from dataclasses import asdict

    import pyarrow.parquet as pq

    from adam_orbit_det_eval.looo.bias_table import (
        BootstrapConfig,
        compute_bias_table,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading sidecar %s", sidecar_path)
    looo_df = pq.read_table(sidecar_path).to_pandas()

    obs_df = None
    if observations_for_provenance is not None:
        obs_df = _load_observations_with_mjd(observations_for_provenance)

    bootstrap_cfg = BootstrapConfig(n_resamples=2000, random_seed=42)
    logger.info("Running compute_bias_table with bootstrap n=%d seed=%d",
                bootstrap_cfg.n_resamples, bootstrap_cfg.random_seed)
    bias_table = compute_bias_table(
        looo_df=looo_df,
        observations_df=obs_df,
        min_obs_per_group=10,
        min_objects_per_group=3,
        max_hold_in_reduced_chi2=100.0,
        max_object_mean_chi2=None,
        bootstrap=bootstrap_cfg,
    )

    parquet_path = out_dir / "bias_table.parquet"
    csv_path = out_dir / "bias_table.csv"
    config_path = out_dir / "bias_table_config.json"
    bias_table.to_parquet(parquet_path, index=False)
    bias_table.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", parquet_path, len(bias_table))

    cfg = {
        "looo_results": str(sidecar_path.relative_to(REPO_ROOT)),
        "observations": str(observations_for_provenance.relative_to(REPO_ROOT))
            if observations_for_provenance else None,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "min_obs_per_group": 10,
        "min_objects_per_group": 3,
        "max_hold_in_reduced_chi2": 100.0,
        "max_object_mean_chi2": None,
        "bootstrap": asdict(bootstrap_cfg),
        "n_rows": int(len(bias_table)),
        "n_observatory_rows": int(bias_table["program_code"].isna().sum()),
        "n_program_rows": int(bias_table["program_code"].notna().sum()),
        "dims_populated": [
            d for d in ("ra", "dec", "at", "ct")
            if not bias_table[f"bias_{d}_arcsec"].isna().all()
        ],
        "source": "refresh_program_code.py — bead 43z",
    }
    config_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote %s", config_path)



def aggregate_program_code_stats(sidecar_path: Path, out_path: Path) -> None:
    """Re-aggregate compute_program_code_stats from the sidecar."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    from adam_orbit_det_eval.looo.core import LOOOResult
    from adam_orbit_det_eval.looo.analysis import compute_program_code_stats

    logger.info("Loading sidecar for program_code_stats: %s", sidecar_path)
    tbl = pq.read_table(sidecar_path)
    # The sidecar carries extra columns (trksub from the program_code refresh;
    # residual_at/ct + speed_deg_per_day + v_ra_unit/v_dec_unit when the
    # source was the AT/CT-augmented residual parquet) that are not in the
    # base LOOOResult schema. Drop any column not in LOOOResult.
    looo_cols = set(LOOOResult.schema.names)
    extras = [c for c in tbl.column_names if c not in looo_cols]
    if extras:
        logger.info("Dropping %d non-LOOOResult columns: %s", len(extras), extras)
        tbl = tbl.drop(extras)
    looo = LOOOResult.from_pyarrow(tbl)
    stats = compute_program_code_stats(
        looo,
        min_obs_per_group=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(str(out_path))
    logger.info("Wrote %s (%d rows)", out_path, len(stats))


def main() -> int:
    args = parse_args()

    if not args.skip_fetch:
        if not ORIGINAL_RESIDUALS.exists():
            logger.error("Missing source residuals: %s", ORIGINAL_RESIDUALS)
            return 1
        fetch_obs_with_prog(
            batch_size=args.batch_size,
            project=args.project,
            dataset_id=args.dataset_id,
        )
    else:
        if not OBS_WITH_PROG.exists():
            logger.error(
                "--skip-fetch but %s does not exist", OBS_WITH_PROG
            )
            return 1
        logger.info("--skip-fetch: reusing existing %s", OBS_WITH_PROG)

    # Build sidecar for the RA/Dec residuals
    build_sidecar(ORIGINAL_RESIDUALS, SIDECAR_RESIDUALS)
    # And for the AT/CT-augmented residuals
    if ORIGINAL_RESIDUALS_ATCT.exists():
        build_sidecar(ORIGINAL_RESIDUALS_ATCT, SIDECAR_RESIDUALS_ATCT)
    else:
        logger.warning(
            "No AT/CT residual parquet at %s — skipping AT/CT sidecar",
            ORIGINAL_RESIDUALS_ATCT,
        )

    # Re-aggregate bias_table (RA/Dec only)
    aggregate_from_sidecar(
        SIDECAR_RESIDUALS,
        BIAS_PUB_DIR,
        observations_for_provenance=None,
    )
    # Re-aggregate bias_table (AT/CT)
    if SIDECAR_RESIDUALS_ATCT.exists():
        aggregate_from_sidecar(
            SIDECAR_RESIDUALS_ATCT,
            BIAS_PUB_ATCT_DIR,
            observations_for_provenance=ORIGINAL_OBS_MERGED,
        )

    # Re-aggregate program_code_stats (lives in bias_catalog_published/ —
    # the file is RA/Dec-only since compute_program_code_stats does not use
    # AT/CT columns; the RA/Dec sidecar is sufficient as input).
    aggregate_program_code_stats(
        SIDECAR_RESIDUALS,
        BIAS_PUB_DIR / "program_code_stats.parquet",
    )

    # Final SHA-256 verification
    final_sha = sha256_of_file(ORIGINAL_RESIDUALS)
    logger.info("Final source residual SHA-256: %s", final_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
