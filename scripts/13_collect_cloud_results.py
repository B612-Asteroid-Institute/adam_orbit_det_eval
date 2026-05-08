#!/usr/bin/env python3
"""
13_collect_cloud_results.py
===========================
Collect per-shard real-data LOOO results from GCS and merge into unified
output files.

Scans the GCS output prefix written by ``12_run_looo_cloud_shard.py`` for
shard directories with SUCCESS/FAILED markers, downloads completed shard
results, and merges them into a single set of output files.

Expected GCS layout (written by the shard runner):

    <gcs_output_prefix>/
        shard_000/
            SUCCESS
            looo_results.parquet
            observatory_stats.parquet
            program_code_stats.parquet
        shard_001/
            FAILED
        shard_002/
            SUCCESS
            ...

Output:

    <output_dir>/
        merged_looo_results.parquet
        observatory_stats.parquet
        program_code_stats.parquet
        collection_report.json

Usage
-----
  python scripts/13_collect_cloud_results.py \\
      --gcs-output-prefix gs://exp-research/mpc-real-data-looo/output \\
      --output-dir data/mpc_scale_results \\
      --min-shards-pct 90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _get_gcs_client():
    from google.cloud import storage
    return storage.Client()


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split gs://bucket/prefix into (bucket_name, blob_prefix)."""
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got {gcs_uri!r}"
    rest = gcs_uri[len("gs://"):]
    if "/" in rest:
        bucket_name, blob_prefix = rest.split("/", 1)
    else:
        bucket_name, blob_prefix = rest, ""
    return bucket_name, blob_prefix.rstrip("/")


def scan_shards(gcs_output_prefix: str) -> tuple[list[str], list[str], list[str]]:
    """Scan GCS for shard directories and classify by marker files.

    Returns (succeeded, failed, unknown) — each a list of shard names.
    """
    client = _get_gcs_client()
    bucket_name, prefix = _parse_gcs_uri(gcs_output_prefix)

    # List all blobs under the prefix to find marker files
    blob_prefix = f"{prefix}/" if prefix else ""
    blobs = client.list_blobs(bucket_name, prefix=blob_prefix)

    shard_markers: dict[str, set[str]] = {}
    for blob in blobs:
        # Extract shard name and filename from blob path
        rel = blob.name[len(blob_prefix):]
        parts = rel.split("/", 1)
        if len(parts) < 2:
            continue
        shard_name, filename = parts[0], parts[1]
        if not shard_name.startswith("shard_"):
            continue
        if shard_name not in shard_markers:
            shard_markers[shard_name] = set()
        shard_markers[shard_name].add(filename)

    succeeded = sorted(s for s, m in shard_markers.items() if "SUCCESS" in m)
    failed = sorted(s for s, m in shard_markers.items() if "FAILED" in m and "SUCCESS" not in m)
    unknown = sorted(s for s, m in shard_markers.items() if "SUCCESS" not in m and "FAILED" not in m)

    return succeeded, failed, unknown


def download_shard_results(
    gcs_output_prefix: str,
    shard_name: str,
    local_dir: Path,
) -> dict[str, Path]:
    """Download a completed shard's result parquet files.

    Returns a dict mapping filename -> local path for files that exist.
    """
    client = _get_gcs_client()
    bucket_name, prefix = _parse_gcs_uri(gcs_output_prefix)
    bucket = client.bucket(bucket_name)

    shard_dir = local_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    result_files = [
        "looo_results.parquet",
        "observatory_stats.parquet",
        "program_code_stats.parquet",
    ]

    downloaded = {}
    for fname in result_files:
        blob_path = f"{prefix}/{shard_name}/{fname}" if prefix else f"{shard_name}/{fname}"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            logger.warning(f"Missing {fname} in {shard_name} (has SUCCESS marker — corrupt?)")
            continue
        local_path = shard_dir / fname
        blob.download_to_filename(str(local_path))
        downloaded[fname] = local_path

    return downloaded


def download_marker(gcs_output_prefix: str, shard_name: str, marker: str) -> str:
    """Download and return the content of a marker file (SUCCESS or FAILED)."""
    client = _get_gcs_client()
    bucket_name, prefix = _parse_gcs_uri(gcs_output_prefix)
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix}/{shard_name}/{marker}" if prefix else f"{shard_name}/{marker}"
    blob = bucket.blob(blob_path)
    if blob.exists():
        return blob.download_as_text()
    return ""


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_looo_results(shard_parquets: list[Path], output_path: Path) -> int:
    """Concatenate per-shard LOOO results, deduplicating by (object_id, obs_id, stn).

    Returns total row count after dedup.
    """
    import pyarrow.parquet as pq

    tables = []
    for p in shard_parquets:
        try:
            tbl = pq.read_table(p)
            if len(tbl) > 0:
                tables.append(tbl)
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")

    if not tables:
        logger.warning("No LOOO result rows to merge")
        return 0

    import pyarrow as pa
    merged = pa.concat_tables(tables, promote_options="default")

    # Deduplicate: if a shard was retried, the same (object_id, obs_id, stn)
    # may appear twice. Keep the first occurrence.
    dedup_cols = []
    available_cols = set(merged.schema.names)
    for col in ("object_id", "obs_id", "stn"):
        if col in available_cols:
            dedup_cols.append(col)

    if dedup_cols:
        import pandas as pd
        df = merged.to_pandas()
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        after = len(df)
        if before != after:
            logger.info(f"Deduplicated: {before} -> {after} rows ({before - after} duplicates)")
        merged = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_table(merged, output_path)
    logger.info(f"Merged LOOO results: {len(merged)} rows -> {output_path}")
    return len(merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect and merge per-shard LOOO results from GCS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gcs-output-prefix",
        required=True,
        help="GCS URI prefix where shard results are stored",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mpc_scale_results"),
        help="Local directory for merged output (default: data/mpc_scale_results)",
    )
    p.add_argument(
        "--min-shards-pct",
        type=float,
        default=90.0,
        help="Minimum percentage of total shards that must have completed "
             "before proceeding (default: 90)",
    )
    p.add_argument(
        "--apply-bias-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the aggregation-time bad-fit filter before computing "
             "per-station stats. Default: enabled. Use --no-apply-bias-filter "
             "to skip (e.g. when re-cutting with custom thresholds downstream).",
    )
    p.add_argument(
        "--bias-filter-max-chi2",
        type=float,
        default=10.0,
        help="Tier 2: max hold_in_reduced_chi2 (default: 10.0).",
    )
    p.add_argument(
        "--bias-filter-max-delta-q",
        type=float,
        default=0.5,
        help="Tier 3: max |delta_q_au| in AU (default: 0.5).",
    )
    p.add_argument(
        "--bias-filter-max-delta-e",
        type=float,
        default=0.3,
        help="Tier 3: max |delta_e| (default: 0.3).",
    )
    p.add_argument(
        "--bias-filter-max-delta-i-deg",
        type=float,
        default=5.0,
        help="Tier 3: max |delta_i_deg| in degrees (default: 5.0).",
    )
    p.add_argument(
        "--bias-filter-mad-factor",
        type=float,
        default=5.0,
        help="Tier 4: per-station MAD multiplier (default: 5.0).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Scan for completed/failed shards ---
    logger.info(f"Scanning shards at {args.gcs_output_prefix} ...")
    succeeded, failed, unknown = scan_shards(args.gcs_output_prefix)

    total = len(succeeded) + len(failed) + len(unknown)
    if total == 0:
        logger.error("No shards found at the given prefix. Check the GCS path.")
        sys.exit(1)

    pct_complete = 100.0 * len(succeeded) / total if total > 0 else 0.0
    logger.info(
        f"Shards: {len(succeeded)} succeeded, {len(failed)} failed, "
        f"{len(unknown)} in-progress/unknown, {total} total "
        f"({pct_complete:.1f}% complete)"
    )

    if failed:
        logger.warning(f"Failed shards: {', '.join(failed)}")

    if pct_complete < args.min_shards_pct:
        logger.error(
            f"Only {pct_complete:.1f}% of shards completed "
            f"(minimum: {args.min_shards_pct}%). Aborting."
        )
        sys.exit(1)

    # --- Download completed shard results ---
    download_dir = args.output_dir / "_shards"
    download_dir.mkdir(parents=True, exist_ok=True)

    looo_parquets: list[Path] = []
    corrupt_shards: list[str] = []

    for shard_name in succeeded:
        downloaded = download_shard_results(args.gcs_output_prefix, shard_name, download_dir)
        if "looo_results.parquet" in downloaded:
            looo_parquets.append(downloaded["looo_results.parquet"])
        else:
            corrupt_shards.append(shard_name)
            logger.warning(f"Shard {shard_name} has SUCCESS marker but no looo_results.parquet")

    if corrupt_shards:
        logger.warning(f"{len(corrupt_shards)} corrupt shard(s): {', '.join(corrupt_shards)}")

    if not looo_parquets:
        logger.error("No valid LOOO result files found across completed shards.")
        sys.exit(1)

    # --- Merge LOOO results ---
    merged_path = args.output_dir / "merged_looo_results.parquet"
    n_rows = merge_looo_results(looo_parquets, merged_path)

    # --- Re-compute analysis on merged results ---
    from adam_orbit_det_eval.looo.analysis import (
        compute_observatory_stats,
        compute_program_code_stats,
    )
    from adam_orbit_det_eval.looo.bias_filter import (
        BiasFilterConfig,
        apply_bias_filter,
        format_filter_audit,
    )
    from adam_orbit_det_eval.looo.core import LOOOResult

    import pyarrow.parquet as pq

    merged_results = LOOOResult(pq.read_table(merged_path))

    # --- Aggregation-time bad-fit filter (bead 7bt) ---
    # Catches the failure modes that bypass per-fit success and chi2 cuts:
    # silent fitter failures, orbit drift, per-station tail outliers. The
    # collector calls this explicitly (rather than via analysis.py) so the
    # unfiltered merge is preserved on disk for downstream re-cutting.
    if args.apply_bias_filter:
        filter_cfg = BiasFilterConfig(
            max_chi2=args.bias_filter_max_chi2,
            max_delta_q=args.bias_filter_max_delta_q,
            max_delta_e=args.bias_filter_max_delta_e,
            max_delta_i_deg=args.bias_filter_max_delta_i_deg,
            mad_factor=args.bias_filter_mad_factor,
        )
        logger.info("Applying aggregation-time bad-fit filter...")
        filtered_results, filter_stats = apply_bias_filter(merged_results, filter_cfg)

        # Persist the filtered parquet alongside the unfiltered merge.
        filtered_path = args.output_dir / "merged_looo_results_filtered.parquet"
        pq.write_table(filtered_results.table, filtered_path)
        logger.info(
            f"Filtered LOOO results: {filter_stats.rows_in} -> "
            f"{filter_stats.rows_out} rows -> {filtered_path}"
        )

        # Persist stats (per-tier + per-station) as JSON.
        bias_filter_stats_path = args.output_dir / "bias_filter_stats.json"
        bias_filter_stats_path.write_text(
            json.dumps(
                {
                    "config": {
                        "max_chi2": filter_cfg.max_chi2,
                        "max_delta_q": filter_cfg.max_delta_q,
                        "max_delta_e": filter_cfg.max_delta_e,
                        "max_delta_i_deg": filter_cfg.max_delta_i_deg,
                        "mad_factor": filter_cfg.mad_factor,
                    },
                    "stats": filter_stats.summary(),
                },
                indent=2,
            )
        )

        stats_input = filtered_results
    else:
        logger.info("Bias filter disabled (--no-apply-bias-filter); using raw merge.")
        filter_cfg = None
        filter_stats = None
        stats_input = merged_results

    logger.info("Computing merged observatory stats...")
    # Pass max_hold_in_reduced_chi2=None when the bias filter has already run —
    # otherwise analysis.py's own chi2 cut would double-filter (and nulls
    # carry the v11 silent-failure trap of fill_null(False)-dropping every row).
    chi2_arg = None if args.apply_bias_filter else 100.0
    obs_stats = compute_observatory_stats(stats_input, max_hold_in_reduced_chi2=chi2_arg)
    obs_stats.to_parquet(args.output_dir / "observatory_stats.parquet")
    logger.info(f"Observatory stats: {len(obs_stats)} stations")

    logger.info("Computing merged program code stats...")
    prog_stats = compute_program_code_stats(stats_input, max_hold_in_reduced_chi2=chi2_arg)
    prog_stats.to_parquet(args.output_dir / "program_code_stats.parquet")
    logger.info(f"Program code stats: {len(prog_stats)} groups")

    # --- Validation report (per-tier and high-MAD station audit) ---
    if filter_stats is not None:
        report_lines = [
            format_filter_audit(filter_stats, filter_cfg),
            "",
            f"Per-station stats computed from filtered set: "
            f"{len(obs_stats)} stations, {len(prog_stats)} program-code groups.",
        ]
        validation_report_path = args.output_dir / "validation_report.txt"
        validation_report_path.write_text("\n".join(report_lines))
        logger.info(f"Wrote {validation_report_path}")

    # --- Count unique objects and observations ---
    import pyarrow.compute as pc
    n_objects = len(pc.unique(merged_results.table.column("object_id")))
    n_observations = n_rows

    # --- Write collection report ---
    elapsed = time.time() - t0
    report = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcs_output_prefix": args.gcs_output_prefix,
        "shards_succeeded": len(succeeded),
        "shards_failed": len(failed),
        "shards_unknown": len(unknown),
        "shards_corrupt": len(corrupt_shards),
        "shards_total": total,
        "completion_pct": round(pct_complete, 1),
        "total_objects": n_objects,
        "total_result_rows": n_rows,
        "observatory_stats_count": len(obs_stats),
        "program_code_stats_count": len(prog_stats),
        "bias_filter_applied": bool(args.apply_bias_filter),
        "bias_filter_rows_in": filter_stats.rows_in if filter_stats is not None else None,
        "bias_filter_rows_out": filter_stats.rows_out if filter_stats is not None else None,
        "bias_filter_loss_fraction": (
            round(filter_stats.loss_fraction, 4)
            if filter_stats is not None
            else None
        ),
        "elapsed_seconds": round(elapsed, 1),
        "failed_shard_names": failed,
        "corrupt_shard_names": corrupt_shards,
    }
    report_path = args.output_dir / "collection_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        f"Collection complete in {elapsed:.1f}s:\n"
        f"  {len(succeeded)} shards -> {n_objects} objects, {n_rows} result rows\n"
        f"  {len(obs_stats)} observatory stats, {len(prog_stats)} program code stats\n"
        f"  Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
