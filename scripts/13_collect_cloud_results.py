#!/usr/bin/env python3
"""
13_collect_cloud_results.py
===========================
Download and assemble results from a completed cloud isolation study run.

After all 81 exp-research shard pods finish, this script:
  1. Downloads all per-scenario outputs from GCS into a local directory
  2. Merges per-scenario recovery_report.csv files into combined_recovery.csv
  3. Prints the same detection-rate summary as the local runner

Usage
-----
    python scripts/13_collect_cloud_results.py \\
        --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
        --output-dir data/sim_products/mpc_scale_run

    # Download only specific scenarios:
    python scripts/13_collect_cloud_results.py \\
        --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
        --output-dir data/sim_products/mpc_scale_run \\
        --station AA00 AA01 --bias constant timing

    # Skip downloading (merge from already-downloaded outputs):
    python scripts/13_collect_cloud_results.py \\
        --output-dir data/sim_products/mpc_scale_run \\
        --no-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers (same Python client approach as the shard runner)
# ---------------------------------------------------------------------------

def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    assert gcs_uri.startswith("gs://"), f"Not a GCS URI: {gcs_uri}"
    without_scheme = gcs_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _download_scenario(gcs_output_prefix: str, scenario: str, local_dir: Path) -> int:
    """Download all blobs for one scenario. Returns number of files downloaded."""
    from google.cloud import storage

    bucket_name, output_prefix = _parse_gcs_uri(gcs_output_prefix)
    prefix = f"{output_prefix.rstrip('/')}/{scenario}/"
    client = storage.Client()
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
    if not blobs:
        return 0
    for blob in blobs:
        rel = blob.name[len(prefix):]
        dest = local_dir / scenario / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    logger.info(f"  {scenario}: downloaded {len(blobs)} files")
    return len(blobs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gcs-output-prefix",
                   help="GCS URI prefix used in the cloud run (omit with --no-download)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Local directory to download outputs into")
    p.add_argument("--no-download", action="store_true",
                   help="Skip GCS download; only merge already-local outputs")
    p.add_argument("--station", nargs="*", metavar="FAKE_CODE",
                   help="Restrict to these fake codes")
    p.add_argument("--bias", nargs="*", metavar="BIAS_NAME",
                   help="Restrict to these bias names")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from adam_orbit_det_eval.isolation_study import list_scenarios, scenario_id, BIAS_NAMES
    scenarios = list_scenarios(args.station, args.bias)
    snames = [scenario_id(fc, bn) for fc, bn in scenarios]

    # --- 1. Download from GCS ---
    if not args.no_download:
        if not args.gcs_output_prefix:
            logger.error("--gcs-output-prefix required unless --no-download is set")
            sys.exit(1)
        logger.info(f"Downloading {len(snames)} scenarios from {args.gcs_output_prefix}...")
        total_files = 0
        missing = []
        for sname in snames:
            n = _download_scenario(args.gcs_output_prefix, sname, args.output_dir)
            total_files += n
            if n == 0:
                missing.append(sname)
        logger.info(f"Downloaded {total_files} total files")
        if missing:
            logger.warning(f"{len(missing)} scenarios had no GCS output: {missing}")

    # --- 2. Merge recovery reports ---
    import pandas as pd

    all_rows = []
    missing_reports = []
    for fc, bn in scenarios:
        sname = scenario_id(fc, bn)
        report_path = args.output_dir / sname / "recovery" / "default" / "recovery_report.csv"
        if not report_path.exists():
            missing_reports.append(sname)
            continue
        df = pd.read_csv(report_path)
        # Ensure scenario metadata columns are present (older runs may lack them)
        if "scenario" not in df.columns:
            df["scenario"] = sname
        if "target_station" not in df.columns:
            df["target_station"] = fc
        if "applied_bias" not in df.columns:
            df["applied_bias"] = bn
        if "is_target" not in df.columns:
            df["is_target"] = df["fake_code"] == fc
        all_rows.append(df)

    if missing_reports:
        logger.warning(f"{len(missing_reports)} scenarios missing recovery report: {missing_reports}")

    if not all_rows:
        logger.error("No recovery reports found — nothing to merge")
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = args.output_dir / "combined_recovery.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Combined recovery → {combined_path}  ({len(combined)} rows, {len(all_rows)} scenarios)")

    # --- 3. Print summary ---
    _print_summary(combined, BIAS_NAMES)


def _print_summary(df, bias_names):
    import numpy as np

    target = df[df["is_target"]].copy()
    if target.empty:
        logger.warning("No target-station rows found in combined results")
        return

    print()
    print("=" * 90)
    print("  ISOLATION STUDY — TARGET STATION RECOVERY")
    print("=" * 90)
    hdr = (f"{'Scenario':<22}  {'BiasType':<20}  "
           f"{'Inj_RA':>7}  {'Rec_RA':>7}  {'Err_RA':>7}  "
           f"{'SNR_RA':>7}  {'N_obs':>6}  {'DetRA':>5}  {'DetDec':>6}")
    print(hdr)
    print("-" * len(hdr))

    for _, row in target.sort_values(["target_station", "applied_bias"]).iterrows():
        def _f(v):
            try:
                return f"{float(v):+7.3f}" if np.isfinite(float(v)) else "   N/A "
            except (TypeError, ValueError):
                return "   N/A "
        print(
            f"{row.get('scenario', ''):<22}  "
            f"{row.get('applied_bias', ''):<20}  "
            f"{_f(row.get('injected_ra_arcsec'))}  "
            f"{_f(row.get('recovered_mean_ra_arcsec'))}  "
            f"{_f(row.get('recovery_error_ra'))}  "
            f"{_f(row.get('detection_snr_ra'))}  "
            f"{int(row.get('n_obs', 0)):>6}  "
            f"{'Y' if row.get('detected_ra') else 'N':>5}  "
            f"{'Y' if row.get('detected_dec') else 'N':>6}"
        )

    print()
    print("Detection rate (RA) by bias type at target station:")
    for bn in bias_names:
        sub = target[target["applied_bias"] == bn]
        if sub.empty:
            continue
        det = sub["detected_ra"].sum()
        tot = len(sub)
        print(f"  {bn:<14}: {det}/{tot} ({100*det//max(tot,1)}%)")
    print()


if __name__ == "__main__":
    main()
