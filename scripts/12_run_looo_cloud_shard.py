#!/usr/bin/env python3
"""
12_run_looo_cloud_shard.py
==========================
Cloud-aware shard runner for the MPC-scale isolation study.

Designed to run inside an exp-research indexed job pod.  The pod's
JOB_COMPLETION_INDEX (0-based) selects which scenario this shard processes
from a manifest JSON stored in GCS.

The manifest (scenarios.json) maps shard index → scenario definition:
    {
      "0":  {"fake_code": "AA00", "bias_name": "clean"},
      "1":  {"fake_code": "AA00", "bias_name": "constant"},
      ...
      "80": {"fake_code": "AA08", "bias_name": "trailing"}
    }

GCS layout:
    <input_prefix>/
        scenarios.json          ← scenario manifest
        mpc_observations.parquet
        mpc_orbits.parquet

    <output_prefix>/
        <scenario>/
            datasets/default/
            looo_results/default/
                checkpoints/   ← per-object checkpoints (GCS-backed)
            analysis/default/
            recovery/default/

Usage
-----
  # Inside a pod (env: JOB_COMPLETION_INDEX=3)
  python scripts/12_run_looo_cloud_shard.py \\
      --gcs-input-prefix  gs://exp-research/mpc-isolation-study/input \\
      --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
      --propagator assist \\
      --max-processes 14

  # Local smoke-test (override shard index)
  JOB_COMPLETION_INDEX=0 python scripts/12_run_looo_cloud_shard.py \\
      --gcs-input-prefix  gs://exp-research/mpc-isolation-study/input \\
      --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
      --shard-index 0 \\
      --propagator twobody \\
      --max-processes 2

Status: STUB — Phase 2 implementation.  See docs/mpc_scale_infra_plan.md §6.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers (thin wrappers; google-cloud-storage assumed installed)
# ---------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, local_path: Path) -> None:
    from google.cloud import storage
    assert gcs_uri.startswith("gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(local_path))
    logger.info(f"Downloaded {gcs_uri} → {local_path}")


def _gcs_upload(local_path: Path, gcs_uri: str) -> None:
    from google.cloud import storage
    assert gcs_uri.startswith("gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
    logger.info(f"Uploaded {local_path} → {gcs_uri}")


def _gcs_sync_down(gcs_prefix: str, local_dir: Path) -> None:
    """Rsync checkpoints from GCS to local before starting (spot-preemption recovery)."""
    import subprocess
    local_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_prefix + "/", str(local_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info(f"Synced checkpoints from {gcs_prefix}")
    else:
        # Not fatal — may be empty on first run
        logger.info(f"Checkpoint sync returned {result.returncode} (ok if first run)")


def _gcs_sync_up(local_dir: Path, gcs_prefix: str) -> None:
    """Sync local checkpoints up to GCS."""
    import subprocess
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_prefix + "/"],
        check=True,
    )
    logger.info(f"Uploaded checkpoints to {gcs_prefix}")


# ---------------------------------------------------------------------------
# Scenario manifest
# ---------------------------------------------------------------------------

def load_manifest(gcs_input_prefix: str, local_dir: Path) -> dict:
    manifest_path = local_dir / "scenarios.json"
    _gcs_download(f"{gcs_input_prefix}/scenarios.json", manifest_path)
    with open(manifest_path) as f:
        return json.load(f)


def generate_manifest() -> dict[str, dict]:
    """
    Generate the default 81-scenario manifest (9 stations × 9 biases).
    Call this once locally and upload to GCS before submitting the job.
    """
    stations = [
        "AA00", "AA01", "AA02", "AA03", "AA04",
        "AA05", "AA06", "AA07", "AA08",
    ]
    biases = [
        "clean", "constant", "timing", "mag_dep", "epoch",
        "seasonal", "step", "dcr", "trailing",
    ]
    manifest: dict[str, dict] = {}
    idx = 0
    for fc in stations:
        for bn in biases:
            manifest[str(idx)] = {"fake_code": fc, "bias_name": bn}
            idx += 1
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcs-input-prefix", required=True,
                   help="GCS URI prefix for input data (scenarios.json, parquet files)")
    p.add_argument("--gcs-output-prefix", required=True,
                   help="GCS URI prefix for output data (one subdir per scenario)")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Override JOB_COMPLETION_INDEX (for local testing)")
    p.add_argument("--propagator", choices=["twobody", "assist"], default="assist")
    p.add_argument("--max-processes", type=int, default=14)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--generate-manifest", action="store_true",
        help="Print the scenario manifest JSON to stdout and exit (for initial setup)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.generate_manifest:
        print(json.dumps(generate_manifest(), indent=2))
        return

    # Determine shard index
    shard_index = args.shard_index
    if shard_index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            logger.error("JOB_COMPLETION_INDEX not set and --shard-index not provided")
            sys.exit(1)
        shard_index = int(raw)

    logger.info(f"Shard index: {shard_index}")

    with tempfile.TemporaryDirectory(prefix="looo_shard_") as tmpdir:
        tmp = Path(tmpdir)
        input_dir = tmp / "input"
        output_dir = tmp / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # --- Load manifest ---
        manifest = load_manifest(args.gcs_input_prefix, tmp)
        key = str(shard_index)
        if key not in manifest:
            logger.error(f"Shard index {shard_index} not in manifest (size={len(manifest)})")
            sys.exit(1)
        fake_code = manifest[key]["fake_code"]
        bias_name = manifest[key]["bias_name"]
        scenario = f"{fake_code}_{bias_name}"
        logger.info(f"Running scenario: {scenario}")

        # --- Download input data ---
        for fname in ("mpc_observations.parquet", "mpc_orbits.parquet"):
            _gcs_download(f"{args.gcs_input_prefix}/{fname}", input_dir / fname)

        # --- Sync existing checkpoints from GCS (spot-preemption recovery) ---
        local_ckpt_dir = output_dir / scenario / "looo_results" / "default" / "checkpoints"
        gcs_ckpt_prefix = f"{args.gcs_output_prefix}/{scenario}/looo_results/default/checkpoints"
        _gcs_sync_down(gcs_ckpt_prefix, local_ckpt_dir)

        # --- Run scenario (reuse logic from 10_run_isolation_study.py) ---
        # Import here to defer heavy imports until after GCS setup
        sys.path.insert(0, str(Path(__file__).parent))
        from _10_run_isolation_study_impl import run_scenario  # noqa: F401
        # TODO: factor run_scenario out of 10_run_isolation_study.py into a shared module
        # For now, inline the call pattern here.
        # See docs/mpc_scale_infra_plan.md §6.3 for refactoring notes.

        # Placeholder: demonstrate the contract
        raise NotImplementedError(
            "Phase 2 implementation pending. "
            "Refactor run_scenario() into adam_orbit_det_eval.looo.runner first. "
            "See docs/mpc_scale_infra_plan.md for the full plan."
        )


if __name__ == "__main__":
    main()
