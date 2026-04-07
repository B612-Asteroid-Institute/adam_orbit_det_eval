#!/usr/bin/env python3
"""
12_run_looo_cloud_shard.py
==========================
Cloud shard runner for the MPC-scale isolation study.

Designed to run inside an exp-research indexed job pod.  Each pod's
``JOB_COMPLETION_INDEX`` (0-based) selects one scenario from a manifest
JSON stored in GCS, then runs the full pipeline for that scenario and
uploads all outputs back to GCS.

Uses the ``google-cloud-storage`` Python client (pre-installed by expctl)
instead of gsutil — no extra system packages needed in the image.

GCS layout
----------
Input (written once before job submission)::

    <input_prefix>/
        scenarios.json              ← manifest: index → {fake_code, bias_name}
        mpc_observations.parquet
        mpc_orbits.parquet

Output (written by each pod)::

    <output_prefix>/
        <scenario>/
            datasets/default/
            looo_results/default/
                checkpoints/       ← synced mid-run for spot-preemption recovery
            analysis/default/
            recovery/default/

Usage
-----
Inside a pod (``JOB_COMPLETION_INDEX`` set by Kubernetes)::

    python scripts/12_run_looo_cloud_shard.py \\
        --gcs-input-prefix  gs://exp-research/mpc-isolation-study/input \\
        --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
        --propagator assist \\
        --max-processes 14

Local smoke-test::

    JOB_COMPLETION_INDEX=0 python scripts/12_run_looo_cloud_shard.py \\
        --gcs-input-prefix  gs://exp-research/mpc-isolation-study/input \\
        --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
        --shard-index 0 \\
        --propagator twobody \\
        --max-processes 2

Preparing a run
---------------
1. Generate and upload the scenario manifest::

    python scripts/12_run_looo_cloud_shard.py --generate-manifest \\
        | gsutil cp - gs://exp-research/mpc-isolation-study/input/scenarios.json

2. Upload input parquet files::

    gsutil -m cp data/looo_sample_3500/mpc_observations.parquet \\
                 data/looo_sample_3500/mpc_orbits.parquet \\
                 gs://exp-research/mpc-isolation-study/input/

3. Submit the job::

    cd /path/to/exp-research
    ./expctl submit -f ../adam_orbit_det_eval/infra/exp-research/mpc-scale-isolation-study.json

4. Collect results::

    python scripts/13_collect_cloud_results.py \\
        --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \\
        --output-dir data/sim_products/mpc_scale_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers (google-cloud-storage client, no gsutil required)
# ---------------------------------------------------------------------------

def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Return ``(bucket_name, blob_prefix)`` from a ``gs://`` URI."""
    assert gcs_uri.startswith("gs://"), f"Not a GCS URI: {gcs_uri}"
    without_scheme = gcs_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _gcs_download(gcs_uri: str, local_path: Path) -> None:
    """Download a single GCS object to a local file."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    logger.info(f"Downloaded gs://{bucket_name}/{blob_name} → {local_path}")


def _gcs_upload(local_path: Path, gcs_uri: str) -> None:
    """Upload a single local file to GCS."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    logger.debug(f"Uploaded {local_path} → gs://{bucket_name}/{blob_name}")


def _gcs_list_blobs(gcs_prefix: str) -> list[str]:
    """Return all blob names under a GCS prefix."""
    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    client = _gcs_client()
    blobs = client.bucket(bucket_name).list_blobs(prefix=prefix)
    return [b.name for b in blobs]


def _gcs_download_prefix(gcs_prefix: str, local_dir: Path) -> int:
    """Download all objects under *gcs_prefix* into *local_dir*, preserving
    relative paths.  Returns the number of files downloaded."""
    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    # Ensure trailing slash so prefix matching is exact
    prefix = prefix.rstrip("/") + "/"
    client = _gcs_client()
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
    if not blobs:
        logger.info(f"No objects found at gs://{bucket_name}/{prefix} (ok on first run)")
        return 0
    for blob in blobs:
        rel = blob.name[len(prefix):]   # strip the shared prefix
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    logger.info(
        f"Downloaded {len(blobs)} objects from gs://{bucket_name}/{prefix} → {local_dir}"
    )
    return len(blobs)


def _gcs_upload_tree(local_dir: Path, gcs_prefix: str) -> int:
    """Upload an entire local directory tree to *gcs_prefix*.  Returns the
    number of files uploaded."""
    if not local_dir.exists():
        logger.warning(f"Upload skipped — {local_dir} does not exist")
        return 0
    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    prefix = prefix.rstrip("/")
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    count = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(local_dir)
        blob_name = f"{prefix}/{rel}"
        bucket.blob(blob_name).upload_from_filename(str(local_file))
        count += 1
    logger.info(
        f"Uploaded {count} files from {local_dir} → gs://{bucket_name}/{prefix}"
    )
    return count


# ---------------------------------------------------------------------------
# Scenario manifest
# ---------------------------------------------------------------------------

def generate_manifest() -> dict[str, dict]:
    """Return the default 81-scenario manifest (9 stations × 9 biases).

    Pipe the output directly to GCS::

        python scripts/12_run_looo_cloud_shard.py --generate-manifest \\
            | gsutil cp - gs://exp-research/mpc-isolation-study/input/scenarios.json
    """
    from adam_orbit_det_eval.isolation_study import BIAS_NAMES, PHASE1_STATIONS

    manifest: dict[str, dict] = {}
    idx = 0
    for stn in PHASE1_STATIONS:
        for bn in BIAS_NAMES:
            manifest[str(idx)] = {"fake_code": stn["fake_code"], "bias_name": bn}
            idx += 1
    return manifest


def load_manifest(gcs_input_prefix: str, local_dir: Path) -> dict:
    manifest_path = local_dir / "scenarios.json"
    _gcs_download(f"{gcs_input_prefix}/scenarios.json", manifest_path)
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gcs-input-prefix",
                   help="GCS URI prefix containing scenarios.json and input parquets")
    p.add_argument("--gcs-output-prefix",
                   help="GCS URI prefix for scenario outputs")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Override JOB_COMPLETION_INDEX (for local testing)")
    p.add_argument("--propagator", choices=["twobody", "assist"], default="assist")
    p.add_argument("--max-processes", type=int, default=14)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--generate-manifest", action="store_true",
        help="Print the 81-scenario manifest JSON to stdout and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.generate_manifest:
        print(json.dumps(generate_manifest(), indent=2))
        return

    if not args.gcs_input_prefix or not args.gcs_output_prefix:
        logger.error("--gcs-input-prefix and --gcs-output-prefix are required")
        sys.exit(1)

    # Resolve shard index
    shard_index = args.shard_index
    if shard_index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            logger.error("JOB_COMPLETION_INDEX not set and --shard-index not provided")
            sys.exit(1)
        shard_index = int(raw)
    logger.info(f"Shard index: {shard_index}")

    with tempfile.TemporaryDirectory(prefix="looo_shard_") as tmpdir:
        tmp        = Path(tmpdir)
        input_dir  = tmp / "input"
        output_dir = tmp / "output"
        cache_dir  = tmp / "cache"
        for d in (input_dir, output_dir, cache_dir):
            d.mkdir()

        # --- Load manifest ---
        logger.info("Loading scenario manifest...")
        manifest = load_manifest(args.gcs_input_prefix, tmp)
        key = str(shard_index)
        if key not in manifest:
            logger.error(f"Shard {shard_index} not in manifest (size={len(manifest)})")
            sys.exit(1)
        fake_code = manifest[key]["fake_code"]
        bias_name = manifest[key]["bias_name"]

        from adam_orbit_det_eval.isolation_study import scenario_id
        sname = scenario_id(fake_code, bias_name)
        logger.info(f"Scenario: {sname}")

        # Paths for this scenario's checkpoints
        local_ckpt_dir = output_dir / sname / "looo_results" / "default" / "checkpoints"
        gcs_ckpt_prefix = f"{args.gcs_output_prefix}/{sname}/looo_results/default/checkpoints"

        # --- Recover checkpoints from a previous (preempted) run ---
        logger.info("Syncing checkpoints from GCS (spot-preemption recovery)...")
        _gcs_download_prefix(gcs_ckpt_prefix, local_ckpt_dir)

        # --- SIGTERM handler: upload checkpoints before pod is killed ---
        def _on_sigterm(signum, frame):
            logger.warning("SIGTERM — uploading checkpoints before exit")
            _gcs_upload_tree(local_ckpt_dir, gcs_ckpt_prefix)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _on_sigterm)

        # --- Download input data ---
        logger.info("Downloading input parquet files...")
        for fname in ("mpc_observations.parquet", "mpc_orbits.parquet"):
            _gcs_download(f"{args.gcs_input_prefix}/{fname}", input_dir / fname)

        # --- Load observations and orbits ---
        from mpcq.observations import MPCObservations
        from mpcq.orbits import MPCOrbits
        obs_template = MPCObservations.from_parquet(input_dir / "mpc_observations.parquet")
        truth_orbits = MPCOrbits.from_parquet(input_dir / "mpc_orbits.parquet")
        object_ids = obs_template.requested_provid.unique().to_pylist()
        logger.info(f"  {len(object_ids)} objects")

        # --- Select propagator ---
        if args.propagator == "twobody":
            from adam_orbit_det_eval.propagators import TwoBodyPropagator
            propagator_class = TwoBodyPropagator
        else:
            from adam_assist import ASSISTPropagator
            propagator_class = ASSISTPropagator

        # --- Run the scenario ---
        logger.info(f"Running pipeline for {sname}...")
        from adam_orbit_det_eval.isolation_study import run_scenario
        rows = run_scenario(
            fake_code=fake_code,
            bias_name=bias_name,
            obs_template=obs_template,
            truth_orbits=truth_orbits,
            object_ids=object_ids,
            propagator_class=propagator_class,
            output_base=output_dir,
            cache_dir=cache_dir,
            max_processes=args.max_processes,
            threshold_arcsec=args.threshold,
            force=args.force,
        )

        if rows is None:
            logger.error(f"Scenario {sname} failed — uploading partial checkpoints")
            _gcs_upload_tree(local_ckpt_dir, gcs_ckpt_prefix)
            sys.exit(1)

        logger.info(f"Scenario {sname} complete — {len(rows)} recovery rows")

        # --- Upload all outputs ---
        logger.info(f"Uploading outputs to {args.gcs_output_prefix}/{sname}/")
        _gcs_upload_tree(output_dir / sname, f"{args.gcs_output_prefix}/{sname}")

        logger.info("Done.")


if __name__ == "__main__":
    main()
