#!/usr/bin/env python3
"""
12_run_looo_cloud_shard.py
==========================
Cloud-aware shard runner for the MPC-scale real-data LOOO study.

Designed to run inside an exp-research indexed job pod.  Each pod processes
ONE shard (a group of ~1000 objects) identified by ``JOB_COMPLETION_INDEX``.

Input layout (produced by ``15_fetch_mpc_scale.py`` + uploaded to GCS):

    <gcs_input_prefix>/
        shard_000/
            mpc_observations.parquet
            mpc_orbits.parquet
        shard_001/
            ...

Output layout:

    <gcs_output_prefix>/
        shard_000/
            looo_results.parquet
            observatory_stats.parquet
            program_code_stats.parquet
            checkpoints/          ← per-object checkpoints (GCS-backed)
            SUCCESS               ← written on successful completion
        shard_001/
            FAILED                ← written on unrecoverable error (includes traceback)
            ...

Usage
-----
  # Inside a pod (env: JOB_COMPLETION_INDEX=3)
  python scripts/12_run_looo_cloud_shard.py \\
      --gcs-input-prefix  gs://exp-research/mpc-real-data-looo/input \\
      --gcs-output-prefix gs://exp-research/mpc-real-data-looo/output \\
      --propagator assist \\
      --orbit-fitter findorb --strict-fitter

  # Local smoke-test (override shard index)
  python scripts/12_run_looo_cloud_shard.py \\
      --gcs-input-prefix  gs://exp-research/mpc-real-data-looo/input \\
      --gcs-output-prefix gs://exp-research/mpc-real-data-looo/output \\
      --shard-index 0 \\
      --propagator twobody \\
      --max-processes 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers (lazy import of google-cloud-storage)
# ---------------------------------------------------------------------------

def _get_gcs_client():
    """Lazy-import google.cloud.storage and return a Client."""
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


def download_shard(gcs_input_prefix: str, shard_name: str, local_dir: Path) -> Path:
    """Download a shard's input parquet files from GCS to a local directory.

    Returns the local shard directory containing the downloaded files.
    """
    client = _get_gcs_client()
    bucket_name, prefix = _parse_gcs_uri(gcs_input_prefix)
    bucket = client.bucket(bucket_name)

    shard_dir = local_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("mpc_observations.parquet", "mpc_orbits.parquet"):
        blob_path = f"{prefix}/{shard_name}/{fname}" if prefix else f"{shard_name}/{fname}"
        local_path = shard_dir / fname
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(local_path))
        logger.info(f"Downloaded gs://{bucket_name}/{blob_path} -> {local_path}")

    return shard_dir


def upload_file(local_path: Path, gcs_uri: str) -> None:
    """Upload a single local file to a GCS URI."""
    client = _get_gcs_client()
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(str(local_path))
    logger.info(f"Uploaded {local_path} -> {gcs_uri}")


def upload_string(content: str, gcs_uri: str) -> None:
    """Upload a string as a blob to GCS (for marker files)."""
    client = _get_gcs_client()
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_string(content)
    logger.info(f"Wrote marker -> {gcs_uri}")


def upload_results(local_output: Path, gcs_output_prefix: str, shard_name: str) -> None:
    """Upload all result files from a shard's local output to GCS."""
    result_files = [
        "looo_results.parquet",
        "observatory_stats.parquet",
        "program_code_stats.parquet",
    ]
    for fname in result_files:
        local_path = local_output / fname
        if local_path.exists():
            gcs_uri = f"{gcs_output_prefix}/{shard_name}/{fname}"
            upload_file(local_path, gcs_uri)
        else:
            logger.warning(f"Expected output file not found: {local_path}")


# ---------------------------------------------------------------------------
# Propagator / orbit-fitter helpers (same pattern as 02_run_looo.py)
# ---------------------------------------------------------------------------

def get_propagator_class(name: str):
    """Return the propagator class for the given name."""
    if name == "twobody":
        from adam_orbit_det_eval.propagators import TwoBodyPropagator
        return TwoBodyPropagator
    elif name == "assist":
        try:
            from adam_assist import ASSISTPropagator
            return ASSISTPropagator
        except ImportError:
            logger.error(
                "ASSISTPropagator not found. Install adam-assist: "
                "pip install adam-assist"
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown propagator: {name}")


def get_orbit_fitter(name: str, strict: bool = True):
    """Build an OrbitFitter instance (or None for scipy DC fallback).

    Parameters
    ----------
    name : str
        Fitter name: "scipy", "findorb", or "native".
    strict : bool
        If True (default), abort when the requested fitter is not importable.
    """
    if name == "scipy":
        return None
    if name == "findorb":
        try:
            from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
            return FindOrbOrbitFitter()
        except ImportError as e:
            msg = (
                f"FindOrbOrbitFitter unavailable ({e}). "
                "Cannot use --orbit-fitter=findorb."
            )
            if strict:
                logger.critical(msg + " Aborting (use --no-strict-fitter to allow fallback).")
                sys.exit(1)
            logger.critical(msg + " Falling back to scipy DC.")
            return None
    if name == "native":
        try:
            from adam_core.orbit_determination.native_orbit_fitter import NativeOrbitFitter
            return NativeOrbitFitter()
        except ImportError as e:
            msg = (
                f"NativeOrbitFitter unavailable ({e}). "
                "Cannot use --orbit-fitter=native."
            )
            if strict:
                logger.critical(msg + " Aborting (use --no-strict-fitter to allow fallback).")
                sys.exit(1)
            logger.critical(msg + " Falling back to scipy DC.")
            return None
    raise ValueError(f"Unknown orbit fitter: {name}")


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------

def resolve_shard_index(cli_shard_index: int | None) -> int:
    """Determine shard index from CLI arg or JOB_COMPLETION_INDEX env var."""
    if cli_shard_index is not None:
        return cli_shard_index
    raw = os.environ.get("JOB_COMPLETION_INDEX")
    if raw is None:
        logger.error("JOB_COMPLETION_INDEX not set and --shard-index not provided")
        sys.exit(1)
    return int(raw)


def shard_name_from_index(index: int) -> str:
    """Format shard index as zero-padded directory name."""
    return f"shard_{index:03d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cloud shard runner for real-data LOOO evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gcs-input-prefix",
        required=True,
        help="GCS URI prefix for sharded input data (e.g. gs://bucket/input)",
    )
    p.add_argument(
        "--gcs-output-prefix",
        required=True,
        help="GCS URI prefix for output data (e.g. gs://bucket/output)",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Override JOB_COMPLETION_INDEX (for local testing)",
    )
    p.add_argument(
        "--propagator",
        choices=["twobody", "assist"],
        default="assist",
        help="Propagator to use (default: assist)",
    )
    p.add_argument(
        "--orbit-fitter",
        choices=["scipy", "findorb", "native"],
        default="findorb",
        help="Orbit fitter for hold-in fits (default: findorb)",
    )
    p.add_argument(
        "--strict-fitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the requested orbit fitter is not importable (default: True). "
             "Use --no-strict-fitter to allow silent fallback to scipy DC.",
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all CPUs)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    # Resolve shard
    shard_index = resolve_shard_index(args.shard_index)
    shard = shard_name_from_index(shard_index)
    logger.info(f"Processing {shard} (index={shard_index})")

    # Local working directories
    local_input = Path(f"/tmp/looo_input/{shard}")
    local_output = Path(f"/tmp/looo_output/{shard}")
    local_input.mkdir(parents=True, exist_ok=True)
    local_output.mkdir(parents=True, exist_ok=True)

    try:
        # --- Download shard input from GCS ---
        shard_dir = download_shard(args.gcs_input_prefix, shard, local_input.parent)
        logger.info(f"Input downloaded to {shard_dir}")

        # --- Load observations and orbits ---
        from mpcq.observations import MPCObservations
        from mpcq.orbits import MPCOrbits

        mpc_obs = MPCObservations.from_parquet(shard_dir / "mpc_observations.parquet")
        mpc_orbits = MPCOrbits.from_parquet(shard_dir / "mpc_orbits.parquet")
        n_objects = len(mpc_obs.requested_provid.unique())
        n_total_obs = len(mpc_obs)
        logger.info(f"Loaded {n_objects} objects, {n_total_obs} observations")

        # --- Setup GCS checkpoint store ---
        from adam_orbit_det_eval.looo.gcs_checkpoint import GCSCheckpointStore

        checkpoint_dir = local_output / "checkpoints"
        gcs_ckpt_prefix = f"{args.gcs_output_prefix}/{shard}/checkpoints"
        gcs_store = GCSCheckpointStore(checkpoint_dir, gcs_ckpt_prefix)
        gcs_store.register_sigterm_handler()

        # --- Configure propagator and orbit fitter ---
        propagator_class = get_propagator_class(args.propagator)
        orbit_fitter = get_orbit_fitter(args.orbit_fitter, strict=args.strict_fitter)

        # --- Configure LOOO pipeline ---
        from adam_orbit_det_eval.looo.core import LOOOConfig

        config = LOOOConfig()
        results_path = local_output / "looo_results.parquet"

        # --- Run LOOO pipeline ---
        from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline

        logger.info(
            f"Starting LOOO pipeline: propagator={args.propagator}, "
            f"orbit_fitter={args.orbit_fitter}, max_processes={args.max_processes}"
        )
        results = run_looo_pipeline(
            mpc_observations=mpc_obs,
            mpc_orbits=mpc_orbits,
            propagator_class=propagator_class,
            output_path=results_path,
            config=config,
            orbit_fitter=orbit_fitter,
            gcs_checkpoint_store=gcs_store,
            max_processes=args.max_processes,
        )
        logger.info(f"LOOO pipeline complete: {len(results)} result rows")

        # --- Run analysis ---
        from adam_orbit_det_eval.looo.analysis import (
            compute_observatory_stats,
            compute_program_code_stats,
        )

        obs_stats = compute_observatory_stats(results)
        obs_stats.to_parquet(local_output / "observatory_stats.parquet")
        logger.info(f"Observatory stats: {len(obs_stats)} stations")

        prog_stats = compute_program_code_stats(results)
        prog_stats.to_parquet(local_output / "program_code_stats.parquet")
        logger.info(f"Program code stats: {len(prog_stats)} groups")

        # --- Upload results to GCS ---
        upload_results(local_output, args.gcs_output_prefix, shard)

        # --- Write SUCCESS marker ---
        elapsed = time.time() - t0
        success_msg = (
            f"shard={shard}\n"
            f"n_objects={n_objects}\n"
            f"n_observations={n_total_obs}\n"
            f"n_result_rows={len(results)}\n"
            f"n_observatory_stats={len(obs_stats)}\n"
            f"n_program_code_stats={len(prog_stats)}\n"
            f"elapsed_seconds={elapsed:.1f}\n"
        )
        upload_string(success_msg, f"{args.gcs_output_prefix}/{shard}/SUCCESS")
        logger.info(f"Shard {shard} completed successfully in {elapsed:.1f}s")

    except Exception:
        # --- Write FAILED marker with traceback ---
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        fail_msg = (
            f"shard={shard}\n"
            f"elapsed_seconds={elapsed:.1f}\n"
            f"error:\n{tb}"
        )
        try:
            upload_string(fail_msg, f"{args.gcs_output_prefix}/{shard}/FAILED")
        except Exception as upload_err:
            logger.error(f"Could not upload FAILED marker: {upload_err}")
        logger.error(f"Shard {shard} FAILED after {elapsed:.1f}s:\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
