#!/usr/bin/env python3
"""
08_run_sim_pipeline.py
======================
Run the full LOOO analysis pipeline on a synthetic dataset.

This is a thin wrapper that calls ``run_looo_pipeline`` and the analysis
functions on a synthetic dataset directory produced by
``07_generate_sim_dataset.py``.  It mirrors the structure of
``02_run_looo.py`` but operates directly on the synthetic dataset directory
layout and calls into the pipeline programmatically rather than via subprocess.

Outputs are written under ``--output-dir/<run-id>/``:
  looo_results/default/looo_results.parquet
  analysis/default/observatory_stats.parquet
  analysis/default/catalog_stats.parquet
  analysis/default/analysis_config.json
  analysis/default/observatory_summary.txt

Usage
-----
    python scripts/08_run_sim_pipeline.py \\
        --dataset-dir data/sim_products/phase1_constant_bias/datasets/default \\
        --output-dir data/sim_products/phase1_constant_bias \\
        --propagator twobody \\
        --max-processes 6
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the synthetic dataset "
             "(mpc_observations.parquet and mpc_orbits.parquet).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory for LOOO results and analysis.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="default",
        help="Dataset identifier used for output subdirectory names (default: default).",
    )
    p.add_argument(
        "--propagator",
        choices=["twobody", "assist"],
        default="twobody",
        help="Propagator backend (default: twobody)",
    )
    p.add_argument(
        "--min-obs-remaining",
        type=int,
        default=6,
        help="Min observations remaining after hold-out (default: 6)",
    )
    p.add_argument(
        "--min-arc-length",
        type=float,
        default=7.0,
        help="Min arc length (days) remaining after hold-out (default: 7.0)",
    )
    p.add_argument(
        "--min-obs-held-out",
        type=int,
        default=1,
        help="Min observations held out per (object, stn) pair (default: 1)",
    )
    p.add_argument(
        "--max-held-out-fraction",
        type=float,
        default=0.8,
        help="Max fraction of total obs held out (default: 0.8)",
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    p.add_argument(
        "--sigma-model",
        choices=["veres2017", "const"],
        default="veres2017",
        help="How to fill missing rmsra/rmsdec values (default: veres2017). "
             "Note: synthetic obs always have rmsra/rmsdec populated, so this "
             "only matters for edge cases.",
    )
    p.add_argument(
        "--object-ids",
        nargs="*",
        help="Restrict to these object IDs (default: all objects in dataset).",
    )
    p.add_argument(
        "--skip-looo",
        action="store_true",
        default=False,
        help="Skip the LOOO step and only run analysis on existing results.",
    )
    p.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip the analysis step (only run LOOO).",
    )
    p.add_argument(
        "--min-obs-per-stn",
        type=int,
        default=5,
        help="Min held-out obs per observatory to report in analysis (default: 5).",
    )
    p.add_argument(
        "--max-chi2",
        type=float,
        default=100.0,
        help="Exclude rows with hold-in reduced-chi2 > threshold (default: 100.0).",
    )
    return p.parse_args()


def get_propagator_class(name: str):
    """Return the propagator class for the given name."""
    if name == "twobody":
        try:
            from adam_core.dynamics.propagation import propagate_2body
            from adam_core.propagator.propagator import Propagator

            class TwoBodyPropagator(Propagator):
                def _propagate_orbits(self, orbits, times, max_iter=1000, tol=1e-14, **kwargs):
                    return propagate_2body(orbits, times, max_iter=max_iter, tol=tol)

            return TwoBodyPropagator
        except Exception as e:
            logger.error(f"Could not set up TwoBodyPropagator: {e}")
            sys.exit(1)
    elif name == "assist":
        try:
            from adam_assist import ASSISTPropagator
            return ASSISTPropagator
        except ImportError:
            logger.error(
                "ASSISTPropagator not found.  Install adam-assist: pip install adam-assist"
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown propagator: {name}")


def run_looo_step(args, dataset_dir: Path, looo_output_dir: Path):
    """Run the LOOO pipeline on the synthetic dataset."""
    obs_path = dataset_dir / "mpc_observations.parquet"
    orbits_path = dataset_dir / "mpc_orbits.parquet"

    if not obs_path.exists():
        logger.error(f"Observations file not found: {obs_path}")
        sys.exit(1)
    if not orbits_path.exists():
        logger.error(f"Orbits file not found: {orbits_path}")
        sys.exit(1)

    from mpcq.observations import MPCObservations
    from mpcq.orbits import MPCOrbits

    logger.info(f"Loading synthetic observations from {obs_path}")
    mpc_observations = MPCObservations.from_parquet(obs_path)
    logger.info(f"Loading truth orbits from {orbits_path}")
    mpc_orbits = MPCOrbits.from_parquet(orbits_path)

    import pyarrow.compute as pc
    n_objects = len(pc.unique(mpc_observations.requested_provid))
    logger.info(
        f"Loaded {len(mpc_observations)} synthetic observations for {n_objects} objects"
    )

    from adam_orbit_det_eval.looo import run_looo_pipeline
    from adam_orbit_det_eval.looo.core import LOOOConfig

    config = LOOOConfig(
        min_obs_held_out=args.min_obs_held_out,
        min_obs_remaining=args.min_obs_remaining,
        min_arc_length_days=args.min_arc_length,
        max_held_out_fraction=args.max_held_out_fraction,
    )

    propagator_class = get_propagator_class(args.propagator)
    logger.info(f"Using propagator: {propagator_class.__name__}")

    looo_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = looo_output_dir / "looo_results.parquet"

    # Write run config
    run_config = {
        "dataset_dir": str(dataset_dir),
        "propagator": args.propagator,
        "sigma_model": args.sigma_model,
        "min_obs_remaining": args.min_obs_remaining,
        "min_arc_length_days": args.min_arc_length,
        "min_obs_held_out": args.min_obs_held_out,
        "max_held_out_fraction": args.max_held_out_fraction,
        "max_processes": args.max_processes,
        "n_input_objects": int(n_objects),
        "n_input_observations": len(mpc_observations),
    }
    (looo_output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2)
    )

    logger.info("Starting LOOO pipeline on synthetic dataset...")
    results = run_looo_pipeline(
        mpc_observations=mpc_observations,
        mpc_orbits=mpc_orbits,
        propagator_class=propagator_class,
        output_path=output_path,
        config=config,
        object_ids=args.object_ids,
        max_processes=args.max_processes,
        sigma_model=args.sigma_model,
    )
    logger.info(
        f"LOOO complete: {len(results)} held-out observation results "
        f"written to {output_path}"
    )
    return output_path


def run_analysis_step(args, looo_results_path: Path, analysis_output_dir: Path):
    """Run the analysis step on LOOO results."""
    if not looo_results_path.exists():
        logger.error(f"LOOO results not found: {looo_results_path}")
        sys.exit(1)

    from adam_orbit_det_eval.looo.core import LOOOResult
    from adam_orbit_det_eval.looo.analysis import (
        compute_observatory_stats,
        compute_catalog_stats,
        print_observatory_summary,
    )

    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import io, contextlib

    logger.info(f"Loading LOOO results from {looo_results_path}")
    results = LOOOResult(pq.read_table(looo_results_path))

    n_obs = len(results)
    n_objects = len(pc.unique(results.object_id))
    n_stns = len(pc.unique(results.stn))
    logger.info(
        f"Loaded {n_obs} held-out observations across "
        f"{n_objects} objects and {n_stns} observatories"
    )

    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    analysis_config = {
        "input_results": str(looo_results_path),
        "n_input_rows": n_obs,
        "n_input_objects": int(n_objects),
        "n_input_stns": int(n_stns),
        "max_hold_in_reduced_chi2": args.max_chi2,
        "min_obs_per_stn": args.min_obs_per_stn,
    }
    (analysis_output_dir / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2)
    )

    logger.info("Computing per-observatory statistics...")
    obs_stats = compute_observatory_stats(
        results,
        max_hold_in_reduced_chi2=args.max_chi2,
        min_obs_per_stn=args.min_obs_per_stn,
    )

    if len(obs_stats) == 0:
        logger.warning("No observatories passed filters — observatory_stats is empty.")
    else:
        logger.info(f"Observatory statistics for {len(obs_stats)} stations.")

    obs_stats_path = analysis_output_dir / "observatory_stats.parquet"
    obs_stats.to_parquet(obs_stats_path)
    logger.info(f"Observatory statistics → {obs_stats_path}")

    # Catalog stats
    logger.info("Computing per-(stn, catalog) statistics...")
    cat_stats = compute_catalog_stats(
        results,
        max_hold_in_reduced_chi2=args.max_chi2,
        min_obs_per_group=args.min_obs_per_stn,
    )
    cat_stats_path = analysis_output_dir / "catalog_stats.parquet"
    cat_stats.to_parquet(cat_stats_path)
    logger.info(f"Catalog statistics → {cat_stats_path}")

    # Human-readable summary
    summary_path = analysis_output_dir / "observatory_summary.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_observatory_summary(obs_stats)
    summary_text = buf.getvalue()
    summary_path.write_text(summary_text)
    print(summary_text)

    return analysis_output_dir


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_id = args.run_id

    looo_output_dir = output_dir / "looo_results" / run_id
    analysis_output_dir = output_dir / "analysis" / run_id

    # --- LOOO step ---
    if not args.skip_looo:
        looo_results_path = run_looo_step(args, dataset_dir, looo_output_dir)
    else:
        looo_results_path = looo_output_dir / "looo_results.parquet"
        logger.info(f"Skipping LOOO step; expecting results at {looo_results_path}")

    # --- Analysis step ---
    if not args.skip_analysis:
        run_analysis_step(args, looo_results_path, analysis_output_dir)
    else:
        logger.info("Skipping analysis step.")

    logger.info("Done.")
    logger.info(f"LOOO results:   {looo_output_dir}")
    logger.info(f"Analysis:       {analysis_output_dir}")


if __name__ == "__main__":
    main()
