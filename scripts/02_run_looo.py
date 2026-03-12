#!/usr/bin/env python3
"""
02_run_looo.py
==============
Run the Leave-One-Observatory-Out cross-validation over the sample
produced by 01_fetch_mpc_sample.py.

For each (object, observatory) pair that passes eligibility filters:
  1. Remove the observatory's observations
  2. Refit orbit via differential correction (starting from MPC orbit)
  3. Predict at held-out observation times
  4. Record residuals and metadata

Output is written incrementally to a Parquet file so partial runs are
preserved and the job can be safely restarted.

Usage
-----
    python scripts/02_run_looo.py \
        --input-dir data/looo_sample \
        --output-dir data/looo_results \
        --propagator twobody \
        --min-obs-remaining 6 \
        --min-arc-length 7.0 \
        --max-held-out-fraction 0.8 \
        --max-processes 8

Propagators
-----------
twobody  : adam_core 2-body propagator (fast, ~ms/orbit, less accurate for
           objects near resonances or with significant perturbations).
           Adequate for residual-level studies where the error is dominated
           by observation noise, not propagation error.
assist   : ASSIST N-body propagator (accurate, ~100ms/orbit). Use for
           higher-precision studies or when validating orbit elements.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/looo_sample"),
        help="Directory with mpc_observations.parquet and mpc_orbits.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_results"),
        help="Directory for output parquet files",
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
        help="Min observations that must remain after hold-out (default: 6)",
    )
    p.add_argument(
        "--min-arc-length",
        type=float,
        default=7.0,
        help="Min arc length (days) that must remain after hold-out (default: 7.0)",
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
        help="Max fraction of total obs that may be held out (default: 0.8)",
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1). "
             "Note: the pipeline currently runs serially; parallelism "
             "over objects will be added in a future iteration.",
    )
    p.add_argument(
        "--object-ids",
        nargs="*",
        help="Restrict to these object IDs (default: all objects in input)",
    )
    p.add_argument(
        "--write-interval",
        type=int,
        default=50,
        help="Write results to disk every N objects (default: 50)",
    )
    return p.parse_args()


def get_propagator_class(name: str):
    if name == "twobody":
        try:
            from adam_core.dynamics.propagation import TwoBodyPropagator
            return TwoBodyPropagator
        except ImportError:
            logger.error(
                "TwoBodyPropagator not found. Make sure adam_core is installed."
            )
            sys.exit(1)
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load input data ---
    obs_path = args.input_dir / "mpc_observations.parquet"
    orbits_path = args.input_dir / "mpc_orbits.parquet"

    if not obs_path.exists():
        logger.error(f"Observations file not found: {obs_path}")
        logger.error("Run 01_fetch_mpc_sample.py first.")
        sys.exit(1)
    if not orbits_path.exists():
        logger.error(f"Orbits file not found: {orbits_path}")
        sys.exit(1)

    logger.info(f"Loading observations from {obs_path}")
    from mpcq.observations import MPCObservations
    mpc_observations = MPCObservations.from_parquet(obs_path)

    logger.info(f"Loading orbits from {orbits_path}")
    from mpcq.orbits import MPCOrbits
    mpc_orbits = MPCOrbits.from_parquet(orbits_path)

    import pyarrow.compute as pc
    n_objects = len(pc.unique(mpc_observations.requested_provid))
    logger.info(
        f"Loaded {len(mpc_observations)} observations for {n_objects} objects"
    )

    # --- Configure LOOO ---
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

    # --- Write run configuration for reproducibility ---
    import json
    run_config = {
        "propagator": args.propagator,
        "min_obs_remaining": args.min_obs_remaining,
        "min_arc_length_days": args.min_arc_length,
        "min_obs_held_out": args.min_obs_held_out,
        "max_held_out_fraction": args.max_held_out_fraction,
        "input_obs": str(obs_path),
        "input_orbits": str(orbits_path),
        "n_input_objects": int(n_objects),
        "n_input_observations": len(mpc_observations),
    }
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    logger.info(f"Run configuration written to {config_path}")

    output_path = args.output_dir / "looo_results.parquet"

    # --- Run the pipeline ---
    logger.info("Starting LOOO pipeline...")
    results = run_looo_pipeline(
        mpc_observations=mpc_observations,
        mpc_orbits=mpc_orbits,
        propagator_class=propagator_class,
        output_path=output_path,
        config=config,
        object_ids=args.object_ids,
        max_processes=args.max_processes,
        write_interval=args.write_interval,
    )

    logger.info(
        f"LOOO complete: {len(results)} held-out observation results "
        f"written to {output_path}"
    )


if __name__ == "__main__":
    main()
