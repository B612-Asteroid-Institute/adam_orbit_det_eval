#!/usr/bin/env python3
"""
04_run_lsst_reference.py
========================
Run the reference-orbit evaluation pipeline over the sample produced by
01_fetch_mpc_sample.py.

Unlike LOOO (which holds one observatory out at a time and refits with the
rest), this pipeline:

  1. Fits the orbit using ONLY the reference station's observations
     (default: X05 = Rubin Observatory / LSST).
  2. Predicts sky-plane positions at ALL other stations' observation times.
  3. Records residuals and metadata, using the same LOOOResult schema so
     03_analyze.py works unchanged.

This is the natural calibration direction for LSST: use the dense, high-
quality X05 arc as ground truth and measure systematic offsets at other
stations relative to that reference orbit.

Output is written incrementally to a Parquet file via per-object checkpoints,
so partial runs are preserved and the job can be safely restarted.

Usage
-----
    python scripts/04_run_lsst_reference.py \\
        --input-dir data/looo_sample_lsst100 \\
        --output-dir data/lsst_reference_results \\
        --propagator assist \\
        --sigma-model veres2017 \\
        --reference-stn X05 \\
        --min-ref-obs 20 \\
        --max-processes 6

Propagators
-----------
twobody  : adam_core 2-body propagator (fast, ~ms/orbit).  Adequate for
           residual-level studies where errors are dominated by observation
           noise rather than propagation error.
assist   : ASSIST N-body propagator (accurate, ~100ms/orbit).  Recommended
           for high-precision studies or when validating orbit elements.
"""

import argparse
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
        "--input-dir",
        type=Path,
        default=Path("data/looo_sample_lsst100"),
        help="Directory with mpc_observations.parquet and mpc_orbits.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lsst_reference_results"),
        help="Base directory for output; results go in a timestamped subdirectory "
             "unless --run-id is provided.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run identifier (e.g. 'run_001').  Defaults to a UTC timestamp "
             "so each run is stored separately and prior results are never overwritten.",
    )
    p.add_argument(
        "--propagator",
        choices=["twobody", "assist"],
        default="twobody",
        help="Propagator backend (default: twobody)",
    )
    p.add_argument(
        "--reference-stn",
        type=str,
        default="X05",
        help="MPC station code for the reference / anchor station whose observations "
             "are used for the DC fit (default: X05 = Rubin Observatory / LSST).",
    )
    p.add_argument(
        "--min-ref-obs",
        type=int,
        default=20,
        help="Minimum number of reference-station observations required per object "
             "before the reference-orbit fit is trusted.  Objects with fewer reference "
             "observations are skipped (default: 20).",
    )
    p.add_argument(
        "--min-obs-held-out",
        type=int,
        default=1,
        help="Minimum observations a non-reference station must have for it to be "
             "evaluated (default: 1).",
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    p.add_argument(
        "--sigma-model",
        choices=["veres2017", "const"],
        default="veres2017",
        help="How to fill missing rmsra/rmsdec values: "
             "'veres2017' uses per-(stn, catalog) sigma lookup from Veres et al. 2017 "
             "(default); 'const' fills with a small constant (inflates chi2).",
    )
    p.add_argument(
        "--object-ids",
        nargs="*",
        help="Restrict to these object IDs (default: all objects in input)",
    )
    return p.parse_args()


def get_propagator_class(name: str):
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
                "ASSISTPropagator not found. Install adam-assist: "
                "pip install adam-assist"
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown propagator: {name}")


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id}  →  {output_dir}")

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
    logger.info(f"Sigma model: {args.sigma_model}")

    import pyarrow.compute as pc
    n_objects = len(pc.unique(mpc_observations.requested_provid))
    logger.info(
        f"Loaded {len(mpc_observations)} observations for {n_objects} objects"
    )

    # --- Configure pipeline ---
    from adam_orbit_det_eval.looo import run_reference_orbit_pipeline
    from adam_orbit_det_eval.looo.core import LOOOConfig

    config = LOOOConfig(
        min_obs_held_out=args.min_obs_held_out,
        # arc_length and held_out_fraction filters are not applied in the
        # reference-orbit pipeline; min_ref_obs takes their place.
    )

    propagator_class = get_propagator_class(args.propagator)
    logger.info(f"Using propagator: {propagator_class.__name__}")
    logger.info(
        f"Reference station: {args.reference_stn}  "
        f"(min_ref_obs={args.min_ref_obs})"
    )

    # --- Write run configuration for reproducibility ---
    import json
    run_config = {
        "run_id": run_id,
        "pipeline": "reference_orbit",
        "propagator": args.propagator,
        "sigma_model": args.sigma_model,
        "reference_stn": args.reference_stn,
        "min_ref_obs": args.min_ref_obs,
        "min_obs_held_out": args.min_obs_held_out,
        "max_processes": args.max_processes,
        "input_obs": str(obs_path),
        "input_orbits": str(orbits_path),
        "n_input_objects": int(n_objects),
        "n_input_observations": len(mpc_observations),
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    logger.info(f"Run configuration written to {config_path}")

    output_path = output_dir / "reference_orbit_results.parquet"

    # --- Run the pipeline ---
    logger.info("Starting reference-orbit pipeline...")
    results = run_reference_orbit_pipeline(
        mpc_observations=mpc_observations,
        mpc_orbits=mpc_orbits,
        propagator_class=propagator_class,
        output_path=output_path,
        config=config,
        object_ids=args.object_ids,
        max_processes=args.max_processes,
        sigma_model=args.sigma_model,
        reference_stn=args.reference_stn,
        min_ref_obs=args.min_ref_obs,
    )

    logger.info(
        f"Reference-orbit pipeline complete: {len(results)} evaluated observation "
        f"results written to {output_path}"
    )


if __name__ == "__main__":
    main()
