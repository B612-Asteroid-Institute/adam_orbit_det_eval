#!/usr/bin/env python3
"""
16_atct_real_data.py
====================
Post-process real-data LOOO results to compute Along-Track / Cross-Track
residual decomposition.

This is a POST-PROCESSING step on existing LOOO results.  It does NOT
modify any pipeline outputs — it reads LOOO results, computes velocity
vectors from the original MPC catalog orbits via generate_ephemeris(),
projects residuals into AT/CT components, and writes an augmented parquet.

Along-track  (AT) = residual component in the direction of apparent motion.
Cross-track  (CT) = residual component perpendicular to motion (90 deg CCW).

Velocity source: the ORIGINAL MPC catalog orbit, using the analytically-
computed vlon/vlat fields from generate_ephemeris().  Research (bead 0at)
confirmed that all velocity source options give identical unit vectors to
well within measurement precision.

Usage
-----
    python scripts/16_atct_real_data.py \\
        --looo-results data/looo_results/run_001/looo_results.parquet \\
        --observations data/looo_sample/mpc_observations.parquet \\
        --orbits data/looo_sample/mpc_orbits.parquet \\
        --output data/looo_results/run_001/looo_results_atct.parquet \\
        --propagator twobody
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="AT/CT decomposition for real-data LOOO results."
    )
    p.add_argument(
        "--looo-results",
        type=Path,
        required=True,
        help="Path to LOOO results parquet (output of 02_run_looo.py)",
    )
    p.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Path to MPC observations parquet (input to the LOOO run)",
    )
    p.add_argument(
        "--orbits",
        type=Path,
        required=True,
        help="Path to MPC orbits parquet (original catalog orbits)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the augmented output parquet",
    )
    p.add_argument(
        "--propagator",
        choices=["twobody", "assist"],
        default="twobody",
        help="Propagator backend for ephemeris generation (default: twobody)",
    )
    return p.parse_args()


def get_propagator(name: str):
    """Instantiate a propagator."""
    if name == "twobody":
        try:
            from adam_orbit_det_eval.propagators import TwoBodyPropagator
            return TwoBodyPropagator()
        except Exception as e:
            logger.error(f"Could not import TwoBodyPropagator: {e}")
            sys.exit(1)
    elif name == "assist":
        try:
            from adam_assist import ASSISTPropagator
            return ASSISTPropagator()
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

    # --- Validate inputs ---
    for path, label in [
        (args.looo_results, "LOOO results"),
        (args.observations, "observations"),
        (args.orbits, "orbits"),
    ]:
        if not path.exists():
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Load LOOO results ---
    logger.info(f"Loading LOOO results from {args.looo_results}")
    looo_df = pq.read_table(args.looo_results).to_pandas()
    logger.info(f"  {len(looo_df)} residual rows")

    if looo_df.empty:
        logger.error("LOOO results are empty, nothing to process.")
        sys.exit(1)

    n_objects = looo_df["object_id"].nunique()
    n_stns = looo_df["stn"].nunique()
    logger.info(f"  {n_objects} unique objects, {n_stns} unique stations")

    # --- Load observations (for times and codes) ---
    logger.info(f"Loading observations from {args.observations}")
    from mpcq.observations import MPCObservations
    mpc_obs = MPCObservations.from_parquet(args.observations)
    logger.info(f"  {len(mpc_obs)} observations loaded")

    obs_ids = mpc_obs.obsid.to_numpy(zero_copy_only=False)
    obs_times = mpc_obs.obstime
    obs_codes = mpc_obs.stn.to_numpy(zero_copy_only=False)
    obs_object_ids = mpc_obs.requested_provid.to_numpy(zero_copy_only=False)

    # --- Load orbits ---
    logger.info(f"Loading orbits from {args.orbits}")
    from mpcq.orbits import MPCOrbits
    mpc_orbits = MPCOrbits.from_parquet(args.orbits)
    orbits = mpc_orbits.orbits()
    logger.info(f"  {len(orbits)} orbits loaded")

    # --- Instantiate propagator ---
    propagator = get_propagator(args.propagator)
    logger.info(f"Using propagator: {type(propagator).__name__}")

    # --- Compute AT/CT ---
    from adam_orbit_det_eval.looo.atct import augment_looo_results_with_atct

    logger.info("Computing AT/CT decomposition...")
    augmented = augment_looo_results_with_atct(
        looo_df=looo_df,
        orbits=orbits,
        obs_times=obs_times,
        obs_ids=obs_ids,
        obs_object_ids=obs_object_ids,
        obs_codes=obs_codes,
        propagator=propagator,
    )

    # --- Summary statistics ---
    valid_at = augmented["residual_at_arcsec"].notna()
    logger.info(f"AT/CT valid: {valid_at.sum()}/{len(augmented)}")

    if valid_at.any():
        at_vals = augmented.loc[valid_at, "residual_at_arcsec"]
        ct_vals = augmented.loc[valid_at, "residual_ct_arcsec"]
        speed_vals = augmented.loc[valid_at, "speed_deg_per_day"]
        logger.info(
            f"  AT: mean={at_vals.mean():+.4f}\", "
            f"std={at_vals.std():.4f}\", "
            f"median={at_vals.median():+.4f}\""
        )
        logger.info(
            f"  CT: mean={ct_vals.mean():+.4f}\", "
            f"std={ct_vals.std():.4f}\", "
            f"median={ct_vals.median():+.4f}\""
        )
        logger.info(
            f"  Speed: median={speed_vals.median():.4f} deg/day, "
            f"min={speed_vals.min():.6f}, max={speed_vals.max():.4f}"
        )

    # --- Write output ---
    logger.info(f"Writing augmented results to {args.output}")
    import pyarrow as pa
    table = pa.Table.from_pandas(augmented, preserve_index=False)
    pq.write_table(table, args.output)
    logger.info(f"Done. {len(augmented)} rows written.")


if __name__ == "__main__":
    main()
