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
        default=Path("data/looo_sample"),
        help="Directory with mpc_observations.parquet and mpc_orbits.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/looo_results"),
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
        help="Number of parallel worker processes (default: 1).",
    )
    p.add_argument(
        "--sigma-model",
        choices=["veres2017", "const"],
        default="veres2017",
        help="How to fill missing rmsra/rmsdec values: "
             "'veres2017' uses per-(stn, catalog) sigma lookup from Veres et al. 2017 (default); "
             "'const' fills with a small constant (original behaviour, inflates chi2).",
    )
    p.add_argument(
        "--orbit-fitter",
        choices=["scipy", "findorb", "native"],
        default="findorb",
        help="Orbit fitter used for the LOOO hold-in fit (default: findorb). "
             "'scipy' uses adam_core's fit_least_squares; "
             "'findorb' uses adam_fo.FindOrbOrbitFitter (requires Find_Orb binary); "
             "'native' uses adam_core's NativeOrbitFitter if available.",
    )
    p.add_argument(
        "--fo-result-dir",
        type=str,
        default=None,
        help="Directory for FindOrb intermediate outputs (default: temp dir under output-dir). "
             "Only used when --orbit-fitter=findorb.",
    )
    p.add_argument(
        "--strict-fitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the requested orbit fitter is not importable (default: True). "
             "Use --no-strict-fitter to allow silent fallback to scipy DC.",
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
    p.add_argument(
        "--group-by",
        type=str,
        default=None,
        help=(
            "Comma-separated LOOO hold-out / aggregation keys "
            "(subset of stn,prog,band,astcat; default: stn,prog,band). "
            "Each unique tuple is held out as a unit per object (bead wl0). "
            "Overrides the --profile value when both are given."
        ),
    )
    p.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Named run profile (e.g. v2_full, v2_full_pre_2017, "
            "v2_full_post_2017) setting group-by + obstime window + output-dir "
            "suffix together (bead tcu). Explicit --group-by/--obstime-min/"
            "--obstime-max flags override the profile's individual values."
        ),
    )
    p.add_argument(
        "--obstime-min",
        type=str,
        default=None,
        help=(
            "ISO date (UTC), e.g. 2017-01-01. Keep only observations with "
            "obstime strictly after this bound, applied BEFORE the LOOO "
            "refits (bead tcu). Overrides the --profile value when both are "
            "given."
        ),
    )
    p.add_argument(
        "--obstime-max",
        type=str,
        default=None,
        help=(
            "ISO date (UTC), e.g. 2017-01-01. Keep only observations with "
            "obstime at or before this bound, applied BEFORE the LOOO "
            "refits (bead tcu). Overrides the --profile value when both are "
            "given."
        ),
    )
    return p.parse_args()


def get_orbit_fitter(name: str, fo_result_dir: str, strict: bool = True, propagator=None):
    """Build an OrbitFitter instance (or None for scipy DC fallback).

    Parameters
    ----------
    name : str
        Fitter name: "scipy", "findorb", or "native".
    fo_result_dir : str
        Directory for FindOrb intermediate outputs.
    strict : bool
        If True (default), abort with sys.exit(1) when the requested fitter
        is not importable. If False, fall back to scipy DC with a critical
        log message.
    propagator : Propagator, optional
        Propagator the fitter uses to evaluate hold-in chi2. Must be the same
        propagator used for the LOOO held-out evaluation; if omitted,
        FindOrbOrbitFitter falls back to 2-body, which inflates
        hold_in_reduced_chi2 by ~7 orders of magnitude on long arcs
        (smoke pilot v13, 2026-06-10).
    """
    if name == "scipy":
        return None
    if name == "findorb":
        try:
            from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
            return FindOrbOrbitFitter(fo_result_dir=fo_result_dir, propagator=propagator)
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


def get_propagator_class(name: str):
    if name == "twobody":
        try:
            from adam_orbit_det_eval.propagators import TwoBodyPropagator
            return TwoBodyPropagator
        except Exception as e:
            logger.error(f"Could not import TwoBodyPropagator: {e}")
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

    # --- Resolve run profile + per-option overrides (bead tcu) ---
    from adam_orbit_det_eval.looo.profiles import resolve_run_options

    explicit_group_by = (
        [k.strip() for k in args.group_by.split(",") if k.strip()]
        if args.group_by is not None
        else None
    )
    options = resolve_run_options(
        profile=args.profile,
        group_by=explicit_group_by,
        obstime_min=args.obstime_min,
        obstime_max=args.obstime_max,
    )
    logger.info(
        f"Run profile: {options.profile or '(ad-hoc)'} — "
        f"group_by={list(options.group_by)}, "
        f"obstime window=({options.obstime_min or '-inf'}, "
        f"{options.obstime_max or '+inf'}]"
    )

    # The profile's output-dir suffix is appended to the default timestamped
    # run id (e.g. 20260612T010203Z_v2_full_pre_2017), matching the
    # data/mpc_scale_results_<date>_<profile>/ convention from v2-scope.md.
    # An explicit --run-id is used verbatim.
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + options.output_dir_suffix
    )
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

    # --- Configure LOOO ---
    from adam_orbit_det_eval.looo import run_looo_pipeline
    from adam_orbit_det_eval.looo.core import LOOOConfig

    group_by = list(options.group_by)
    logger.info(f"LOOO group-by keys: {group_by}")

    config = LOOOConfig(
        min_obs_held_out=args.min_obs_held_out,
        min_obs_remaining=args.min_obs_remaining,
        min_arc_length_days=args.min_arc_length,
        max_held_out_fraction=args.max_held_out_fraction,
    )

    propagator_class = get_propagator_class(args.propagator)
    logger.info(f"Using propagator: {propagator_class.__name__}")

    # --- Configure orbit fitter ---
    fo_result_dir = args.fo_result_dir or str(output_dir / "findorb_work")
    orbit_fitter = get_orbit_fitter(
        args.orbit_fitter,
        fo_result_dir,
        strict=args.strict_fitter,
        propagator=propagator_class(),
    )
    fitter_name = type(orbit_fitter).__name__ if orbit_fitter is not None else "scipy_fit_least_squares"
    logger.info(f"Using orbit fitter: {fitter_name}")

    # --- Write run configuration for reproducibility ---
    import json
    run_config = {
        "run_id": run_id,
        "propagator": args.propagator,
        "orbit_fitter": args.orbit_fitter,
        "orbit_fitter_class": fitter_name,
        "sigma_model": args.sigma_model,
        "min_obs_remaining": args.min_obs_remaining,
        "min_arc_length_days": args.min_arc_length,
        "min_obs_held_out": args.min_obs_held_out,
        "max_held_out_fraction": args.max_held_out_fraction,
        # Resolved run profile (bead tcu): the effective values after
        # explicit-flag > profile > default precedence, for reproducibility.
        "profile": options.profile,
        "output_dir_suffix": options.output_dir_suffix,
        "group_by": group_by,
        "obstime_min": options.obstime_min,
        "obstime_max": options.obstime_max,
        "max_processes": args.max_processes,
        "input_obs": str(obs_path),
        "input_orbits": str(orbits_path),
        "n_input_objects": int(n_objects),
        "n_input_observations": len(mpc_observations),
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    logger.info(f"Run configuration written to {config_path}")

    output_path = output_dir / "looo_results.parquet"

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
        sigma_model=args.sigma_model,
        orbit_fitter=orbit_fitter,
        group_by=group_by,
        obstime_min=options.obstime_min,
        obstime_max=options.obstime_max,
    )

    logger.info(
        f"LOOO complete: {len(results)} held-out observation results "
        f"written to {output_path}"
    )


if __name__ == "__main__":
    main()
