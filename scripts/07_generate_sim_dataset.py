#!/usr/bin/env python3
"""
07_generate_sim_dataset.py
==========================
Generate a synthetic observation dataset from MPC templates.

For each object in the input sample, propagates the truth orbit to the exact
observation times of the real data and re-generates synthetic positions with:
  - Controlled Gaussian noise (per fake observatory)
  - Injected systematic biases (per fake observatory)
  - Fake station codes (AA00–AA08 by default)

The default observatory map (Phase 1 test matrix) can be overridden via a JSON
file passed to ``--observatory-map``.

Outputs (written to ``--output-dir/<run-id>/datasets/default/``):
  mpc_observations.parquet  — synthetic observations (same schema as real data)
  mpc_orbits.parquet         — truth orbits (pass-through from input)
  truth_biases.csv           — injected bias truth table (for evaluate step)
  sim_config.json            — full reproducibility record

Usage
-----
    python scripts/07_generate_sim_dataset.py \\
        --input-dir data/sim_sample \\
        --output-dir data/sim_products \\
        --run-id phase1_constant_bias \\
        --propagator twobody \\
        --noise-seed 42 \\
        --cache-dir data/sim_products/cache

Propagators
-----------
twobody  : adam_core 2-body propagator (fast, use during development).
assist   : ASSIST N-body propagator (accurate, use for publication).
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
        "--input-dir",
        type=Path,
        default=Path("data/sim_sample"),
        help="Directory with mpc_observations.parquet and mpc_orbits.parquet "
             "(default: data/sim_sample)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sim_products"),
        help="Base output directory (default: data/sim_products)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier.  Defaults to a UTC timestamp.",
    )
    p.add_argument(
        "--propagator",
        choices=["twobody", "assist"],
        default="twobody",
        help="Propagator backend (default: twobody for speed during dev)",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="RNG seed for noise draws (default: 42)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for truth ephemeris parquet cache "
             "(reused across different bias variants).  "
             "Defaults to <output-dir>/cache.",
    )
    p.add_argument(
        "--observatory-map",
        type=Path,
        default=None,
        help="Path to a JSON file defining the ObservatoryMap.  "
             "If not provided, uses the built-in Phase 1 default map.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-generate even if output already exists.",
    )
    p.add_argument(
        "--object-ids",
        nargs="*",
        default=None,
        help="Restrict to these object IDs (default: all objects in input).",
    )
    return p.parse_args()


def get_propagator_class(name: str):
    """Return the propagator class for the given name."""
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
                "ASSISTPropagator not found.  Install adam-assist: pip install adam-assist"
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown propagator: {name}")


def build_default_observatory_map():
    """
    Return the Phase 1 test matrix ObservatoryMap.

    Stations
    --------
    AA00  F51  None (clean reference)
    AA01  G96  ConstantBias(0.5, 0.0)
    AA02  F52  TimingBias(1.0 sec)
    AA03  703  MagnitudeDependentBias(0.05, 0.02)
    AA04  691  CatalogEpochBias(5.0 years)
    AA05  W84  SeasonalBias(0.3, 0.1)
    AA06  W68  StepChangeBias (split arc)   [was 568 (24 obs); T08 co-located with T09 (Hawaii)]
    AA07  T09  DCRBias
    AA08  V00  TrailingBias(0.1)

    Noise sigmas are from empirical Gaia-calibrated estimates:
      Modern wide-field surveys (F51, G96, F52, W84):  ~0.15"
      Older/smaller aperture stations (703, 691, 568, T09, V00): ~0.35"
    """
    from adam_orbit_det_eval.simulation import (
        FakeObservatory,
        ObservatoryMap,
        ConstantBias,
        TimingBias,
        MagnitudeDependentBias,
        CatalogEpochBias,
        SeasonalBias,
        StepChangeBias,
        DCRBias,
        TrailingBias,
    )

    # MJD for a mid-arc split (approximate 2023-01-01 = MJD 59945)
    STEP_MJD = 59945.0

    assignments = [
        FakeObservatory(
            fake_code="AA00",
            real_code="F51",
            noise_sigma_ra=0.15,
            noise_sigma_dec=0.15,
            biases=[],  # clean reference
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA01",
            real_code="G96",
            noise_sigma_ra=0.15,
            noise_sigma_dec=0.15,
            biases=[ConstantBias(delta_ra=0.5, delta_dec=0.0)],
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA02",
            real_code="F52",
            noise_sigma_ra=0.15,
            noise_sigma_dec=0.15,
            biases=[TimingBias(delta_t_sec=1.0)],
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA03",
            real_code="703",
            noise_sigma_ra=0.35,
            noise_sigma_dec=0.35,
            biases=[MagnitudeDependentBias(slope_ra=0.05, slope_dec=0.02)],
            astcat="Gaia2",
        ),
        FakeObservatory(
            fake_code="AA04",
            real_code="691",
            noise_sigma_ra=0.35,
            noise_sigma_dec=0.35,
            biases=[CatalogEpochBias(epoch_error_years=5.0)],
            astcat="UCAC4",
        ),
        FakeObservatory(
            fake_code="AA05",
            real_code="W84",
            noise_sigma_ra=0.20,
            noise_sigma_dec=0.20,
            biases=[SeasonalBias(amplitude_ra=0.3, amplitude_dec=0.1)],
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA06",
            real_code="W68",  # ATLAS Chile, Rio Hurtado (4.8k obs; T08 co-located with T09)
            noise_sigma_ra=0.25,
            noise_sigma_dec=0.25,
            biases=[
                StepChangeBias(
                    delta_ra_before=0.0,
                    delta_dec_before=0.0,
                    delta_ra_after=0.4,
                    delta_dec_after=-0.2,
                    change_mjd=STEP_MJD,
                )
            ],
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA07",
            real_code="T09",
            noise_sigma_ra=0.25,
            noise_sigma_dec=0.25,
            biases=[DCRBias(bandpass_nm=200.0, ref_wavelength_nm=550.0)],
            astcat="Gaia3E",
        ),
        FakeObservatory(
            fake_code="AA08",
            real_code="V00",
            noise_sigma_ra=0.35,
            noise_sigma_dec=0.35,
            biases=[TrailingBias(trailing_factor=0.1)],
            astcat="Gaia2",
        ),
    ]

    return ObservatoryMap(assignments)


def load_observatory_map_from_json(path: Path):
    """
    Load an ObservatoryMap from a JSON file.

    The JSON file must contain a list of fake observatory definitions.
    Bias models are specified by type name and params dict.

    Expected format::

        [
          {
            "fake_code": "AA00",
            "real_code": "F51",
            "noise_sigma_ra": 0.15,
            "noise_sigma_dec": 0.15,
            "astcat": "Gaia3E",
            "biases": [
              {"type": "ConstantBias", "params": {"delta_ra": 0.5, "delta_dec": 0.0}}
            ]
          },
          ...
        ]
    """
    import adam_orbit_det_eval.simulation.bias_models as bm_module
    from adam_orbit_det_eval.simulation import FakeObservatory, ObservatoryMap

    data = json.loads(path.read_text())
    assignments = []
    for entry in data:
        biases = []
        for b_spec in entry.get("biases", []):
            bias_cls = getattr(bm_module, b_spec["type"], None)
            if bias_cls is None:
                logger.warning(f"Unknown bias type {b_spec['type']!r}, skipping.")
                continue
            biases.append(bias_cls(**b_spec.get("params", {})))

        assignments.append(
            FakeObservatory(
                fake_code=entry["fake_code"],
                real_code=entry["real_code"],
                noise_sigma_ra=entry["noise_sigma_ra"],
                noise_sigma_dec=entry["noise_sigma_dec"],
                astcat=entry.get("astcat", "Gaia3E"),
                biases=biases,
            )
        )
    return ObservatoryMap(assignments)


def main():
    args = parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id / "datasets" / "default"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id}  →  {output_dir}")

    # --- Input files ---
    obs_path = args.input_dir / "mpc_observations.parquet"
    orbits_path = args.input_dir / "mpc_orbits.parquet"

    for p, label in [(obs_path, "observations"), (orbits_path, "orbits")]:
        if not p.exists():
            logger.error(f"{label.capitalize()} file not found: {p}")
            logger.error("Run 06_fetch_sim_sample.py first.")
            sys.exit(1)

    logger.info(f"Loading observations from {obs_path}")
    from mpcq.observations import MPCObservations
    mpc_observations = MPCObservations.from_parquet(obs_path)

    logger.info(f"Loading orbits from {orbits_path}")
    from mpcq.orbits import MPCOrbits
    mpc_orbits = MPCOrbits.from_parquet(orbits_path)

    import pyarrow.compute as pc
    all_provids = mpc_observations.requested_provid.unique().to_pylist()
    logger.info(
        f"Loaded {len(mpc_observations)} observations for {len(all_provids)} objects"
    )

    # Restrict to requested object IDs
    if args.object_ids:
        object_ids = args.object_ids
        logger.info(f"Restricting to {len(object_ids)} user-specified objects.")
    else:
        object_ids = all_provids
    logger.info(f"Processing {len(object_ids)} objects.")

    # --- Observatory map ---
    if args.observatory_map is not None:
        logger.info(f"Loading observatory map from {args.observatory_map}")
        obs_map = load_observatory_map_from_json(args.observatory_map)
    else:
        logger.info("Using built-in Phase 1 default observatory map.")
        obs_map = build_default_observatory_map()

    logger.info(
        f"Observatory map: {len(obs_map.assignments)} fake stations "
        f"← real codes: {obs_map.all_real_codes()}"
    )

    # --- Cache directory ---
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Truth ephemeris cache: {cache_dir}")

    # --- Propagator ---
    propagator_class = get_propagator_class(args.propagator)
    logger.info(f"Using propagator: {propagator_class.__name__}")

    # --- SimulationConfig ---
    from adam_orbit_det_eval.simulation import SimulationConfig, SimulationDataset
    from adam_orbit_det_eval.looo.core import LOOOConfig

    config = SimulationConfig(
        run_id=run_id,
        objects=object_ids,
        observatory_map=obs_map,
        propagator_class=propagator_class,
        noise_seed=args.noise_seed,
        looo_config=LOOOConfig(),
    )

    # --- Check existing outputs ---
    obs_out = output_dir / "mpc_observations.parquet"
    if obs_out.exists() and not args.force:
        logger.info(
            f"Output already exists at {obs_out}.  "
            "Skipping generation (use --force to regenerate)."
        )
        return

    # --- Generate ---
    dataset = SimulationDataset(config)
    out_path = dataset.generate(
        obs_template=mpc_observations,
        truth_orbits=mpc_orbits,
        output_dir=output_dir,
        cache_dir=cache_dir,
        force=args.force,
    )

    logger.info(f"Dataset written to {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
