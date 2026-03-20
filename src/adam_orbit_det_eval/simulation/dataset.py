"""
SimulationConfig and SimulationDataset.

``SimulationConfig`` stores all parameters needed to reproduce a simulation
run.  ``SimulationDataset`` orchestrates multi-object synthetic observation
generation and writes the standard output files used by downstream scripts.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..looo.core import LOOOConfig
from .observatory_map import ObservatoryMap
from .synthetic_obs import generate_synthetic_observations

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """
    All parameters required to reproduce a simulation scenario.

    Parameters
    ----------
    run_id : str
        Human-readable scenario identifier (e.g. ``"phase1_constant_bias"``).
    objects : list of str
        Provid/object identifiers to generate synthetic observations for.
    observatory_map : ObservatoryMap
        Fake-to-real observatory assignments with noise and bias parameters.
    propagator_class : type
        Propagator class (not instance) to use for ephemeris generation.
    noise_seed : int
        Global RNG seed.  Each object gets ``noise_seed + i`` to avoid
        correlation between objects while remaining fully reproducible.
    looo_config : LOOOConfig
        Configuration forwarded to the LOOO pipeline.
    """

    run_id: str
    objects: List[str]
    observatory_map: ObservatoryMap
    propagator_class: Type
    noise_seed: int = 42
    looo_config: LOOOConfig = field(default_factory=LOOOConfig)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this config."""
        return {
            "run_id": self.run_id,
            "objects": self.objects,
            "noise_seed": self.noise_seed,
            "propagator_class": (
                f"{self.propagator_class.__module__}."
                f"{self.propagator_class.__qualname__}"
            ),
            "observatory_map": [
                {
                    "fake_code": a.fake_code,
                    "real_code": a.real_code,
                    "noise_sigma_ra": a.noise_sigma_ra,
                    "noise_sigma_dec": a.noise_sigma_dec,
                    "astcat": a.astcat,
                    "bias_type": a.bias_type,
                    "bias_params": a.bias_params_dict(),
                }
                for a in self.observatory_map.assignments
            ],
            "looo_config": {
                "min_obs_held_out": self.looo_config.min_obs_held_out,
                "min_obs_remaining": self.looo_config.min_obs_remaining,
                "min_arc_length_days": self.looo_config.min_arc_length_days,
                "max_held_out_fraction": self.looo_config.max_held_out_fraction,
                "ls_kwargs": self.looo_config.ls_kwargs,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        """
        Reconstruct a ``SimulationConfig`` from its ``to_dict()`` representation.

        Note: ``observatory_map`` bias models are not fully deserialised
        (they are stored as type/params dicts for human reference only).
        This method reconstructs the config metadata but the bias model
        objects will be no-ops unless you reconstruct them externally.
        """
        import importlib

        # Reconstruct propagator class
        fqn = d.get("propagator_class", "")
        try:
            module_name, class_name = fqn.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            propagator_class = getattr(mod, class_name)
        except Exception:
            logger.warning(
                f"Could not import propagator class {fqn!r}; using placeholder."
            )
            from adam_core.propagator.propagator import Propagator
            propagator_class = Propagator

        # Reconstruct ObservatoryMap with zero-bias fake observatories
        from .observatory_map import FakeObservatory, ObservatoryMap

        assignments = []
        for entry in d.get("observatory_map", []):
            assignments.append(
                FakeObservatory(
                    fake_code=entry["fake_code"],
                    real_code=entry["real_code"],
                    noise_sigma_ra=entry["noise_sigma_ra"],
                    noise_sigma_dec=entry["noise_sigma_dec"],
                    astcat=entry.get("astcat", "Gaia3E"),
                    biases=[],  # bias models not deserialised
                )
            )

        # Reconstruct LOOOConfig
        lc = d.get("looo_config", {})
        looo_config = LOOOConfig(
            min_obs_held_out=lc.get("min_obs_held_out", 1),
            min_obs_remaining=lc.get("min_obs_remaining", 6),
            min_arc_length_days=lc.get("min_arc_length_days", 7.0),
            max_held_out_fraction=lc.get("max_held_out_fraction", 0.8),
            ls_kwargs=lc.get("ls_kwargs", {}),
        )

        return cls(
            run_id=d["run_id"],
            objects=d.get("objects", []),
            observatory_map=ObservatoryMap(assignments),
            propagator_class=propagator_class,
            noise_seed=d.get("noise_seed", 42),
            looo_config=looo_config,
        )


class SimulationDataset:
    """
    Generate a complete synthetic observation dataset for one scenario.

    Handles per-object ephemeris caching, noise/bias injection, and writing
    the standard set of output files consumed by the LOOO pipeline.

    Parameters
    ----------
    config : SimulationConfig
        Scenario configuration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def generate(
        self,
        obs_template: "MPCObservations",  # noqa: F821
        truth_orbits: "MPCOrbits",  # noqa: F821
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """
        Generate synthetic observations for all objects in the config.

        Writes the following files to *output_dir*:
          - ``mpc_observations.parquet``   — synthetic observations
          - ``mpc_orbits.parquet``          — truth orbits (same schema as input)
          - ``truth_biases.csv``            — injected bias truth table
          - ``sim_config.json``             — full config for reproducibility

        Parameters
        ----------
        obs_template : MPCObservations
            Real MPC observations used as timing/cadence template.
        truth_orbits : MPCOrbits
            MPC orbits; each object's nominal orbit is taken as ground truth.
        output_dir : Path
            Directory to write outputs.  Created if necessary.
        cache_dir : Path, optional
            Directory for truth ephemeris parquet cache.  Shared across
            bias variants for the same objects.
        force : bool
            If False (default) and ``mpc_observations.parquet`` already exists
            in *output_dir*, skip generation and return immediately.

        Returns
        -------
        Path
            Absolute path to *output_dir*.
        """
        import quivr as qv
        from mpcq.observations import MPCObservations
        from mpcq.orbits import MPCOrbits

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        obs_out_path = output_dir / "mpc_observations.parquet"
        if obs_out_path.exists() and not force:
            logger.info(
                f"Output already exists at {obs_out_path}; skipping. "
                "Use force=True to regenerate."
            )
            return output_dir

        if cache_dir is not None:
            cache_dir = Path(cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate propagator
        propagator = self.config.propagator_class()

        synthetic_chunks: list = []
        truth_orbit_chunks: list = []

        for i, object_id in enumerate(self.config.objects):
            logger.info(
                f"[{i + 1}/{len(self.config.objects)}] Generating synthetic obs for {object_id!r}"
            )

            # Filter template observations for this object
            obj_obs = _filter_by_provid(obs_template, object_id)
            obj_orbit = _filter_orbit_by_provid(truth_orbits, object_id)

            if len(obj_obs) == 0:
                logger.warning(f"{object_id}: no template observations found, skipping.")
                continue
            if len(obj_orbit) == 0:
                logger.warning(f"{object_id}: no reference orbit found, skipping.")
                continue

            try:
                truth_orbit = obj_orbit.orbits()
            except Exception as exc:
                logger.warning(f"{object_id}: could not get Orbits from MCPOrbits: {exc}")
                continue

            # Per-object seed for reproducibility without inter-object correlation
            obj_seed = self.config.noise_seed + i

            try:
                syn_obs = generate_synthetic_observations(
                    truth_orbit=truth_orbit,
                    obs_template=obj_obs,
                    observatory_map=self.config.observatory_map,
                    propagator=propagator,
                    noise_seed=obj_seed,
                    cache_dir=cache_dir,
                    object_id=object_id,
                )
            except Exception as exc:
                logger.error(
                    f"{object_id}: generate_synthetic_observations failed: {exc}",
                    exc_info=True,
                )
                continue

            if len(syn_obs) == 0:
                logger.warning(f"{object_id}: no synthetic observations generated, skipping.")
                continue

            synthetic_chunks.append(syn_obs)
            truth_orbit_chunks.append(obj_orbit)

        if not synthetic_chunks:
            logger.error("No synthetic observations were generated for any object.")
            return output_dir

        # Merge and write observations
        all_syn_obs = qv.concatenate(synthetic_chunks)
        if all_syn_obs.fragmented():
            all_syn_obs = qv.defragment(all_syn_obs)
        all_syn_obs.to_parquet(obs_out_path)
        logger.info(
            f"Wrote {len(all_syn_obs)} synthetic observations → {obs_out_path}"
        )

        # Write truth orbits (same objects, pass-through)
        all_orbits = qv.concatenate(truth_orbit_chunks)
        orbits_path = output_dir / "mpc_orbits.parquet"
        all_orbits.to_parquet(orbits_path)
        logger.info(f"Wrote truth orbits → {orbits_path}")

        # Write truth biases CSV
        truth_df = self.config.observatory_map.truth_table()
        truth_csv = output_dir / "truth_biases.csv"
        truth_df.to_csv(truth_csv, index=False)
        logger.info(f"Wrote truth biases → {truth_csv}")

        # Write sim config JSON
        config_path = output_dir / "sim_config.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2))
        logger.info(f"Wrote sim config → {config_path}")

        return output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_by_provid(
    obs: "MPCObservations", object_id: str  # noqa: F821
) -> "MPCObservations":  # noqa: F821
    mask = pc.equal(obs.requested_provid, object_id)
    return obs.apply_mask(mask)


def _filter_orbit_by_provid(
    orbits: "MPCOrbits", object_id: str  # noqa: F821
) -> "MPCOrbits":  # noqa: F821
    mask = pc.equal(orbits.requested_provid, object_id)
    return orbits.apply_mask(mask)
