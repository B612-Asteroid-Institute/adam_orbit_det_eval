"""
Isolation sensitivity study pipeline.

This module contains the station definitions, bias catalogue, and the
``run_scenario()`` entry point shared by the local CLI script
(``scripts/10_run_isolation_study.py``) and the cloud shard runner
(``scripts/12_run_looo_cloud_shard.py``).

The isolation study runs one scenario per (station, bias) pair: the target
station has a single bias injected; every other station is clean.  The LOOO
residuals at the held-out target station are then compared to the clean
baseline to assess whether the bias is detectable.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import List, Optional, Type

from adam_core.propagator.propagator import Propagator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Station definitions
# ---------------------------------------------------------------------------

PHASE1_STATIONS: List[dict] = [
    dict(fake_code="AA00", real_code="F51", noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA01", real_code="G96", noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA02", real_code="F52", noise_ra=0.15, noise_dec=0.15, astcat="Gaia3E"),
    dict(fake_code="AA03", real_code="703", noise_ra=0.35, noise_dec=0.35, astcat="Gaia2"),
    dict(fake_code="AA04", real_code="691", noise_ra=0.35, noise_dec=0.35, astcat="UCAC4"),
    dict(fake_code="AA05", real_code="W84", noise_ra=0.20, noise_dec=0.20, astcat="Gaia3E"),
    dict(fake_code="AA06", real_code="W68", noise_ra=0.25, noise_dec=0.25, astcat="Gaia3E"),
    dict(fake_code="AA07", real_code="T09", noise_ra=0.25, noise_dec=0.25, astcat="Gaia3E"),
    dict(fake_code="AA08", real_code="V00", noise_ra=0.35, noise_dec=0.35, astcat="Gaia2"),
]

BIAS_NAMES: List[str] = [
    "clean", "constant", "timing", "mag_dep", "epoch",
    "seasonal", "step", "dcr", "trailing",
]

_STEP_MJD = 59945.0  # 2023-01-01


# ---------------------------------------------------------------------------
# Bias catalogue
# ---------------------------------------------------------------------------

def get_bias_catalogue() -> dict:
    """Return the canonical bias parameter dict.  Imports are deferred so the
    module loads quickly without pulling in simulation deps at import time."""
    import adam_orbit_det_eval.simulation.bias_models as bm

    return {
        "clean":    [],
        "constant": [bm.ConstantBias(delta_ra=0.5, delta_dec=0.0)],
        "timing":   [bm.TimingBias(delta_t_sec=1.0)],
        "mag_dep":  [bm.MagnitudeDependentBias(slope_ra=0.05, slope_dec=0.02)],
        "epoch":    [bm.CatalogEpochBias(epoch_error_years=5.0)],
        "seasonal": [bm.SeasonalBias(amplitude_ra=0.3, amplitude_dec=0.1)],
        "step":     [bm.StepChangeBias(
            delta_ra_before=0.0, delta_dec_before=0.0,
            delta_ra_after=0.4, delta_dec_after=-0.2,
            change_mjd=_STEP_MJD,
        )],
        "dcr":      [bm.DCRBias(bandpass_nm=200.0, ref_wavelength_nm=550.0)],
        "trailing": [bm.TrailingBias(trailing_factor=0.1)],
    }


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def scenario_id(fake_code: str, bias_name: str) -> str:
    return f"{fake_code}_{bias_name}"


def list_scenarios(
    stations: Optional[List[str]] = None,
    biases: Optional[List[str]] = None,
) -> List[tuple]:
    """Return a list of ``(fake_code, bias_name)`` pairs to run.

    Parameters
    ----------
    stations : list of str, optional
        Restrict to these fake station codes.  Defaults to all Phase 1 stations.
    biases : list of str, optional
        Restrict to these bias names.  Defaults to all ``BIAS_NAMES``.
    """
    all_fake = [s["fake_code"] for s in PHASE1_STATIONS]
    target_stations = stations if stations else all_fake
    target_biases = biases if biases else BIAS_NAMES
    return [
        (fc, bn)
        for fc in all_fake
        for bn in BIAS_NAMES
        if fc in target_stations and bn in target_biases
    ]


def build_isolation_map(fake_code: str, bias_name: str):
    """Return an ObservatoryMap where *fake_code* has *bias_name* applied
    and all other stations are clean."""
    from adam_orbit_det_eval.simulation import FakeObservatory, ObservatoryMap

    catalogue = get_bias_catalogue()
    assignments = [
        FakeObservatory(
            fake_code=stn["fake_code"],
            real_code=stn["real_code"],
            noise_sigma_ra=stn["noise_ra"],
            noise_sigma_dec=stn["noise_dec"],
            biases=catalogue[bias_name] if stn["fake_code"] == fake_code and bias_name != "clean" else [],
            astcat=stn["astcat"],
        )
        for stn in PHASE1_STATIONS
    ]
    return ObservatoryMap(assignments)


# ---------------------------------------------------------------------------
# Per-scenario pipeline
# ---------------------------------------------------------------------------

def run_scenario(
    fake_code: str,
    bias_name: str,
    obs_template,
    truth_orbits,
    object_ids: List[str],
    propagator_class: Type[Propagator],
    output_base: Path,
    cache_dir: Path,
    max_processes: int = 6,
    threshold_arcsec: float = 0.1,
    force: bool = False,
    orbit_fitter=None,
    sigma_model: str = "veres2017",
) -> Optional[List[dict]]:
    """Run the full pipeline for one isolation scenario.

    Steps: dataset generation → LOOO cross-validation → observatory statistics
    → recovery evaluation.

    Parameters
    ----------
    fake_code : str
        Target station fake code (e.g. ``"AA01"``).
    bias_name : str
        Bias type to inject at the target station (e.g. ``"constant"``).
    obs_template : MPCObservations
        Real MPC observations used as a timing/pointing template.
    truth_orbits : MPCOrbits
        Reference orbits corresponding to ``obs_template``.
    object_ids : list of str
        Object IDs to process.
    propagator_class : Type[Propagator]
        Propagator class used for ephemeris generation and DC.
    output_base : Path
        Root directory for all scenario outputs.
    cache_dir : Path
        Directory for truth ephemeris cache (shared across scenarios).
    max_processes : int
        Number of parallel worker processes for LOOO.
    threshold_arcsec : float
        Detection threshold in arcseconds for recovery evaluation.
    force : bool
        Re-run even if outputs already exist.
    orbit_fitter : OrbitFitter, optional
        OD backend for the DC step.  Defaults to ``fit_least_squares``.
    sigma_model : str
        Observation uncertainty model (``"veres2017"`` or ``"constant"``).

    Returns
    -------
    list of dict or None
        Recovery row dicts (one per station in the scenario), or ``None`` on
        failure.
    """
    from adam_orbit_det_eval.looo.core import LOOOConfig
    from adam_orbit_det_eval.simulation import SimulationConfig, SimulationDataset

    sname = scenario_id(fake_code, bias_name)
    scenario_dir = output_base / sname
    dataset_dir  = scenario_dir / "datasets"   / "default"
    looo_dir     = scenario_dir / "looo_results" / "default"
    analysis_dir = scenario_dir / "analysis"   / "default"
    recovery_dir = scenario_dir / "recovery"   / "default"

    obs_out       = dataset_dir  / "mpc_observations.parquet"
    looo_out      = looo_dir     / "looo_results.parquet"
    obs_stats_path = analysis_dir / "observatory_stats.parquet"

    # --- 1. Dataset generation ---
    if obs_out.exists() and not force:
        logger.info(f"  [{sname}] Dataset exists, skipping generation.")
    else:
        obs_map = build_isolation_map(fake_code, bias_name)
        config = SimulationConfig(
            run_id=sname,
            objects=object_ids,
            observatory_map=obs_map,
            propagator_class=propagator_class,
            noise_seed=42,
            looo_config=LOOOConfig(),
        )
        SimulationDataset(config).generate(
            obs_template=obs_template,
            truth_orbits=truth_orbits,
            output_dir=dataset_dir,
            cache_dir=cache_dir,
            force=force,
        )

    # --- 2. LOOO pipeline ---
    if looo_out.exists() and not force:
        logger.info(f"  [{sname}] LOOO results exist, skipping pipeline.")
    else:
        looo_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_looo(
                dataset_dir=dataset_dir,
                looo_out=looo_out,
                propagator_class=propagator_class,
                max_processes=max_processes,
                force=force,
                orbit_fitter=orbit_fitter,
                sigma_model=sigma_model,
            )
        except Exception as exc:
            logger.error(f"  [{sname}] LOOO failed: {exc}", exc_info=True)
            return None

    # --- 3. Observatory statistics ---
    if obs_stats_path.exists() and not force:
        logger.info(f"  [{sname}] Analysis exists, skipping.")
    else:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_analysis(looo_out=looo_out, analysis_dir=analysis_dir)
        except Exception as exc:
            logger.error(f"  [{sname}] Analysis failed: {exc}", exc_info=True)
            return None

    # --- 4. Recovery evaluation ---
    truth_csv = dataset_dir / "truth_biases.csv"
    if not obs_stats_path.exists() or not truth_csv.exists():
        logger.warning(f"  [{sname}] Missing inputs for evaluation, skipping.")
        return None

    try:
        from adam_orbit_det_eval.simulation import evaluate_recovery, print_recovery_summary

        recovery_df = evaluate_recovery(
            observatory_stats_parquet=obs_stats_path,
            truth_biases_csv=truth_csv,
            threshold_arcsec=threshold_arcsec,
        )
        recovery_df["scenario"]       = sname
        recovery_df["target_station"] = fake_code
        recovery_df["applied_bias"]   = bias_name
        recovery_df["is_target"]      = recovery_df["fake_code"] == fake_code

        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_df.to_csv(recovery_dir / "recovery_report.csv", index=False)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_recovery_summary(recovery_df)
        (recovery_dir / "recovery_summary.txt").write_text(buf.getvalue())

        return recovery_df.to_dict("records")

    except Exception as exc:
        logger.error(f"  [{sname}] Evaluation failed: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_looo(
    dataset_dir: Path,
    looo_out: Path,
    propagator_class: Type[Propagator],
    max_processes: int,
    force: bool = False,
    orbit_fitter=None,
    sigma_model: str = "veres2017",
) -> None:
    import shutil

    from mpcq.observations import MPCObservations
    from mpcq.orbits import MPCOrbits

    from adam_orbit_det_eval.looo import run_looo_pipeline
    from adam_orbit_det_eval.looo.core import LOOOConfig

    if force:
        looo_dir = looo_out.parent
        for stale in ("_obs_input.parquet", "_orbits_input.parquet"):
            p = looo_dir / stale
            if p.exists():
                p.unlink()
                logger.info(f"    Removed stale {stale}")
        ckpt_dir = looo_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info("    Removed stale checkpoints/")

    obs    = MPCObservations.from_parquet(dataset_dir / "mpc_observations.parquet")
    orbits = MPCOrbits.from_parquet(dataset_dir / "mpc_orbits.parquet")
    config = LOOOConfig(
        min_obs_held_out=1,
        min_obs_remaining=6,
        min_arc_length_days=7.0,
        max_held_out_fraction=0.8,
    )
    run_looo_pipeline(
        mpc_observations=obs,
        mpc_orbits=orbits,
        propagator_class=propagator_class,
        output_path=looo_out,
        config=config,
        object_ids=None,
        max_processes=max_processes,
        sigma_model=sigma_model,
        orbit_fitter=orbit_fitter,
    )


def _run_analysis(looo_out: Path, analysis_dir: Path) -> None:
    import pyarrow.parquet as pq

    from adam_orbit_det_eval.looo.analysis import compute_observatory_stats
    from adam_orbit_det_eval.looo.core import LOOOResult

    results   = LOOOResult(pq.read_table(looo_out))
    obs_stats = compute_observatory_stats(results, min_obs_per_stn=5)
    obs_stats.to_parquet(analysis_dir / "observatory_stats.parquet")
