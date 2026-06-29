#!/usr/bin/env python3
"""
run_wide_variant_sweep_v2.py
============================

v2-catalog re-run of the wide-population variant sweep — bead
``od_experiments_setup-3bx``. A faithful clone of
``run_wide_variant_sweep.py`` (7en) that swaps the **v1** high-confidence bias
catalog for the **v2** ``bias_catalog_v2_full_no_prog`` catalog and adds one
new principled lever, ``v2_empirical_covar``.

What changes vs 7en
-------------------
1. **Catalog**: loads the v2 per-station rollup
   (``band IS NULL AND astcat IS NULL AND program_code IS NULL``, 1,123
   stations) via :func:`adam_orbit_det_eval.utils.load_v2_bias_catalog`. v1
   published only ~520 high-confidence stations; v2 ``_full_`` carries every
   station and exposes ``high_confidence`` as a per-station column.
2. **EFCC18 semantics**: v2 bias values were measured POST-EFCC18, so every
   ``v2_*`` variant applies EFCC18 catalog-debiasing to the observations
   *first*, then applies its σ-modification. ``no_bias``/``veres_only`` stay
   raw; ``efcc18_only`` is EFCC18 prep with no σ-modification. The pure-filter
   variants (``drop_*``) and ``uniform_sigma`` do NOT consume v2 bias values,
   so they stay EFCC18-free (matching their 7en behaviour) — this keeps them
   directly comparable to 7en.
3. **drop_non_HC_stations** now filters on the v2 ``high_confidence`` *column*
   (not table membership, which under v2 would keep every station).
4. **AT/CT** floor reads ``bias_at``/``bias_ct`` from the same v2 rollup (no
   separate AT/CT parquet).
5. **Dropped variant**: 7en's ``v1_sigma_floor+efcc18`` is degenerate under v2
   (EFCC18 is already baked into every ``v2_*`` variant — it would be a
   byte-identical alias of ``v2_sigma_floor``). It is dropped rather than
   carried as a misleading duplicate. Net: 17 v2 variants (16 carried over +
   ``v2_empirical_covar``), matching 7en's 17-variant count.
6. **Ray**: runs 8 actors, each owning a propagator + FindOrb fitter + BQ
   client and processing one object (all variants) per task.

Headline question
-----------------
Does ``v2_empirical_covar`` — which inflates the obs covariance with the
station's *measured* 2×2 residual covariance instead of the *assumed*
``outer(b, b)`` — beat ``no_bias`` on the cph discrepant subset AND keep
short-arc control regressions at or below the ``no_bias`` baseline? That is
the thing ``v1_covar_inflation`` failed (29.2 median Δr/σ on cph discrepants
but refuted on the si3 hold-out; 2 control regressions in 7en).

Outputs (default ``data/wide_variant_sweep_v2/``)
-------------------------------------------------
    variant_comparison.parquet    one row per (object, variant); mirrors 7en
    REPORT.md                     narrative incl. v1-vs-v2 side-by-side

Population: re-uses ``data/wide_variant_sweep/population_manifest.parquet``
(149 objects) — NOT re-derived.

Usage
-----
    pdm run python scripts/run_wide_variant_sweep_v2.py \\
        [--output-dir data/wide_variant_sweep_v2] [--n-workers 8] \\
        [--max-objects N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv
import ray

from adam_assist import ASSISTPropagator
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.efcc18 import (
    compute_efcc18_corrections,
    load_efcc18_biases,
    n_observations_covered,
)
from adam_orbit_det_eval.jpl_compare import (
    compute_orbit_gap,
    fetch_jpl_orbit,
    propagate_to_epoch,
)
from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    load_v2_bias_catalog,
    mpc_to_od_observations,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("wide_variant_sweep_v2")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
DEFAULT_POP_MANIFEST = Path("data/wide_variant_sweep/population_manifest.parquet")
DEFAULT_V1_VC = Path("data/wide_variant_sweep/variant_comparison.parquet")

MAX_OBS_PER_OBJECT = 2000
HIGH_RMS_ARCSEC = 0.5
CHI2_OUTLIER_RATIO = 3.0
CHI2_PATHOLOGICAL_LOW = 0.3
CHI2_PATHOLOGICAL_HIGH = 3.0
RESID_COV_N_THRESHOLD = 30  # empirical_covar: min residual sample to trust the 2×2

# v1 → v2 variant-name map for the side-by-side report. v1's
# ``v1_sigma_floor+efcc18`` maps to ``v2_sigma_floor`` (EFCC18 baked in under
# v2); ``v2_empirical_covar`` has no v1 analog.
V1_TO_V2_NAME: Dict[str, str] = {
    "no_bias": "no_bias",
    "veres_only": "veres_only",
    "v1_sigma_floor": "v2_sigma_floor",
    "v1_subtract": "v2_subtract",
    "efcc18_only": "efcc18_only",
    "uniform_sigma": "uniform_sigma",
    "drop_non_HC_stations": "drop_non_HC_stations",
    "v1_RSS_additive": "v2_RSS_additive",
    "v1_performance_weighted": "v2_performance_weighted",
    "drop_bias_significant": "drop_bias_significant",
    "v1_bayes_shrinkage": "v2_bayes_shrinkage",
    "v1_at_ct_floor": "v2_at_ct_floor",
    "v1_chi2_outlier_reject": "v2_chi2_outlier_reject",
    "veres_v1_max_floor": "veres_v2_max_floor",
    "drop_high_rms_stations": "drop_high_rms_stations",
    "v1_covar_inflation": "v2_covar_inflation",
    "v1_sigma_floor+efcc18": "v2_sigma_floor",
}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    sigma_model: str = "veres2017"
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_efcc18: bool = False
    pre_filter: Optional[str] = None
    post_processing: Optional[str] = None
    uniform_sigma_arcsec: float = 0.5
    use_station_chi2: bool = False
    use_station_sem: bool = False
    use_atct: bool = False
    use_resid_covar: bool = False
    is_legacy: bool = False


VARIANTS: List[VariantConfig] = [
    # --- anchors (4) ---
    VariantConfig("no_bias"),
    VariantConfig("veres_only"),
    VariantConfig(
        "v2_sigma_floor",
        bias_application="sigma_floor",
        use_bias_table=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_subtract",  # LEGACY / REFERENCE ANCHOR ONLY — never a candidate
        bias_application="subtract",
        use_bias_table=True,
        use_efcc18=True,
        is_legacy=True,
    ),
    # --- EFCC18 (1; v2 drops the stacked sigma_floor+efcc18) ---
    VariantConfig("efcc18_only", use_efcc18=True),
    # --- diagnostic / filters (no v2 bias values → no EFCC18, matches 7en) ---
    VariantConfig("uniform_sigma", sigma_model="uniform", uniform_sigma_arcsec=0.5),
    VariantConfig("drop_non_HC_stations", pre_filter="hc_stations_only"),
    VariantConfig("drop_bias_significant", pre_filter="drop_bias_significant"),
    VariantConfig("drop_high_rms_stations", pre_filter="drop_high_rms"),
    # --- v2 σ-modification levers (all EFCC18-prepped) ---
    VariantConfig(
        "v2_RSS_additive",
        bias_application="rss_additive",
        use_bias_table=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_performance_weighted",
        bias_application="performance_weighted",
        use_bias_table=True,
        use_station_chi2=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_bayes_shrinkage",
        bias_application="bayes_shrinkage",
        use_bias_table=True,
        use_station_sem=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_at_ct_floor",
        bias_application="at_ct_floor",
        use_atct=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_chi2_outlier_reject",
        post_processing="chi2_outlier_reject",
        use_station_chi2=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "veres_v2_max_floor",
        bias_application="veres_v1_max_floor",  # mode is catalog-agnostic
        use_bias_table=True,
        use_efcc18=True,
    ),
    VariantConfig(
        "v2_covar_inflation",
        bias_application="covar_inflation",
        use_bias_table=True,
        use_efcc18=True,
    ),
    # --- NEW lever (the headline of this bead) ---
    VariantConfig(
        "v2_empirical_covar",
        bias_application="empirical_covar",
        use_resid_covar=True,
        use_efcc18=True,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/wide_variant_sweep_v2"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--population-manifest", type=Path, default=DEFAULT_POP_MANIFEST)
    p.add_argument("--v1-variant-comparison", type=Path, default=DEFAULT_V1_VC)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_wide_v2_runs")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--per-object-timeout", type=float, default=1800.0,
                   help="Hard wall-clock cap (s) for ALL variants of one object. "
                        "If exceeded (e.g. a wedged FindOrb subprocess), the actor "
                        "is killed + replaced and the object is recorded as a "
                        "timeout failure so it can never stall the whole sweep.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None, help="Smoke cap")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# v2 catalog → per-mode sub-dicts
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class V2Catalogs:
    """All per-station sub-dicts the variants need, derived from the v2 rollup."""
    bias_table: Dict[str, Tuple[float, float]]
    station_chi2: Dict[str, float]
    bias_significant: Dict[str, bool]
    station_sem: Dict[str, Tuple[float, float]]
    station_rms: Dict[str, Tuple[float, float]]
    high_confidence: Dict[str, bool]
    atct_bias_table: Dict[str, Tuple[float, float]]
    station_resid_covar: Dict[str, Tuple[float, float, float, float]]
    n_stations: int


def build_v2_catalogs(path: str) -> V2Catalogs:
    cat = load_v2_bias_catalog(path, rollup_only=True)
    bias_table: Dict[str, Tuple[float, float]] = {}
    station_chi2: Dict[str, float] = {}
    bias_significant: Dict[str, bool] = {}
    station_sem: Dict[str, Tuple[float, float]] = {}
    station_rms: Dict[str, Tuple[float, float]] = {}
    high_confidence: Dict[str, bool] = {}
    atct_bias_table: Dict[str, Tuple[float, float]] = {}
    station_resid_covar: Dict[str, Tuple[float, float, float, float]] = {}
    for code, rec in cat.items():
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"]):
            bias_table[code] = (rec["bias_ra"], rec["bias_dec"])
        if np.isfinite(rec["chi2_per_obs"]):
            station_chi2[code] = rec["chi2_per_obs"]
        bias_significant[code] = bool(rec["bias_significant"])
        if np.isfinite(rec["sem_ra"]) and np.isfinite(rec["sem_dec"]):
            station_sem[code] = (rec["sem_ra"], rec["sem_dec"])
        if np.isfinite(rec["rms_ra"]) and np.isfinite(rec["rms_dec"]):
            station_rms[code] = (rec["rms_ra"], rec["rms_dec"])
        high_confidence[code] = bool(rec["high_confidence"])
        if np.isfinite(rec["bias_at"]) and np.isfinite(rec["bias_ct"]):
            atct_bias_table[code] = (abs(rec["bias_at"]), abs(rec["bias_ct"]))
        station_resid_covar[code] = (
            rec["resid_var_ra"],
            rec["resid_var_dec"],
            rec["resid_cov_ra_dec"],
            rec["resid_cov_n"],
        )
    return V2Catalogs(
        bias_table=bias_table,
        station_chi2=station_chi2,
        bias_significant=bias_significant,
        station_sem=station_sem,
        station_rms=station_rms,
        high_confidence=high_confidence,
        atct_bias_table=atct_bias_table,
        station_resid_covar=station_resid_covar,
        n_stations=len(cat),
    )


# ────────────────────────────────────────────────────────────────────────────
# Observation prep (mirrors 7en)
# ────────────────────────────────────────────────────────────────────────────


def _dedupe_close_obs(obs: MPCObservations) -> MPCObservations:
    stns = obs.stn.to_pylist()
    times_sec = obs.obstime.mjd().to_numpy(zero_copy_only=False) * 86400.0
    order = np.lexsort((times_sec, np.asarray(stns, dtype=object)))
    keep = np.zeros(len(obs), dtype=bool)
    last_kept: Dict[str, float] = {}
    for idx in order:
        s = stns[idx]
        t = float(times_sec[idx])
        prev = last_kept.get(s)
        if prev is None or abs(t - prev) > 1.5:
            keep[idx] = True
            last_kept[s] = t
    if keep.sum() == len(obs):
        return obs
    deduped = obs.apply_mask(pa.array(keep))
    if deduped.fragmented():
        deduped = qv.concatenate([deduped])
    return deduped


@dataclass
class PreparedObservations:
    obs: MPCObservations
    n_obs_input: int
    n_stations_input: int
    arc_days: float
    n_obs_in_bias_table: int
    n_obs_efcc18_covered: int
    efcc18_corrections: np.ndarray
    atct_unit_vectors: np.ndarray


def _compute_atct_unit_vectors(
    obs: MPCObservations, jpl_orbit: Orbits, propagator: ASSISTPropagator
) -> np.ndarray:
    n = len(obs)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    if jpl_orbit is None or len(jpl_orbit) == 0:
        return out
    try:
        observers = Observers.from_codes(codes=obs.stn, times=obs.obstime)
        ephemeris = propagator.generate_ephemeris(
            orbits=jpl_orbit, observers=observers, max_processes=1
        )
    except Exception as e:
        logger.warning("AT/CT ephemeris failed: %s", e)
        return out
    vlon = ephemeris.coordinates.vlon.to_numpy(zero_copy_only=False)
    vlat = ephemeris.coordinates.vlat.to_numpy(zero_copy_only=False)
    dec_deg = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))
    v_dec = vlat
    speed = np.sqrt(v_ra_cosdec**2 + v_dec**2)
    valid = speed >= 1e-6
    out[valid, 0] = v_ra_cosdec[valid] / speed[valid]
    out[valid, 1] = v_dec[valid] / speed[valid]
    return out


def prepare_observations(
    provid: str,
    client: BigQueryMPCClient,
    spacebased: set,
    bias_table: Dict[str, Tuple[float, float]],
    efcc18_bias_table,
    jpl_orbit: Optional[Orbits],
    propagator: ASSISTPropagator,
) -> Optional[PreparedObservations]:
    raw = client.query_observations([provid])
    if raw is None or len(raw) == 0:
        return None
    stns_raw = raw.stn.to_pylist()
    ground_mask = np.array(
        [(s is not None) and (s not in spacebased) for s in stns_raw], dtype=bool
    )
    if not ground_mask.any():
        return None
    ground = raw.apply_mask(pa.array(ground_mask))
    if ground.fragmented():
        ground = qv.concatenate([ground])
    deduped = _dedupe_close_obs(ground)
    if len(deduped) > MAX_OBS_PER_OBJECT:
        times = deduped.obstime.mjd().to_numpy(zero_copy_only=False)
        order = np.argsort(times)
        keep_idx = order[-MAX_OBS_PER_OBJECT:]
        keep_mask = np.zeros(len(deduped), dtype=bool)
        keep_mask[keep_idx] = True
        deduped = deduped.apply_mask(pa.array(keep_mask))
        if deduped.fragmented():
            deduped = qv.concatenate([deduped])

    stns = deduped.stn.to_pylist()
    times_mjd = deduped.obstime.mjd().to_numpy(zero_copy_only=False)
    astcats = deduped.astcat.to_pylist()
    ra_deg = deduped.ra.to_numpy(zero_copy_only=False)
    dec_deg = deduped.dec.to_numpy(zero_copy_only=False)
    jd_tdb = times_mjd + 2400000.5
    efcc_corrections = compute_efcc18_corrections(
        ra_deg, dec_deg, astcats, jd_tdb, bias_table=efcc18_bias_table
    )
    n_efcc_covered = n_observations_covered(astcats)
    if jpl_orbit is not None and len(jpl_orbit) > 0:
        atct_uv = _compute_atct_unit_vectors(deduped, jpl_orbit, propagator)
    else:
        atct_uv = np.full((len(deduped), 2), np.nan, dtype=np.float64)

    return PreparedObservations(
        obs=deduped,
        n_obs_input=len(deduped),
        n_stations_input=len(set(stns)),
        arc_days=float(times_mjd.max() - times_mjd.min()),
        n_obs_in_bias_table=sum(1 for s in stns if s in bias_table),
        n_obs_efcc18_covered=n_efcc_covered,
        efcc18_corrections=efcc_corrections,
        atct_unit_vectors=atct_uv,
    )


# ────────────────────────────────────────────────────────────────────────────
# Pre-filters + 2-pass outlier reject
# ────────────────────────────────────────────────────────────────────────────


def apply_pre_filter(
    obs: MPCObservations,
    pre_filter: Optional[str],
    *,
    high_confidence: Dict[str, bool],
    bias_significant: Dict[str, bool],
    station_rms: Dict[str, Tuple[float, float]],
    rms_threshold: float = HIGH_RMS_ARCSEC,
) -> MPCObservations:
    """Filter obs BEFORE mpc_to_od_observations.

    ``hc_stations_only`` uses the v2 ``high_confidence`` column (NOT table
    membership, which under v2 keeps every measured station). The other two
    mirror 7en.
    """
    if pre_filter is None:
        return obs
    stns = obs.stn.to_pylist()
    if pre_filter == "hc_stations_only":
        mask_list = [bool(high_confidence.get(s, False)) for s in stns]
    elif pre_filter == "drop_bias_significant":
        mask_list = [not bias_significant.get(s, False) for s in stns]
    elif pre_filter == "drop_high_rms":
        def _keep(s: str) -> bool:
            r = station_rms.get(s)
            if r is None:
                return True
            return r[0] <= rms_threshold and r[1] <= rms_threshold
        mask_list = [_keep(s) for s in stns]
    else:
        raise ValueError(f"Unknown pre_filter={pre_filter!r}")
    return obs.apply_mask(pa.array(mask_list, type=pa.bool_()))


def chi2_outlier_reject_mask(
    od_obs,
    fitted_orbit: Orbits,
    propagator: ASSISTPropagator,
    station_chi2: Dict[str, float],
    stns_list: List[str],
    ratio_threshold: float = CHI2_OUTLIER_RATIO,
) -> np.ndarray:
    observers = Observers.from_codes(
        codes=od_obs.coordinates.origin.code, times=od_obs.coordinates.time
    )
    eph = propagator.generate_ephemeris(
        orbits=fitted_orbit, observers=observers, max_processes=1
    )
    obs_lon = od_obs.coordinates.lon.to_numpy(zero_copy_only=False)
    obs_lat = od_obs.coordinates.lat.to_numpy(zero_copy_only=False)
    pred_lon = eph.coordinates.lon.to_numpy(zero_copy_only=False)
    pred_lat = eph.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_dec = np.cos(np.deg2rad(obs_lat))
    res_ra_arcsec = (obs_lon - pred_lon) * cos_dec * 3600.0
    res_ra_arcsec = np.where(
        res_ra_arcsec > 180.0 * 3600.0, res_ra_arcsec - 360.0 * 3600.0,
        np.where(res_ra_arcsec < -180.0 * 3600.0, res_ra_arcsec + 360.0 * 3600.0,
                 res_ra_arcsec),
    )
    res_dec_arcsec = (obs_lat - pred_lat) * 3600.0
    sigmas = od_obs.coordinates.covariance.sigmas
    sigma_ra_cosdec_arcsec = sigmas[:, 1] * cos_dec * 3600.0
    sigma_dec_arcsec = sigmas[:, 2] * 3600.0
    factor = np.ones(len(stns_list))
    for i, code in enumerate(stns_list):
        c = station_chi2.get(code)
        if c is None or not np.isfinite(c):
            continue
        factor[i] = float(np.sqrt(max(float(c), 1.0)))
    sigma_ra_pred = sigma_ra_cosdec_arcsec * factor
    sigma_dec_pred = sigma_dec_arcsec * factor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ra = np.where(sigma_ra_pred > 0, np.abs(res_ra_arcsec) / sigma_ra_pred, 0.0)
        ratio_dec = np.where(sigma_dec_pred > 0, np.abs(res_dec_arcsec) / sigma_dec_pred, 0.0)
    return (ratio_ra <= ratio_threshold) & (ratio_dec <= ratio_threshold)


# ────────────────────────────────────────────────────────────────────────────
# Per (object, variant)
# ────────────────────────────────────────────────────────────────────────────


def _jpl_pos_sigma_au(jpl_orbit: Orbits) -> float:
    try:
        cov = jpl_orbit.coordinates.covariance.to_matrix()
    except Exception:
        return float("nan")
    if cov is None or cov.size == 0:
        return float("nan")
    diag = np.array([cov[0, 0, 0], cov[0, 1, 1], cov[0, 2, 2]], dtype=np.float64)
    if not np.all(np.isfinite(diag)) or np.any(diag < 0):
        return float("nan")
    return float(np.sqrt(diag.sum()))


@dataclass
class VariantResult:
    object_id: str
    designation: str
    stratum: str
    variant: str
    is_legacy: bool
    in_cph_set: bool
    is_discrepant_in_cph: bool
    is_control_in_cph: bool
    converged: bool = False
    failure_reason: str = ""
    n_obs_input: int = 0
    n_obs_surviving: int = 0
    n_stations_input: int = 0
    n_stations_surviving: int = 0
    arc_days: float = float("nan")
    n_obs_in_bias_table: int = 0
    n_obs_efcc18_covered: int = 0
    hold_in_reduced_chi2: float = float("nan")
    epoch_mjd_tdb: float = float("nan")
    delta_a_au: float = float("nan")
    delta_e: float = float("nan")
    delta_i_deg: float = float("nan")
    delta_raan_deg: float = float("nan")
    delta_ap_deg: float = float("nan")
    delta_M_deg: float = float("nan")
    delta_q_au: float = float("nan")
    cartesian_dr_au: float = float("nan")
    cartesian_dv_au_per_day: float = float("nan")
    dr_over_sigma: float = float("nan")
    jpl_sigma_units_a: float = float("nan")
    jpl_sigma_units_e: float = float("nan")
    jpl_sigma_units_i: float = float("nan")
    chi2_pathological: bool = False


def _build_od_obs(
    obs: MPCObservations,
    variant: VariantConfig,
    *,
    cats: V2Catalogs,
    efcc18_corrections: Optional[np.ndarray],
    atct_unit_vectors: Optional[np.ndarray],
):
    kw: Dict[str, object] = {
        "prevent_nans": True,
        "sigma_model": variant.sigma_model,
        "uniform_sigma_arcsec": variant.uniform_sigma_arcsec,
        "bias_application": variant.bias_application,
    }
    if variant.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if variant.use_efcc18 and efcc18_corrections is not None:
        kw["catalog_debias_arcsec"] = efcc18_corrections
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if variant.use_station_sem:
        kw["station_sem_arcsec"] = cats.station_sem
    if variant.use_atct:
        kw["atct_bias_table"] = cats.atct_bias_table
        kw["atct_unit_vectors"] = atct_unit_vectors
    if variant.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


def run_single(
    rec: VariantResult,
    variant: VariantConfig,
    obs: MPCObservations,
    efcc18_corrections: Optional[np.ndarray],
    atct_unit_vectors: Optional[np.ndarray],
    jpl_orbit: Orbits,
    jpl_pos_sigma_au: float,
    *,
    cats: V2Catalogs,
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
    efcc18_bias_table,
) -> VariantResult:
    try:
        variant_obs = apply_pre_filter(
            obs, variant.pre_filter,
            high_confidence=cats.high_confidence,
            bias_significant=cats.bias_significant,
            station_rms=cats.station_rms,
        )
        if variant_obs.fragmented():
            variant_obs = qv.concatenate([variant_obs])
        n_kept = len(variant_obs)
        if n_kept == 0:
            rec.failure_reason = "pre_filter dropped all observations"
            return rec
        rec.n_obs_surviving = n_kept
        rec.n_stations_surviving = len(set(variant_obs.stn.to_pylist()))

        # Re-align EFCC18/AT-CT slices if a pre_filter dropped rows.
        if variant.pre_filter is not None:
            if variant.use_efcc18:
                eff_efcc = compute_efcc18_corrections(
                    variant_obs.ra.to_numpy(zero_copy_only=False),
                    variant_obs.dec.to_numpy(zero_copy_only=False),
                    variant_obs.astcat.to_pylist(),
                    variant_obs.obstime.jd().to_numpy(zero_copy_only=False),
                    bias_table=efcc18_bias_table,
                )
            else:
                eff_efcc = None
            eff_atct_uv = None  # at_ct_floor has no pre_filter
        else:
            eff_efcc = efcc18_corrections if variant.use_efcc18 else None
            eff_atct_uv = atct_unit_vectors if variant.use_atct else None

        od_obs = _build_od_obs(
            variant_obs, variant,
            cats=cats, efcc18_corrections=eff_efcc, atct_unit_vectors=eff_atct_uv,
        )
        if od_obs is None or len(od_obs) == 0:
            rec.failure_reason = "mpc_to_od_observations returned None/empty"
            return rec

        fitted, _ = fitter.initial_fit(rec.object_id, od_obs)
        if len(fitted) == 0:
            rec.failure_reason = "FindOrb returned empty FittedOrbits"
            return rec

        # chi2_outlier_reject 2-pass — 2nd pass keeps EFCC18 (v2 semantics).
        if variant.post_processing == "chi2_outlier_reject":
            fitted_orbit_first = Orbits.from_kwargs(
                orbit_id=fitted.orbit_id, object_id=fitted.object_id,
                coordinates=fitted.coordinates,
            )
            keep_mask = chi2_outlier_reject_mask(
                od_obs, fitted_orbit_first, propagator,
                cats.station_chi2, variant_obs.stn.to_pylist(),
            )
            n_kept2 = int(keep_mask.sum())
            if n_kept2 < 6:
                rec.failure_reason = f"chi2_outlier_reject left only {n_kept2} obs"
                return rec
            variant_obs2 = variant_obs.apply_mask(pa.array(keep_mask))
            if variant_obs2.fragmented():
                variant_obs2 = qv.concatenate([variant_obs2])
            eff_efcc2 = (
                eff_efcc[keep_mask] if (variant.use_efcc18 and eff_efcc is not None)
                else None
            )
            # Second pass: veres σ, no bias table, EFCC18 retained if set.
            od_obs2 = mpc_to_od_observations(
                variant_obs2,
                prevent_nans=True,
                sigma_model="veres2017",
                bias_application="sigma_floor",
                catalog_debias_arcsec=eff_efcc2,
            )
            if od_obs2 is None or len(od_obs2) == 0:
                rec.failure_reason = "2nd-pass mpc_to_od returned empty"
                return rec
            fitted, _ = fitter.initial_fit(rec.object_id, od_obs2)
            if len(fitted) == 0:
                rec.failure_reason = "2nd-pass FindOrb returned empty"
                return rec
            rec.n_obs_surviving = len(variant_obs2)
            rec.n_stations_surviving = len(set(variant_obs2.stn.to_pylist()))

        rec.hold_in_reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
        rec.chi2_pathological = bool(
            (not np.isfinite(rec.hold_in_reduced_chi2))
            or rec.hold_in_reduced_chi2 < CHI2_PATHOLOGICAL_LOW
            or rec.hold_in_reduced_chi2 > CHI2_PATHOLOGICAL_HIGH
        )
        fitted_orbit = Orbits.from_kwargs(
            orbit_id=fitted.orbit_id, object_id=fitted.object_id,
            coordinates=fitted.coordinates,
        )
        propagated = propagate_to_epoch(
            fitted_orbit, jpl_orbit.coordinates.time, propagator
        )
        gap = compute_orbit_gap(propagated, jpl_orbit, variant.variant_id)
        rec.converged = True
        rec.epoch_mjd_tdb = gap.epoch_mjd_tdb
        rec.delta_a_au = gap.delta_a_au
        rec.delta_e = gap.delta_e
        rec.delta_i_deg = gap.delta_i_deg
        rec.delta_raan_deg = gap.delta_raan_deg
        rec.delta_ap_deg = gap.delta_ap_deg
        rec.delta_M_deg = gap.delta_M_deg
        rec.delta_q_au = gap.delta_q_au
        rec.cartesian_dr_au = gap.cartesian_dr_au
        rec.cartesian_dv_au_per_day = gap.cartesian_dv_au_per_day
        rec.jpl_sigma_units_a = gap.delta_a_in_sigma
        rec.jpl_sigma_units_e = gap.delta_e_in_sigma
        rec.jpl_sigma_units_i = gap.delta_i_in_sigma
        if np.isfinite(jpl_pos_sigma_au) and jpl_pos_sigma_au > 0:
            rec.dr_over_sigma = gap.cartesian_dr_au / jpl_pos_sigma_au
    except Exception as e:
        rec.failure_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    return rec


# ────────────────────────────────────────────────────────────────────────────
# Ray actor — one actor owns a propagator/fitter/client; processes one obj/task
# ────────────────────────────────────────────────────────────────────────────


@ray.remote
class ObjectWorker:
    def __init__(self, idx: int, args_dict: dict, cats: V2Catalogs, efcc18_bias_table):
        self.idx = idx
        logging.basicConfig(level=logging.WARNING)
        self.cats = cats
        self.spacebased = set(get_spacebased_stns())
        # EFCC18 table is loaded once in main() and shipped via the object
        # store (Ray auto-dereferences the ObjectRef into the real array here),
        # so every actor shares it — no per-actor file load, no cache race.
        self.efcc18_bias_table = efcc18_bias_table
        self.client = BigQueryMPCClient(
            dataset_id=args_dict["dataset_id"],
            views_dataset_id=args_dict["views_dataset_id"],
            project=args_dict["project"],
        )
        self.propagator = ASSISTPropagator()
        fo_dir = f"{args_dict['fo_result_dir']}/actor_{idx}"
        Path(fo_dir).mkdir(parents=True, exist_ok=True)
        self.fitter = FindOrbOrbitFitter(
            fo_result_dir=fo_dir, clean_up_fo_dir=True, propagator=self.propagator
        )

    def process(self, row: dict) -> List[dict]:
        provid = row["provid"]
        stratum = row.get("stratum", "")
        in_cph = bool(row.get("in_cph_set", False))
        is_disc = bool(row.get("is_discrepant_in_cph", False))
        is_ctrl = bool(row.get("is_control_in_cph", False))

        def _mk(reason: str = "", **kw) -> VariantResult:
            return VariantResult(
                object_id=provid, designation=provid, stratum=stratum,
                variant="", is_legacy=False, in_cph_set=in_cph,
                is_discrepant_in_cph=is_disc, is_control_in_cph=is_ctrl,
                failure_reason=reason, **kw,
            )

        try:
            jpl_orbit = fetch_jpl_orbit(provid)
        except Exception as e:
            jpl_orbit = None
            logger.warning("%s: JPL fetch failed: %s", provid, e)
        jpl_sigma = (
            _jpl_pos_sigma_au(jpl_orbit)
            if jpl_orbit is not None and len(jpl_orbit) > 0 else float("nan")
        )
        try:
            prepared = prepare_observations(
                provid, self.client, self.spacebased, self.cats.bias_table,
                self.efcc18_bias_table, jpl_orbit, self.propagator,
            )
        except Exception as e:
            logger.warning("%s: prepare_observations raised: %s", provid, e)
            prepared = None

        if prepared is None or jpl_orbit is None or len(jpl_orbit) == 0:
            reason = "observation prep failed" if prepared is None else "JPL orbit fetch failed"
            out = []
            for v in VARIANTS:
                r = _mk(reason)
                r.variant = v.variant_id
                r.is_legacy = v.is_legacy
                out.append(asdict(r))
            return out

        out: List[dict] = []
        for v in VARIANTS:
            rec = VariantResult(
                object_id=provid, designation=provid, stratum=stratum,
                variant=v.variant_id, is_legacy=v.is_legacy, in_cph_set=in_cph,
                is_discrepant_in_cph=is_disc, is_control_in_cph=is_ctrl,
                n_obs_input=prepared.n_obs_input,
                n_stations_input=prepared.n_stations_input,
                arc_days=prepared.arc_days,
                n_obs_in_bias_table=prepared.n_obs_in_bias_table,
                n_obs_efcc18_covered=prepared.n_obs_efcc18_covered,
            )
            rec = run_single(
                rec, v, prepared.obs, prepared.efcc18_corrections,
                prepared.atct_unit_vectors, jpl_orbit, jpl_sigma,
                cats=self.cats, fitter=self.fitter, propagator=self.propagator,
                efcc18_bias_table=self.efcc18_bias_table,
            )
            out.append(asdict(rec))
        return out


# ────────────────────────────────────────────────────────────────────────────
# Persistence + REPORT
# ────────────────────────────────────────────────────────────────────────────


def write_parquet(records: List[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    path = output_dir / "variant_comparison.parquet"
    df.to_parquet(path, index=False)
    return path


def _failure_rows(row: dict, reason: str) -> List[dict]:
    """One failure VariantResult per variant for an object that never returned
    (orchestration-level: actor crash or per-object timeout)."""
    out: List[dict] = []
    for v in VARIANTS:
        out.append(asdict(VariantResult(
            object_id=row["provid"], designation=row["provid"],
            stratum=row.get("stratum", ""), variant=v.variant_id,
            is_legacy=v.is_legacy, in_cph_set=bool(row.get("in_cph_set", False)),
            is_discrepant_in_cph=bool(row.get("is_discrepant_in_cph", False)),
            is_control_in_cph=bool(row.get("is_control_in_cph", False)),
            failure_reason=reason,
        )))
    return out


def _med(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if not s.empty else float("nan")


def _regression_count(ctrl_df: pd.DataFrame, variant: str, baseline_variant: str) -> int:
    base_by = ctrl_df.loc[ctrl_df.variant == baseline_variant].set_index("object_id")["dr_over_sigma"]
    sub = ctrl_df.loc[ctrl_df.variant == variant].set_index("object_id")
    n = 0
    for oid, r in sub.iterrows():
        b = base_by.get(oid)
        c = r.dr_over_sigma
        if b is not None and np.isfinite(b) and np.isfinite(c) and b > 0 and c > 2.0 * b:
            n += 1
    return n


def write_report(
    records: List[dict], population_df: pd.DataFrame, output_dir: Path,
    v2_catalog_path: Path, v1_vc_path: Path, n_v2_stations: int,
) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n")
        return

    md: List[str] = []
    md.append("# Wide-population v2-catalog sweep — 3bx")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-3bx` · Branch: `kk/od-bias-experiments`")
    md.append("")
    md.append(f"v2 catalog: `{v2_catalog_path}`")
    md.append(f"v2 stations in per-station rollup: {n_v2_stations}")
    md.append(
        "Note: every `v2_*` variant applies EFCC18 preprocessing before its "
        "σ-modification (v2 bias values are measured post-EFCC18). `no_bias`/"
        "`veres_only` are raw; `efcc18_only` is EFCC18 prep only. The `drop_*` "
        "filters + `uniform_sigma` consume no v2 bias values, so they stay "
        "EFCC18-free (matching 7en). Unlike v1's ~520-station high-confidence "
        f"table, the v2 rollup carries all {n_v2_stations} measured stations; "
        "`drop_non_HC_stations` filters on the `high_confidence` column."
    )
    md.append("")

    # Population
    md.append("## Population")
    md.append("")
    md.append(f"Total objects: {len(population_df)} (re-used 7en manifest)")
    for s, c in sorted(population_df.groupby("stratum").size().to_dict().items()):
        md.append(f"- {s}: {c}")
    md.append("")

    # Convergence
    md.append("## Convergence")
    md.append("")
    md.append("| variant | converged / total | χ²-pathological |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        sub = df[df.variant == v.variant_id]
        md.append(f"| `{v.variant_id}` | {int(sub.converged.sum())}/{len(sub)} | "
                  f"{int(sub.chi2_pathological.sum())} |")
    md.append("")

    # Per-stratum ranking
    md.append("## Per-stratum variant ranking (median Δr/σ, lower = better)")
    md.append("")
    md.append("Restricted to converged, χ²_in ∈ [0.3, 3] (non-pathological) fits.")
    md.append("")
    for stratum in sorted(df.stratum.dropna().unique()):
        sdf = df[(df.stratum == stratum) & df.converged & ~df.chi2_pathological]
        if sdf.empty:
            continue
        md.append(f"### {stratum} (n_objects = {sdf.object_id.nunique()})")
        md.append("")
        md.append("| variant | median Δr/σ | median Δr (AU) | n | n_path |")
        md.append("|---|---|---|---|---|")
        for v in VARIANTS:
            vsub = df[(df.stratum == stratum) & (df.variant == v.variant_id)]
            vclean = vsub[vsub.converged & ~vsub.chi2_pathological]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {_med(vclean.dr_over_sigma):.3f} | "
                      f"{_med(vclean.cartesian_dr_au):.2e} | {len(vsub)} | "
                      f"{int(vsub.chi2_pathological.sum())} |")
        md.append("")

    # Headline — cph discrepant subset
    md.append("## Headline — cph discrepant subset")
    md.append("")
    disc = df[df.is_discrepant_in_cph & df.converged & ~df.chi2_pathological]
    base_nb = _med(disc.loc[disc.variant == "no_bias", "dr_over_sigma"]) if not disc.empty else float("nan")
    base_sub = _med(disc.loc[disc.variant == "v2_subtract", "dr_over_sigma"]) if not disc.empty else float("nan")
    if not disc.empty:
        md.append(f"cph's 14 discrepant NEOs (n_evaluated = {disc.object_id.nunique()}).")
        md.append("")
        md.append("| variant | median Δr/σ | vs no_bias | vs v2_subtract |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            m = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"])
            r_nb = base_nb / m if np.isfinite(base_nb) and np.isfinite(m) and m > 0 else float("nan")
            r_sub = base_sub / m if np.isfinite(base_sub) and np.isfinite(m) and m > 0 else float("nan")
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {m:.3f} | {r_nb:.2f}× | {r_sub:.2f}× |")
        md.append("")
        md.append("`vs no_bias` > 1 ⇒ moved closer to JPL than no-bias. "
                  "`vs v2_subtract` > 1 ⇒ beat the legacy/reference anchor.")
        md.append("")

    # Control regression
    md.append("## Control regression — controls in cph + short-arc stratum")
    md.append("")
    md.append("Per variant: number of controls whose Δr/σ exceeds **2× the "
              "per-object baseline**, computed against BOTH `veres_only` (7en's "
              "metric, for joinability) and `no_bias` (this bead's headline "
              "criterion).")
    md.append("")
    ctrl_df = df[(df.is_control_in_cph | (df.stratum == "short_arc_mod_obs")) & df.converged]
    n_ctrl = ctrl_df.object_id.nunique()
    md.append(f"n_controls = {n_ctrl}")
    md.append("")
    md.append("| variant | reg vs veres_only | reg vs no_bias |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        if v.variant_id in ("no_bias", "veres_only"):
            continue
        rv = _regression_count(ctrl_df, v.variant_id, "veres_only")
        rn = _regression_count(ctrl_df, v.variant_id, "no_bias")
        tag = " (LEGACY)" if v.is_legacy else ""
        md.append(f"| `{v.variant_id}`{tag} | {rv} | {rn} |")
    md.append("")

    # MBA-only
    md.append("## Main-belt-only analysis (cleanest test)")
    md.append("")
    mba = df[(df.stratum == "main_belt") & df.converged & ~df.chi2_pathological]
    if not mba.empty:
        md.append("| variant | median Δr/σ | median Δr (AU) | n |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            vsub = mba[mba.variant == v.variant_id]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {_med(vsub.dr_over_sigma):.3f} | "
                      f"{_med(vsub.cartesian_dr_au):.2e} | {len(vsub)} |")
    else:
        md.append("- No converged non-pathological MBA fits.")
    md.append("")

    # v1-vs-v2 side-by-side
    md.append("## v1-vs-v2 side-by-side (cph discrepant subset)")
    md.append("")
    md.append("For each variant present in both sweeps: v1 (7en, pre-EFCC18 bias "
              "values) vs v2 (this bead, EFCC18-baked-in bias values) median "
              "Δr/σ on the SAME cph discrepant subset. Shows whether the v2 "
              "catalog flips any ranking.")
    md.append("")
    if v1_vc_path.exists() and not disc.empty:
        v1 = pd.read_parquet(v1_vc_path)
        v1d = v1[v1.is_discrepant_in_cph & v1.converged & ~v1.chi2_pathological]
        md.append("| v1 variant | v2 variant | v1 median Δr/σ | v2 median Δr/σ | Δ(v2−v1) |")
        md.append("|---|---|---|---|---|")
        for v1name, v2name in V1_TO_V2_NAME.items():
            m1 = _med(v1d.loc[v1d.variant == v1name, "dr_over_sigma"])
            m2 = _med(disc.loc[disc.variant == v2name, "dr_over_sigma"])
            delta = m2 - m1 if np.isfinite(m1) and np.isfinite(m2) else float("nan")
            note = " *(v2 alias)*" if v1name == "v1_sigma_floor+efcc18" else ""
            md.append(f"| `{v1name}`{note} | `{v2name}` | {m1:.3f} | {m2:.3f} | {delta:+.3f} |")
        m2_emp = _med(disc.loc[disc.variant == "v2_empirical_covar", "dr_over_sigma"])
        md.append(f"| *(none)* | `v2_empirical_covar` | — | {m2_emp:.3f} | NEW |")
        md.append("")
    else:
        md.append("(v1 variant_comparison not found — skipping side-by-side.)")
        md.append("")

    # HEADLINE verdict
    md.append("## HEADLINE — does `v2_empirical_covar` work?")
    md.append("")
    if not disc.empty and n_ctrl > 0:
        m_emp = _med(disc.loc[disc.variant == "v2_empirical_covar", "dr_over_sigma"])
        reg_emp_nb = _regression_count(ctrl_df, "v2_empirical_covar", "no_bias")
        reg_nb_nb = _regression_count(ctrl_df, "no_bias", "veres_only")  # = 0 by construction
        beats_nb = np.isfinite(m_emp) and np.isfinite(base_nb) and m_emp < base_nb
        keeps_controls = reg_emp_nb <= 0  # at or below no_bias (no_bias regresses 0 vs itself)
        verdict = "**YES**" if (beats_nb and keeps_controls) else "**NO**"
        md.append(
            f"- `v2_empirical_covar` median Δr/σ on cph discrepants = "
            f"**{m_emp:.3f}** vs `no_bias` **{base_nb:.3f}** "
            f"({'beats no_bias' if beats_nb else 'does NOT beat no_bias'})."
        )
        md.append(f"- Control regressions vs no_bias = **{reg_emp_nb}** "
                  f"(target: ≤ 0, i.e. at/below the no_bias baseline).")
        md.append(f"- **Verdict: {verdict}** — beats no_bias on discrepants AND "
                  f"keeps controls at/below no_bias.")
        md.append("")
        md.append("Reference — `v1_covar_inflation` on 7en: median Δr/σ ≈ 29.2 on "
                  "cph discrepants, 2 control regressions; refuted on the si3 "
                  "hold-out (the assumed `outer(b, b)` over-coupled PS1/PS2/H21). "
                  "`v2_empirical_covar` replaces that assumed structure with the "
                  "measured residual covariance.")
        md.append("")
        # v2_covar_inflation for direct comparison
        m_cov = _med(disc.loc[disc.variant == "v2_covar_inflation", "dr_over_sigma"])
        reg_cov_nb = _regression_count(ctrl_df, "v2_covar_inflation", "no_bias")
        md.append(f"- For comparison, `v2_covar_inflation` (assumed outer(b,b), "
                  f"v2 values): median Δr/σ = {m_cov:.3f}, control regressions "
                  f"vs no_bias = {reg_cov_nb}.")
        md.append("")

    (output_dir / "REPORT.md").write_text("\n".join(md))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)

    if not args.population_manifest.exists():
        logger.error("Population manifest not found: %s", args.population_manifest)
        return 1
    population_df = pd.read_parquet(args.population_manifest)
    logger.info("Re-using 7en population manifest: %d objects", len(population_df))
    if args.max_objects is not None:
        population_df = population_df.head(args.max_objects)
        logger.info("Smoke cap: %d objects", len(population_df))

    cats = build_v2_catalogs(str(args.v2_catalog_path))
    logger.info(
        "v2 catalog: %d rollup stations (bias_table=%d, chi2=%d, sem=%d, "
        "rms=%d, HC=%d, atct=%d, resid_covar=%d)",
        cats.n_stations, len(cats.bias_table), len(cats.station_chi2),
        len(cats.station_sem), len(cats.station_rms),
        sum(cats.high_confidence.values()), len(cats.atct_bias_table),
        len(cats.station_resid_covar),
    )

    # Resume
    existing: set = set()
    records: List[dict] = []
    out_parquet = args.output_dir / "variant_comparison.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        records = prev.to_dict(orient="records")
        existing = set(prev.object_id.unique().tolist())
        logger.info("Resume: %d objects already in output", len(existing))

    targets = [r for r in population_df.to_dict(orient="records")
               if r["provid"] not in existing]
    logger.info("Objects to process: %d", len(targets))

    if not targets:
        write_parquet(records, args.output_dir)
        write_report(records, population_df, args.output_dir,
                     args.v2_catalog_path, args.v1_variant_comparison, cats.n_stations)
        return 0

    efcc18_bias_table = load_efcc18_biases()  # load once; warms the .npy cache
    logger.info("Loaded EFCC18 catalog (%s)", efcc18_bias_table.shape)

    ray.init(num_cpus=args.n_workers, ignore_reinit_error=True,
             include_dashboard=False, logging_level=logging.WARNING)
    cats_ref = ray.put(cats)
    efcc18_ref = ray.put(efcc18_bias_table)
    args_dict = {
        "project": args.project, "dataset_id": args.dataset_id,
        "views_dataset_id": args.views_dataset_id,
        "fo_result_dir": args.fo_result_dir,
    }
    workers = [ObjectWorker.remote(i, args_dict, cats_ref, efcc18_ref)
               for i in range(args.n_workers)]

    # Manual scheduler with a per-object timeout. ActorPool.map_unordered blocks
    # forever if one object wedges (a FindOrb subprocess that never returns took
    # down the first 3bx run for ~1.5 days on 4 Vesta). Here we ray.wait with a
    # short poll, and any object exceeding --per-object-timeout has its actor
    # killed + replaced and is recorded as a timeout failure.
    start = time.time()
    n_done = len(existing)
    n_total = len(population_df)
    timeout_s = args.per_object_timeout
    queue = list(targets)
    inflight: Dict = {}  # ObjectRef -> [worker_idx, row, submit_time]

    def _submit(widx: int) -> None:
        if queue:
            row = queue.pop(0)
            ref = workers[widx].process.remote(row)
            inflight[ref] = [widx, row, time.time()]

    def _record(obj_rows: List[dict]) -> None:
        nonlocal n_done
        records.extend(obj_rows)
        n_done += 1
        if obj_rows:
            oid = obj_rows[0]["object_id"]
            nconv = sum(1 for r in obj_rows if r["converged"])
            logger.info("[%d/%d] %s done (%d/%d converged, cum %.1f min)",
                        n_done, n_total, oid, nconv, len(obj_rows),
                        (time.time() - start) / 60.0)
        if n_done % 5 == 0:
            write_parquet(records, args.output_dir)
            try:
                write_report(records, population_df, args.output_dir,
                             args.v2_catalog_path, args.v1_variant_comparison,
                             cats.n_stations)
            except Exception as e:
                logger.warning("Incremental REPORT failed: %s", e)

    for widx in range(len(workers)):
        _submit(widx)

    while inflight:
        done, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=20.0)
        now = time.time()
        if done:
            ref = done[0]
            widx, row, _ = inflight.pop(ref)
            try:
                obj_rows = ray.get(ref)
            except Exception as e:
                logger.warning("%s: actor task errored: %s", row["provid"], e)
                obj_rows = _failure_rows(row, f"actor error: {type(e).__name__}: {e}")
            _record(obj_rows)
            _submit(widx)
        for ref in list(inflight.keys()):
            widx, row, submit_t = inflight[ref]
            if now - submit_t > timeout_s:
                logger.warning("[timeout] %s exceeded %.0fs on actor %d — killing "
                               "+ replacing", row["provid"], timeout_s, widx)
                try:
                    ray.kill(workers[widx])
                except Exception:
                    pass
                inflight.pop(ref)
                _record(_failure_rows(row, f"per-object timeout >{int(timeout_s)}s"))
                workers[widx] = ObjectWorker.remote(widx, args_dict, cats_ref, efcc18_ref)
                _submit(widx)

    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows → %s", len(records), parq)
    write_report(records, population_df, args.output_dir,
                 args.v2_catalog_path, args.v1_variant_comparison, cats.n_stations)
    logger.info("Wrote REPORT.md · total runtime %.1f min", (time.time() - start) / 60.0)
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
