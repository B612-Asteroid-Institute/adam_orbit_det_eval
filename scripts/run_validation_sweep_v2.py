#!/usr/bin/env python3
"""
run_validation_sweep_v2.py
==========================

v2-catalog re-run of the si3 hold-out validation — bead
``od_experiments_setup-3bx``. A faithful clone of ``run_validation_sweep.py``
(si3) that swaps the v1 high-confidence catalog for the v2
``bias_catalog_v2_full_no_prog`` catalog and adds the new ``v2_empirical_covar``
lever.

What changes vs si3
-------------------
1. **Catalog**: v2 per-station rollup via
   :func:`adam_orbit_det_eval.utils.load_v2_bias_catalog`.
2. **EFCC18**: every ``v2_*`` variant applies EFCC18 catalog-debiasing first
   (v2 bias values are post-EFCC18). ``no_bias`` stays raw. si3 had no EFCC18
   at all — this is the key behavioural change.
3. **New variant**: ``v2_empirical_covar`` (measured 2×2 residual covariance).
4. **Ray**: 8 actors, one object per task.

Cohort: re-uses ``data/validation_sweep/cohort_manifest.parquet`` (the SAME
100 fresh NEOs si3 used — zero overlap with cph/7en), so this is a true v1-vs-v2
comparison on identical objects.

Variants (5)
------------
    no_bias                  baseline (raw, Veres σ, no EFCC18)
    v2_subtract              LEGACY / REFERENCE anchor (EFCC18 + subtract)
    v2_performance_weighted  EFCC18 + σ×√χ²_per_obs
    v2_covar_inflation       EFCC18 + assumed outer(b, b)
    v2_empirical_covar       EFCC18 + MEASURED 2×2 residual covariance (NEW)

Diagnostic: 2026 DX / 2025 UA3 — the two short-arc controls v1_covar_inflation
regressed in 7en (PS1/PS2 failure cases). They live in the **wide** sweep
(7en/3bx) population, not this cohort, so their v2 fits are pulled from
``data/wide_variant_sweep_v2/variant_comparison.parquet`` for a per-object
regress/no-regress check under ``v2_empirical_covar``.

Outputs (default ``data/validation_sweep_v2/``)
-----------------------------------------------
    variant_comparison.parquet    500 rows (100 obj × 5 variants); mirrors si3
    REPORT.md                     ranking + DX/UA3 diagnostic + v1-vs-v2

Usage
-----
    pdm run python scripts/run_validation_sweep_v2.py \\
        [--output-dir data/validation_sweep_v2] [--n-workers 8] [--max-objects N]
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
logger = logging.getLogger("validation_sweep_v2")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
DEFAULT_COHORT_MANIFEST = Path("data/validation_sweep/cohort_manifest.parquet")
DEFAULT_SI3_VC = Path("data/validation_sweep/variant_comparison.parquet")
DEFAULT_WIDE_V2_VC = Path("data/wide_variant_sweep_v2/variant_comparison.parquet")

MAX_OBS_PER_OBJECT = 2000
CHI2_PATHOLOGICAL_LOW = 0.3
CHI2_PATHOLOGICAL_HIGH = 3.0
RESID_COV_N_THRESHOLD = 30

# DX/UA3 are 7en/wide-sweep controls; diagnostic is cross-referenced from there.
DXUA3_OBJECTS = ["2026 DX", "2025 UA3"]

# si3 (v1) → v2 name map for the v1-vs-v2 side-by-side on the identical cohort.
V1_TO_V2_NAME: Dict[str, str] = {
    "no_bias": "no_bias",
    "v1_subtract": "v2_subtract",
    "v1_performance_weighted": "v2_performance_weighted",
    "v1_covar_inflation": "v2_covar_inflation",
}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_efcc18: bool = False
    use_station_chi2: bool = False
    use_resid_covar: bool = False
    is_legacy: bool = False


VARIANTS: List[VariantConfig] = [
    VariantConfig("no_bias"),
    VariantConfig(
        "v2_subtract", bias_application="subtract",
        use_bias_table=True, use_efcc18=True, is_legacy=True,
    ),
    VariantConfig(
        "v2_performance_weighted", bias_application="performance_weighted",
        use_bias_table=True, use_station_chi2=True, use_efcc18=True,
    ),
    VariantConfig(
        "v2_covar_inflation", bias_application="covar_inflation",
        use_bias_table=True, use_efcc18=True,
    ),
    VariantConfig(
        "v2_empirical_covar", bias_application="empirical_covar",
        use_resid_covar=True, use_efcc18=True,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/validation_sweep_v2"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    p.add_argument("--si3-variant-comparison", type=Path, default=DEFAULT_SI3_VC)
    p.add_argument("--wide-v2-variant-comparison", type=Path, default=DEFAULT_WIDE_V2_VC)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_validation_v2_runs")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--per-object-timeout", type=float, default=1800.0,
                   help="Hard wall-clock cap (s) per object; a wedged actor is "
                        "killed + replaced and the object recorded as a timeout "
                        "failure so it can never stall the sweep.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None)
    return p.parse_args()


# ── v2 catalog sub-dicts (only what these 5 variants need) ──


@dataclass
class V2Catalogs:
    bias_table: Dict[str, Tuple[float, float]]
    station_chi2: Dict[str, float]
    station_resid_covar: Dict[str, Tuple[float, float, float, float]]
    n_stations: int


def build_v2_catalogs(path: str) -> V2Catalogs:
    cat = load_v2_bias_catalog(path, rollup_only=True)
    bias_table: Dict[str, Tuple[float, float]] = {}
    station_chi2: Dict[str, float] = {}
    station_resid_covar: Dict[str, Tuple[float, float, float, float]] = {}
    for code, rec in cat.items():
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"]):
            bias_table[code] = (rec["bias_ra"], rec["bias_dec"])
        if np.isfinite(rec["chi2_per_obs"]):
            station_chi2[code] = rec["chi2_per_obs"]
        station_resid_covar[code] = (
            rec["resid_var_ra"], rec["resid_var_dec"],
            rec["resid_cov_ra_dec"], rec["resid_cov_n"],
        )
    return V2Catalogs(bias_table, station_chi2, station_resid_covar, len(cat))


# ── observation prep (mirrors si3 + EFCC18 corrections) ──


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
    mean_obs_epoch_mjd: float
    efcc18_corrections: np.ndarray


def prepare_observations(
    provid: str, client: BigQueryMPCClient, spacebased: set, efcc18_bias_table,
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
    efcc = compute_efcc18_corrections(
        deduped.ra.to_numpy(zero_copy_only=False),
        deduped.dec.to_numpy(zero_copy_only=False),
        deduped.astcat.to_pylist(),
        times_mjd + 2400000.5,
        bias_table=efcc18_bias_table,
    )
    return PreparedObservations(
        obs=deduped,
        n_obs_input=len(deduped),
        n_stations_input=len(set(stns)),
        arc_days=float(times_mjd.max() - times_mjd.min()),
        mean_obs_epoch_mjd=float(times_mjd.mean()),
        efcc18_corrections=efcc,
    )


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
    converged: bool = False
    failure_reason: str = ""
    n_obs_input: int = 0
    n_obs_surviving: int = 0
    n_stations_input: int = 0
    n_stations_surviving: int = 0
    arc_days: float = float("nan")
    mean_obs_epoch_mjd: float = float("nan")
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


def _build_od_obs(obs, variant: VariantConfig, *, cats: V2Catalogs, efcc18):
    kw: Dict[str, object] = {
        "prevent_nans": True,
        "sigma_model": "veres2017",
        "bias_application": variant.bias_application,
    }
    if variant.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if variant.use_efcc18 and efcc18 is not None:
        kw["catalog_debias_arcsec"] = efcc18
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if variant.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


def run_single(rec, variant, obs, efcc18, jpl_orbit, jpl_sigma, *, cats, fitter, propagator):
    try:
        od_obs = _build_od_obs(obs, variant, cats=cats, efcc18=efcc18)
        if od_obs is None or len(od_obs) == 0:
            rec.failure_reason = "mpc_to_od_observations returned None/empty"
            return rec
        rec.n_obs_surviving = len(obs)
        rec.n_stations_surviving = len(set(obs.stn.to_pylist()))
        fitted, _ = fitter.initial_fit(rec.object_id, od_obs)
        if len(fitted) == 0:
            rec.failure_reason = "FindOrb returned empty FittedOrbits"
            return rec
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
        propagated = propagate_to_epoch(fitted_orbit, jpl_orbit.coordinates.time, propagator)
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
        if np.isfinite(jpl_sigma) and jpl_sigma > 0:
            rec.dr_over_sigma = gap.cartesian_dr_au / jpl_sigma
    except Exception as e:
        rec.failure_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    return rec


@ray.remote
class ObjectWorker:
    def __init__(self, idx: int, args_dict: dict, cats: V2Catalogs, efcc18_bias_table):
        logging.basicConfig(level=logging.WARNING)
        self.cats = cats
        self.spacebased = set(get_spacebased_stns())
        # Loaded once in main() and shipped via the object store (see wide v2).
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
                provid, self.client, self.spacebased, self.efcc18_bias_table
            )
        except Exception as e:
            logger.warning("%s: prepare_observations raised: %s", provid, e)
            prepared = None

        if prepared is None or jpl_orbit is None or len(jpl_orbit) == 0:
            reason = "observation prep failed" if prepared is None else "JPL orbit fetch failed"
            out = []
            for v in VARIANTS:
                out.append(asdict(VariantResult(
                    object_id=provid, designation=provid, stratum=stratum,
                    variant=v.variant_id, is_legacy=v.is_legacy, failure_reason=reason,
                )))
            return out

        out: List[dict] = []
        for v in VARIANTS:
            rec = VariantResult(
                object_id=provid, designation=provid, stratum=stratum,
                variant=v.variant_id, is_legacy=v.is_legacy,
                n_obs_input=prepared.n_obs_input,
                n_stations_input=prepared.n_stations_input,
                arc_days=prepared.arc_days,
                mean_obs_epoch_mjd=prepared.mean_obs_epoch_mjd,
            )
            rec = run_single(
                rec, v, prepared.obs, prepared.efcc18_corrections,
                jpl_orbit, jpl_sigma, cats=self.cats,
                fitter=self.fitter, propagator=self.propagator,
            )
            out.append(asdict(rec))
        return out


# ── persistence + REPORT ──


def write_parquet(records: List[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_dir / "variant_comparison.parquet", index=False)
    return output_dir / "variant_comparison.parquet"


def _failure_rows(row: dict, reason: str) -> List[dict]:
    """One failure VariantResult per variant for an object that never returned."""
    out: List[dict] = []
    for v in VARIANTS:
        out.append(asdict(VariantResult(
            object_id=row["provid"], designation=row["provid"],
            stratum=row.get("stratum", ""), variant=v.variant_id,
            is_legacy=v.is_legacy, failure_reason=reason,
        )))
    return out


def _med(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if not s.empty else float("nan")


def _regression_count(ctrl_df: pd.DataFrame, variant: str, baseline: str) -> int:
    base_by = ctrl_df.loc[ctrl_df.variant == baseline].set_index("object_id")["dr_over_sigma"]
    sub = ctrl_df.loc[ctrl_df.variant == variant].set_index("object_id")
    n = 0
    for oid, r in sub.iterrows():
        b = base_by.get(oid)
        c = r.dr_over_sigma
        if b is not None and np.isfinite(b) and np.isfinite(c) and b > 0 and c > 2.0 * b:
            n += 1
    return n


def write_report(
    records: List[dict], cohort_df: pd.DataFrame, output_dir: Path,
    v2_catalog_path: Path, si3_vc_path: Path, wide_v2_vc_path: Path,
    n_v2_stations: int,
) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n")
        return

    md: List[str] = []
    md.append("# Hold-out validation v2-catalog re-run — 3bx")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-3bx` · Branch: `kk/od-bias-experiments`")
    md.append("")
    md.append(f"v2 catalog: `{v2_catalog_path}` ({n_v2_stations} rollup stations)")
    md.append("Cohort: re-used si3 `cohort_manifest.parquet` (100 fresh NEOs, "
              "zero overlap with cph/7en) — identical objects to si3, so the "
              "v1-vs-v2 comparison below is on the same cohort.")
    md.append("Every `v2_*` variant applies EFCC18 preprocessing before its "
              "σ-modification; `no_bias` is raw (si3 used NO EFCC18 anywhere).")
    md.append("")

    # Cohort
    md.append("## Cohort")
    md.append("")
    md.append(f"Total NEOs: {len(cohort_df)}")
    for s, n in sorted(cohort_df.stratum.value_counts().to_dict().items()):
        md.append(f"- {s}: {n}")
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
    for stratum in sorted(df.stratum.dropna().unique()):
        sdf = df[(df.stratum == stratum) & df.converged & ~df.chi2_pathological]
        if sdf.empty:
            continue
        md.append(f"### {stratum} (n = {sdf.object_id.nunique()})")
        md.append("")
        md.append("| variant | median Δr/σ | median Δr (AU) | n_path |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            vsub = df[(df.stratum == stratum) & (df.variant == v.variant_id)]
            vclean = vsub[vsub.converged & ~vsub.chi2_pathological]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {_med(vclean.dr_over_sigma):.3f} | "
                      f"{_med(vclean.cartesian_dr_au):.2e} | "
                      f"{int(vsub.chi2_pathological.sum())} |")
        md.append("")

    # Headline — discrepant subset (impact_monitor + long_arc_well_obs), per si3
    md.append("## Headline — discrepant subset (impact_monitor + long_arc_well_obs)")
    md.append("")
    disc = df[df.stratum.isin(["impact_monitor", "long_arc_well_obs"])
              & df.converged & ~df.chi2_pathological]
    base_nb = _med(disc.loc[disc.variant == "no_bias", "dr_over_sigma"]) if not disc.empty else float("nan")
    base_sub = _med(disc.loc[disc.variant == "v2_subtract", "dr_over_sigma"]) if not disc.empty else float("nan")
    if not disc.empty:
        md.append("| variant | median Δr/σ | vs no_bias | vs v2_subtract |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            m = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"])
            r_nb = base_nb / m if np.isfinite(base_nb) and np.isfinite(m) and m > 0 else float("nan")
            r_sub = base_sub / m if np.isfinite(base_sub) and np.isfinite(m) and m > 0 else float("nan")
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {m:.3f} | {r_nb:.2f}× | {r_sub:.2f}× |")
        md.append("")

    # Control regression — short_arc_mod_obs (vs no_bias)
    md.append("## Control regression — short_arc_mod_obs stratum (vs no_bias)")
    md.append("")
    ctrl = df[(df.stratum == "short_arc_mod_obs") & df.converged]
    n_ctrl = ctrl.object_id.nunique()
    md.append(f"n_controls = {n_ctrl}")
    md.append("")
    md.append("| variant | n_controls_regressed (>2× no_bias) | rate |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        if v.variant_id == "no_bias":
            continue
        nreg = _regression_count(ctrl, v.variant_id, "no_bias")
        rate = nreg / n_ctrl if n_ctrl else 0.0
        tag = " (LEGACY)" if v.is_legacy else ""
        md.append(f"| `{v.variant_id}`{tag} | {nreg} | {100*rate:.1f}% |")
    md.append("")

    # Per-variant ranking summary (overall, discrepant subset)
    md.append("## Per-variant ranking summary (discrepant subset, best→worst)")
    md.append("")
    if not disc.empty:
        ranking = disc.groupby("variant")["dr_over_sigma"].median().sort_values()
        for i, (vname, mval) in enumerate(ranking.items(), 1):
            md.append(f"{i}. `{vname}` — median Δr/σ = {mval:.3f}")
        md.append("")

    # DX/UA3 diagnostic (cross-referenced from the wide v2 sweep)
    md.append("## Diagnostic — 2026 DX / 2025 UA3 under `v2_empirical_covar`")
    md.append("")
    md.append("These two short-arc controls are the v1_covar_inflation "
              "regressions from 7en (PS1/PS2 failure cases). They are members "
              "of the **wide** sweep population, so their v2 fits are pulled "
              "from `data/wide_variant_sweep_v2/`. Question: does "
              "`v2_empirical_covar` keep their Δr/σ at or below `no_bias`?")
    md.append("")
    if wide_v2_vc_path.exists():
        wv2 = pd.read_parquet(wide_v2_vc_path)
        md.append("| object | no_bias Δr/σ | v2_covar_inflation Δr/σ | "
                  "v2_empirical_covar Δr/σ | empirical ≤ no_bias? |")
        md.append("|---|---|---|---|---|")
        for obj in DXUA3_OBJECTS:
            sub = wv2[wv2.object_id == obj]
            def _g(var):
                r = sub[(sub.variant == var) & sub.converged]
                return float(r.dr_over_sigma.iloc[0]) if len(r) else float("nan")
            nb = _g("no_bias")
            cov = _g("v2_covar_inflation")
            emp = _g("v2_empirical_covar")
            ok = ("YES" if (np.isfinite(emp) and np.isfinite(nb) and emp <= nb * 1.0)
                  else ("—" if not (np.isfinite(emp) and np.isfinite(nb)) else "NO"))
            md.append(f"| `{obj}` | {nb:.4f} | {cov:.4f} | {emp:.4f} | {ok} |")
        md.append("")
        md.append("(`≤ no_bias` uses the same 'regression' lens as 7en: "
                  "empirical_covar must not push Δr/σ above the no-bias fit. "
                  "v1_covar_inflation pushed both objects to ~2.8–2.9× their "
                  "veres_only/no_bias baseline in 7en.)")
        md.append("")
    else:
        md.append(f"(wide v2 variant_comparison not found at {wide_v2_vc_path} — "
                  "run the wide sweep first.)")
        md.append("")

    # v1-vs-v2 side-by-side on the identical cohort
    md.append("## v1-vs-v2 side-by-side (identical cohort, discrepant subset)")
    md.append("")
    if si3_vc_path.exists() and not disc.empty:
        si3 = pd.read_parquet(si3_vc_path)
        si3d = si3[si3.stratum.isin(["impact_monitor", "long_arc_well_obs"])
                   & si3.converged & ~si3.chi2_pathological]
        md.append("| si3 (v1) variant | v2 variant | si3 median Δr/σ | v2 median Δr/σ | Δ(v2−v1) |")
        md.append("|---|---|---|---|---|")
        for v1name, v2name in V1_TO_V2_NAME.items():
            m1 = _med(si3d.loc[si3d.variant == v1name, "dr_over_sigma"])
            m2 = _med(disc.loc[disc.variant == v2name, "dr_over_sigma"])
            delta = m2 - m1 if np.isfinite(m1) and np.isfinite(m2) else float("nan")
            md.append(f"| `{v1name}` | `{v2name}` | {m1:.3f} | {m2:.3f} | {delta:+.3f} |")
        m2_emp = _med(disc.loc[disc.variant == "v2_empirical_covar", "dr_over_sigma"])
        md.append(f"| *(none)* | `v2_empirical_covar` | — | {m2_emp:.3f} | NEW |")
        md.append("")
    else:
        md.append("(si3 variant_comparison not found — skipping side-by-side.)")
        md.append("")

    # Verdict
    md.append("## Verdict on `v2_empirical_covar`")
    md.append("")
    if not disc.empty and n_ctrl > 0:
        m_emp = _med(disc.loc[disc.variant == "v2_empirical_covar", "dr_over_sigma"])
        reg_emp = _regression_count(ctrl, "v2_empirical_covar", "no_bias")
        rate_emp = reg_emp / n_ctrl if n_ctrl else 0.0
        beats = np.isfinite(m_emp) and np.isfinite(base_nb) and m_emp < base_nb
        threshold = 0.04
        keeps = rate_emp <= threshold
        verdict = "**GO**" if (beats and keeps) else "**NO-GO**"
        md.append(
            f"{verdict} — v2_empirical_covar discrepant median Δr/σ = {m_emp:.3f} "
            f"vs no_bias {base_nb:.3f} ({'improves' if beats else 'no improvement'}); "
            f"control regression rate vs no_bias = {100*rate_emp:.1f}% "
            f"({'≤' if keeps else '>'} 4% target). si3's v1_covar_inflation was "
            "NO-GO (tied no_bias, regressed ~20% of controls)."
        )
        md.append("")

    (output_dir / "REPORT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)

    if not args.cohort_manifest.exists():
        logger.error("Cohort manifest not found: %s", args.cohort_manifest)
        return 1
    cohort_df = pd.read_parquet(args.cohort_manifest)
    logger.info("Re-using si3 cohort manifest: %d objects", len(cohort_df))
    if args.max_objects is not None:
        cohort_df = cohort_df.head(args.max_objects)
        logger.info("Smoke cap: %d objects", len(cohort_df))

    cats = build_v2_catalogs(str(args.v2_catalog_path))
    logger.info("v2 catalog: %d rollup stations (bias_table=%d, chi2=%d, resid_covar=%d)",
                cats.n_stations, len(cats.bias_table), len(cats.station_chi2),
                len(cats.station_resid_covar))

    existing: set = set()
    records: List[dict] = []
    out_parquet = args.output_dir / "variant_comparison.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        records = prev.to_dict(orient="records")
        existing = set(prev.object_id.unique().tolist())
        logger.info("Resume: %d objects already done", len(existing))

    targets = [r for r in cohort_df.to_dict(orient="records") if r["provid"] not in existing]
    logger.info("Objects to process: %d", len(targets))

    if targets:
        efcc18_bias_table = load_efcc18_biases()  # load once; warms .npy cache
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
        # Manual scheduler with a per-object timeout (see run_wide_variant_sweep_v2
        # for the rationale: a wedged FindOrb fit must never stall the sweep).
        start = time.time()
        n_done = len(existing)
        n_total = len(cohort_df)
        timeout_s = args.per_object_timeout
        queue = list(targets)
        inflight: Dict = {}

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
            if n_done % 10 == 0:
                write_parquet(records, args.output_dir)

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
                    logger.warning("[timeout] %s exceeded %.0fs on actor %d — "
                                   "killing + replacing", row["provid"], timeout_s, widx)
                    try:
                        ray.kill(workers[widx])
                    except Exception:
                        pass
                    inflight.pop(ref)
                    _record(_failure_rows(row, f"per-object timeout >{int(timeout_s)}s"))
                    workers[widx] = ObjectWorker.remote(widx, args_dict, cats_ref, efcc18_ref)
                    _submit(widx)
        ray.shutdown()

    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows → %s", len(records), parq)
    write_report(records, cohort_df, args.output_dir, args.v2_catalog_path,
                 args.si3_variant_comparison, args.wide_v2_variant_comparison,
                 cats.n_stations)
    logger.info("Wrote REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
