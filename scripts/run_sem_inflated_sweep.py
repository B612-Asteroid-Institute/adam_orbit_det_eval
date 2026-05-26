#!/usr/bin/env python3
"""
run_sem_inflated_sweep.py
=========================

Joint cph + si3 cohort sweep testing the new ``v1_subtract_sem_inflated``
variant.

Implements bead ``od_experiments_setup-8d9``. si3 (commit ``7e9b9de``)
invalidated ``v1_covar_inflation`` as the principled default: the
off-diagonal ``b_RA · b_Dec`` term over-couples RA/Dec when a station with
small-but-isotropic bias dominates a short arc (PS1/PS2/H21 on
2025/2026-designation NEOs). The new variant addresses this by using the
v1 CI half-width as a diagonal-only σ inflation paired with v1_subtract
for the mean correction. No off-diagonal terms.

Variants (5)
------------
    no_bias                       baseline (Veres 2017 σ fallback)
    v1_subtract                   LEGACY/REF anchor (position correction)
    v1_performance_weighted       qsd YR4 winner; partial generaliser
    v1_covar_inflation            7en "winner" — si3 refuted; re-run for direct cmp
    v1_subtract_sem_inflated      NEW — subtract + diagonal σ_b inflation

Cohort (joint, 124 objects)
---------------------------
    cph's 24 (data/od_discrepancy_population/variant_comparison.parquet)
    si3's 100 (data/validation_sweep/cohort_manifest.parquet)
    Cross-checked: cph ∩ si3 = ∅ (si3 was built excluding cph).

σ_b convention
--------------
The v1 HC bias catalog publishes ``bias_{ra,dec}_ci_low`` and
``bias_{ra,dec}_ci_high`` columns. We assume these are 95% CIs (bootstrap
percentile, per the LOOO study methodology) and convert to 1-σ by:

    σ_b = (ci_high - ci_low) / 2 / 1.96

If the published CI level is something else (e.g. 68%), the conversion
factor changes accordingly. This bead documents the choice in REPORT.md
and surfaces the σ_b distribution so the calibration is visible.

Outputs (default ``data/sem_inflated_sweep/``)
----------------------------------------------
    joint_manifest.parquet         124 rows; source_cohort + stratum
    variant_comparison.parquet     620 rows; per (object, variant) result
    REPORT.md                      narrative + diagnostics

Usage
-----
    pdm run python scripts/run_sem_inflated_sweep.py \\
        [--output-dir data/sem_inflated_sweep]
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
import pyarrow.parquet as pq
import quivr as qv

from adam_assist import ASSISTPropagator
from adam_core.orbits import Orbits
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.jpl_compare import (
    compute_orbit_gap,
    fetch_jpl_orbit,
    propagate_to_epoch,
)
from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    mpc_to_od_observations,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sem_inflated_sweep")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_BIAS_TABLE = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/mpc_scale_results_20260510/bias_catalog_published/"
    "high_confidence_bias_table.parquet"
)
DEFAULT_CPH_PARQUET = Path("data/od_discrepancy_population/variant_comparison.parquet")
DEFAULT_SI3_MANIFEST = Path("data/validation_sweep/cohort_manifest.parquet")
DEFAULT_SI3_VC = Path("data/validation_sweep/variant_comparison.parquet")

# CI → 1-σ conversion. The v1 catalog publishes 95% CIs (bootstrap percentile).
CI_TO_ONE_SIGMA = 1.96  # 95% CI half-width / 1.96 = 1-σ

# Cap obs/object — matches cph/7en/si3.
MAX_OBS_PER_OBJECT = 2000

# χ² hold-in pathology range — matches 7en/si3.
CHI2_PATHOLOGICAL_LOW = 0.3
CHI2_PATHOLOGICAL_HIGH = 3.0


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_station_chi2: bool = False
    use_station_bias_ci: bool = False
    is_legacy: bool = False


VARIANTS: List[VariantConfig] = [
    VariantConfig("no_bias"),
    VariantConfig(
        "v1_subtract",
        bias_application="subtract",
        use_bias_table=True,
        is_legacy=True,
    ),
    VariantConfig(
        "v1_performance_weighted",
        bias_application="performance_weighted",
        use_bias_table=True,
        use_station_chi2=True,
    ),
    VariantConfig(
        "v1_covar_inflation",
        bias_application="covar_inflation",
        use_bias_table=True,
    ),
    VariantConfig(
        "v1_subtract_sem_inflated",
        bias_application="subtract_sem_inflated",
        use_bias_table=True,
        use_station_bias_ci=True,
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Argparse
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/sem_inflated_sweep"))
    p.add_argument("--bias-table-path", type=Path,
                   default=Path(DEFAULT_BIAS_TABLE))
    p.add_argument("--cph-parquet", type=Path, default=DEFAULT_CPH_PARQUET)
    p.add_argument("--si3-manifest", type=Path, default=DEFAULT_SI3_MANIFEST)
    p.add_argument("--si3-vc", type=Path, default=DEFAULT_SI3_VC,
                   help="si3 variant_comparison.parquet — used to import the "
                        "stratum classification of si3 objects")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_sem_inflated_runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Hard cap for smoke testing")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Bias-catalog loaders — HC table with CI columns
# ────────────────────────────────────────────────────────────────────────────


def load_bias_table_with_ci(
    path: Path, ci_to_one_sigma: float = CI_TO_ONE_SIGMA
) -> Tuple[Dict[str, Tuple[float, float]],
           Dict[str, Tuple[float, float]],
           Dict[str, float]]:
    """Load v1 HC table → (bias_mean_dict, bias_ci_one_sigma_dict, chi2_dict).

    bias_mean: {obs_code: (bias_ra, bias_dec)} arcsec.
    bias_ci_one_sigma: {obs_code: (σ_b_ra, σ_b_dec)} arcsec; converted from
                       the published 95% CI half-width via /ci_to_one_sigma.
    chi2:      {obs_code: chi2_per_obs} for the perf-weighted variant.
    """
    t = pq.read_table(
        path,
        columns=[
            "obs_code",
            "bias_ra_arcsec", "bias_dec_arcsec",
            "bias_ra_ci_low", "bias_ra_ci_high",
            "bias_dec_ci_low", "bias_dec_ci_high",
            "chi2_per_obs",
        ],
    )
    codes = t.column("obs_code").to_pylist()
    bras = t.column("bias_ra_arcsec").to_pylist()
    bdecs = t.column("bias_dec_arcsec").to_pylist()
    ra_lo = t.column("bias_ra_ci_low").to_pylist()
    ra_hi = t.column("bias_ra_ci_high").to_pylist()
    dec_lo = t.column("bias_dec_ci_low").to_pylist()
    dec_hi = t.column("bias_dec_ci_high").to_pylist()
    chi2s = t.column("chi2_per_obs").to_pylist()

    bias_mean: Dict[str, Tuple[float, float]] = {}
    bias_ci_one_sigma: Dict[str, Tuple[float, float]] = {}
    chi2_dict: Dict[str, float] = {}
    for code, bra, bdec, rl, rh, dl, dh, c in zip(
        codes, bras, bdecs, ra_lo, ra_hi, dec_lo, dec_hi, chi2s
    ):
        if code is None:
            continue
        if bra is not None and bdec is not None:
            bias_mean[str(code)] = (float(bra), float(bdec))
        if rl is not None and rh is not None and dl is not None and dh is not None:
            sb_ra = (float(rh) - float(rl)) / 2.0 / ci_to_one_sigma
            sb_dec = (float(dh) - float(dl)) / 2.0 / ci_to_one_sigma
            bias_ci_one_sigma[str(code)] = (sb_ra, sb_dec)
        if c is not None:
            chi2_dict[str(code)] = float(c)
    return bias_mean, bias_ci_one_sigma, chi2_dict


# ────────────────────────────────────────────────────────────────────────────
# Joint cohort assembly
# ────────────────────────────────────────────────────────────────────────────


def assemble_joint_cohort(
    cph_parquet: Path,
    si3_manifest: Path,
    si3_vc: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Stitch cph's 24 + si3's 100 into a single 124-object manifest.

    Returns (df, xcheck_counts). The df has columns:
        provid, source_cohort, stratum, is_discrepant_in_cph, is_control_in_cph.
    Strata are mapped from cph's 9d7 strata for cph objects, and from si3's
    cohort_manifest for si3 objects.
    """
    if not cph_parquet.exists():
        raise FileNotFoundError(f"cph parquet not found: {cph_parquet}")
    if not si3_manifest.exists():
        raise FileNotFoundError(f"si3 manifest not found: {si3_manifest}")

    cph_df = pd.read_parquet(cph_parquet)
    # cph's variant_comparison.parquet rolls up to 24 unique objects.
    cph_objs = (
        cph_df[["object_id", "is_control", "discrepant_in_9d7"]]
        .drop_duplicates(subset=["object_id"])
        .rename(columns={"object_id": "provid"})
    )
    cph_objs["source_cohort"] = "cph"
    cph_objs["stratum"] = np.where(
        cph_objs.is_control,
        "short_arc_mod_obs",
        "impact_monitor",  # cph's 14 discrepant are all impact-monitor profile
    )
    cph_objs["is_discrepant_in_cph"] = ~cph_objs.is_control
    cph_objs["is_control_in_cph"] = cph_objs.is_control
    cph_objs = cph_objs[
        ["provid", "source_cohort", "stratum",
         "is_discrepant_in_cph", "is_control_in_cph"]
    ]

    si3_manifest_df = pd.read_parquet(si3_manifest)
    si3_objs = si3_manifest_df[["provid", "stratum"]].copy()
    si3_objs["source_cohort"] = "si3"
    si3_objs["is_discrepant_in_cph"] = False
    si3_objs["is_control_in_cph"] = False
    si3_objs = si3_objs[
        ["provid", "source_cohort", "stratum",
         "is_discrepant_in_cph", "is_control_in_cph"]
    ]

    # Cross-check: zero overlap is expected (si3 built with cph as exclusion).
    overlap = set(cph_objs.provid) & set(si3_objs.provid)
    if overlap:
        raise RuntimeError(
            f"cph/si3 unexpected overlap: {sorted(overlap)}. "
            "Did si3 build its cohort correctly?"
        )

    joint = pd.concat([cph_objs, si3_objs], ignore_index=True)
    xcheck = {
        "n_cph": len(cph_objs),
        "n_si3": len(si3_objs),
        "n_overlap": len(overlap),
        "n_total": len(joint),
    }
    return joint, xcheck


# ────────────────────────────────────────────────────────────────────────────
# Observation prep — same pattern as si3
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


def prepare_observations(
    provid: str,
    client: BigQueryMPCClient,
    spacebased: set[str],
) -> Optional[PreparedObservations]:
    raw = client.query_observations([provid])
    if raw is None or len(raw) == 0:
        return None
    stns_raw = raw.stn.to_pylist()
    ground_mask = np.array(
        [(s is not None) and (s not in spacebased) for s in stns_raw],
        dtype=bool,
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
    return PreparedObservations(
        obs=deduped,
        n_obs_input=len(deduped),
        n_stations_input=len(set(stns)),
        arc_days=float(times_mjd.max() - times_mjd.min()),
    )


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
    diag = np.array(
        [cov[0, 0, 0], cov[0, 1, 1], cov[0, 2, 2]], dtype=np.float64
    )
    if not np.all(np.isfinite(diag)) or np.any(diag < 0):
        return float("nan")
    return float(np.sqrt(diag.sum()))


@dataclass
class VariantResult:
    object_id: str
    designation: str
    source_cohort: str
    stratum: str
    variant: str
    is_legacy: bool
    is_discrepant_in_cph: bool
    is_control_in_cph: bool
    converged: bool = False
    failure_reason: str = ""
    n_obs_input: int = 0
    n_obs_surviving: int = 0
    n_stations_input: int = 0
    n_stations_surviving: int = 0
    arc_days: float = float("nan")
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
    bias_table: Dict[str, Tuple[float, float]],
    station_chi2: Dict[str, float],
    station_bias_ci: Dict[str, Tuple[float, float]],
):
    kw: Dict[str, object] = {
        "prevent_nans": True,
        "sigma_model": "veres2017",
        "bias_application": variant.bias_application,
    }
    if variant.use_bias_table:
        kw["bias_table"] = bias_table
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = station_chi2
    if variant.use_station_bias_ci:
        kw["station_bias_ci_arcsec"] = station_bias_ci
    return mpc_to_od_observations(obs, **kw)


def run_single(
    rec: VariantResult,
    variant: VariantConfig,
    obs: MPCObservations,
    jpl_orbit: Orbits,
    jpl_pos_sigma_au: float,
    *,
    bias_table: Dict[str, Tuple[float, float]],
    station_chi2: Dict[str, float],
    station_bias_ci: Dict[str, Tuple[float, float]],
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
) -> VariantResult:
    try:
        od_obs = _build_od_obs(
            obs, variant,
            bias_table=bias_table,
            station_chi2=station_chi2,
            station_bias_ci=station_bias_ci,
        )
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
            orbit_id=fitted.orbit_id,
            object_id=fitted.object_id,
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
# Reporting
# ────────────────────────────────────────────────────────────────────────────


def write_parquet(records: List[VariantResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])
    path = output_dir / "variant_comparison.parquet"
    df.to_parquet(path, index=False)
    return path


def _med(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if not s.empty else float("nan")


def write_report(
    records: List[VariantResult],
    joint_df: pd.DataFrame,
    xcheck: Dict[str, int],
    output_dir: Path,
    bias_table_path: Path,
    bias_ci_one_sigma: Dict[str, Tuple[float, float]],
) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n")
        return

    md: List[str] = []
    md.append("# `v1_subtract_sem_inflated` joint cohort sweep — 8d9")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-8d9` · Branch: `kk/od-bias-experiments`")
    md.append("")

    # Cohort
    md.append("## Cohort")
    md.append("")
    md.append(
        f"Joint cohort: cph ({xcheck['n_cph']}) + si3 ({xcheck['n_si3']}) "
        f"= {xcheck['n_total']} objects. Overlap: {xcheck['n_overlap']} "
        "(expected zero since si3 was built with cph as exclusion)."
    )
    md.append("")
    md.append("Per-cohort × stratum counts:")
    md.append("")
    md.append("| cohort | stratum | n |")
    md.append("|---|---|---|")
    for (coh, strat), n in (
        joint_df.groupby(["source_cohort", "stratum"]).size().items()
    ):
        md.append(f"| {coh} | {strat} | {n} |")
    md.append("")
    md.append(f"v1 RA/Dec bias catalog: `{bias_table_path}`")
    md.append("")

    # σ_b convention
    md.append("## σ_b convention")
    md.append("")
    md.append(
        f"σ_b is derived from the published 95% CI half-width as "
        f"`σ_b = (ci_high - ci_low) / 2 / {CI_TO_ONE_SIGMA}`. The v1 LOOO "
        "study publishes percentile-bootstrap CIs at the 95% level "
        "(matching the `bias_{ra,dec}_ci_{low,high}` columns in the HC "
        "table). If a future catalog revision changes the CI level (e.g. "
        "68%), update `CI_TO_ONE_SIGMA` in the driver accordingly."
    )
    md.append("")
    # σ_b histogram (text-table)
    sb_ra = np.array([v[0] for v in bias_ci_one_sigma.values()])
    sb_dec = np.array([v[1] for v in bias_ci_one_sigma.values()])
    md.append(
        f"Per-station σ_b distribution across {len(bias_ci_one_sigma)} HC stations:"
    )
    md.append("")
    md.append("| pct  | σ_b_ra (arcsec) | σ_b_dec (arcsec) |")
    md.append("|---|---|---|")
    for q in (5, 25, 50, 75, 95):
        md.append(
            f"| p{q} | {np.percentile(sb_ra, q):.3f} | "
            f"{np.percentile(sb_dec, q):.3f} |"
        )
    md.append("")
    md.append(
        f"Interpretation: when σ_b ≪ σ_baseline (≈ 0.15-0.3″ for modern "
        "catalogs), `subtract_sem_inflated` ≈ `v1_subtract`. When σ_b ≫ "
        "σ_baseline (poorly measured stations), it ≈ `no_bias`."
    )
    md.append("")

    # Convergence
    md.append("## Convergence")
    md.append("")
    md.append("| variant | converged / total | χ²-pathological |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        sub = df[df.variant == v.variant_id]
        nc = int(sub.converged.sum())
        npath = int(sub.chi2_pathological.sum())
        md.append(f"| `{v.variant_id}` | {nc}/{len(sub)} | {npath} |")
    md.append("")

    # Per-cohort variant ranking
    md.append("## Per-cohort variant ranking (median Δr/σ, lower = better)")
    md.append("")
    md.append("χ²-pathological fits excluded from the medians.")
    md.append("")
    for cohort in ["cph", "si3"]:
        cdf = df[(df.source_cohort == cohort) & df.converged & ~df.chi2_pathological]
        if cdf.empty:
            continue
        md.append(f"### {cohort} (n = {cdf.object_id.nunique()})")
        md.append("")
        md.append("| variant | median Δr/σ | median Δr (AU) | n_path |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            vsub_all = df[(df.source_cohort == cohort) & (df.variant == v.variant_id)]
            vsub_clean = vsub_all[vsub_all.converged & ~vsub_all.chi2_pathological]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | "
                f"{_med(vsub_clean.dr_over_sigma):.3f} | "
                f"{_med(vsub_clean.cartesian_dr_au):.2e} | "
                f"{int(vsub_all.chi2_pathological.sum())} |"
            )
        md.append("")

    # Joint
    joint_clean = df[df.converged & ~df.chi2_pathological]
    md.append(f"### Joint (n = {joint_clean.object_id.nunique()})")
    md.append("")
    md.append("| variant | median Δr/σ | median Δr (AU) | n_path |")
    md.append("|---|---|---|---|")
    for v in VARIANTS:
        vsub_all = df[df.variant == v.variant_id]
        vsub_clean = vsub_all[vsub_all.converged & ~vsub_all.chi2_pathological]
        tag = " (LEGACY)" if v.is_legacy else ""
        md.append(
            f"| `{v.variant_id}`{tag} | "
            f"{_med(vsub_clean.dr_over_sigma):.3f} | "
            f"{_med(vsub_clean.cartesian_dr_au):.2e} | "
            f"{int(vsub_all.chi2_pathological.sum())} |"
        )
    md.append("")

    # Headline (a) — discrepant set
    md.append("## HEADLINE (a) — cph discrepant set vs `v1_subtract`")
    md.append("")
    md.append(
        "Target: `v1_subtract_sem_inflated` median Δr/σ within ~10% of "
        "`v1_subtract` on cph's 14 discrepant NEOs. This is the regime "
        "where v1_subtract has its biggest absolute win."
    )
    md.append("")
    disc = df[
        (df.source_cohort == "cph") & df.is_discrepant_in_cph
        & df.converged & ~df.chi2_pathological
    ]
    if not disc.empty:
        md.append("| variant | median Δr/σ | vs v1_subtract |")
        md.append("|---|---|---|")
        base_sub = _med(disc.loc[disc.variant == "v1_subtract", "dr_over_sigma"])
        for v in VARIANTS:
            m = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"])
            ratio = base_sub / m if np.isfinite(m) and m > 0 else float("nan")
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {m:.3f} | {ratio:.2f}× |")
        md.append("")
        m_sem = _med(disc.loc[disc.variant == "v1_subtract_sem_inflated", "dr_over_sigma"])
        within_10pct = (
            np.isfinite(m_sem) and np.isfinite(base_sub) and base_sub > 0
            and abs(m_sem - base_sub) / base_sub <= 0.10
        )
        md.append(
            "**Verdict (a)**: "
            + ("✓ within 10% of v1_subtract" if within_10pct
               else "✗ NOT within 10% of v1_subtract")
            + f" — sem_inflated median {m_sem:.3f} vs v1_subtract {base_sub:.3f}"
            + f" (Δ = {100*abs(m_sem - base_sub)/max(base_sub, 1e-12):.1f}%)."
        )
        md.append("")

    # Headline (b) — si3 short-arc control regression
    md.append("## HEADLINE (b) — si3 short-arc control regression rate")
    md.append("")
    md.append(
        "Target: `v1_subtract_sem_inflated` regresses ≤ 2 controls (≤ 7% "
        "of si3's 30 short-arc controls, matching v1_subtract's *baseline* "
        "regression rate would mean we kept the subtract win without "
        "paying the off-diagonal degeneracy cost). Regression criterion: "
        "Δr/σ > 2× the per-object `no_bias` baseline."
    )
    md.append("")
    si3_ctrl = df[
        (df.source_cohort == "si3") & (df.stratum == "short_arc_mod_obs")
        & df.converged
    ]
    n_ctrl = si3_ctrl.object_id.nunique()
    md.append(f"n_controls = {n_ctrl}")
    md.append("")
    md.append("| variant | n_regressed (>2× no_bias) | rate |")
    md.append("|---|---|---|")
    no_bias_by_obj = si3_ctrl[si3_ctrl.variant == "no_bias"].set_index("object_id")["dr_over_sigma"]
    n_reg_sem = 0
    for v in VARIANTS:
        if v.variant_id == "no_bias":
            continue
        sub = si3_ctrl[si3_ctrl.variant == v.variant_id].set_index("object_id")
        n_reg = 0
        for obj_id, row in sub.iterrows():
            base = no_bias_by_obj.get(obj_id)
            cand = row.dr_over_sigma
            if (
                base is not None and np.isfinite(base) and np.isfinite(cand)
                and base > 0 and cand > 2.0 * base
            ):
                n_reg += 1
        if v.variant_id == "v1_subtract_sem_inflated":
            n_reg_sem = n_reg
        rate = n_reg / n_ctrl if n_ctrl else 0.0
        tag = " (LEGACY)" if v.is_legacy else ""
        md.append(f"| `{v.variant_id}`{tag} | {n_reg} | {100*rate:.1f}% |")
    md.append("")
    md.append(
        "**Verdict (b)**: "
        + (f"✓ ≤ 2 controls regressed ({n_reg_sem})" if n_reg_sem <= 2
           else f"✗ > 2 controls regressed ({n_reg_sem})")
        + f" — sem_inflated regression rate {100*n_reg_sem/max(n_ctrl,1):.1f}% on {n_ctrl} controls."
    )
    md.append("")

    # 2026 DX / 2025 UA3 specific check
    md.append("## Specific check — 2026 DX and 2025 UA3")
    md.append("")
    md.append(
        "These are the two short-arc controls regressed under "
        "`v1_covar_inflation` in both 7en and si3 (PS1/PS2/H21 station-mix "
        "failure mode). Target: `v1_subtract_sem_inflated` keeps their "
        "Δr/σ at or below `no_bias`."
    )
    md.append("")
    for obj_id in ("2026 DX", "2025 UA3"):
        sub = df[df.object_id == obj_id]
        if sub.empty:
            md.append(f"### `{obj_id}` — not in joint cohort")
            md.append("")
            continue
        md.append(f"### `{obj_id}`")
        md.append("")
        md.append("| variant | Δr/σ | Δr (AU) | χ² | path? |")
        md.append("|---|---|---|---|---|")
        for v in VARIANTS:
            r = sub[sub.variant == v.variant_id]
            if r.empty:
                continue
            r = r.iloc[0]
            tag = " (LEGACY)" if v.is_legacy else ""
            chi2_str = f"{r.hold_in_reduced_chi2:.2g}" if np.isfinite(r.hold_in_reduced_chi2) else "—"
            dr_sig = f"{r.dr_over_sigma:.3f}" if np.isfinite(r.dr_over_sigma) else "—"
            dr_au = f"{r.cartesian_dr_au:.2e}" if np.isfinite(r.cartesian_dr_au) else "—"
            md.append(f"| `{v.variant_id}`{tag} | {dr_sig} | {dr_au} | {chi2_str} | {r.chi2_pathological} |")
        md.append("")

    # Recommendation
    md.append("## Recommendation — go / no-go on `v1_subtract_sem_inflated`")
    md.append("")
    a_pass = False
    b_pass = False
    if not disc.empty:
        m_sem = _med(disc.loc[disc.variant == "v1_subtract_sem_inflated", "dr_over_sigma"])
        m_sub = _med(disc.loc[disc.variant == "v1_subtract", "dr_over_sigma"])
        a_pass = (
            np.isfinite(m_sem) and np.isfinite(m_sub) and m_sub > 0
            and abs(m_sem - m_sub) / m_sub <= 0.10
        )
    b_pass = n_reg_sem <= 2

    if a_pass and b_pass:
        md.append(
            "**GO** — `v1_subtract_sem_inflated` retains v1_subtract's "
            "discrepant-set advantage (within 10%) AND keeps short-arc "
            "control regressions ≤ 2. Elevate this variant to the principled "
            "default for v1 catalog applications."
        )
    elif a_pass and not b_pass:
        md.append(
            f"**MIXED** — Retains discrepant-set advantage but regresses "
            f"{n_reg_sem} short-arc controls (target ≤ 2). Diagonal σ "
            "inflation softens but does not eliminate the survey-dominated "
            "failure mode; investigate per-station σ_b distribution."
        )
    elif not a_pass and b_pass:
        md.append(
            f"**NO-GO** — Loses too much of v1_subtract's discrepant-set "
            "advantage (sem_inflated diverges by >10% from v1_subtract). "
            "The σ inflation is too aggressive for well-measured stations. "
            "Consider an asymmetric weighting (smaller σ_b scale factor)."
        )
    else:
        md.append(
            "**NO-GO** — Fails both criteria: discrepant-set lead lost AND "
            f"{n_reg_sem} short-arc controls regressed. The variant doesn't "
            "thread the needle. Path forward: station-conditional "
            "application or a different mean-correction strategy."
        )
    md.append("")

    (output_dir / "REPORT.md").write_text("\n".join(md))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)

    # ── Joint cohort manifest ──
    manifest_path = args.output_dir / "joint_manifest.parquet"
    if manifest_path.exists() and args.resume:
        joint_df = pd.read_parquet(manifest_path)
        # Reconstruct xcheck from manifest contents
        xcheck = {
            "n_cph": int((joint_df.source_cohort == "cph").sum()),
            "n_si3": int((joint_df.source_cohort == "si3").sum()),
            "n_overlap": 0,
            "n_total": len(joint_df),
        }
        logger.info("Resume: reusing joint manifest at %s", manifest_path)
    else:
        joint_df, xcheck = assemble_joint_cohort(
            args.cph_parquet, args.si3_manifest, args.si3_vc,
        )
        joint_df.to_parquet(manifest_path, index=False)
        logger.info(
            "Wrote joint manifest → %s (cph=%d, si3=%d, overlap=%d, total=%d)",
            manifest_path, xcheck["n_cph"], xcheck["n_si3"],
            xcheck["n_overlap"], xcheck["n_total"],
        )

    if args.max_objects is not None:
        joint_df = joint_df.head(args.max_objects)
        logger.info("Capped to %d objects (smoke mode)", len(joint_df))

    # ── Bias catalog with CI columns ──
    bias_mean, bias_ci_one_sigma, station_chi2 = load_bias_table_with_ci(
        args.bias_table_path
    )
    logger.info(
        "Loaded HC catalog: %d stations with bias, %d with CI, %d with chi2",
        len(bias_mean), len(bias_ci_one_sigma), len(station_chi2),
    )

    # ── Resume state ──
    existing_pairs: set[Tuple[str, str]] = set()
    records: List[VariantResult] = []
    out_parquet = args.output_dir / "variant_comparison.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        for r in prev.to_dict(orient="records"):
            existing_pairs.add((r["object_id"], r["variant"]))
            records.append(VariantResult(**{
                k: v for k, v in r.items()
                if k in VariantResult.__dataclass_fields__
            }))
        logger.info("Resume: loaded %d prior pairs", len(existing_pairs))

    # ── Shared resources ──
    spacebased = set(get_spacebased_stns())
    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
        project=args.project,
    )
    propagator = ASSISTPropagator()
    fitter = FindOrbOrbitFitter(
        fo_result_dir=str(args.fo_result_dir),
        clean_up_fo_dir=True,
        propagator=propagator,
    )

    start = time.time()
    targets = joint_df.to_dict(orient="records")
    for i, row in enumerate(targets, 1):
        provid = row["provid"]
        cohort = row["source_cohort"]
        stratum = row["stratum"]
        is_disc = bool(row.get("is_discrepant_in_cph", False))
        is_ctrl = bool(row.get("is_control_in_cph", False))
        pending = [v for v in VARIANTS if (provid, v.variant_id) not in existing_pairs]
        if not pending:
            logger.info("[%d/%d] %s — already complete", i, len(targets), provid)
            continue
        t0 = time.time()
        logger.info(
            "[%d/%d] %s (cohort=%s, stratum=%s, %d variants pending)",
            i, len(targets), provid, cohort, stratum, len(pending),
        )

        try:
            jpl_orbit = fetch_jpl_orbit(provid)
        except Exception as e:
            logger.warning("%s: JPL fetch failed: %s", provid, e)
            jpl_orbit = None
        jpl_sigma = (
            _jpl_pos_sigma_au(jpl_orbit)
            if jpl_orbit is not None and len(jpl_orbit) > 0
            else float("nan")
        )

        try:
            prepared = prepare_observations(provid, client, spacebased)
        except Exception as e:
            logger.exception("%s: prepare_observations raised: %s", provid, e)
            prepared = None

        if prepared is None or jpl_orbit is None or len(jpl_orbit) == 0:
            for v in pending:
                rec = VariantResult(
                    object_id=provid, designation=provid,
                    source_cohort=cohort, stratum=stratum,
                    variant=v.variant_id, is_legacy=v.is_legacy,
                    is_discrepant_in_cph=is_disc, is_control_in_cph=is_ctrl,
                    failure_reason=(
                        "observation prep failed" if prepared is None
                        else "JPL orbit fetch failed"
                    ),
                )
                records.append(rec)
                existing_pairs.add((provid, v.variant_id))
            continue

        for v in pending:
            rec = VariantResult(
                object_id=provid, designation=provid,
                source_cohort=cohort, stratum=stratum,
                variant=v.variant_id, is_legacy=v.is_legacy,
                is_discrepant_in_cph=is_disc, is_control_in_cph=is_ctrl,
                n_obs_input=prepared.n_obs_input,
                n_stations_input=prepared.n_stations_input,
                arc_days=prepared.arc_days,
            )
            rec = run_single(
                rec, v, prepared.obs, jpl_orbit, jpl_sigma,
                bias_table=bias_mean,
                station_chi2=station_chi2,
                station_bias_ci=bias_ci_one_sigma,
                fitter=fitter, propagator=propagator,
            )
            records.append(rec)
            existing_pairs.add((provid, v.variant_id))
            logger.info(
                "  variant=%-26s converged=%s Δr=%s Δr/σ=%s χ²=%s",
                v.variant_id, rec.converged,
                f"{rec.cartesian_dr_au:.2e}"
                if np.isfinite(rec.cartesian_dr_au) else "—",
                f"{rec.dr_over_sigma:.2f}"
                if np.isfinite(rec.dr_over_sigma) else "—",
                f"{rec.hold_in_reduced_chi2:.2g}"
                if np.isfinite(rec.hold_in_reduced_chi2) else "—",
            )

        logger.info(
            "[%d/%d] %s done in %.1fs (cum %.1f min)",
            i, len(targets), provid, time.time() - t0,
            (time.time() - start) / 60.0,
        )

        if i % 5 == 0:
            write_parquet(records, args.output_dir)
            try:
                write_report(records, joint_df, xcheck, args.output_dir,
                             args.bias_table_path, bias_ci_one_sigma)
            except Exception as e:
                logger.warning("Incremental REPORT failed: %s", e)

    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows → %s", len(records), parq)
    write_report(records, joint_df, xcheck, args.output_dir,
                 args.bias_table_path, bias_ci_one_sigma)
    logger.info("Wrote REPORT.md")
    logger.info("Total runtime: %.1f min", (time.time() - start) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
