#!/usr/bin/env python3
"""
run_validation_sweep.py
=======================

Hold-out validation of v1_covar_inflation on 100 fresh NEOs, no MBAs.

Implements bead ``od_experiments_setup-si3``. 7en (commit ``aaa7243``)
identified v1_covar_inflation as the principled winner — 1.41× better than
no_bias on the cph discrepant subset with only 2/45 control regressions (vs
v1_subtract's 9/45). This bead validates that result on a fresh 100-NEO
cohort with ZERO overlap with cph (24 obj) or 7en (149 obj). If
v1_covar_inflation continues to lead the discrepant subset AND keeps control
regressions at or below 4/100, it becomes the publishable principled
deliverable.

Variants (4)
------------
    no_bias                 baseline (Veres 2017 σ fallback, no bias)
    v1_subtract             legacy / reference (subtract bias from RA/Dec)
    v1_performance_weighted qsd YR4 winner; 7en runner-up principled
    v1_covar_inflation      7en principled winner — target of this validation

All four bias_application modes are already landed in
``adam_orbit_det_eval.utils.mpc_to_od_observations`` via prior beads. This
script writes no new utils.py code.

Cohort (100 NEOs, no MBAs)
--------------------------
- ~35 impact-monitor NEOs (q < 1.3, arc > 1000 d, n_obs ≥ 100; excluding
                            cph + 7en manifests)
- ~35 long-arc well-observed NEOs (q < 1.3, arc > 5000 d, n_obs ≥ 200)
- ~30 short-arc moderate-obs NEOs (q < 1.3, arc 30-500 d, n_obs 50-300)

cph + 7en manifests are subtracted as exclusion sets before sampling, with
the cross-check documented in the REPORT.

Runtime
-------
Sequential — mirrors 7en's actual pattern (no Ray in this workspace yet).
~400 fits at ~10 s/fit ≈ 70 min wall-clock.

Outputs (default ``data/validation_sweep/``)
--------------------------------------------
    cohort_manifest.parquet         100 rows; selected objects with strata
    variant_comparison.parquet      400 rows; per (object, variant) result
    REPORT.md                       narrative including 7en-control diagnostic

Usage
-----
    pdm run python scripts/run_validation_sweep.py \\
        [--output-dir data/validation_sweep] \\
        [--cohort-cap 100]
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
from google.cloud import bigquery as bq_lib
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
logger = logging.getLogger("validation_sweep")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────
# Defaults — match 7en for direct comparability
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_BIAS_TABLE = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/mpc_scale_results_20260510/bias_catalog_published/"
    "high_confidence_bias_table.parquet"
)
DEFAULT_CPH_PARQUET = Path("data/od_discrepancy_population/variant_comparison.parquet")
DEFAULT_7EN_MANIFEST = Path("data/wide_variant_sweep/population_manifest.parquet")

# Same caps as cph/7en so per-object obs sets line up.
MAX_OBS_PER_OBJECT = 2000

# Same pathology thresholds as 7en.
CHI2_PATHOLOGICAL_LOW = 0.3
CHI2_PATHOLOGICAL_HIGH = 3.0


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_station_chi2: bool = False
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
]


# ────────────────────────────────────────────────────────────────────────────
# Argparse
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/validation_sweep"))
    p.add_argument("--cohort-cap", type=int, default=100)
    p.add_argument("--bias-table-path", type=Path,
                   default=Path(DEFAULT_BIAS_TABLE))
    p.add_argument("--cph-parquet", type=Path, default=DEFAULT_CPH_PARQUET,
                   help="cph variant_comparison.parquet (exclusion source)")
    p.add_argument("--seven-en-manifest", type=Path,
                   default=DEFAULT_7EN_MANIFEST,
                   help="7en population_manifest.parquet (exclusion source)")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_validation_runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cohort-manifest", type=Path, default=None,
                   help="Reuse an existing cohort_manifest.parquet")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Hard cap for smoke testing")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Bias-catalog loaders — match 7en signatures
# ────────────────────────────────────────────────────────────────────────────


def load_bias_table(path: Path) -> Dict[str, Tuple[float, float]]:
    t = pq.read_table(
        path, columns=["obs_code", "bias_ra_arcsec", "bias_dec_arcsec"]
    )
    out: Dict[str, Tuple[float, float]] = {}
    for code, ra, dec in zip(
        t.column("obs_code").to_pylist(),
        t.column("bias_ra_arcsec").to_pylist(),
        t.column("bias_dec_arcsec").to_pylist(),
    ):
        if code is None or ra is None or dec is None:
            continue
        out[str(code)] = (float(ra), float(dec))
    return out


def load_station_chi2_per_obs(path: Path) -> Dict[str, float]:
    t = pq.read_table(path, columns=["obs_code", "chi2_per_obs"])
    out: Dict[str, float] = {}
    for code, c in zip(
        t.column("obs_code").to_pylist(),
        t.column("chi2_per_obs").to_pylist(),
    ):
        if code is None or c is None:
            continue
        out[str(code)] = float(c)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Exclusion list — cph + 7en
# ────────────────────────────────────────────────────────────────────────────


def load_exclusion_set(
    cph_parquet: Path, seven_en_manifest: Path
) -> Tuple[set[str], int, int]:
    """Return the union of cph + 7en object_ids, plus the per-source counts.

    Both source files are gitignored / produced by prior beads. Missing files
    fail loudly — we *cannot* validate independence without them.
    """
    cph_ids: set[str] = set()
    if cph_parquet.exists():
        cph_ids = set(pd.read_parquet(cph_parquet).object_id.unique().tolist())
    else:
        raise FileNotFoundError(
            f"cph parquet not found at {cph_parquet}. Cohort independence "
            "cannot be verified."
        )
    seven_en_ids: set[str] = set()
    if seven_en_manifest.exists():
        seven_en_ids = set(
            pd.read_parquet(seven_en_manifest).provid.unique().tolist()
        )
    else:
        raise FileNotFoundError(
            f"7en manifest not found at {seven_en_manifest}. Cohort "
            "independence cannot be verified."
        )
    return cph_ids | seven_en_ids, len(cph_ids), len(seven_en_ids)


# ────────────────────────────────────────────────────────────────────────────
# Cohort selection — 100 NEOs, no MBAs
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CohortStratum:
    name: str
    where_clause: str
    target_count: int


# BQ strata. Each excludes cph/7en via an in-Python second pass; SQL-level
# exclusion would need NOT IN with the full id list, which is fine for ~200
# ids but cleaner to filter post-fetch.
COHORT_STRATA: List[CohortStratum] = [
    CohortStratum(
        "impact_monitor",
        # NEO regime, multi-opposition arc, well-observed. cph/7en's
        # impact-monitor stratum used a manual extension list; here we let
        # BQ pick from the NEO catalog so we get truly independent objects.
        """
        WHERE q < 1.3
          AND arc_length_total > 1000
          AND nobs_total >= 100
          AND nobs_total <= 5000
          AND arc_length_total IS NOT NULL
          AND unpacked_primary_provisional_designation IS NOT NULL
          AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
          AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
        """,
        target_count=35,
    ),
    CohortStratum(
        "long_arc_well_obs",
        """
        WHERE q < 1.3
          AND arc_length_total > 5000
          AND nobs_total >= 200
          AND nobs_total <= 5000
          AND arc_length_total IS NOT NULL
          AND unpacked_primary_provisional_designation IS NOT NULL
          AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
          AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
        """,
        target_count=35,
    ),
    CohortStratum(
        "short_arc_mod_obs",
        """
        WHERE q < 1.3
          AND arc_length_total BETWEEN 30 AND 500
          AND nobs_total BETWEEN 50 AND 300
          AND arc_length_total IS NOT NULL
          AND unpacked_primary_provisional_designation IS NOT NULL
          AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
          AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
        """,
        target_count=30,
    ),
]


def _bq_sample_stratum(
    bq: bq_lib.Client,
    project: str,
    dataset_id: str,
    stratum: CohortStratum,
    exclude: set[str],
    seed_offset: int = 0,
) -> List[Dict[str, object]]:
    """Sample provids from one stratum, excluding cph + 7en members.

    Uses a different FARM_FINGERPRINT modulus than 7en so we don't sample
    the same objects under a different stratum classification.
    """
    # Pull enough candidates so we have headroom for the exclusion filter
    overdraw = max(stratum.target_count * 5, 200)
    sql = f"""
SELECT unpacked_primary_provisional_designation AS provid,
       nobs_total, arc_length_total, q, e
FROM `{project}.{dataset_id}.public_mpc_orbits`
{stratum.where_clause}
ORDER BY MOD(ABS(FARM_FINGERPRINT(
    CONCAT(unpacked_primary_provisional_designation, '_si3_{seed_offset}'))),
    1000003)
LIMIT {overdraw}
""".strip()
    picked: List[Dict[str, object]] = []
    for r in bq.query(sql).result():
        if r.provid is None:
            continue
        if r.provid in exclude:
            continue
        picked.append({
            "provid": r.provid,
            "stratum": stratum.name,
            "n_obs_orbit": int(r.nobs_total) if r.nobs_total is not None else 0,
            "arc_days_orbit": float(r.arc_length_total) if r.arc_length_total is not None else 0.0,
            "q_au": float(r.q) if r.q is not None else float("nan"),
            "e_orbit": float(r.e) if r.e is not None else float("nan"),
        })
        exclude.add(r.provid)
        if len(picked) >= stratum.target_count:
            break
    logger.info(
        "stratum %s -> %d objects (target=%d, overdraw=%d)",
        stratum.name, len(picked), stratum.target_count, overdraw,
    )
    return picked


def select_cohort(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build the 100-NEO cohort, returning the manifest + xcheck counts."""
    exclude, n_cph, n_7en = load_exclusion_set(
        args.cph_parquet, args.seven_en_manifest
    )
    starting_exclude_size = len(exclude)
    logger.info(
        "Exclusion set: cph=%d, 7en=%d, union=%d (overlap=%d)",
        n_cph, n_7en, starting_exclude_size, n_cph + n_7en - starting_exclude_size,
    )

    bq = bq_lib.Client(project=args.project)
    rows: List[Dict[str, object]] = []
    for stratum in COHORT_STRATA:
        cap_remaining = max(0, args.cohort_cap - len(rows))
        if cap_remaining == 0:
            break
        local = CohortStratum(
            stratum.name,
            stratum.where_clause,
            min(stratum.target_count, cap_remaining),
        )
        picked = _bq_sample_stratum(bq, args.project, args.dataset_id, local, exclude)
        rows.extend(picked)
    df = pd.DataFrame(rows)
    xcheck = {
        "n_cph": n_cph,
        "n_7en": n_7en,
        "n_exclude_union": starting_exclude_size,
        "n_in_cph": int(df.provid.isin(
            pd.read_parquet(args.cph_parquet).object_id.unique()
        ).sum()) if not df.empty else 0,
        "n_in_7en": int(df.provid.isin(
            pd.read_parquet(args.seven_en_manifest).provid.unique()
        ).sum()) if not df.empty else 0,
    }
    return df, xcheck


# ────────────────────────────────────────────────────────────────────────────
# Observation prep — copy 7en's pattern
# ────────────────────────────────────────────────────────────────────────────


def _dedupe_close_obs(obs: MPCObservations) -> MPCObservations:
    """Sequential ≤1.5 s same-station dedupe."""
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
        mean_obs_epoch_mjd=float(times_mjd.mean()),
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


def _build_od_obs(
    obs: MPCObservations,
    variant: VariantConfig,
    *,
    bias_table: Dict[str, Tuple[float, float]],
    station_chi2: Dict[str, float],
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
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
) -> VariantResult:
    try:
        od_obs = _build_od_obs(
            obs, variant,
            bias_table=bias_table, station_chi2=station_chi2,
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
# 7en-control diagnostic
# ────────────────────────────────────────────────────────────────────────────


def diagnose_seven_en_regressed_controls(
    seven_en_parquet: Path,
    client: BigQueryMPCClient,
    spacebased: set[str],
    bias_table: Dict[str, Tuple[float, float]],
) -> List[Dict[str, object]]:
    """Identify the 2 controls v1_covar_inflation regressed in 7en, and
    profile their station mix vs un-regressed controls.

    Returns a list of dicts (one per regressed control + summary entries).
    """
    df = pd.read_parquet(seven_en_parquet)
    ctrl = df[
        (df.is_control_in_cph | (df.stratum == "short_arc_mod_obs"))
        & df.converged
    ]
    veres_by_obj = ctrl[ctrl.variant == "veres_only"].set_index("object_id")["dr_over_sigma"]
    cov = ctrl[ctrl.variant == "v1_covar_inflation"].set_index("object_id")

    regressed: List[str] = []
    other: List[str] = []
    for obj_id, row in cov.iterrows():
        base = veres_by_obj.get(obj_id)
        cand = row.dr_over_sigma
        if base is None or not np.isfinite(base) or not np.isfinite(cand) or base <= 0:
            continue
        if cand > 2.0 * base:
            regressed.append(obj_id)
        else:
            other.append(obj_id)

    logger.info("7en regressed controls: %s", regressed)
    logger.info(
        "  (compared to %d un-regressed controls for station-mix profile)",
        len(other),
    )

    # Profile each regressed object's station mix
    profiles: List[Dict[str, object]] = []
    regressed_stations: set[str] = set()
    for obj_id in regressed:
        try:
            obs = client.query_observations([obj_id])
        except Exception as e:
            profiles.append({"object_id": obj_id, "error": str(e)})
            continue
        if obs is None or len(obs) == 0:
            profiles.append({"object_id": obj_id, "error": "no obs"})
            continue
        stns = obs.stn.to_pylist()
        ground_stns = [s for s in stns if s is not None and s not in spacebased]
        from collections import Counter
        counts = Counter(ground_stns)
        in_bias = sum(1 for s in ground_stns if s in bias_table)
        unique_stns = sorted(counts.keys())
        regressed_stations.update(unique_stns)
        profiles.append({
            "object_id": obj_id,
            "n_obs_ground": len(ground_stns),
            "n_unique_stations": len(counts),
            "n_obs_in_bias_table": in_bias,
            "frac_obs_in_bias_table": in_bias / len(ground_stns) if ground_stns else 0.0,
            "top_5_stations": counts.most_common(5),
            "all_stations": unique_stns,
        })

    # Compare with un-regressed control station sets — find stations that
    # appear in BOTH regressed objects but RARELY in un-regressed controls
    # (suggests a station-specific failure mode).
    if len(regressed) >= 2:
        shared = set(profiles[0].get("all_stations", []))
        for p in profiles[1:]:
            shared &= set(p.get("all_stations", []))
        profiles.append({
            "object_id": "__shared_across_regressed__",
            "stations": sorted(shared),
        })

    return profiles


# ────────────────────────────────────────────────────────────────────────────
# Persistence + REPORT
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
    cohort_df: pd.DataFrame,
    xcheck: Dict[str, int],
    output_dir: Path,
    bias_table_path: Path,
    seven_en_diagnostic: List[Dict[str, object]],
) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n")
        return

    md: List[str] = []
    md.append("# Hold-out validation of `v1_covar_inflation` — si3")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-si3` · Branch: `kk/od-bias-experiments`")
    md.append("")

    # Cohort summary + cross-check
    md.append("## Cohort")
    md.append("")
    md.append(f"Total NEOs sampled: {len(cohort_df)}")
    for s, n in sorted(cohort_df.stratum.value_counts().to_dict().items()):
        md.append(f"- {s}: {n}")
    md.append("")
    md.append(
        f"**Cross-check vs cph + 7en**: exclusion set built from cph "
        f"({xcheck['n_cph']} objects) and 7en ({xcheck['n_7en']} objects), "
        f"union = {xcheck['n_exclude_union']}. Post-sampling membership check: "
        f"{xcheck['n_in_cph']} cohort objects also in cph; "
        f"{xcheck['n_in_7en']} also in 7en. "
        + ("**Cohort is clean.**" if xcheck['n_in_cph'] == 0 and xcheck['n_in_7en'] == 0
           else "**OVERLAP DETECTED — investigate.**")
    )
    md.append("")
    md.append(f"v1 RA/Dec bias catalog: `{bias_table_path}`")
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

    # Per-stratum ranking
    md.append("## Per-stratum variant ranking (median Δr/σ, lower = better)")
    md.append("")
    md.append("Ranking restricted to χ²-non-pathological fits.")
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
            vsub_clean = vsub[vsub.converged & ~vsub.chi2_pathological]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | "
                f"{_med(vsub_clean.dr_over_sigma):.3f} | "
                f"{_med(vsub_clean.cartesian_dr_au):.2e} | "
                f"{int(vsub.chi2_pathological.sum())} |"
            )
        md.append("")

    # Headline — discrepant subset = impact_monitor + long_arc_well_obs
    md.append("## Headline — discrepant subset (impact_monitor + long_arc_well_obs)")
    md.append("")
    md.append(
        "Per 7en's framing, the 'discrepant subset' is the combination of "
        "impact-monitor and long-arc objects (the regime where bias matters)."
    )
    md.append("")
    disc = df[
        df.stratum.isin(["impact_monitor", "long_arc_well_obs"])
        & df.converged & ~df.chi2_pathological
    ]
    if not disc.empty:
        md.append("| variant | median Δr/σ | vs no_bias | vs v1_subtract |")
        md.append("|---|---|---|---|")
        base_nb = _med(disc.loc[disc.variant == "no_bias", "dr_over_sigma"])
        base_sub = _med(disc.loc[disc.variant == "v1_subtract", "dr_over_sigma"])
        for v in VARIANTS:
            m = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"])
            r_nb = base_nb / m if np.isfinite(m) and m > 0 else float("nan")
            r_sub = base_sub / m if np.isfinite(m) and m > 0 else float("nan")
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | {m:.3f} | {r_nb:.2f}× | {r_sub:.2f}× |"
            )
        md.append("")
    else:
        md.append("- No converged non-pathological discrepant rows.")
        md.append("")

    # Control regression — short_arc_mod_obs stratum
    md.append("## Control regression — short_arc_mod_obs stratum")
    md.append("")
    md.append(
        "For each principled variant, count of objects whose Δr/σ exceeds "
        "**2× the per-object `no_bias` baseline**. (Mirrors 7en's regression "
        "metric but uses `no_bias` here since there's no `veres_only` row in "
        "this 4-variant matrix — `no_bias` and `veres_only` are coded "
        "identically in this codebase.)"
    )
    md.append("")
    ctrl = df[(df.stratum == "short_arc_mod_obs") & df.converged]
    n_ctrl = ctrl.object_id.nunique()
    md.append(f"n_controls = {n_ctrl}")
    md.append("")
    md.append("| variant | n_controls_regressed (>2× no_bias) | rate |")
    md.append("|---|---|---|")
    no_bias_by_obj = ctrl[ctrl.variant == "no_bias"].set_index("object_id")["dr_over_sigma"]
    for v in VARIANTS:
        if v.variant_id == "no_bias":
            continue
        sub = ctrl[ctrl.variant == v.variant_id].set_index("object_id")
        n_reg = 0
        for obj_id, row in sub.iterrows():
            base = no_bias_by_obj.get(obj_id)
            cand = row.dr_over_sigma
            if (
                base is not None and np.isfinite(base) and np.isfinite(cand)
                and base > 0 and cand > 2.0 * base
            ):
                n_reg += 1
        tag = " (LEGACY)" if v.is_legacy else ""
        rate = n_reg / n_ctrl if n_ctrl else 0.0
        md.append(f"| `{v.variant_id}`{tag} | {n_reg} | {100*rate:.1f}% |")
    md.append("")

    # Comparison to 7en
    md.append("## Comparison to 7en's discrepant-set ranking")
    md.append("")
    md.append(
        "Per-variant median Δr/σ on this validation cohort's discrepant "
        "subset (impact_monitor + long_arc) versus 7en's reported numbers "
        "on its 14-object cph discrepant set. Same 4 variants only."
    )
    md.append("")
    SEVEN_EN_NUMBERS = {
        "no_bias": 41.22,
        "v1_subtract": 25.66,
        "v1_performance_weighted": 28.55,
        "v1_covar_inflation": 29.22,
    }
    md.append("| variant | 7en median Δr/σ | si3 median Δr/σ |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        sm = SEVEN_EN_NUMBERS.get(v.variant_id, float("nan"))
        m = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"]) if not disc.empty else float("nan")
        tag = " (LEGACY)" if v.is_legacy else ""
        md.append(f"| `{v.variant_id}`{tag} | {sm:.2f} | {m:.3f} |")
    md.append("")

    # 7en-control diagnostic
    md.append("## Diagnostic — 7en's 2 regressed controls under v1_covar_inflation")
    md.append("")
    md.append(
        "These are the two short-arc controls whose `v1_covar_inflation` "
        "fit moved >2× away from the `veres_only` baseline in 7en. Note: "
        "both have very small *absolute* veres_only Δr/σ (0.013 and 0.079), "
        "so the 2× threshold catches tiny absolute changes."
    )
    md.append("")
    for p in seven_en_diagnostic:
        if "error" in p:
            md.append(f"- `{p['object_id']}`: query error — {p['error']}")
            continue
        if p.get("object_id") == "__shared_across_regressed__":
            md.append(
                f"- **Stations common to both regressed controls**: "
                + (", ".join(p.get("stations", [])) or "(none)")
            )
            continue
        md.append(f"### `{p['object_id']}`")
        md.append("")
        md.append(
            f"- n_obs_ground = {p['n_obs_ground']}, "
            f"n_unique_stations = {p['n_unique_stations']}, "
            f"n_obs_in_v1_bias_table = {p['n_obs_in_bias_table']} "
            f"({100*p['frac_obs_in_bias_table']:.0f}%)"
        )
        md.append("- Top 5 stations by obs count:")
        for stn, n in p["top_5_stations"]:
            md.append(f"  - `{stn}` ({n} obs)")
        md.append("")

    # Recommendation
    md.append("## Recommendation — go / no-go on `v1_covar_inflation`")
    md.append("")
    if not disc.empty:
        m_cov = _med(disc.loc[disc.variant == "v1_covar_inflation", "dr_over_sigma"])
        m_nb = base_nb
        m_sub = base_sub
        improves_vs_no_bias = (
            np.isfinite(m_cov) and np.isfinite(m_nb) and m_cov < m_nb
        )
        # Recompute the no_bias-baselined regression count for v1_covar_inflation
        n_cov_reg = 0
        if n_ctrl > 0:
            sub = ctrl[ctrl.variant == "v1_covar_inflation"].set_index("object_id")
            for obj_id, row in sub.iterrows():
                base = no_bias_by_obj.get(obj_id)
                cand = row.dr_over_sigma
                if (
                    base is not None and np.isfinite(base) and np.isfinite(cand)
                    and base > 0 and cand > 2.0 * base
                ):
                    n_cov_reg += 1
        rate_cov = n_cov_reg / n_ctrl if n_ctrl else 0.0
        threshold = 0.04  # 7en's 2/45 ≈ 4.4%; cap at 4%
        keeps_low_regressions = rate_cov <= threshold
        verdict = "**GO**" if (improves_vs_no_bias and keeps_low_regressions) else "**NO-GO**"
        md.append(
            f"Decision: {verdict} — v1_covar_inflation lead on discrepant subset: "
            f"median Δr/σ = {m_cov:.2f} vs no_bias {m_nb:.2f} "
            f"({'improves' if improves_vs_no_bias else 'no improvement'}); "
            f"control regression rate = {100*rate_cov:.1f}% "
            f"({'≤' if keeps_low_regressions else '>'} 4% target)."
        )
        md.append("")
        if improves_vs_no_bias and keeps_low_regressions:
            md.append(
                "v1_covar_inflation validates on a fresh cohort. It can be "
                "treated as a publishable principled result."
            )
        else:
            md.append(
                "v1_covar_inflation does NOT validate cleanly on this cohort. "
                "Review the per-stratum tables before declaring it the "
                "v1 principled deliverable."
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

    # ── Cohort manifest ──
    manifest_path = args.output_dir / "cohort_manifest.parquet"
    if args.cohort_manifest is not None and args.cohort_manifest.exists():
        cohort_df = pd.read_parquet(args.cohort_manifest)
        # Best-effort: reload xcheck counts from sources
        xcheck = {
            "n_cph": int(pd.read_parquet(args.cph_parquet).object_id.nunique()),
            "n_7en": int(pd.read_parquet(args.seven_en_manifest).provid.nunique()),
            "n_exclude_union": 0,
            "n_in_cph": int(cohort_df.provid.isin(
                pd.read_parquet(args.cph_parquet).object_id.unique()
            ).sum()),
            "n_in_7en": int(cohort_df.provid.isin(
                pd.read_parquet(args.seven_en_manifest).provid.unique()
            ).sum()),
        }
        xcheck["n_exclude_union"] = xcheck["n_cph"] + xcheck["n_7en"]
        logger.info("Reusing cohort manifest at %s", args.cohort_manifest)
    elif manifest_path.exists() and args.resume:
        cohort_df = pd.read_parquet(manifest_path)
        xcheck = {
            "n_cph": int(pd.read_parquet(args.cph_parquet).object_id.nunique()),
            "n_7en": int(pd.read_parquet(args.seven_en_manifest).provid.nunique()),
            "n_exclude_union": 0,
            "n_in_cph": int(cohort_df.provid.isin(
                pd.read_parquet(args.cph_parquet).object_id.unique()
            ).sum()),
            "n_in_7en": int(cohort_df.provid.isin(
                pd.read_parquet(args.seven_en_manifest).provid.unique()
            ).sum()),
        }
        xcheck["n_exclude_union"] = xcheck["n_cph"] + xcheck["n_7en"]
        logger.info("Resuming with cohort manifest at %s", manifest_path)
    else:
        logger.info("Building cohort (cap=%d)…", args.cohort_cap)
        cohort_df, xcheck = select_cohort(args)
        cohort_df.to_parquet(manifest_path, index=False)
        logger.info("Wrote cohort manifest → %s (%d objects)",
                    manifest_path, len(cohort_df))

    if args.max_objects is not None:
        cohort_df = cohort_df.head(args.max_objects)
        logger.info("Capped to %d objects (smoke mode)", len(cohort_df))

    if xcheck["n_in_cph"] != 0 or xcheck["n_in_7en"] != 0:
        logger.warning(
            "Cohort overlap detected: cph=%d, 7en=%d. Continuing but flagging in REPORT.",
            xcheck["n_in_cph"], xcheck["n_in_7en"],
        )

    # ── Bias catalogs ──
    bias_table = load_bias_table(args.bias_table_path)
    station_chi2 = load_station_chi2_per_obs(args.bias_table_path)
    logger.info("Loaded v1 HC bias catalog (%d stations)", len(bias_table))

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

    # ── 7en-control diagnostic (cheap, do before the main loop) ──
    diagnostic = []
    seven_en_vc_parquet = Path("data/wide_variant_sweep/variant_comparison.parquet")
    if seven_en_vc_parquet.exists():
        try:
            diagnostic = diagnose_seven_en_regressed_controls(
                seven_en_vc_parquet, client, spacebased, bias_table,
            )
        except Exception as e:
            logger.warning("7en-control diagnostic failed: %s", e)
    else:
        logger.warning(
            "7en variant_comparison.parquet not found; skipping diagnostic"
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
        logger.info("Resume: loaded %d (object, variant) prior pairs", len(existing_pairs))

    # ── Main loop ──
    start = time.time()
    targets = cohort_df.to_dict(orient="records")
    for i, row in enumerate(targets, 1):
        provid = row["provid"]
        stratum = row.get("stratum", "")
        pending = [v for v in VARIANTS if (provid, v.variant_id) not in existing_pairs]
        if not pending:
            logger.info("[%d/%d] %s — already complete", i, len(targets), provid)
            continue
        t0 = time.time()
        logger.info("[%d/%d] %s (stratum=%s, %d variants pending)",
                    i, len(targets), provid, stratum, len(pending))

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
                    object_id=provid, designation=provid, stratum=stratum,
                    variant=v.variant_id, is_legacy=v.is_legacy,
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
                object_id=provid, designation=provid, stratum=stratum,
                variant=v.variant_id, is_legacy=v.is_legacy,
                n_obs_input=prepared.n_obs_input,
                n_stations_input=prepared.n_stations_input,
                arc_days=prepared.arc_days,
                mean_obs_epoch_mjd=prepared.mean_obs_epoch_mjd,
            )
            rec = run_single(
                rec, v, prepared.obs, jpl_orbit, jpl_sigma,
                bias_table=bias_table, station_chi2=station_chi2,
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

        logger.info("[%d/%d] %s done in %.1fs (cum %.1f min)",
                    i, len(targets), provid, time.time() - t0,
                    (time.time() - start) / 60.0)

        if i % 5 == 0:
            write_parquet(records, args.output_dir)
            try:
                write_report(records, cohort_df, xcheck, args.output_dir,
                             args.bias_table_path, diagnostic)
            except Exception as e:
                logger.warning("Incremental REPORT failed: %s", e)

    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows → %s", len(records), parq)
    write_report(records, cohort_df, xcheck, args.output_dir,
                 args.bias_table_path, diagnostic)
    logger.info("Wrote REPORT.md")
    logger.info("Total runtime: %.1f min", (time.time() - start) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
