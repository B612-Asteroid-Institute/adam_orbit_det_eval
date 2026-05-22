#!/usr/bin/env python3
"""
run_wide_variant_sweep.py
=========================

Wider-population non-subtract sweep: 17 variants × ~150 NEOs + MBAs.

Implements bead ``od_experiments_setup-7en``. Extends cph's 24-object × 6-variant
matrix with the qsd new levers (already landed in ``utils.py``) plus six new
non-subtract levers, evaluated across a stratified ~150-object population.

Headline question
-----------------
Does ``v1_performance_weighted`` (qsd's top principled variant on YR4) beat
``no_bias`` on the discrepant population *without* regressing the short-arc
controls that ``v1_subtract`` damages? And how close does it get to
``v1_subtract`` on the discrepant set?

Variant matrix (17 total)
-------------------------
Anchors (4): no_bias, veres_only, v1_sigma_floor, v1_subtract (LEGACY/REF)
EFCC18 retained (2): efcc18_only, v1_sigma_floor+efcc18
qsd levers (5): uniform_sigma, drop_non_HC_stations, v1_RSS_additive,
                v1_performance_weighted, drop_bias_significant
New non-subtract (6): v1_bayes_shrinkage, v1_at_ct_floor,
                      v1_chi2_outlier_reject, veres_v1_max_floor,
                      drop_high_rms_stations, v1_covar_inflation

Population (~150 stratified)
----------------------------
- Impact-monitor NEOs   (target 30-40): cph's 14 discrepant + extension list
- Long-arc well-observed (target 30-40): arc > 5000 d, n_obs > 200 (BQ sample)
- Short-arc moderately-observed (target 30-40): arc 30-500 d, n_obs 50-300 +
                                                 cph's 4 short-arc regressors
- Main-belt asteroids   (target 20-30): numbered MBAs, varied arc (BQ sample)

cph's 24 objects are forcibly included as a comparability subset.

Outputs (default ``data/wide_variant_sweep/``)
----------------------------------------------
    population_manifest.parquet   selected objects + stratum + provenance
    variant_comparison.parquet    one row per (object, variant)
    REPORT.md                     narrative

Usage
-----
    pdm run python scripts/run_wide_variant_sweep.py \\
        [--output-dir data/wide_variant_sweep] \\
        [--population-cap 150]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv

from adam_assist import ASSISTPropagator
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from google.cloud import bigquery as bq_lib
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
    mpc_to_od_observations,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from od_discrepancy_test_set import TEST_SET  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("wide_variant_sweep")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_BIAS_TABLE = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/mpc_scale_results_20260510/bias_catalog_published/"
    "high_confidence_bias_table.parquet"
)
DEFAULT_ATCT_BIAS_TABLE = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/mpc_scale_results_20260510/bias_catalog_published_atct/"
    "bias_table.parquet"
)
DEFAULT_9D7_PARQUET = Path("data/od_discrepancy_population/discrepancy_ranking.parquet")
DEFAULT_CPH_PARQUET = Path("data/od_discrepancy_population/variant_comparison.parquet")

# Cap obs/object to bound fit time. Matches cph/9d7.
MAX_OBS_PER_OBJECT = 2000

# drop_high_rms_stations threshold (arcsec).
HIGH_RMS_ARCSEC = 0.5

# chi2-outlier-reject ratio.
CHI2_OUTLIER_RATIO = 3.0

# χ² hold-in range outside which fits are flagged "pathological" — too low =
# sigmas too wide (underconstrained), too high = sigmas too narrow.
CHI2_PATHOLOGICAL_LOW = 0.3
CHI2_PATHOLOGICAL_HIGH = 3.0

# Hardcoded extension to the impact-monitor / well-known NEO stratum. cph's 24
# objects already contribute Apophis (2004 MN4), Bennu (1999 RQ36), Eros
# (A898 PA), Ivar (1929 SH), Toutatis-class etc.; add a few that aren't in cph.
IMPACT_MONITOR_EXTENSION = [
    "1996 GT",    # 25143 Itokawa
    "1996 FG3",   # binary NEA (Hera target)
    "2017 BX",    # already may be present
    "2022 AE1",   # impact-monitor candidate
    "2019 PR2",
    "2003 SD220", # Aten impact-monitor
    "2010 RF12",
    "1950 DA",    # 29075
    "2001 FB",
    "1998 OR2",
    "1998 KY26",
    "2002 TC70",
    "2005 ED224",
    "2007 FT3",
    "2006 QV89",  # impact-monitor
    "2008 JL3",
    "2009 JF1",
    "2010 GZ60",
    "2011 AG5",
    "2012 HG2",
]

# Hardcoded MBAs to anchor that stratum. Numbered MBAs with multi-opposition arcs.
MBA_SEEDS = [
    "1801 AA",    # 1 Ceres (provid; designation packed)
    "A847 NA",    # 7 Iris
    "A801 AA",    # 1 Ceres alternate
    "A802 FA",    # 2 Pallas
    "A802 PA",    # 3 Juno
    "A807 FA",    # 4 Vesta
    "A847 OA",    # 8 Flora
    "A852 UA",    # 14 Irene
    "A858 RA",    # 51 Nemausa
    "A861 EA",    # 67 Asia
]


@dataclass(frozen=True)
class VariantConfig:
    """A single variant.

    Encodes (sigma_model, bias_application, pre_filter, post_processing)
    so the driver can dispatch without sprinkling string comparisons.
    """
    variant_id: str
    sigma_model: str = "veres2017"
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_efcc18: bool = False
    pre_filter: Optional[str] = None  # 'hc_stations_only' | 'drop_bias_significant' | 'drop_high_rms'
    post_processing: Optional[str] = None  # 'chi2_outlier_reject'
    uniform_sigma_arcsec: float = 0.5
    use_station_chi2: bool = False
    use_station_sem: bool = False
    use_atct: bool = False
    is_legacy: bool = False


VARIANTS: List[VariantConfig] = [
    # --- anchors (4) ---
    VariantConfig("no_bias"),
    VariantConfig("veres_only"),
    VariantConfig(
        "v1_sigma_floor",
        bias_application="sigma_floor",
        use_bias_table=True,
    ),
    VariantConfig(
        "v1_subtract",
        bias_application="subtract",
        use_bias_table=True,
        is_legacy=True,
    ),
    # --- EFCC18 (2) ---
    VariantConfig("efcc18_only", use_efcc18=True),
    VariantConfig(
        "v1_sigma_floor+efcc18",
        bias_application="sigma_floor",
        use_bias_table=True,
        use_efcc18=True,
    ),
    # --- qsd levers (5; modes already in utils.py) ---
    VariantConfig(
        "uniform_sigma",
        sigma_model="uniform",
        uniform_sigma_arcsec=0.5,
    ),
    VariantConfig(
        "drop_non_HC_stations",
        pre_filter="hc_stations_only",
    ),
    VariantConfig(
        "v1_RSS_additive",
        bias_application="rss_additive",
        use_bias_table=True,
    ),
    VariantConfig(
        "v1_performance_weighted",
        bias_application="performance_weighted",
        use_bias_table=True,
        use_station_chi2=True,
    ),
    VariantConfig(
        "drop_bias_significant",
        pre_filter="drop_bias_significant",
    ),
    # --- new non-subtract (6; modes added in this bead) ---
    VariantConfig(
        "v1_bayes_shrinkage",
        bias_application="bayes_shrinkage",
        use_bias_table=True,
        use_station_sem=True,
    ),
    VariantConfig(
        "v1_at_ct_floor",
        bias_application="at_ct_floor",
        use_atct=True,
    ),
    VariantConfig(
        "v1_chi2_outlier_reject",
        # First pass = veres_only; post_processing handles the 2-pass logic.
        post_processing="chi2_outlier_reject",
        use_station_chi2=True,
    ),
    VariantConfig(
        "veres_v1_max_floor",
        bias_application="veres_v1_max_floor",
        use_bias_table=True,
    ),
    VariantConfig(
        "drop_high_rms_stations",
        pre_filter="drop_high_rms",
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wide_variant_sweep"),
    )
    p.add_argument("--population-cap", type=int, default=150,
                   help="Target population size (default: 150)")
    p.add_argument("--bias-table-path", type=Path,
                   default=Path(DEFAULT_BIAS_TABLE))
    p.add_argument("--atct-bias-table-path", type=Path,
                   default=Path(DEFAULT_ATCT_BIAS_TABLE))
    p.add_argument("--ninedeseven-parquet", type=Path,
                   default=DEFAULT_9D7_PARQUET)
    p.add_argument("--cph-parquet", type=Path, default=DEFAULT_CPH_PARQUET,
                   help="cph variant_comparison.parquet (24-object subset)")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_wide_runs")
    p.add_argument("--resume", action="store_true",
                   help="Pick up from an existing variant_comparison.parquet")
    p.add_argument("--population-manifest", type=Path, default=None,
                   help="Reuse an existing population_manifest.parquet")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Hard cap for smoke testing")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Bias-catalog loaders
# ────────────────────────────────────────────────────────────────────────────


def load_bias_table(path: Path) -> Dict[str, Tuple[float, float]]:
    """v1 high-confidence RA/Dec bias rollup → {stn: (bias_ra, bias_dec)} arcsec."""
    t = pq.read_table(
        path, columns=["obs_code", "bias_ra_arcsec", "bias_dec_arcsec"]
    )
    codes = t.column("obs_code").to_pylist()
    ras = t.column("bias_ra_arcsec").to_pylist()
    decs = t.column("bias_dec_arcsec").to_pylist()
    out: Dict[str, Tuple[float, float]] = {}
    for code, ra, dec in zip(codes, ras, decs):
        if code is None or ra is None or dec is None:
            continue
        out[str(code)] = (float(ra), float(dec))
    return out


def load_station_chi2_per_obs(path: Path) -> Dict[str, float]:
    """v1 catalog chi2_per_obs → {stn: chi2_per_obs}."""
    t = pq.read_table(path, columns=["obs_code", "chi2_per_obs"])
    out: Dict[str, float] = {}
    for code, c in zip(
        t.column("obs_code").to_pylist(), t.column("chi2_per_obs").to_pylist()
    ):
        if code is None or c is None:
            continue
        out[str(code)] = float(c)
    return out


def load_bias_significant(path: Path) -> Dict[str, bool]:
    """v1 catalog bias_significant flag → {stn: bool}."""
    t = pq.read_table(path, columns=["obs_code", "bias_significant"])
    out: Dict[str, bool] = {}
    for code, f in zip(
        t.column("obs_code").to_pylist(), t.column("bias_significant").to_pylist()
    ):
        if code is None or f is None:
            continue
        out[str(code)] = bool(f)
    return out


def load_station_sem(path: Path) -> Dict[str, Tuple[float, float]]:
    """v1 catalog SEM (standard error of the bias estimate) → {stn: (sem_ra, sem_dec)}."""
    t = pq.read_table(path, columns=["obs_code", "sem_ra_arcsec", "sem_dec_arcsec"])
    out: Dict[str, Tuple[float, float]] = {}
    for code, sr, sd in zip(
        t.column("obs_code").to_pylist(),
        t.column("sem_ra_arcsec").to_pylist(),
        t.column("sem_dec_arcsec").to_pylist(),
    ):
        if code is None or sr is None or sd is None:
            continue
        out[str(code)] = (float(sr), float(sd))
    return out


def load_station_rms(path: Path) -> Dict[str, Tuple[float, float]]:
    """v1 catalog per-station RMS → {stn: (rms_ra, rms_dec)} arcsec."""
    t = pq.read_table(path, columns=["obs_code", "rms_ra_arcsec", "rms_dec_arcsec"])
    out: Dict[str, Tuple[float, float]] = {}
    for code, rr, rd in zip(
        t.column("obs_code").to_pylist(),
        t.column("rms_ra_arcsec").to_pylist(),
        t.column("rms_dec_arcsec").to_pylist(),
    ):
        if code is None or rr is None or rd is None:
            continue
        out[str(code)] = (float(rr), float(rd))
    return out


def load_atct_bias_table(path: Path) -> Dict[str, Tuple[float, float]]:
    """v1 AT/CT bias catalog (per-station rollup, program_code IS NULL) →
    {stn: (|bias_AT|, |bias_CT|)} arcsec.

    The published AT/CT parquet keys rows by ``(obs_code, program_code)``.
    The ``program_code IS NULL`` rows are the per-station rollups (one per
    station).
    """
    t = pq.read_table(
        path,
        columns=[
            "obs_code", "program_code", "bias_at_arcsec", "bias_ct_arcsec"
        ],
    )
    mask = pa.compute.is_null(t.column("program_code"))
    t = t.filter(mask)
    codes = t.column("obs_code").to_pylist()
    ats = t.column("bias_at_arcsec").to_pylist()
    cts = t.column("bias_ct_arcsec").to_pylist()
    out: Dict[str, Tuple[float, float]] = {}
    for code, a, c in zip(codes, ats, cts):
        if code is None or a is None or c is None:
            continue
        out[str(code)] = (float(abs(a)), float(abs(c)))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Population selection
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class PopulationStratum:
    name: str
    where_clause: str
    target_count: int


# BigQuery strata definitions. Each stratum is a SQL fragment against the
# public_mpc_orbits view (in DATASET, not VIEWS_DATASET — same as
# build_od_discrepancy_population.py).
BQ_STRATA: List[PopulationStratum] = [
    PopulationStratum(
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
    PopulationStratum(
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
        target_count=35,
    ),
    PopulationStratum(
        "main_belt",
        """
        WHERE q >= 1.66 AND q < 3.3   -- inner main belt
          AND e < 0.4
          AND arc_length_total > 1000
          AND nobs_total >= 100
          AND nobs_total <= 5000
          AND arc_length_total IS NOT NULL
          AND unpacked_primary_provisional_designation IS NOT NULL
          AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
          AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
        """,
        target_count=25,
    ),
]


def _bq_sample(
    bq: bq_lib.Client,
    project: str,
    dataset_id: str,
    stratum: PopulationStratum,
    exclude: set[str],
    seed: int = 7,
) -> List[Dict[str, object]]:
    """Sample provids from one stratum. Deterministic via FARM_FINGERPRINT seed."""
    sql = f"""
SELECT unpacked_primary_provisional_designation AS provid,
       nobs_total, arc_length_total, q, e
FROM `{project}.{dataset_id}.public_mpc_orbits`
{stratum.where_clause}
ORDER BY MOD(ABS(FARM_FINGERPRINT(unpacked_primary_provisional_designation)), 1000000)
LIMIT {stratum.target_count * 3}
""".strip()
    rows = list(bq.query(sql).result())
    picked: List[Dict[str, object]] = []
    for r in rows:
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
    logger.info("stratum %s -> %d objects", stratum.name, len(picked))
    return picked


def select_population(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Assemble the 7en population.

    Step 1: cph's 24 objects (preserves direct comparability).
    Step 2: hardcoded impact-monitor extension list.
    Step 3: hardcoded MBA seed list.
    Step 4: BigQuery strata fill-ins (long_arc, short_arc, main_belt).

    Returns a DataFrame keyed by provid with columns:
        provid, stratum, source, in_cph_set, is_discrepant_in_cph,
        is_control_in_cph, ...
    """
    rows: List[Dict[str, object]] = []
    seen: set[str] = set()

    # Step 1 — cph's 24 objects
    cph_provids: List[str] = []
    if args.cph_parquet.exists():
        try:
            cdf = pd.read_parquet(args.cph_parquet)
            cph_provids = sorted(cdf.object_id.unique().tolist())
            cph_disc = set(cdf.loc[~cdf.is_control, "object_id"].unique().tolist())
            cph_ctrl = set(cdf.loc[cdf.is_control, "object_id"].unique().tolist())
        except Exception as e:
            logger.warning("Couldn't read cph parquet %s: %s", args.cph_parquet, e)
            cph_disc, cph_ctrl = set(), set()
    else:
        cph_disc, cph_ctrl = set(), set()
    for p in cph_provids:
        if p in seen:
            continue
        is_disc = p in cph_disc
        # Stratum classification: cph discrepant → impact-monitor; cph controls
        # are mostly short-arc 2025/2026 designations.
        if is_disc:
            stratum = "impact_monitor"
        else:
            stratum = "short_arc_mod_obs"
        rows.append({
            "provid": p,
            "stratum": stratum,
            "source": "cph",
            "in_cph_set": True,
            "is_discrepant_in_cph": is_disc,
            "is_control_in_cph": p in cph_ctrl,
        })
        seen.add(p)

    # Step 2 — impact-monitor extension
    for p in IMPACT_MONITOR_EXTENSION:
        if p in seen or len(rows) >= args.population_cap:
            continue
        rows.append({
            "provid": p, "stratum": "impact_monitor", "source": "extension",
            "in_cph_set": False, "is_discrepant_in_cph": False,
            "is_control_in_cph": False,
        })
        seen.add(p)

    # Step 3 — MBA seeds
    for p in MBA_SEEDS:
        if p in seen or len(rows) >= args.population_cap:
            continue
        rows.append({
            "provid": p, "stratum": "main_belt", "source": "extension",
            "in_cph_set": False, "is_discrepant_in_cph": False,
            "is_control_in_cph": False,
        })
        seen.add(p)

    # Step 4 — BigQuery fill-ins, only as needed
    bq = bq_lib.Client(project=args.project)
    for stratum in BQ_STRATA:
        cap_remaining = max(0, args.population_cap - len(rows))
        if cap_remaining == 0:
            break
        # Reduce target if we'd exceed cap
        stratum_local = PopulationStratum(
            stratum.name, stratum.where_clause,
            min(stratum.target_count, cap_remaining),
        )
        picked = _bq_sample(bq, args.project, args.dataset_id, stratum_local, seen)
        for entry in picked:
            rows.append({
                "provid": entry["provid"],
                "stratum": stratum.name,
                "source": "bigquery",
                "in_cph_set": False,
                "is_discrepant_in_cph": False,
                "is_control_in_cph": False,
            })

    df = pd.DataFrame(rows)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Per-object observation prep
# ────────────────────────────────────────────────────────────────────────────


def _dedupe_close_obs(obs: MPCObservations) -> MPCObservations:
    """Sequential ≤1.5s same-station dedupe (mirrors 9d7/cph)."""
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
    efcc18_corrections: np.ndarray  # (N, 2), zero rows for unsupported astcats
    # AT/CT velocity unit vectors (N, 2): (u_ra_cosdec, u_dec). NaN per row
    # where velocity could not be computed (stationary or ephemeris failure).
    atct_unit_vectors: np.ndarray


def _compute_atct_unit_vectors(
    obs: MPCObservations,
    jpl_orbit: Orbits,
    propagator: ASSISTPropagator,
) -> np.ndarray:
    """Per-observation cos(dec)-frame velocity unit vector from the JPL orbit.

    Returns ``(N, 2)`` with rows ``(u_ra_cosdec, u_dec)``. Stationary obs
    (speed < 1e-6 deg/day) get NaN rows; same for ephemeris failures.
    """
    n = len(obs)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    if jpl_orbit is None or len(jpl_orbit) == 0:
        return out
    times = obs.obstime
    codes = obs.stn
    try:
        observers = Observers.from_codes(codes=codes, times=times)
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
    MIN_SPEED = 1e-6
    valid = speed >= MIN_SPEED
    out[valid, 0] = v_ra_cosdec[valid] / speed[valid]
    out[valid, 1] = v_dec[valid] / speed[valid]
    return out


def prepare_observations(
    provid: str,
    client: BigQueryMPCClient,
    spacebased: set[str],
    bias_table: Dict[str, Tuple[float, float]],
    efcc18_bias_table: np.ndarray,
    jpl_orbit: Optional[Orbits],
    propagator: ASSISTPropagator,
) -> Optional[PreparedObservations]:
    """Fetch + clean obs once; return everything the variants need."""
    raw = client.query_observations([provid])
    if raw is None or len(raw) == 0:
        return None

    # Drop space-based + null STN
    stns_raw = raw.stn.to_pylist()
    ground_mask_arr = np.array(
        [(s is not None) and (s not in spacebased) for s in stns_raw],
        dtype=bool,
    )
    if not ground_mask_arr.any():
        return None
    ground = raw.apply_mask(pa.array(ground_mask_arr))
    if ground.fragmented():
        ground = qv.concatenate([ground])

    # Dedupe ≤1.5s same-station
    deduped = _dedupe_close_obs(ground)

    # Cap to most-recent MAX_OBS_PER_OBJECT
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
    n_obs = len(deduped)
    n_stations = len(set(stns))
    arc_days = float(times_mjd.max() - times_mjd.min())
    n_in_bias_table = sum(1 for s in stns if s in bias_table)

    # EFCC18 corrections (zero rows for un-debiased catalogs)
    astcats = deduped.astcat.to_pylist()
    ra_deg = deduped.ra.to_numpy(zero_copy_only=False)
    dec_deg = deduped.dec.to_numpy(zero_copy_only=False)
    jd_tdb = times_mjd + 2400000.5  # MJD→JD
    efcc_corrections = compute_efcc18_corrections(
        ra_deg, dec_deg, astcats, jd_tdb, bias_table=efcc18_bias_table
    )
    n_efcc_covered = n_observations_covered(astcats)

    # AT/CT unit vectors from the JPL orbit (one-shot per object)
    if jpl_orbit is not None and len(jpl_orbit) > 0:
        atct_uv = _compute_atct_unit_vectors(deduped, jpl_orbit, propagator)
    else:
        atct_uv = np.full((n_obs, 2), np.nan, dtype=np.float64)

    return PreparedObservations(
        obs=deduped,
        n_obs_input=n_obs,
        n_stations_input=n_stations,
        arc_days=arc_days,
        n_obs_in_bias_table=n_in_bias_table,
        n_obs_efcc18_covered=n_efcc_covered,
        efcc18_corrections=efcc_corrections,
        atct_unit_vectors=atct_uv,
    )


# ────────────────────────────────────────────────────────────────────────────
# Pre-filters and 2-pass outlier reject
# ────────────────────────────────────────────────────────────────────────────


def apply_pre_filter(
    obs: MPCObservations,
    pre_filter: Optional[str],
    *,
    bias_table: Dict[str, Tuple[float, float]],
    bias_significant: Dict[str, bool],
    station_rms: Dict[str, Tuple[float, float]],
    rms_threshold: float = HIGH_RMS_ARCSEC,
) -> MPCObservations:
    """Filter obs BEFORE they reach mpc_to_od_observations.

    Implemented filters:
      ``hc_stations_only``       keep only obs from stations in v1 HC table
      ``drop_bias_significant``  drop obs from stations with bias_significant=True
                                 (stations absent from the catalog are kept)
      ``drop_high_rms``          drop obs from stations where v1 rms_ra > threshold
                                 OR rms_dec > threshold (stations absent are kept)
    """
    if pre_filter is None:
        return obs
    stns = obs.stn.to_pylist()
    if pre_filter == "hc_stations_only":
        mask_list = [s in bias_table for s in stns]
    elif pre_filter == "drop_bias_significant":
        mask_list = [not bias_significant.get(s, False) for s in stns]
    elif pre_filter == "drop_high_rms":
        def _keep(s: str) -> bool:
            r = station_rms.get(s)
            if r is None:
                return True  # not measured → kept (conservative)
            return r[0] <= rms_threshold and r[1] <= rms_threshold
        mask_list = [_keep(s) for s in stns]
    else:
        raise ValueError(f"Unknown pre_filter={pre_filter!r}")
    mask = pa.array(mask_list, type=pa.bool_())
    return obs.apply_mask(mask)


def chi2_outlier_reject_mask(
    od_obs,
    fitted_orbit: Orbits,
    propagator: ASSISTPropagator,
    bias_table: Dict[str, Tuple[float, float]],
    station_chi2: Dict[str, float],
    stns_list: List[str],
    ratio_threshold: float = CHI2_OUTLIER_RATIO,
) -> np.ndarray:
    """Compute residuals, predict σ from v1 chi2_per_obs, return keep-mask.

    Steps:
      1. Generate ephemeris of the first-pass orbit at each obs time/station.
      2. Compute |residual| / σ_predicted where σ_predicted =
         sqrt(σ_baseline² × max(chi2_per_obs_station, 1.0)).
      3. Reject rows where the per-axis ratio exceeds ``ratio_threshold``.
    """
    from adam_core.observers import Observers as _Observers

    obs_times = od_obs.coordinates.time
    obs_codes = od_obs.coordinates.origin.code
    observers = _Observers.from_codes(codes=obs_codes, times=obs_times)
    eph = propagator.generate_ephemeris(
        orbits=fitted_orbit, observers=observers, max_processes=1
    )

    obs_lon = od_obs.coordinates.lon.to_numpy(zero_copy_only=False)
    obs_lat = od_obs.coordinates.lat.to_numpy(zero_copy_only=False)
    pred_lon = eph.coordinates.lon.to_numpy(zero_copy_only=False)
    pred_lat = eph.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_dec = np.cos(np.deg2rad(obs_lat))
    # residuals in cos(dec)-corrected arcsec
    res_ra_arcsec = (obs_lon - pred_lon) * cos_dec * 3600.0
    # wrap RA differences into [-180, 180] deg before scaling
    res_ra_arcsec = np.where(
        res_ra_arcsec > 180.0 * 3600.0,
        res_ra_arcsec - 360.0 * 3600.0,
        np.where(
            res_ra_arcsec < -180.0 * 3600.0,
            res_ra_arcsec + 360.0 * 3600.0,
            res_ra_arcsec,
        ),
    )
    res_dec_arcsec = (obs_lat - pred_lat) * 3600.0

    sigmas = od_obs.coordinates.covariance.sigmas
    # sigmas[:, 1] is σ_lon in deg (RA without cos(dec)); convert to arcsec cos(dec).
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
    keep = (ratio_ra <= ratio_threshold) & (ratio_dec <= ratio_threshold)
    return keep


# ────────────────────────────────────────────────────────────────────────────
# Per (object, variant) fit + gap
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
    bias_table: Dict[str, Tuple[float, float]],
    efcc18_corrections: Optional[np.ndarray],
    station_chi2: Dict[str, float],
    station_sem: Dict[str, Tuple[float, float]],
    atct_bias_table: Dict[str, Tuple[float, float]],
    atct_unit_vectors: Optional[np.ndarray],
):
    """Single dispatch into mpc_to_od_observations with variant-appropriate kwargs."""
    kw: Dict[str, object] = {
        "prevent_nans": True,
        "sigma_model": variant.sigma_model,
        "uniform_sigma_arcsec": variant.uniform_sigma_arcsec,
    }
    if variant.use_bias_table:
        kw["bias_table"] = bias_table
        kw["bias_application"] = variant.bias_application
    else:
        # Modes like at_ct_floor / chi2_outlier_reject don't need RA/Dec
        # bias_table but still must thread the bias_application string.
        kw["bias_application"] = variant.bias_application
    if variant.use_efcc18 and efcc18_corrections is not None:
        kw["catalog_debias_arcsec"] = efcc18_corrections
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = station_chi2
    if variant.use_station_sem:
        kw["station_sem_arcsec"] = station_sem
    if variant.use_atct:
        kw["atct_bias_table"] = atct_bias_table
        kw["atct_unit_vectors"] = atct_unit_vectors
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
    bias_table: Dict[str, Tuple[float, float]],
    bias_significant: Dict[str, bool],
    station_chi2: Dict[str, float],
    station_sem: Dict[str, Tuple[float, float]],
    station_rms: Dict[str, Tuple[float, float]],
    atct_bias_table: Dict[str, Tuple[float, float]],
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
    veres_only_variant: VariantConfig,
) -> VariantResult:
    """Run one (object, variant). Mutates and returns rec."""
    try:
        # Pre-filter
        variant_obs = apply_pre_filter(
            obs,
            variant.pre_filter,
            bias_table=bias_table,
            bias_significant=bias_significant,
            station_rms=station_rms,
        )
        if variant_obs.fragmented():
            variant_obs = qv.concatenate([variant_obs])
        n_kept = len(variant_obs)
        if n_kept == 0:
            rec.failure_reason = "pre_filter dropped all observations"
            return rec
        rec.n_obs_surviving = n_kept
        rec.n_stations_surviving = len(set(variant_obs.stn.to_pylist()))

        # If pre_filter dropped rows, the per-row arrays must be re-aligned.
        if variant.pre_filter is not None:
            # Recompute EFCC18 / AT/CT slices for the kept rows
            if efcc18_corrections is not None and variant.use_efcc18:
                # rebuild from variant_obs (cheap)
                from adam_orbit_det_eval.efcc18 import load_efcc18_biases as _l
                _eb = _l()
                eff_efcc = compute_efcc18_corrections(
                    variant_obs.ra.to_numpy(zero_copy_only=False),
                    variant_obs.dec.to_numpy(zero_copy_only=False),
                    variant_obs.astcat.to_pylist(),
                    variant_obs.obstime.jd().to_numpy(zero_copy_only=False),
                    bias_table=_eb,
                )
            else:
                eff_efcc = None
            eff_atct_uv = None  # at_ct_floor doesn't combine with pre_filters
        else:
            eff_efcc = efcc18_corrections if variant.use_efcc18 else None
            eff_atct_uv = atct_unit_vectors if variant.use_atct else None

        # Build OD obs for the first (or only) pass
        od_obs = _build_od_obs(
            variant_obs,
            variant,
            bias_table=bias_table,
            efcc18_corrections=eff_efcc,
            station_chi2=station_chi2,
            station_sem=station_sem,
            atct_bias_table=atct_bias_table,
            atct_unit_vectors=eff_atct_uv,
        )
        if od_obs is None or len(od_obs) == 0:
            rec.failure_reason = "mpc_to_od_observations returned None/empty"
            return rec

        # First-pass fit
        fitted, _ = fitter.initial_fit(rec.object_id, od_obs)
        if len(fitted) == 0:
            rec.failure_reason = "FindOrb returned empty FittedOrbits"
            return rec

        # ---- chi2_outlier_reject second pass ----
        if variant.post_processing == "chi2_outlier_reject":
            fitted_orbit_first = Orbits.from_kwargs(
                orbit_id=fitted.orbit_id,
                object_id=fitted.object_id,
                coordinates=fitted.coordinates,
            )
            keep_mask = chi2_outlier_reject_mask(
                od_obs, fitted_orbit_first, propagator,
                bias_table, station_chi2,
                variant_obs.stn.to_pylist(),
            )
            n_kept_pass2 = int(keep_mask.sum())
            if n_kept_pass2 < 6:
                rec.failure_reason = (
                    f"chi2_outlier_reject left only {n_kept_pass2} obs"
                )
                return rec
            variant_obs2 = variant_obs.apply_mask(pa.array(keep_mask))
            if variant_obs2.fragmented():
                variant_obs2 = qv.concatenate([variant_obs2])
            # Build second-pass OD obs with veres_only settings (no bias) +
            # the same observation set sans outliers.
            od_obs2 = _build_od_obs(
                variant_obs2,
                veres_only_variant,
                bias_table=bias_table,
                efcc18_corrections=None,
                station_chi2=station_chi2,
                station_sem=station_sem,
                atct_bias_table=atct_bias_table,
                atct_unit_vectors=None,
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
    population_df: pd.DataFrame,
    output_dir: Path,
    bias_table_path: Path,
    atct_bias_table_path: Path,
) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n")
        return

    md: List[str] = []
    md.append("# Wide-population non-subtract sweep — 7en")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-7en` · Branch: `kk/od-bias-experiments`")
    md.append("")

    # Population summary
    md.append("## Population")
    md.append("")
    stratum_counts = (
        population_df.groupby("stratum").size().to_dict()
        if not population_df.empty
        else {}
    )
    md.append(f"Total objects: {len(population_df)}")
    for s, c in sorted(stratum_counts.items()):
        md.append(f"- {s}: {c}")
    cph_count = int(population_df.in_cph_set.sum()) if "in_cph_set" in population_df else 0
    md.append(f"cph subset (forced inclusion): {cph_count}")
    md.append("")
    md.append(f"v1 RA/Dec bias catalog: `{bias_table_path}`")
    md.append(f"v1 AT/CT bias catalog: `{atct_bias_table_path}`")
    md.append("")

    # Convergence per variant
    md.append("## Convergence")
    md.append("")
    md.append("| variant | converged / total | χ²-pathological |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        sub = df[df.variant == v.variant_id]
        n = len(sub)
        nc = int(sub.converged.sum())
        npath = int(sub.chi2_pathological.sum())
        md.append(f"| `{v.variant_id}` | {nc}/{n} | {npath} |")
    md.append("")

    # Per-stratum variant ranking
    md.append("## Per-stratum variant ranking (median Δr/σ, lower = better)")
    md.append("")
    md.append("Ranking restricted to fits with χ²_in ∈ [0.3, 3] (non-pathological).")
    md.append("")
    for stratum in sorted(df.stratum.dropna().unique()):
        sdf = df[(df.stratum == stratum) & df.converged]
        if sdf.empty:
            continue
        md.append(f"### {stratum} (n_objects = {sdf.object_id.nunique()})")
        md.append("")
        md.append("| variant | median Δr/σ | median Δr (AU) | n_converged | n_path |")
        md.append("|---|---|---|---|---|")
        for v in VARIANTS:
            vsub = sdf[sdf.variant == v.variant_id]
            vsub_clean = vsub[~vsub.chi2_pathological]
            med_ratio = _med(vsub_clean.dr_over_sigma)
            med_dr = _med(vsub_clean.cartesian_dr_au)
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | {med_ratio:.3f} | "
                f"{med_dr:.2e} | {len(vsub)} | "
                f"{int(vsub.chi2_pathological.sum())} |"
            )
        md.append("")

    # Headline: v1_performance_weighted vs no_bias vs v1_subtract on discrepant
    md.append("## Headline — discrepant set")
    md.append("")
    disc = df[df.is_discrepant_in_cph & df.converged & ~df.chi2_pathological]
    if not disc.empty:
        md.append(
            "Discrepant set = cph's 14 discrepant NEOs (forced inclusion, "
            f"n_objects_evaluated = {disc.object_id.nunique()})."
        )
        md.append("")
        md.append("| variant | median Δr/σ | vs no_bias | vs v1_subtract |")
        md.append("|---|---|---|---|")
        base_nb = _med(disc.loc[disc.variant == "no_bias", "dr_over_sigma"])
        base_sub = _med(disc.loc[disc.variant == "v1_subtract", "dr_over_sigma"])
        for v in VARIANTS:
            vmed = _med(disc.loc[disc.variant == v.variant_id, "dr_over_sigma"])
            ratio_nb = (
                base_nb / vmed if np.isfinite(base_nb) and np.isfinite(vmed) and vmed > 0
                else float("nan")
            )
            ratio_sub = (
                base_sub / vmed if np.isfinite(base_sub) and np.isfinite(vmed) and vmed > 0
                else float("nan")
            )
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | {vmed:.3f} | "
                f"{ratio_nb:.2f}× | {ratio_sub:.2f}× |"
            )
        md.append("")
        md.append(
            "Reading: a `vs no_bias` ratio > 1 means the variant moved the "
            "fit closer to JPL than the no-bias baseline. A `vs v1_subtract` "
            "ratio > 1 means it beat the legacy/reference variant."
        )
        md.append("")
    else:
        md.append("- No (converged, non-pathological) discrepant rows yet.")
        md.append("")

    # Control-regression table
    md.append("## Control regression — controls in cph + short-arc stratum")
    md.append("")
    md.append(
        "For each principled (non-legacy, non-anchor) variant, count of "
        "controls whose Δr/σ exceeds 2× the per-object `veres_only` baseline."
    )
    md.append("")
    md.append("| variant | n_controls_regressed (>2× veres_only) |")
    md.append("|---|---|")
    ctrl_df = df[
        (df.is_control_in_cph | (df.stratum == "short_arc_mod_obs"))
        & df.converged
    ]
    if not ctrl_df.empty:
        veres_by_obj = (
            ctrl_df.loc[ctrl_df.variant == "veres_only"]
            .set_index("object_id")["dr_over_sigma"]
        )
        for v in VARIANTS:
            if v.variant_id in ("no_bias", "veres_only"):
                continue
            sub = ctrl_df.loc[ctrl_df.variant == v.variant_id]
            n_reg = 0
            for obj_id, row in sub.set_index("object_id").iterrows():
                base = veres_by_obj.get(obj_id)
                cand = row.dr_over_sigma
                if (
                    base is not None and np.isfinite(base) and np.isfinite(cand)
                    and base > 0 and cand > 2.0 * base
                ):
                    n_reg += 1
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(f"| `{v.variant_id}`{tag} | {n_reg} |")
    md.append("")

    # MBA-only
    md.append("## Main-belt-only analysis (cleanest test)")
    md.append("")
    md.append(
        "MBAs have no NEO-specific dynamical confounders (no Yarkovsky-dominant "
        "objects, well-separated from planets). Differences here are dominated "
        "by observational systematics."
    )
    md.append("")
    mba = df[(df.stratum == "main_belt") & df.converged & ~df.chi2_pathological]
    if not mba.empty:
        md.append("| variant | median Δr/σ | median Δr (AU) | n |")
        md.append("|---|---|---|---|")
        for v in VARIANTS:
            vsub = mba[mba.variant == v.variant_id]
            tag = " (LEGACY)" if v.is_legacy else ""
            md.append(
                f"| `{v.variant_id}`{tag} | "
                f"{_med(vsub.dr_over_sigma):.3f} | "
                f"{_med(vsub.cartesian_dr_au):.2e} | {len(vsub)} |"
            )
    else:
        md.append("- No converged non-pathological MBA fits.")
    md.append("")

    # Obs-survival
    md.append("## Filter obs-survival diagnostic")
    md.append("")
    md.append(
        "For filter variants, what fraction of input obs survive? Objects "
        "where a filter dropped > 50% of obs are flagged for downstream "
        "verification."
    )
    md.append("")
    md.append("| variant | median frac surviving | n_objects > 50% dropped |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        if v.pre_filter is None:
            continue
        vsub = df[df.variant == v.variant_id]
        if vsub.empty:
            continue
        fracs = vsub.n_obs_surviving / np.where(
            vsub.n_obs_input > 0, vsub.n_obs_input, np.nan
        )
        frac_med = float(np.nanmedian(fracs))
        n_severe = int((fracs < 0.5).sum())
        md.append(f"| `{v.variant_id}` | {frac_med:.2f} | {n_severe} |")
    md.append("")

    # cph comparability check
    md.append("## cph comparability check")
    md.append("")
    md.append(
        "For the cph subset, the variants `no_bias`, `veres_only`, "
        "`v1_sigma_floor`, `v1_subtract` should produce numerically identical "
        "Δr to cph (same code path, same noise model, same observations). "
        "Anything > 1e-12 AU indicates a regression in shared infrastructure."
    )
    md.append("")
    cph_path = Path("data/od_discrepancy_population/variant_comparison.parquet")
    if cph_path.exists():
        try:
            cph_df = pd.read_parquet(cph_path)
            shared = df[df.in_cph_set & df.converged].merge(
                cph_df[["object_id", "variant", "cartesian_dr_au"]],
                on=["object_id", "variant"],
                suffixes=("", "_cph"),
                how="inner",
            )
            shared["abs_diff_au"] = (
                shared.cartesian_dr_au - shared.cartesian_dr_au_cph
            ).abs()
            max_diff_by_variant = (
                shared.groupby("variant")["abs_diff_au"].max().sort_values(ascending=False)
            )
            md.append("| variant | n_compared | max |diff| (AU) |")
            md.append("|---|---|---|")
            for v, diff in max_diff_by_variant.items():
                n_cmp = int((shared.variant == v).sum())
                md.append(f"| `{v}` | {n_cmp} | {diff:.2e} |")
            md.append("")
            severe = max_diff_by_variant[max_diff_by_variant > 1e-12]
            if not severe.empty:
                md.append("**Flagged divergences (> 1e-12 AU):**")
                for v, diff in severe.items():
                    md.append(f"- `{v}`: max diff {diff:.2e} AU")
                md.append("")
        except Exception as e:
            md.append(f"(Comparison failed: {e})")
            md.append("")
    else:
        md.append("(cph parquet not found at `data/od_discrepancy_population/`)")
        md.append("")

    # Recommendation
    md.append("## Recommendation")
    md.append("")
    md.append("Auto-summarized from headline + control tables above:")
    md.append("")
    if not disc.empty:
        # Best principled variant by median Δr/σ on the discrepant set
        principled = disc[disc.variant != "v1_subtract"]
        if not principled.empty:
            grp = principled.groupby("variant")["dr_over_sigma"].median().sort_values()
            best = grp.index[0]
            best_val = grp.iloc[0]
            md.append(
                f"- Best principled variant on discrepant set: **`{best}`** "
                f"(median Δr/σ = {best_val:.3f})."
            )
            sub_val = _med(disc.loc[disc.variant == "v1_subtract", "dr_over_sigma"])
            md.append(
                f"- v1_subtract (legacy) median Δr/σ = {sub_val:.3f}."
            )
            if np.isfinite(sub_val) and np.isfinite(best_val) and sub_val > 0:
                md.append(
                    f"- Gap: best principled is {best_val / sub_val:.2f}× of v1_subtract."
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

    # ── Population manifest ──
    manifest_path = args.output_dir / "population_manifest.parquet"
    if args.population_manifest is not None and args.population_manifest.exists():
        population_df = pd.read_parquet(args.population_manifest)
        logger.info("Reusing population manifest at %s (%d rows)",
                    args.population_manifest, len(population_df))
    elif manifest_path.exists() and args.resume:
        population_df = pd.read_parquet(manifest_path)
        logger.info("Reusing population manifest at %s (%d rows)",
                    manifest_path, len(population_df))
    else:
        logger.info("Building population (cap=%d)…", args.population_cap)
        population_df = select_population(args)
        population_df.to_parquet(manifest_path, index=False)
        logger.info("Wrote population manifest → %s (%d objects)",
                    manifest_path, len(population_df))

    if args.max_objects is not None:
        population_df = population_df.head(args.max_objects)
        logger.info("Capped to %d objects (smoke mode)", len(population_df))

    # ── Bias catalogs ──
    bias_table = load_bias_table(args.bias_table_path)
    station_chi2 = load_station_chi2_per_obs(args.bias_table_path)
    bias_significant = load_bias_significant(args.bias_table_path)
    station_sem = load_station_sem(args.bias_table_path)
    station_rms = load_station_rms(args.bias_table_path)
    atct_bias_table = load_atct_bias_table(args.atct_bias_table_path)
    logger.info(
        "Loaded v1 catalogs: %d HC stations, %d AT/CT stations",
        len(bias_table), len(atct_bias_table),
    )
    efcc18_bias_table = load_efcc18_biases()
    logger.info("Loaded EFCC18 catalog")

    # ── Resume support ──
    existing_pairs: set[Tuple[str, str]] = set()
    records: List[VariantResult] = []
    out_parquet = args.output_dir / "variant_comparison.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        # Keep prior rows (only converged or with non-empty failure_reason);
        # re-attempt rows that look unfinished.
        for r in prev.to_dict(orient="records"):
            existing_pairs.add((r["object_id"], r["variant"]))
            records.append(VariantResult(**{
                k: v for k, v in r.items() if k in VariantResult.__dataclass_fields__
            }))
        logger.info("Resume: %d prior (object, variant) pairs loaded", len(existing_pairs))

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
    veres_only_variant = next(v for v in VARIANTS if v.variant_id == "veres_only")

    # ── Iterate objects ──
    start = time.time()
    targets = population_df.to_dict(orient="records")
    for i, row in enumerate(targets, 1):
        provid = row["provid"]
        stratum = row.get("stratum", "")
        in_cph = bool(row.get("in_cph_set", False))
        is_disc = bool(row.get("is_discrepant_in_cph", False))
        is_ctrl = bool(row.get("is_control_in_cph", False))

        # Skip object if ALL 17 (obj, variant) rows already exist
        pending_variants = [
            v for v in VARIANTS if (provid, v.variant_id) not in existing_pairs
        ]
        if not pending_variants:
            logger.info("[%d/%d] %s — already complete (skipping)",
                        i, len(targets), provid)
            continue

        t0 = time.time()
        logger.info("[%d/%d] %s (stratum=%s, in_cph=%s, %d variants pending)",
                    i, len(targets), provid, stratum, in_cph, len(pending_variants))

        # Fetch JPL orbit (needed for AT/CT axes)
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

        # Prepare observations
        try:
            prepared = prepare_observations(
                provid, client, spacebased, bias_table, efcc18_bias_table,
                jpl_orbit, propagator,
            )
        except Exception as e:
            logger.exception("%s: prepare_observations raised: %s", provid, e)
            prepared = None
        if prepared is None or jpl_orbit is None or len(jpl_orbit) == 0:
            for v in pending_variants:
                r = VariantResult(
                    object_id=provid, designation=provid, stratum=stratum,
                    variant=v.variant_id, is_legacy=v.is_legacy,
                    in_cph_set=in_cph, is_discrepant_in_cph=is_disc,
                    is_control_in_cph=is_ctrl,
                    failure_reason=(
                        "observation prep failed" if prepared is None
                        else "JPL orbit fetch failed"
                    ),
                )
                records.append(r)
                existing_pairs.add((provid, v.variant_id))
            continue

        # Run each variant
        for v in pending_variants:
            rec = VariantResult(
                object_id=provid, designation=provid, stratum=stratum,
                variant=v.variant_id, is_legacy=v.is_legacy,
                in_cph_set=in_cph, is_discrepant_in_cph=is_disc,
                is_control_in_cph=is_ctrl,
                n_obs_input=prepared.n_obs_input,
                n_stations_input=prepared.n_stations_input,
                arc_days=prepared.arc_days,
                n_obs_in_bias_table=prepared.n_obs_in_bias_table,
                n_obs_efcc18_covered=prepared.n_obs_efcc18_covered,
            )
            rec = run_single(
                rec, v, prepared.obs,
                prepared.efcc18_corrections,
                prepared.atct_unit_vectors,
                jpl_orbit, jpl_sigma,
                bias_table=bias_table,
                bias_significant=bias_significant,
                station_chi2=station_chi2,
                station_sem=station_sem,
                station_rms=station_rms,
                atct_bias_table=atct_bias_table,
                fitter=fitter, propagator=propagator,
                veres_only_variant=veres_only_variant,
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

        # Incremental safety write every 5 objects
        if i % 5 == 0:
            write_parquet(records, args.output_dir)
            try:
                write_report(records, population_df, args.output_dir,
                             args.bias_table_path, args.atct_bias_table_path)
            except Exception as e:
                logger.warning("Incremental REPORT failed: %s", e)

    # Final write
    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows → %s", len(records), parq)
    write_report(records, population_df, args.output_dir,
                 args.bias_table_path, args.atct_bias_table_path)
    logger.info("Wrote REPORT.md")
    logger.info("Total runtime: %.1f min", (time.time() - start) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
