#!/usr/bin/env python3
"""
run_discrepancy_variant_matrix.py
=================================

Variant-matrix sweep on the 9d7 discrepant set (14 objects) plus 10
non-discrepant controls. Re-fits each object under 6 bias/debiasing
variants, propagates to the JPL/SBDB epoch via ASSIST, and reports the
element + Cartesian gap for each (object, variant) pair.

Implements bead ``od_experiments_setup-cph``.

Variants
--------
1. ``no_bias``                — Veres 2017 sigma fallback, no bias, no EFCC18
2. ``veres_only``             — Veres 2017 sigmas, no bias, no EFCC18
3. ``v1_sigma_floor``         — Veres 2017 + v1 bias as σ-floor
4. ``efcc18_only``            — Veres 2017 + EFCC18 catalog debiasing
5. ``v1_sigma_floor+efcc18``  — Veres 2017 + v1 σ-floor + EFCC18 (stacked)
6. ``v1_subtract`` (LEGACY)   — Veres 2017 + v1 subtracted from observed RA/Dec

All five principled variants share ``sigma_model='veres2017'`` — a uniform
sigma policy was mandated by the cph bead so that aggregate hold-in χ²
stays O(1) (the 9d7 ``diag_nan=1e-9`` path inflated it by ~5 decades).

Outputs (under --output-dir, default data/od_discrepancy_population/):

    variant_comparison.parquet  one row per (object, variant) = 144 rows
    REPORT_variants.md          narrative

Usage
-----
    pdm run python scripts/run_discrepancy_variant_matrix.py \\
        [--output-dir data/od_discrepancy_population]
"""

from __future__ import annotations

import argparse
import json
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
    OrbitGap,
    compute_orbit_gap,
    fetch_jpl_orbit,
    propagate_to_epoch,
)
from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    mpc_to_od_observations,
)

# Load the frozen test set (14 discrepant) from the sibling script. Importing
# this way works whether the script is run as a module or directly (we add the
# scripts/ dir to sys.path below for the direct-run case).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from od_discrepancy_test_set import TEST_SET  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("discrepancy_variants")
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
DEFAULT_NINEDESEVEN_PARQUET = Path("data/od_discrepancy_population/discrepancy_ranking.parquet")

# Cap obs/object to bound fit time. Matches 9d7's choice.
MAX_OBS_PER_OBJECT = 2000


@dataclass(frozen=True)
class VariantConfig:
    """A single variant of the (sigma_model, bias_application, ...) bundle."""
    variant_id: str
    label: str
    sigma_model: str
    use_bias_table: bool
    bias_application: str           # 'sigma_floor' or 'subtract'
    use_efcc18: bool
    is_legacy: bool                  # True for v1_subtract (annotated as such)


# All five principled variants use Veres 2017 (uniform sigma policy mandated
# by cph). The legacy v1_subtract row is also evaluated under the same sigma
# policy so the column has a fair baseline.
VARIANTS: List[VariantConfig] = [
    VariantConfig("no_bias",
                  "no_bias",
                  sigma_model="veres2017",
                  use_bias_table=False,
                  bias_application="sigma_floor",
                  use_efcc18=False,
                  is_legacy=False),
    VariantConfig("veres_only",
                  "veres_only",
                  sigma_model="veres2017",
                  use_bias_table=False,
                  bias_application="sigma_floor",
                  use_efcc18=False,
                  is_legacy=False),
    VariantConfig("v1_sigma_floor",
                  "v1_sigma_floor",
                  sigma_model="veres2017",
                  use_bias_table=True,
                  bias_application="sigma_floor",
                  use_efcc18=False,
                  is_legacy=False),
    VariantConfig("efcc18_only",
                  "efcc18_only",
                  sigma_model="veres2017",
                  use_bias_table=False,
                  bias_application="sigma_floor",
                  use_efcc18=True,
                  is_legacy=False),
    VariantConfig("v1_sigma_floor+efcc18",
                  "v1_sigma_floor+efcc18",
                  sigma_model="veres2017",
                  use_bias_table=True,
                  bias_application="sigma_floor",
                  use_efcc18=True,
                  is_legacy=False),
    VariantConfig("v1_subtract",
                  "v1_subtract (legacy/reference)",
                  sigma_model="veres2017",
                  use_bias_table=True,
                  bias_application="subtract",
                  use_efcc18=False,
                  is_legacy=True),
]

# Note: no_bias and veres_only are coded identically (both with
# use_bias_table=False, use_efcc18=False, sigma_model='veres2017'). cph's
# original prompt listed them with the same sigma policy. We keep them as
# distinct rows so the comparison report has the conventional 'no_bias'
# anchor; downstream code dedupes when computing variant-vs-variant ratios.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/od_discrepancy_population"),
    )
    p.add_argument("--ninedeseven-parquet", type=Path,
                   default=DEFAULT_NINEDESEVEN_PARQUET,
                   help="9d7 parquet used to select controls + cross-reference")
    p.add_argument("--bias-table-path", type=Path,
                   default=Path(DEFAULT_BIAS_TABLE))
    p.add_argument("--n-controls", type=int, default=10,
                   help="Number of non-discrepant controls to draw (default: 10)")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_variant_runs")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Control selection
# ────────────────────────────────────────────────────────────────────────────


def select_controls(ninedeseven_parquet: Path, n: int) -> List[Dict[str, object]]:
    """Pick controls from the 9d7 converged set.

    Strategy: smallest |Δr/σ| among converged non-discrepant rows with JPL
    covariance. Rationale (documented in the report): these are objects
    where the no-bias fit was statistically consistent with JPL. They serve
    as negative controls — any principled bias variant that pushes them
    away from JPL is a regression signal.
    """
    df = pd.read_parquet(ninedeseven_parquet)
    cand = df[
        (df.fit_status == "converged")
        & (~df.discrepant)
        & df.delta_r_over_sigma.notna()
    ].copy()
    cand["abs_ratio"] = cand.delta_r_over_sigma.abs()
    cand = cand.sort_values("abs_ratio").head(n)
    return [
        {
            "provid": row.object_id,
            "stratum": row.stratum,
            "arc_days": float(row.arc_days),
            "n_obs": int(row.n_obs_ground),
            "n_stations": int(row.n_stations),
            "baseline_delta_r_over_sigma": float(row.delta_r_over_sigma),
        }
        for row in cand.itertuples(index=False)
    ]


# ────────────────────────────────────────────────────────────────────────────
# Bias table + EFCC18 helpers
# ────────────────────────────────────────────────────────────────────────────


def load_bias_table(path: Path) -> Dict[str, Tuple[float, float]]:
    """Read the v1 high-confidence bias table into {stn: (bias_ra, bias_dec)}."""
    t = pq.read_table(path, columns=["obs_code", "bias_ra_arcsec", "bias_dec_arcsec"])
    codes = t.column("obs_code").to_pylist()
    ras = t.column("bias_ra_arcsec").to_pylist()
    decs = t.column("bias_dec_arcsec").to_pylist()
    out: Dict[str, Tuple[float, float]] = {}
    for code, ra, dec in zip(codes, ras, decs):
        if code is None or ra is None or dec is None:
            continue
        out[str(code)] = (float(ra), float(dec))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Per-object observation prep (run once; shared across the 6 variants)
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class PreparedObservations:
    obs: MPCObservations
    n_obs_used: int
    n_stations: int
    arc_days: float
    n_obs_in_bias_table: int
    n_obs_efcc18_covered: int
    efcc18_corrections: np.ndarray  # (N, 2) — zero if no astcat match


def _dedupe_close_obs(obs: MPCObservations) -> MPCObservations:
    """Sequential ≤1.5 s same-station dedupe (cf. 9d7)."""
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


def prepare_observations(
    provid: str,
    client: BigQueryMPCClient,
    spacebased: set[str],
    bias_table: Dict[str, Tuple[float, float]],
    efcc18_bias_table: np.ndarray,
) -> Optional[PreparedObservations]:
    """Fetch and clean obs once; return everything the 6 variants need."""
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

    # Dedupe near-simultaneous same-station obs (FindOrb requirement)
    deduped = _dedupe_close_obs(ground)

    # Cap to most-recent MAX_OBS_PER_OBJECT to bound fit time
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

    # Pre-compute EFCC18 corrections (per-obs, in arcsec, (N,2))
    astcats = deduped.astcat.to_pylist()
    ra_deg = deduped.ra.to_numpy(zero_copy_only=False)
    dec_deg = deduped.dec.to_numpy(zero_copy_only=False)
    jd_tdb = times_mjd + 2400000.5  # MJD→JD; obstime is already TDB
    efcc_corrections = compute_efcc18_corrections(
        ra_deg, dec_deg, astcats, jd_tdb, bias_table=efcc18_bias_table
    )
    n_efcc_covered = n_observations_covered(astcats)

    return PreparedObservations(
        obs=deduped,
        n_obs_used=n_obs,
        n_stations=n_stations,
        arc_days=arc_days,
        n_obs_in_bias_table=n_in_bias_table,
        n_obs_efcc18_covered=n_efcc_covered,
        efcc18_corrections=efcc_corrections,
    )


# ────────────────────────────────────────────────────────────────────────────
# Per (object, variant) fit + gap
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class VariantResult:
    object_id: str
    designation: str           # provid (== object_id; kept for clarity)
    variant: str
    is_legacy: bool
    is_control: bool
    discrepant_in_9d7: bool
    converged: bool = False
    failure_reason: str = ""
    n_obs_used: int = 0
    n_stations: int = 0
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
    dr_over_sigma: float = float("nan")              # extra: |Δr| / σ_pos_au
    jpl_sigma_units_a: float = float("nan")
    jpl_sigma_units_e: float = float("nan")
    jpl_sigma_units_i: float = float("nan")


def _jpl_pos_sigma_au(jpl_orbit: Orbits) -> float:
    """Cartesian position 1-σ scalar (AU) from the JPL covariance.

    Same definition as 9d7: sqrt(σ_xx² + σ_yy² + σ_zz²). NaN if unavailable.
    """
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


def run_single(
    rec: VariantResult,
    variant: VariantConfig,
    prepared: PreparedObservations,
    jpl_orbit: Orbits,
    jpl_pos_sigma_au: float,
    bias_table: Dict[str, Tuple[float, float]],
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
) -> VariantResult:
    """Run one variant: build OD obs, fit, propagate, compute gap. Mutates rec."""
    try:
        catalog_debias = (
            prepared.efcc18_corrections if variant.use_efcc18 else None
        )
        bias_kw = bias_table if variant.use_bias_table else None
        od_obs = mpc_to_od_observations(
            prepared.obs,
            prevent_nans=True,
            bias_table=bias_kw,
            sigma_model=variant.sigma_model,
            bias_application=variant.bias_application,
            catalog_debias_arcsec=catalog_debias,
        )
        if od_obs is None or len(od_obs) == 0:
            rec.failure_reason = "mpc_to_od_observations returned None/empty"
            return rec

        fitted, _members = fitter.initial_fit(rec.object_id, od_obs)
        if len(fitted) == 0:
            rec.failure_reason = "FindOrb returned empty FittedOrbits"
            return rec

        rec.hold_in_reduced_chi2 = float(fitted.reduced_chi2[0].as_py())

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
        rec.failure_reason = (
            f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        )
    return rec


# ────────────────────────────────────────────────────────────────────────────
# Aggregation + reporting
# ────────────────────────────────────────────────────────────────────────────


def write_parquet(records: List[VariantResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])
    path = output_dir / "variant_comparison.parquet"
    df.to_parquet(path, index=False)
    return path


def _safe_log_ratio(baseline: float, candidate: float) -> float:
    if (
        not np.isfinite(baseline)
        or not np.isfinite(candidate)
        or baseline <= 0
        or candidate <= 0
    ):
        return float("nan")
    return float(np.log(baseline / candidate))


def write_report(
    records: List[VariantResult],
    output_dir: Path,
    controls_meta: List[Dict[str, object]],
    bias_table_path: Path,
) -> None:
    df = pd.DataFrame([asdict(r) for r in records])

    md: List[str] = []
    md.append("# Variant-matrix sweep on 9d7 discrepant set + 10 controls")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("Bead: `od_experiments_setup-cph` · Branch: `kk/od-bias-experiments`")
    md.append("")
    md.append("## Population")
    md.append("")
    md.append(
        f"- 14 discrepant objects (frozen in `scripts/od_discrepancy_test_set.py`)."
    )
    md.append(
        f"- 10 controls drawn from the 9d7 converged non-discrepant set, ranked"
        f" by smallest `|Δr/σ_r|`. Rationale: these objects' no-bias fits"
        f" were statistically consistent with JPL — they serve as negative"
        f" controls for regression detection if a principled bias variant"
        f" pushes them away from JPL."
    )
    md.append("")
    md.append("Controls picked (baseline |Δr/σ| from 9d7):")
    for c in controls_meta:
        md.append(
            f"- `{c['provid']}` — stratum={c['stratum']}, arc={c['arc_days']:.0f} d,"
            f" n_obs={c['n_obs']}, baseline Δr/σ={c['baseline_delta_r_over_sigma']:.3f}"
        )
    md.append("")
    md.append(f"Bias catalog: `{bias_table_path}`")
    md.append("")

    # Hold-in χ² sanity check
    chi2 = df.loc[df.converged & df.hold_in_reduced_chi2.notna(), "hold_in_reduced_chi2"]
    md.append("## Sanity check — uniform Veres-2017 sigma policy")
    md.append("")
    md.append(
        f"All variants share `sigma_model='veres2017'` (cph mandate). Hold-in"
        f" reduced χ² across {len(chi2)} converged fits:"
    )
    md.append(
        f"- median = {chi2.median():.3f}, p95 = {np.percentile(chi2, 95):.3f}, "
        f"max = {chi2.max():.3f}"
    )
    md.append(
        "Expected O(1) if the sigma model is realistic; the 9d7 `diag_nan=1e-9`"
        " path had this in the 1e5–1e12 range."
    )
    md.append("")

    # Per-variant counts
    md.append("## Convergence per variant")
    md.append("")
    md.append("| variant | converged / total | failures |")
    md.append("|---|---|---|")
    for v in VARIANTS:
        sub = df[df.variant == v.variant_id]
        n_total = len(sub)
        n_conv = int(sub.converged.sum())
        fails = sub.loc[~sub.converged, "object_id"].tolist()
        md.append(f"| `{v.variant_id}` | {n_conv}/{n_total} | {', '.join(fails) if fails else '—'} |")
    md.append("")

    # Per-object table for the 14 discrepant
    md.append("## Discrepant set — per-object Δr (AU) and Δr/σ by variant")
    md.append("")
    disc_df = df[~df.is_control].copy()
    pivot_dr = disc_df.pivot(index="object_id", columns="variant", values="cartesian_dr_au")
    # Preserve discrepant-set TEST_SET order
    order = [obj.provid for obj in TEST_SET if obj.provid in pivot_dr.index]
    pivot_dr = pivot_dr.reindex(order)
    pivot_ratio = disc_df.pivot(index="object_id", columns="variant", values="dr_over_sigma").reindex(order)

    md.append("Δr (AU):")
    md.append("")
    cols = [v.variant_id for v in VARIANTS]
    md.append("| object | " + " | ".join(cols) + " |")
    md.append("|---|" + "|".join("---" for _ in cols) + "|")
    for obj in order:
        row_vals = []
        for c in cols:
            val = pivot_dr.loc[obj, c] if c in pivot_dr.columns else float("nan")
            row_vals.append(f"{val:.2e}" if np.isfinite(val) else "—")
        md.append(f"| `{obj}` | " + " | ".join(row_vals) + " |")
    md.append("")

    md.append("Δr / σ_r (Cartesian position σ from JPL covariance):")
    md.append("")
    md.append("| object | " + " | ".join(cols) + " |")
    md.append("|---|" + "|".join("---" for _ in cols) + "|")
    for obj in order:
        row_vals = []
        for c in cols:
            val = pivot_ratio.loc[obj, c] if c in pivot_ratio.columns else float("nan")
            row_vals.append(f"{val:.2f}" if np.isfinite(val) else "—")
        md.append(f"| `{obj}` | " + " | ".join(row_vals) + " |")
    md.append("")

    # Aggregate variant comparison
    md.append("## Aggregate Δr/σ on the discrepant set")
    md.append("")
    md.append(
        "Two baselines reported per cph spec: "
        "(a) vs `no_bias`, (b) vs `veres_only` (the more defensible baseline"
        " given the 9d7 diag_nan history). Improvement = baseline median / variant"
        " median; values > 1 mean the variant moved the fit closer to JPL."
    )
    md.append("")
    md.append("| variant | median Δr/σ | vs no_bias | vs veres_only |")
    md.append("|---|---|---|---|")
    baseline_no_bias = disc_df.loc[
        (disc_df.variant == "no_bias") & disc_df.dr_over_sigma.notna(),
        "dr_over_sigma",
    ].median()
    baseline_veres = disc_df.loc[
        (disc_df.variant == "veres_only") & disc_df.dr_over_sigma.notna(),
        "dr_over_sigma",
    ].median()
    for v in VARIANTS:
        sub = disc_df.loc[
            (disc_df.variant == v.variant_id) & disc_df.dr_over_sigma.notna(),
            "dr_over_sigma",
        ]
        if sub.empty:
            med = float("nan")
        else:
            med = float(sub.median())
        vs_nb = baseline_no_bias / med if np.isfinite(med) and med > 0 else float("nan")
        vs_vo = baseline_veres / med if np.isfinite(med) and med > 0 else float("nan")
        legacy_tag = " (LEGACY)" if v.is_legacy else ""
        md.append(
            f"| `{v.variant_id}`{legacy_tag} | {med:.3f} | {vs_nb:.2f}× | {vs_vo:.2f}× |"
        )
    md.append("")

    # Control regression
    md.append("## Control regression check")
    md.append("")
    md.append(
        "For each control × variant, flag if Δr/σ exceeds 2× the per-object"
        " `veres_only` baseline. A regression in a control suggests the"
        " variant is over-correcting on otherwise well-fit data."
    )
    md.append("")
    ctrl_df = df[df.is_control].copy()
    veres_by_obj = ctrl_df.loc[ctrl_df.variant == "veres_only"].set_index("object_id")["dr_over_sigma"]
    flags: List[Tuple[str, str, float, float]] = []
    for v in VARIANTS:
        if v.variant_id in ("no_bias", "veres_only"):
            continue
        sub = ctrl_df.loc[ctrl_df.variant == v.variant_id].set_index("object_id")
        for obj_id, row in sub.iterrows():
            base = veres_by_obj.get(obj_id, float("nan"))
            cand = row.dr_over_sigma
            if (
                np.isfinite(base) and np.isfinite(cand)
                and base > 0 and cand > 2.0 * base
            ):
                flags.append((obj_id, v.variant_id, base, cand))
    if not flags:
        md.append("- No regressions flagged.")
    else:
        md.append("| control | variant | veres_only Δr/σ | variant Δr/σ | ratio |")
        md.append("|---|---|---|---|---|")
        for obj_id, vid, base, cand in flags:
            md.append(f"| `{obj_id}` | `{vid}` | {base:.3f} | {cand:.3f} | {cand/base:.2f}× |")
    md.append("")

    # Stacking analysis
    md.append("## Stacking analysis — v1_sigma_floor + EFCC18")
    md.append("")
    md.append(
        "Compare the stacked variant against each component in isolation."
        " 'Complementary' means stacking improves on both single corrections;"
        " 'double-correcting' means stacking degrades vs at least one"
        " component (suggests the corrections overlap and partially cancel)."
    )
    md.append("")
    def _med_dr_over_sigma(variant_id: str, subset: pd.DataFrame) -> float:
        s = subset.loc[
            (subset.variant == variant_id) & subset.dr_over_sigma.notna(),
            "dr_over_sigma",
        ]
        return float(s.median()) if not s.empty else float("nan")

    for label, subset in [("Discrepant", disc_df), ("Controls", ctrl_df)]:
        m_stack = _med_dr_over_sigma("v1_sigma_floor+efcc18", subset)
        m_v1 = _med_dr_over_sigma("v1_sigma_floor", subset)
        m_efcc = _med_dr_over_sigma("efcc18_only", subset)
        md.append(
            f"- {label}: stack median={m_stack:.3f}; v1_only={m_v1:.3f};"
            f" efcc18_only={m_efcc:.3f}"
        )
        if np.isfinite(m_stack) and np.isfinite(m_v1) and np.isfinite(m_efcc):
            if m_stack < m_v1 and m_stack < m_efcc:
                verdict = "**complementary** (stack improves on both)"
            elif m_stack > m_v1 or m_stack > m_efcc:
                verdict = "**double-correcting / overlap** (stack worse than ≥1 component)"
            else:
                verdict = "neutral"
            md.append(f"  - verdict: {verdict}")
    md.append("")

    # EFCC18 coverage diagnostic (carrying through the 9f2 YR4 finding that
    # EFCC18 covers only a small fraction of modern obs).
    cov_df = df[df.variant == "veres_only"].copy()  # any single variant row works
    md.append("## EFCC18 coverage diagnostic")
    md.append("")
    md.append("| object | n_obs | n_efcc18_covered | coverage |")
    md.append("|---|---|---|---|")
    for _, r in cov_df.sort_values("object_id").iterrows():
        cov = r.n_obs_efcc18_covered / r.n_obs_used if r.n_obs_used else 0.0
        md.append(
            f"| `{r.object_id}` | {r.n_obs_used} | {r.n_obs_efcc18_covered} | "
            f"{100*cov:.0f}% |"
        )
    md.append("")

    md.append("## Notes")
    md.append("")
    md.append(
        "- `no_bias` and `veres_only` are coded identically (both"
        " `sigma_model='veres2017'` with no bias and no EFCC18). Both rows"
        " are kept so the conventional `no_bias` anchor is present in the"
        " comparison; numerical results across the pair should match."
    )
    md.append("")

    (output_dir / "REPORT_variants.md").write_text("\n".join(md))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)

    # Build target list: 14 discrepant + N controls
    disc_provids = [obj.provid for obj in TEST_SET]
    controls = select_controls(args.ninedeseven_parquet, args.n_controls)
    ctrl_provids = [c["provid"] for c in controls]
    logger.info("Discrepant set: %d objects; controls: %d objects",
                len(disc_provids), len(ctrl_provids))
    targets: List[Tuple[str, bool, bool]] = []  # (provid, is_control, was_discrepant)
    for p in disc_provids:
        targets.append((p, False, True))
    for p in ctrl_provids:
        targets.append((p, True, False))

    # Shared resources
    spacebased = set(get_spacebased_stns())
    bias_table = load_bias_table(args.bias_table_path)
    logger.info("Loaded v1 bias table: %d stations", len(bias_table))
    efcc18_bias_table = load_efcc18_biases()
    logger.info("Loaded EFCC18 bias table")
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

    records: List[VariantResult] = []
    start = time.time()
    for i, (provid, is_control, was_discrepant) in enumerate(targets, 1):
        t0 = time.time()
        logger.info("[%d/%d] %s (%s)", i, len(targets), provid,
                    "control" if is_control else "discrepant")

        # Prepare obs once per object
        try:
            prepared = prepare_observations(
                provid, client, spacebased, bias_table, efcc18_bias_table
            )
        except Exception as e:
            logger.exception("%s: observation prep raised: %s", provid, e)
            prepared = None
        if prepared is None:
            for v in VARIANTS:
                r = VariantResult(
                    object_id=provid, designation=provid,
                    variant=v.variant_id, is_legacy=v.is_legacy,
                    is_control=is_control, discrepant_in_9d7=was_discrepant,
                    failure_reason="observation prep returned None",
                )
                records.append(r)
            continue

        # Fetch JPL orbit once per object
        try:
            jpl_orbit = fetch_jpl_orbit(provid)
        except Exception as e:
            logger.exception("%s: fetch_jpl_orbit raised: %s", provid, e)
            jpl_orbit = None
        if jpl_orbit is None or len(jpl_orbit) == 0:
            for v in VARIANTS:
                r = VariantResult(
                    object_id=provid, designation=provid,
                    variant=v.variant_id, is_legacy=v.is_legacy,
                    is_control=is_control, discrepant_in_9d7=was_discrepant,
                    n_obs_used=prepared.n_obs_used,
                    n_stations=prepared.n_stations,
                    arc_days=prepared.arc_days,
                    n_obs_in_bias_table=prepared.n_obs_in_bias_table,
                    n_obs_efcc18_covered=prepared.n_obs_efcc18_covered,
                    failure_reason="JPL/SBDB fetch failed",
                )
                records.append(r)
            continue
        jpl_sigma = _jpl_pos_sigma_au(jpl_orbit)

        # Run each of the 6 variants
        for v in VARIANTS:
            rec = VariantResult(
                object_id=provid, designation=provid,
                variant=v.variant_id, is_legacy=v.is_legacy,
                is_control=is_control, discrepant_in_9d7=was_discrepant,
                n_obs_used=prepared.n_obs_used,
                n_stations=prepared.n_stations,
                arc_days=prepared.arc_days,
                n_obs_in_bias_table=prepared.n_obs_in_bias_table,
                n_obs_efcc18_covered=prepared.n_obs_efcc18_covered,
            )
            rec = run_single(
                rec, v, prepared, jpl_orbit, jpl_sigma,
                bias_table, fitter, propagator,
            )
            records.append(rec)
            logger.info(
                "  variant=%-22s converged=%s Δr=%s Δr/σ=%s",
                v.variant_id, rec.converged,
                f"{rec.cartesian_dr_au:.2e}" if np.isfinite(rec.cartesian_dr_au) else "—",
                f"{rec.dr_over_sigma:.2f}" if np.isfinite(rec.dr_over_sigma) else "—",
            )

        logger.info("[%d/%d] %s done in %.1fs (cum %.1f min)",
                    i, len(targets), provid, time.time() - t0,
                    (time.time() - start) / 60.0)

        # Incremental safety write every 5 objects
        if i % 5 == 0:
            write_parquet(records, args.output_dir)
            write_report(records, args.output_dir, controls, args.bias_table_path)

    logger.info("Total runtime: %.1f min", (time.time() - start) / 60.0)
    parq = write_parquet(records, args.output_dir)
    logger.info("Wrote %d rows to %s", len(records), parq)
    write_report(records, args.output_dir, controls, args.bias_table_path)
    logger.info("Wrote REPORT_variants.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
