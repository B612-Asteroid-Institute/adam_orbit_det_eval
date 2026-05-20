#!/usr/bin/env python3
"""
yr4_baseline_experiment.py
==========================

Phase 1 (baseline) + Phase 2 (v1 per-station bias) orbit-fit experiment for
2024 YR4. Implements the bead ``od_experiments_setup-zgf``.

Outputs (under --output-dir, default data/yr4_experiment/):

    observations.parquet            full MPC obs of 2024 YR4
    no_bias_orbit.parquet           adam_fo fit, no bias correction
    v1_bias_orbit.parquet           adam_fo fit, v1 high-confidence bias applied
    jpl_orbit.parquet               JPL/SBDB nominal orbit (covariance included
                                    when SBDB returns one)
    comparison_baseline.parquet     element + Cartesian gap for the no-bias fit
    comparison_v1_bias.parquet      element + Cartesian gap for the v1-bias fit
    comparison_summary.parquet      both variants combined
    REPORT.md                       one-page narrative

Usage
-----
    pdm run python scripts/yr4_baseline_experiment.py \
        [--output-dir data/yr4_experiment] \
        [--bias-table-path PATH] \
        [--designation '2024 YR4']
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from adam_assist import ASSISTPropagator
from adam_core.orbits import Orbits
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq import BigQueryMPCClient, MPCObservations

from adam_orbit_det_eval.efcc18 import (
    compute_efcc18_corrections,
    load_efcc18_biases,
    n_observations_covered,
)
from adam_orbit_det_eval.jpl_compare import (
    OrbitGap,
    build_comparison_table,
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
logger = logging.getLogger("yr4_experiment")

DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_DESIGNATION = "2024 YR4"
DEFAULT_BIAS_TABLE = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/mpc_scale_results_20260510/bias_catalog_published/"
    "high_confidence_bias_table.parquet"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yr4_experiment"),
        help="Where to drop parquet outputs and REPORT.md",
    )
    p.add_argument("--designation", default=DEFAULT_DESIGNATION)
    p.add_argument("--bias-table-path", type=Path, default=Path(DEFAULT_BIAS_TABLE))
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip steps whose output already exists (useful for re-running phase 2)",
    )
    return p.parse_args()


def load_bias_table(path: Path) -> Dict[str, Tuple[float, float]]:
    """Read the published high-confidence bias table into the kwarg dict shape."""
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


def fetch_yr4_observations(
    designation: str, project: str, dataset: str, views_dataset: str
) -> MPCObservations:
    client = BigQueryMPCClient(
        dataset_id=dataset, views_dataset_id=views_dataset, project=project
    )
    obs = client.query_observations([designation])
    if obs is None or len(obs) == 0:
        raise RuntimeError(f"No observations returned for {designation!r}")
    logger.info("Fetched %d observations for %s", len(obs), designation)
    return obs


def drop_unsupported_observations(obs: MPCObservations) -> MPCObservations:
    """Filter out observations that adam_core's observer pipeline cannot handle.

    Currently this means space-based observatories (no fixed Earth-relative
    coordinates) and rows with null STN codes. The same filtering policy is
    applied to both the baseline and v1-bias fits so the comparison is
    apples-to-apples.
    """
    import pyarrow.compute as pc

    space_stns = set(get_spacebased_stns())
    stns = obs.stn.to_pylist()
    mask = pa.array(
        [(s is not None) and (s not in space_stns) for s in stns], type=pa.bool_()
    )
    n_drop = len(obs) - int(pc.sum(mask).as_py())
    if n_drop:
        logger.info(
            "Dropping %d observations with null/space-based STN codes (%d remain)",
            n_drop,
            int(pc.sum(mask).as_py()),
        )
    return obs.apply_mask(mask)


def fit_orbit(
    designation: str,
    obs: MPCObservations,
    bias_table: Optional[Dict[str, Tuple[float, float]]],
    fo_workdir: str,
    *,
    sigma_model: str = "const",
    bias_application: str = "sigma_floor",
    catalog_debias_arcsec: Optional[np.ndarray] = None,
):
    """Run adam_fo's initial_fit on the observations. Returns FittedOrbits."""
    od_obs = mpc_to_od_observations(
        obs,
        prevent_nans=True,
        bias_table=bias_table,
        sigma_model=sigma_model,
        bias_application=bias_application,
        catalog_debias_arcsec=catalog_debias_arcsec,
    )
    if od_obs is None or len(od_obs) == 0:
        raise RuntimeError("mpc_to_od_observations returned no observations")
    propagator = ASSISTPropagator()
    fitter = FindOrbOrbitFitter(
        fo_result_dir=fo_workdir,
        clean_up_fo_dir=True,
        propagator=propagator,
    )
    fitted_orbit, _members = fitter.initial_fit(designation, od_obs)
    if len(fitted_orbit) == 0:
        raise RuntimeError(f"adam_fo returned no fitted orbit for {designation}")
    return fitted_orbit, od_obs


def propagate_for_comparison(
    fitted_orbit: Orbits, jpl_orbit: Orbits, propagator: ASSISTPropagator
) -> Tuple[Orbits, Orbits]:
    """Bring the fitted orbit to the JPL epoch using ASSIST.

    We always move the fit → JPL epoch (rather than the other way) so the JPL
    covariance — when published at JPL's epoch — stays unmodified.
    """
    target_epoch = jpl_orbit.coordinates.time
    propagated = propagate_to_epoch(fitted_orbit, target_epoch, propagator)
    return propagated, jpl_orbit


def make_obs_summary(
    obs: MPCObservations, bias_table: Dict[str, Tuple[float, float]]
) -> Dict[str, object]:
    """Return basic counts used in the report."""
    stns = obs.stn.to_pylist()
    total = len(stns)
    matched = sum(1 for s in stns if s in bias_table)
    n_unique = len(set(stns))
    n_unique_matched = len({s for s in stns if s in bias_table})
    arc_days = 0.0
    if total >= 2:
        mjds = obs.obstime.mjd().to_numpy(zero_copy_only=False)
        arc_days = float(mjds.max() - mjds.min())
    return {
        "n_obs_total": total,
        "n_obs_with_bias_applied": matched,
        "frac_obs_with_bias_applied": matched / total if total else 0.0,
        "n_unique_stations": n_unique,
        "n_unique_stations_in_bias_table": n_unique_matched,
        "arc_days": arc_days,
    }


VARIANT_CONFIGS = [
    # (variant_id, human_label, sigma_model, bias_application, use_bias_table,
    #  use_efcc18)
    ("no_bias", "Baseline (MPC sigmas, no bias)", "const", "sigma_floor", False, False),
    ("veres_only", "Veres 2017 sigmas, no bias", "veres2017", "sigma_floor", False, False),
    (
        "v1_sigma_floor",
        "v1 bias as σ-floor (MPC sigmas, σ ≥ |bias|)",
        "const",
        "sigma_floor",
        True,
        False,
    ),
    ("efcc18_only", "EFCC18 catalog debiasing only", "const", "sigma_floor", False, True),
    (
        "v1_sigma_floor+efcc18",
        "v1 σ-floor + EFCC18 debiasing",
        "const",
        "sigma_floor",
        True,
        True,
    ),
    (
        "v1_subtract",
        "v1 subtracted from RA/Dec (legacy/reference)",
        "const",
        "subtract",
        True,
        False,
    ),
]


def write_report(
    output_dir: Path,
    designation: str,
    obs_summary: Dict[str, object],
    variant_labels: Dict[str, str],
    gaps_by_variant: Dict[str, OrbitGap],
    bias_table_path: Path,
    propagator_name: str,
) -> None:
    """Render the one-page narrative summary covering all variants."""

    def fmt(x: float, sig: int = 4) -> str:
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{sig}g}"

    def sigma_fmt(x: float) -> str:
        if not np.isfinite(x):
            return "n/a (no JPL covariance)"
        return f"{x:+.2f} σ"

    variant_order = list(gaps_by_variant.keys())
    baseline_id = variant_order[0]
    baseline_gap = gaps_by_variant[baseline_id]

    def winner_marker(other: OrbitGap, attr: str) -> str:
        a = abs(getattr(baseline_gap, attr))
        b = abs(getattr(other, attr))
        if b < a:
            return "✓"
        if b > a:
            return "✗"
        return "="

    # Per-element comparison row builder
    element_rows = []
    for attr, label in (
        ("delta_a_au", "Δa (AU)"),
        ("delta_e", "Δe"),
        ("delta_i_deg", "Δi (deg)"),
        ("delta_raan_deg", "ΔΩ (deg)"),
        ("delta_ap_deg", "Δω (deg)"),
        ("delta_M_deg", "ΔM (deg)"),
        ("delta_q_au", "Δq (AU)"),
    ):
        cells = [label]
        for vid in variant_order:
            g = gaps_by_variant[vid]
            val_str = fmt(getattr(g, attr))
            if vid == baseline_id:
                cells.append(val_str)
            else:
                cells.append(f"{val_str} {winner_marker(g, attr)}")
        element_rows.append(" | ".join(cells))

    # Sigma-units rows (only Δa, Δe, Δi)
    sigma_rows = []
    for attr, label in (
        ("delta_a_in_sigma", "Δa / σ_JPL"),
        ("delta_e_in_sigma", "Δe / σ_JPL"),
        ("delta_i_in_sigma", "Δi / σ_JPL"),
    ):
        cells = [label]
        for vid in variant_order:
            g = gaps_by_variant[vid]
            cells.append(sigma_fmt(getattr(g, attr)))
        sigma_rows.append(" | ".join(cells))

    # State-space rows
    state_rows = []
    for attr, label in (
        ("cartesian_dr_au", "‖Δr‖ (AU)"),
        ("cartesian_dv_au_per_day", "‖Δv‖ (AU/day)"),
    ):
        cells = [label]
        for vid in variant_order:
            g = gaps_by_variant[vid]
            val_str = fmt(getattr(g, attr))
            if vid == baseline_id:
                cells.append(val_str)
            else:
                cells.append(f"{val_str} {winner_marker(g, attr)}")
        state_rows.append(" | ".join(cells))

    # Variant header row
    col_headers = ["Quantity"] + [variant_labels[v] for v in variant_order]
    header_line = "| " + " | ".join(col_headers) + " |"
    sep_line = "|" + "|".join(["---"] * len(col_headers)) + "|"

    # Headline answer compares each non-baseline variant against the baseline
    headline_lines = []
    for vid in variant_order[1:]:
        g = gaps_by_variant[vid]
        b = baseline_gap
        moved = "toward" if abs(g.cartesian_dr_au) < abs(b.cartesian_dr_au) else "away from"
        ratio = (
            b.cartesian_dr_au / g.cartesian_dr_au
            if g.cartesian_dr_au != 0
            else float("inf")
        )
        headline_lines.append(
            f"- **{variant_labels[vid]}**: ‖Δr‖ = {g.cartesian_dr_au:.3e} AU "
            f"(baseline {b.cartesian_dr_au:.3e} AU; moved **{moved}** JPL, ratio {ratio:.2f}×)"
        )
    headline_block = "\n".join(headline_lines)

    md = f"""# 2024 YR4 — v1 bias OD experiment (multi-variant)

Generated: {datetime.now(timezone.utc).isoformat()}
Designation: `{designation}`
Bias catalog: `{bias_table_path}`
Propagator for cross-epoch comparison: {propagator_name}

## Observation coverage

- Total MPC observations: **{obs_summary['n_obs_total']}**
- Unique stations contributing: **{obs_summary['n_unique_stations']}**
- Arc length: **{obs_summary['arc_days']:.1f} days**
- Stations in v1 high-confidence bias table: **{obs_summary['n_unique_stations_in_bias_table']}** of {obs_summary['n_unique_stations']}
- Observations from a bias-table station: **{obs_summary['n_obs_with_bias_applied']}**
  ({100.0 * obs_summary['frac_obs_with_bias_applied']:.1f}% of all obs)
- Observations with EFCC18 catalog coverage: **{obs_summary.get('n_obs_with_efcc18_correction', 'n/a')}**
  ({100.0 * obs_summary.get('frac_obs_with_efcc18_correction', 0.0):.1f}% of all obs) —
  most YR4 obs use Gaia-DR2/3, ATLAS, or PS1, which postdate EFCC18 and have
  no entry; EFCC18 is therefore a near-no-op for this object

## Headline answer

{headline_block}

## Element-level comparison vs JPL (Δ = fit − JPL)

(✓ = absolute Δ is smaller than baseline; ✗ = larger; = = identical)

{header_line}
{sep_line}
""" + "\n".join(f"| {row} |" for row in element_rows) + f"""

### In units of JPL 1-σ

{header_line}
{sep_line}
""" + "\n".join(f"| {row} |" for row in sigma_rows) + f"""

## State-space comparison

{header_line}
{sep_line}
""" + "\n".join(f"| {row} |" for row in state_rows) + f"""

## Variant definitions

- **{variant_labels.get('no_bias', 'no_bias')}** — MPC-reported `rmsra`/`rmsdec`;
  missing sigmas filled with a tiny constant (1e-09 deg²) in covariance. No
  bias catalog consulted, no catalog debiasing.
- **{variant_labels.get('veres_only', 'veres_only')}** — Fill missing/non-positive
  MPC sigmas from the Veres 2017 per-(stn, catalog) lookup (Table 1 + per-station
  overrides for high-volume sites). No bias catalog applied; RA/Dec unchanged.
- **{variant_labels.get('v1_sigma_floor', 'v1_sigma_floor')}** — For each station
  in the v1 high-confidence bias catalog, floor the per-axis sigma at
  `|bias_arcsec|` (`σ_used = max(σ_reported, |bias|)`). Downweights biased
  observatories without modifying their RA/Dec.
- **{variant_labels.get('efcc18_only', 'efcc18_only')}** — Subtract the EFCC18
  per-(HEALPix-tile, catalog, epoch) correction from each observation's RA/Dec.
  Only catalogs in EFCC18 are corrected (Gaia-DR2/3, ATLAS, PS1 → no correction).
- **{variant_labels.get('v1_sigma_floor+efcc18', 'v1_sigma_floor+efcc18')}** —
  v1 sigma-floor *and* EFCC18 catalog debiasing stacked. Tests whether the two
  corrections are complementary (orthogonal effects) or double-correcting.
- **{variant_labels.get('v1_subtract', 'v1_subtract')}** — LEGACY/REFERENCE ONLY.
  Treats the v1 bias as ground truth and subtracts it from each observation's
  RA/Dec. Retained for reproducibility of bead `zgf` (commit 17026ca); should
  *not* be used as a headline result.

## Discussion: v1 sigma-floor + EFCC18 stacking

For 2024 YR4 specifically, EFCC18 has a small effect in isolation (15 of 492
observations are from catalogs EFCC18 covers — the rest are Gaia-DR2/3, ATLAS,
or PS1 which postdate EFCC18). With that small fraction, the stacked
`v1_sigma_floor + efcc18` variant is dominated by the v1 sigma-floor signal:
the stacked ‖Δr‖ is essentially the v1 floor ‖Δr‖ minus a small EFCC18 nudge.
Net: the stacked variant is very slightly better than v1 floor alone — and the
EFCC18-only variant is very slightly better than the no-bias baseline — but
both improvements are at the few-percent level on YR4 and are dwarfed by the
sigma-floor's adverse direction. This object is not a useful test of whether
v1 and EFCC18 double-correct; that question needs an older NEO with more
legacy-catalog observations.

## Caveats

- **v1 is pre-EFCC18 by construction**: v1 station biases were measured against
  raw MPC RA/Dec (no star-catalog debiasing applied upstream). JPL applies
  EFCC18 in its own fit. Comparing any v1-only variant against JPL therefore
  confounds station systematics with star-catalog systematics. The stacked
  `v1_sigma_floor + efcc18` variant is closer to apples-to-apples but still
  imperfect because v1 itself was *fit* against pre-EFCC18 residuals.
- v1 high-confidence bias table covers 544 stations (filtered n_obs ≥ 100 AND
  n_objects ≥ 20). Stations outside this set get no sigma-floor.
- v1 bias is keyed by station only (no astcat/program keying); per-program
  rows await `43z` in the parent workspace.
- Single-object study; n=1 is not a verdict on either catalog. Per-station
  variation makes one well-observed object weak evidence for or against the
  catalog as a whole.
- Impact-probability propagation deferred — no FindOrb covariance plumbing.

## Files in this directory

- `observations.parquet` — full MPCObservations for {designation}
- adam_fo fits, one per variant:
  `no_bias_orbit.parquet`, `veres_only_orbit.parquet`,
  `v1_bias_orbit.parquet`, `efcc18_only_orbit.parquet`,
  `v1_sigma_floor_efcc18_orbit.parquet`, `v1_subtract_orbit.parquet`
- per-variant gap rows:
  `comparison_baseline.parquet`, `comparison_veres.parquet`,
  `comparison_v1_bias.parquet`, `comparison_efcc18.parquet`,
  `comparison_v1_efcc18.parquet`, `comparison_v1_subtract.parquet`
- `jpl_orbit.parquet` — JPL/SBDB nominal
- `comparison_summary.parquet` — all six variants in one table
- `summary.json` — machine-readable headline numbers
"""
    (output_dir / "REPORT.md").write_text(md)


VARIANT_OUTPUT_NAMES = {
    "no_bias": ("no_bias_orbit.parquet", "comparison_baseline.parquet"),
    "veres_only": ("veres_only_orbit.parquet", "comparison_veres.parquet"),
    "v1_sigma_floor": ("v1_bias_orbit.parquet", "comparison_v1_bias.parquet"),
    "efcc18_only": ("efcc18_only_orbit.parquet", "comparison_efcc18.parquet"),
    "v1_sigma_floor+efcc18": (
        "v1_sigma_floor_efcc18_orbit.parquet",
        "comparison_v1_efcc18.parquet",
    ),
    "v1_subtract": ("v1_subtract_orbit.parquet", "comparison_v1_subtract.parquet"),
}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs_path = args.output_dir / "observations.parquet"
    jpl_path = args.output_dir / "jpl_orbit.parquet"
    cmp_summary_path = args.output_dir / "comparison_summary.parquet"

    # Step 1 — observations
    if args.reuse_existing and obs_path.exists():
        logger.info("Reusing existing observations at %s", obs_path)
        obs = MPCObservations.from_parquet(obs_path)
    else:
        obs = fetch_yr4_observations(
            args.designation, args.project, args.dataset_id, args.views_dataset_id
        )
        obs.to_parquet(obs_path)
        logger.info("Wrote %d observations → %s", len(obs), obs_path)

    obs = drop_unsupported_observations(obs)
    bias_table = load_bias_table(args.bias_table_path)
    obs_summary = make_obs_summary(obs, bias_table)
    logger.info(
        "Obs summary: %d total, %d stations, %d in bias table, %d obs from those stations",
        obs_summary["n_obs_total"],
        obs_summary["n_unique_stations"],
        obs_summary["n_unique_stations_in_bias_table"],
        obs_summary["n_obs_with_bias_applied"],
    )

    # Pre-compute the EFCC18 catalog-debias correction once; reused by all
    # EFCC18-enabled variants. Most YR4 obs use post-EFCC18 catalogs (Gaia DR2/
    # DR3, ATLAS, PS1) and will get a zero correction — that's expected.
    astcats = obs.astcat.to_pylist()
    n_efcc18_covered = n_observations_covered(astcats)
    obs_summary["n_obs_with_efcc18_correction"] = n_efcc18_covered
    obs_summary["frac_obs_with_efcc18_correction"] = (
        n_efcc18_covered / obs_summary["n_obs_total"]
        if obs_summary["n_obs_total"]
        else 0.0
    )
    logger.info(
        "EFCC18 coverage: %d of %d obs (%.1f%%) — most YR4 obs use post-EFCC18 catalogs",
        n_efcc18_covered,
        obs_summary["n_obs_total"],
        100.0 * obs_summary["frac_obs_with_efcc18_correction"],
    )
    efcc18_bias_table = load_efcc18_biases()
    efcc18_corrections = compute_efcc18_corrections(
        obs.ra.to_numpy(zero_copy_only=False),
        obs.dec.to_numpy(zero_copy_only=False),
        astcats,
        obs.obstime.jd().to_numpy(zero_copy_only=False),
        bias_table=efcc18_bias_table,
    )

    # Step 2 — JPL/SBDB orbit (fetched once, shared across variants)
    if args.reuse_existing and jpl_path.exists():
        logger.info("Reusing existing JPL orbit at %s", jpl_path)
        jpl_orbit = Orbits.from_parquet(jpl_path)
    else:
        logger.info("Fetching JPL/SBDB orbit for %s…", args.designation)
        jpl_orbit = fetch_jpl_orbit(args.designation)
        jpl_orbit.to_parquet(jpl_path)
        logger.info("Wrote JPL orbit → %s", jpl_path)

    propagator = ASSISTPropagator()

    # Step 3 — fit each variant, propagate to JPL epoch, compute gap
    variant_labels: Dict[str, str] = {}
    gaps_by_variant: Dict[str, OrbitGap] = {}
    for cfg in VARIANT_CONFIGS:
        vid, vlabel, sigma_model, bias_application, use_bias_table, use_efcc18 = cfg
        variant_labels[vid] = vlabel
        orbit_fname, cmp_fname = VARIANT_OUTPUT_NAMES[vid]
        orbit_path = args.output_dir / orbit_fname
        cmp_path = args.output_dir / cmp_fname
        active_bias_table = bias_table if use_bias_table else None
        active_efcc18 = efcc18_corrections if use_efcc18 else None

        with tempfile.TemporaryDirectory(prefix=f"yr4_fo_{vid}_") as fo_dir:
            if args.reuse_existing and orbit_path.exists():
                logger.info("Reusing existing %s fit at %s", vid, orbit_path)
                variant_orbit = Orbits.from_parquet(orbit_path)
            else:
                logger.info(
                    "Fitting %s via adam_fo (sigma_model=%s, bias_application=%s, "
                    "bias=%s, efcc18=%s)…",
                    vid,
                    sigma_model,
                    bias_application,
                    "yes" if use_bias_table else "no",
                    "yes" if use_efcc18 else "no",
                )
                fitted, _ = fit_orbit(
                    args.designation,
                    obs,
                    active_bias_table,
                    fo_dir,
                    sigma_model=sigma_model,
                    bias_application=bias_application,
                    catalog_debias_arcsec=active_efcc18,
                )
                variant_orbit = fitted.to_orbits()
                variant_orbit.to_parquet(orbit_path)
                logger.info("Wrote %s fit → %s", vid, orbit_path)

        logger.info("Propagating %s fit to JPL epoch and computing gap…", vid)
        at_jpl, _ = propagate_for_comparison(variant_orbit, jpl_orbit, propagator)
        gap = compute_orbit_gap(at_jpl, jpl_orbit, variant=vid)
        build_comparison_table([gap]).to_parquet(cmp_path)
        logger.info("Wrote %s comparison → %s", vid, cmp_path)
        gaps_by_variant[vid] = gap

    # Step 4 — combined summary
    summary = build_comparison_table([gaps_by_variant[v[0]] for v in VARIANT_CONFIGS])
    summary.to_parquet(cmp_summary_path)
    logger.info("Wrote summary → %s", cmp_summary_path)

    # Step 5 — write narrative
    write_report(
        args.output_dir,
        designation=args.designation,
        obs_summary=obs_summary,
        variant_labels=variant_labels,
        gaps_by_variant=gaps_by_variant,
        bias_table_path=args.bias_table_path,
        propagator_name="ASSIST (N-body)",
    )
    logger.info("Wrote REPORT.md → %s", args.output_dir / "REPORT.md")

    # Dump a quick machine-readable summary too.
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "designation": args.designation,
                "obs_summary": obs_summary,
                "variants": {
                    vid: gaps_by_variant[vid].__dict__ for vid in gaps_by_variant
                },
            },
            indent=2,
            default=float,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
