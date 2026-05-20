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

# The wheel of ``adam_fo`` installed via PDM (``adam_fo>=0.x``) is older than
# the symlinked source tree in this workspace and does not export
# ``FindOrbOrbitFitter``. Inject the symlinked source onto ``sys.path`` ahead
# of site-packages so we pick up the newer wrapper. Match the convention used
# by ``build_od_discrepancy_population.py``.
_ADAM_FO_SRC = "/Users/kathleenkiker/od_experiments_setup/adam_fo/src"
if _ADAM_FO_SRC not in sys.path:
    sys.path.insert(0, _ADAM_FO_SRC)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from adam_assist import ASSISTPropagator  # noqa: E402
from adam_core.orbits import Orbits  # noqa: E402
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter  # noqa: E402
from mpcq import BigQueryMPCClient, MPCObservations  # noqa: E402

from adam_orbit_det_eval.jpl_compare import (  # noqa: E402
    OrbitGap,
    build_comparison_table,
    compute_orbit_gap,
    fetch_jpl_orbit,
    propagate_to_epoch,
)
from adam_orbit_det_eval.utils import (  # noqa: E402
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
):
    """Run adam_fo's initial_fit on the observations. Returns FittedOrbits."""
    od_obs = mpc_to_od_observations(obs, prevent_nans=True, bias_table=bias_table)
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


def write_report(
    output_dir: Path,
    designation: str,
    obs_summary: Dict[str, object],
    baseline_gap: OrbitGap,
    v1_gap: OrbitGap,
    bias_table_path: Path,
    propagator_name: str,
) -> None:
    """Render the one-page narrative summary."""

    def fmt(x: float, sig: int = 4) -> str:
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{sig}g}"

    def sigma_fmt(x: float) -> str:
        if not np.isfinite(x):
            return "n/a (no JPL covariance)"
        return f"{x:+.2f} σ"

    moved_toward = abs(v1_gap.cartesian_dr_au) < abs(baseline_gap.cartesian_dr_au)
    direction = "toward" if moved_toward else "away from"
    delta_dr = v1_gap.cartesian_dr_au - baseline_gap.cartesian_dr_au

    md = f"""# 2024 YR4 — v1 bias-correction OD experiment

Generated: {datetime.now(timezone.utc).isoformat()}
Designation: `{designation}`
Bias catalog: `{bias_table_path}`
Propagator for cross-epoch comparison: {propagator_name}

## Observation coverage

- Total MPC observations: **{obs_summary['n_obs_total']}**
- Unique stations contributing: **{obs_summary['n_unique_stations']}**
- Arc length: **{obs_summary['arc_days']:.1f} days**
- Stations in v1 high-confidence bias table: **{obs_summary['n_unique_stations_in_bias_table']}** of {obs_summary['n_unique_stations']}
- Observations receiving a bias correction: **{obs_summary['n_obs_with_bias_applied']}**
  ({100.0 * obs_summary['frac_obs_with_bias_applied']:.1f}% of all obs)

## Headline answer

Applying v1 station biases moved the YR4 fit **{direction}** JPL/SBDB by
**{abs(delta_dr):.3e} AU** in Cartesian state-space distance at the JPL epoch
(baseline dr = {baseline_gap.cartesian_dr_au:.3e} AU → v1 dr = {v1_gap.cartesian_dr_au:.3e} AU).

## Element-level comparison vs JPL (Δ = fit − JPL)

| Quantity | Baseline (no bias) | v1 bias applied | Better? |
|---|---|---|---|
| Δa (AU)         | {fmt(baseline_gap.delta_a_au)}      | {fmt(v1_gap.delta_a_au)}      | {"yes" if abs(v1_gap.delta_a_au) < abs(baseline_gap.delta_a_au) else "no"} |
| Δe              | {fmt(baseline_gap.delta_e)}         | {fmt(v1_gap.delta_e)}         | {"yes" if abs(v1_gap.delta_e) < abs(baseline_gap.delta_e) else "no"} |
| Δi (deg)        | {fmt(baseline_gap.delta_i_deg)}     | {fmt(v1_gap.delta_i_deg)}     | {"yes" if abs(v1_gap.delta_i_deg) < abs(baseline_gap.delta_i_deg) else "no"} |
| ΔΩ (deg)        | {fmt(baseline_gap.delta_raan_deg)}  | {fmt(v1_gap.delta_raan_deg)}  | {"yes" if abs(v1_gap.delta_raan_deg) < abs(baseline_gap.delta_raan_deg) else "no"} |
| Δω (deg)        | {fmt(baseline_gap.delta_ap_deg)}    | {fmt(v1_gap.delta_ap_deg)}    | {"yes" if abs(v1_gap.delta_ap_deg) < abs(baseline_gap.delta_ap_deg) else "no"} |
| ΔM (deg)        | {fmt(baseline_gap.delta_M_deg)}     | {fmt(v1_gap.delta_M_deg)}     | {"yes" if abs(v1_gap.delta_M_deg) < abs(baseline_gap.delta_M_deg) else "no"} |
| Δq (AU)         | {fmt(baseline_gap.delta_q_au)}      | {fmt(v1_gap.delta_q_au)}      | {"yes" if abs(v1_gap.delta_q_au) < abs(baseline_gap.delta_q_au) else "no"} |

### In units of JPL 1-σ

| Element | Baseline | v1 bias |
|---|---|---|
| Δa | {sigma_fmt(baseline_gap.delta_a_in_sigma)} | {sigma_fmt(v1_gap.delta_a_in_sigma)} |
| Δe | {sigma_fmt(baseline_gap.delta_e_in_sigma)} | {sigma_fmt(v1_gap.delta_e_in_sigma)} |
| Δi | {sigma_fmt(baseline_gap.delta_i_in_sigma)} | {sigma_fmt(v1_gap.delta_i_in_sigma)} |

## State-space comparison

| Quantity | Baseline | v1 bias |
|---|---|---|
| ‖Δr‖ (AU)            | {fmt(baseline_gap.cartesian_dr_au)}      | {fmt(v1_gap.cartesian_dr_au)} |
| ‖Δv‖ (AU/day)        | {fmt(baseline_gap.cartesian_dv_au_per_day)} | {fmt(v1_gap.cartesian_dv_au_per_day)} |

## Caveats

- **v1 is pre-EFCC18**: JPL applies the EFCC18 star-catalog debiasing upstream
  of its fit; v1 station biases here are measured against MPC RA/Dec with no
  star-catalog correction. As a result, comparing a v1-corrected fit against
  JPL is **not** a clean test of v1's quality — it confounds star-catalog and
  station systematics. Use this as an indicative signal only; pair with an
  EFCC18-stripped JPL fit (out of scope for this bead) before reading much
  into the magnitude.
- v1 high-confidence bias table contains 544 stations (filtered n_obs ≥ 100
  AND n_objects ≥ 20). Stations outside this set get no correction here.
- Bias keyed by station only (no astcat/program keying); per-program rows
  await `43z` in the parent workspace.
- Impact-probability propagation is deferred (no covariance plumbing for
  the FindOrb fit in this bead).

## Files in this directory

- `observations.parquet` — full MPCObservations for {designation}
- `no_bias_orbit.parquet` / `v1_bias_orbit.parquet` — adam_fo fits
- `jpl_orbit.parquet` — JPL/SBDB nominal
- `comparison_baseline.parquet` / `comparison_v1_bias.parquet` — gap rows
- `comparison_summary.parquet` — both variants combined
"""
    (output_dir / "REPORT.md").write_text(md)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs_path = args.output_dir / "observations.parquet"
    no_bias_path = args.output_dir / "no_bias_orbit.parquet"
    v1_bias_path = args.output_dir / "v1_bias_orbit.parquet"
    jpl_path = args.output_dir / "jpl_orbit.parquet"
    cmp_base_path = args.output_dir / "comparison_baseline.parquet"
    cmp_v1_path = args.output_dir / "comparison_v1_bias.parquet"
    cmp_summary_path = args.output_dir / "comparison_summary.parquet"

    # Step 1 — observations
    if args.reuse_existing and obs_path.exists():
        logger.info("Reusing existing observations at %s", obs_path)
        obs = MPCObservations.from_parquet(obs_path)
    else:
        obs = fetch_yr4_observations(
            args.designation, args.project, args.dataset_id, args.views_dataset_id
        )
        # Persist the full fetch (including space-based) for record-keeping;
        # only the supported subset feeds the fitter below.
        obs.to_parquet(obs_path)
        logger.info("Wrote %d observations → %s", len(obs), obs_path)

    obs = drop_unsupported_observations(obs)
    bias_table = load_bias_table(args.bias_table_path)
    obs_summary = make_obs_summary(obs, bias_table)
    logger.info(
        "Obs summary: %d total, %d stations, %d in bias table, %d obs corrected",
        obs_summary["n_obs_total"],
        obs_summary["n_unique_stations"],
        obs_summary["n_unique_stations_in_bias_table"],
        obs_summary["n_obs_with_bias_applied"],
    )

    # Step 2 — fit no-bias orbit
    with tempfile.TemporaryDirectory(prefix="yr4_fo_") as fo_dir:
        if args.reuse_existing and no_bias_path.exists():
            logger.info("Reusing existing no-bias orbit at %s", no_bias_path)
            no_bias_fit = Orbits.from_parquet(no_bias_path)
        else:
            logger.info("Fitting no-bias orbit via adam_fo…")
            fitted_no_bias, _ = fit_orbit(args.designation, obs, None, fo_dir)
            # FittedOrbits → Orbits for saving; covariance not surfaced by adam_fo.
            no_bias_fit = fitted_no_bias.to_orbits()
            no_bias_fit.to_parquet(no_bias_path)
            logger.info("Wrote no-bias fit → %s", no_bias_path)

    # Step 3 — JPL/SBDB orbit
    if args.reuse_existing and jpl_path.exists():
        logger.info("Reusing existing JPL orbit at %s", jpl_path)
        jpl_orbit = Orbits.from_parquet(jpl_path)
    else:
        logger.info("Fetching JPL/SBDB orbit for %s…", args.designation)
        jpl_orbit = fetch_jpl_orbit(args.designation)
        jpl_orbit.to_parquet(jpl_path)
        logger.info("Wrote JPL orbit → %s", jpl_path)

    propagator = ASSISTPropagator()

    # Step 4 — propagate baseline fit to JPL epoch and compute gap
    logger.info("Propagating no-bias fit to JPL epoch and computing gap…")
    no_bias_at_jpl, _ = propagate_for_comparison(no_bias_fit, jpl_orbit, propagator)
    baseline_gap = compute_orbit_gap(no_bias_at_jpl, jpl_orbit, variant="no_bias")
    build_comparison_table([baseline_gap]).to_parquet(cmp_base_path)
    logger.info("Wrote baseline comparison → %s", cmp_base_path)

    # Step 5 — fit v1-bias orbit
    with tempfile.TemporaryDirectory(prefix="yr4_fo_v1_") as fo_dir:
        if args.reuse_existing and v1_bias_path.exists():
            logger.info("Reusing existing v1-bias orbit at %s", v1_bias_path)
            v1_bias_fit = Orbits.from_parquet(v1_bias_path)
        else:
            logger.info("Fitting v1-bias orbit via adam_fo…")
            fitted_v1, _ = fit_orbit(args.designation, obs, bias_table, fo_dir)
            v1_bias_fit = fitted_v1.to_orbits()
            v1_bias_fit.to_parquet(v1_bias_path)
            logger.info("Wrote v1-bias fit → %s", v1_bias_path)

    # Step 6 — propagate v1-bias fit to JPL epoch and compute gap
    logger.info("Propagating v1-bias fit to JPL epoch and computing gap…")
    v1_at_jpl, _ = propagate_for_comparison(v1_bias_fit, jpl_orbit, propagator)
    v1_gap = compute_orbit_gap(v1_at_jpl, jpl_orbit, variant="v1_bias")
    build_comparison_table([v1_gap]).to_parquet(cmp_v1_path)
    logger.info("Wrote v1-bias comparison → %s", cmp_v1_path)

    # Step 7 — combined summary
    summary = build_comparison_table([baseline_gap, v1_gap])
    summary.to_parquet(cmp_summary_path)
    logger.info("Wrote summary → %s", cmp_summary_path)

    # Step 8 — write narrative
    write_report(
        args.output_dir,
        designation=args.designation,
        obs_summary=obs_summary,
        baseline_gap=baseline_gap,
        v1_gap=v1_gap,
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
                "baseline_gap": baseline_gap.__dict__,
                "v1_bias_gap": v1_gap.__dict__,
            },
            indent=2,
            default=float,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
