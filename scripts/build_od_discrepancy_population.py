#!/usr/bin/env python3
"""
build_od_discrepancy_population.py
==================================

Assemble a NEO discrepancy population for downstream bias-variant testing.

For each of 50-100 selected NEOs:
  1. Fetch MPC observations via mpcq, drop space-based stations
  2. Fit a nominal adam_fo orbit (no bias correction) via FindOrbOrbitFitter
  3. Fetch the JPL/SBDB nominal orbit (with covariance when available)
  4. Propagate the adam_fo orbit to the JPL epoch via ASSIST
  5. Compute element gaps and Cartesian Δr / Δv at the common epoch
  6. Classify as "discrepant" using Δr / σ_r > 3 (or |Δr| > 1e-7 AU if no σ)

Outputs (written to <output_dir>/):
  discrepancy_ranking.parquet   - one row per attempted object
  failures.json                 - per-object failure log
  REPORT.md                     - narrative report

This script is the work product of bead od_experiments_setup-9d7. It does NOT
apply bias corrections; that's a follow-on bead.

SCRATCH helper note: SBDB orbit fetching is delegated to
`adam_core.orbits.query.query_sbdb_new` (which already returns covariance), so
no scratch SBDB helper is needed here.

Usage
-----
    python scripts/build_od_discrepancy_population.py \\
        --output-dir data/od_discrepancy_population \\
        --population-cap 70
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Inject the symlinked adam_fo source so we get FindOrbOrbitFitter from the
# kk/reference-orbit-warm-start branch (the wheel installed in the venv is an
# older release that pre-dates that symbol).
_ADAM_FO_SRC = "/Users/kathleenkiker/od_experiments_setup/adam_fo/src"
if _ADAM_FO_SRC not in sys.path:
    sys.path.insert(0, _ADAM_FO_SRC)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from adam_assist import ASSISTPropagator
from adam_core.coordinates import KeplerianCoordinates
from adam_core.orbits import Orbits
from adam_core.orbits.query import query_sbdb_new
from adam_core.time import Timestamp
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from google.cloud import bigquery as bq_lib
from mpcq.client import BigQueryMPCClient

from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    mpc_to_od_observations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_od_discrepancy_population")
# Silence noisy adam_core covariance warnings
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────
# Population selection
# ────────────────────────────────────────────────────────────────────────────

# Impact-monitor / well-known NEOs to always include. These are the high-priority
# targets explicitly named in the bead. Bead zgf (2024 YR4) runs in parallel
# against the same designation — overlap is intentional as a cross-check.
IMPACT_MONITOR_PROVIDS = [
    "2004 MN4",   # 99942 Apophis
    "1999 RQ36",  # 101955 Bennu  (also resolves under '101955', use unpacked form)
    "2017 BX",
    "2022 AE1",
    "2019 PR2",
    "2024 YR4",
]

# Other well-known NEOs from the existing 06_fetch_neo_sample.py priority list.
# Keeps continuity with the v1 LOOO study population.
PRIORITY_PROVIDS = [
    "A898 PA",    # 433 Eros
    "1929 SH",    # Amor
    "1998 OH",    # Apollo
    "1985 DO2",   # Amor
    "1994 LY",    # Apollo
    "1972 XA",    # Apollo
    "1981 ET3",   # Toutatis
    "2001 MZ7",   # Amor
]


@dataclass
class PopulationStratum:
    name: str
    min_nobs: int
    max_nobs: int | None
    min_arc_days: float
    max_arc_days: float | None
    target_count: int


# Stratification — chosen to cover the diversity axes the bead asks for:
# arc length, total observations, and (implicitly) station mix (well-observed
# objects necessarily have larger station mixes).
STRATA = [
    PopulationStratum("long_arc_well_obs",  500, None,   3650.0, None,   12),
    PopulationStratum("mod_arc_mod_obs",    100,  500,    365.0, 3650.0, 15),
    PopulationStratum("short_arc_few_obs",   20,  100,     30.0,  365.0, 15),
    PopulationStratum("very_short_arc",      10,   50,      0.0,   30.0, 10),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/od_discrepancy_population"))
    p.add_argument("--population-cap", type=int, default=70,
                   help="Soft cap on total NEOs attempted (default: 70)")
    p.add_argument("--project", default="moeyens-thor-dev")
    p.add_argument("--dataset-id", default="mpc_sbn_aurora")
    p.add_argument("--views-dataset-id", default="mpc_sbn_aurora_views")
    p.add_argument("--fo-result-dir", default="/tmp/fo_discrepancy_runs",
                   help="Working dir for Find_Orb runs")
    p.add_argument("--min-stations", type=int, default=3,
                   help="Minimum distinct ground-based stations per object")
    p.add_argument("--max-obs-per-object", type=int, default=2000,
                   help="Cap obs per object to bound fit time (later obs kept)")
    p.add_argument("--discrepant-sigma-threshold", type=float, default=3.0,
                   help="|Δr| / σ_r above this is discrepant (when σ available)")
    p.add_argument("--discrepant-fallback-au", type=float, default=1.0e-7,
                   help="|Δr| above this is discrepant when σ unavailable")
    return p.parse_args()


def _bq_sample_stratum(
    bq: bq_lib.Client,
    project: str,
    dataset_id: str,
    stratum: PopulationStratum,
    exclude: set[str],
) -> list[str]:
    """Return up to stratum.target_count provids matching the stratum bounds."""
    arc_max_clause = (
        f" AND arc_length_total <= {stratum.max_arc_days}"
        if stratum.max_arc_days is not None else ""
    )
    nobs_max_clause = (
        f" AND nobs_total <= {stratum.max_nobs}"
        if stratum.max_nobs is not None else ""
    )
    # ORDER BY RAND() gives a representative sample within each stratum.
    sql = f"""
SELECT unpacked_primary_provisional_designation AS provid,
       nobs_total, arc_length_total
FROM `{project}.{dataset_id}.public_mpc_orbits`
WHERE q < 1.3
  AND nobs_total >= {stratum.min_nobs}
  {nobs_max_clause}
  AND arc_length_total >= {stratum.min_arc_days}
  {arc_max_clause}
  AND arc_length_total IS NOT NULL
  AND unpacked_primary_provisional_designation IS NOT NULL
  AND unpacked_primary_provisional_designation NOT LIKE 'C/%'
  AND unpacked_primary_provisional_designation NOT LIKE 'P/%'
ORDER BY RAND()
LIMIT {stratum.target_count * 3}
""".strip()
    out: list[str] = []
    for r in bq.query(sql).result():
        if r.provid and r.provid not in exclude:
            out.append(r.provid)
            exclude.add(r.provid)
        if len(out) >= stratum.target_count:
            break
    logger.info(
        "stratum %s -> %d objects (nobs %d-%s, arc %s-%s d)",
        stratum.name, len(out), stratum.min_nobs, stratum.max_nobs,
        stratum.min_arc_days, stratum.max_arc_days,
    )
    return out


def select_population(args: argparse.Namespace) -> tuple[list[str], dict]:
    """Pick a diverse NEO list combining impact-monitor + priority + BQ strata."""
    seen: set[str] = set()
    selected: list[str] = []
    strata_membership: dict[str, str] = {}

    for p in IMPACT_MONITOR_PROVIDS:
        if p not in seen and len(selected) < args.population_cap:
            seen.add(p)
            selected.append(p)
            strata_membership[p] = "impact_monitor"
    for p in PRIORITY_PROVIDS:
        if p not in seen and len(selected) < args.population_cap:
            seen.add(p)
            selected.append(p)
            strata_membership[p] = "priority_well_known"

    bq = bq_lib.Client(project=args.project)
    for stratum in STRATA:
        cap_remaining = max(0, args.population_cap - len(selected))
        if cap_remaining == 0:
            break
        # Don't over-sample any single stratum beyond the cap.
        adj = PopulationStratum(
            name=stratum.name,
            min_nobs=stratum.min_nobs, max_nobs=stratum.max_nobs,
            min_arc_days=stratum.min_arc_days, max_arc_days=stratum.max_arc_days,
            target_count=min(stratum.target_count, cap_remaining),
        )
        for p in _bq_sample_stratum(bq, args.project, args.dataset_id, adj, seen):
            selected.append(p)
            strata_membership[p] = stratum.name

    logger.info("Selected %d total NEOs across strata", len(selected))
    return selected, strata_membership


# ────────────────────────────────────────────────────────────────────────────
# Per-object processing
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectRecord:
    """One row in discrepancy_ranking.parquet."""
    object_id: str
    stratum: str
    fit_status: str  # "converged", "no_obs", "no_ground_obs", "no_od_obs",
                     # "fit_no_solution", "exception"
    failure_reason: str = ""
    n_obs_raw: int = 0
    n_obs_ground: int = 0
    n_stations: int = 0
    arc_days: float = float("nan")
    mean_obs_epoch_mjd: float = float("nan")
    primary_obs_class: str = ""        # SBDB orbit class
    sbdb_object_id: str = ""
    # Our adam_fo orbit (Keplerian @ fit epoch)
    our_epoch_mjd: float = float("nan")
    our_a_au: float = float("nan")
    our_e: float = float("nan")
    our_i_deg: float = float("nan")
    our_raan_deg: float = float("nan")
    our_argp_deg: float = float("nan")
    our_M_deg: float = float("nan")
    our_reduced_chi2: float = float("nan")
    # JPL/SBDB orbit (Keplerian @ JPL epoch)
    jpl_epoch_mjd: float = float("nan")
    jpl_a_au: float = float("nan")
    jpl_e: float = float("nan")
    jpl_i_deg: float = float("nan")
    jpl_raan_deg: float = float("nan")
    jpl_argp_deg: float = float("nan")
    jpl_M_deg: float = float("nan")
    jpl_has_covariance: bool = False
    # Gaps @ common epoch (JPL epoch)
    common_epoch_mjd: float = float("nan")
    delta_a_au: float = float("nan")
    delta_e: float = float("nan")
    delta_i_deg: float = float("nan")
    delta_raan_deg: float = float("nan")
    delta_argp_deg: float = float("nan")
    delta_M_deg: float = float("nan")
    delta_r_au: float = float("nan")          # Cartesian position gap magnitude
    delta_v_au_per_day: float = float("nan")  # Cartesian velocity gap magnitude
    jpl_pos_sigma_au: float = float("nan")    # JPL 1-σ position at JPL epoch
    delta_r_over_sigma: float = float("nan")
    discrepant: bool = False
    # Comma-joined list of distinct station codes
    station_codes: str = ""


def _wrap_deg(x: float) -> float:
    """Wrap angle difference into [-180, 180]."""
    if not np.isfinite(x):
        return x
    return ((x + 180.0) % 360.0) - 180.0


def _jpl_pos_sigma_au(orbit: Orbits) -> float:
    """1-σ Cartesian position uncertainty (AU) from the Orbits covariance.

    The SBDB orbit returned by query_sbdb_new has Cartesian covariance derived
    from the cometary covariance via the Jacobian (handled inside adam_core).
    We take sqrt(σ_xx² + σ_yy² + σ_zz²) as the radial 1-σ scalar.

    Returns NaN if covariance isn't available or contains non-finite entries.
    """
    try:
        cov = orbit.coordinates.covariance.to_matrix()
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


def _fetch_obs(client: BigQueryMPCClient, provid: str):
    """Fetch MPC observations for a single provid (single-element batch)."""
    return client.query_observations([provid])


def _process_object(
    provid: str,
    stratum: str,
    client: BigQueryMPCClient,
    fitter: FindOrbOrbitFitter,
    propagator: ASSISTPropagator,
    spacebased: set[str],
    max_obs: int,
    min_stations: int,
) -> ObjectRecord:
    rec = ObjectRecord(object_id=provid, stratum=stratum, fit_status="exception")

    # 1. Fetch obs
    try:
        all_obs = _fetch_obs(client, provid)
    except Exception as e:
        rec.failure_reason = f"mpcq query_observations failed: {e}"
        return rec

    rec.n_obs_raw = len(all_obs)
    if rec.n_obs_raw == 0:
        rec.fit_status = "no_obs"
        rec.failure_reason = "mpcq returned 0 observations"
        return rec

    # 2. Drop space-based stations
    try:
        ground_mask = pc.invert(
            pc.is_in(all_obs.stn, value_set=pa.array(list(spacebased), type=pa.large_utf8()))
        )
        ground_obs = all_obs.apply_mask(ground_mask)
        # apply_mask preserves fragmentation; downstream Observers.from_codes
        # converts to StructArrays which fails on fragmented tables. Defrag.
        if ground_obs.fragmented():
            ground_obs = qv.concatenate([ground_obs])
    except Exception as e:
        rec.failure_reason = f"space-based filter failed: {e}"
        return rec

    rec.n_obs_ground = len(ground_obs)
    if rec.n_obs_ground == 0:
        rec.fit_status = "no_ground_obs"
        rec.failure_reason = "all observations were space-based"
        return rec

    # Diversity / metadata
    stn_list = ground_obs.stn.to_pylist()
    rec.n_stations = len(set(stn_list))
    times_mjd = ground_obs.obstime.mjd().to_numpy(zero_copy_only=False)
    rec.arc_days = float(times_mjd.max() - times_mjd.min())
    rec.mean_obs_epoch_mjd = float(np.mean(times_mjd))
    rec.station_codes = ",".join(sorted(set(stn_list)))

    if rec.n_stations < min_stations:
        rec.fit_status = "too_few_stations"
        rec.failure_reason = f"n_stations={rec.n_stations} < {min_stations}"
        return rec

    # Dedupe near-duplicate obs at the same station within ~1.5 s. FindOrb's
    # rejection-matching asserts that each rejected obs maps to exactly one
    # input obs within ±1 s, so two near-simultaneous obs from the same
    # station crash the fit. Observed for 2004 MN4 (B72 at 59279.85585 and
    # 59279.85586, 0.86 s apart). Strategy: sort by (stn, time) and keep an
    # obs only if its same-station predecessor is >1.5 s away.
    time_secs_f = times_mjd * 86400.0
    order = np.lexsort((time_secs_f, np.asarray(stn_list, dtype=object)))
    keep = np.zeros(len(ground_obs), dtype=bool)
    last_kept_by_stn: dict[str, float] = {}
    for idx in order:
        stn = stn_list[idx]
        t = float(time_secs_f[idx])
        prev = last_kept_by_stn.get(stn)
        if prev is None or abs(t - prev) > 1.5:
            keep[idx] = True
            last_kept_by_stn[stn] = t
    if keep.sum() < len(ground_obs):
        dropped = len(ground_obs) - int(keep.sum())
        ground_obs = ground_obs.apply_mask(pa.array(keep))
        if ground_obs.fragmented():
            ground_obs = qv.concatenate([ground_obs])
        logger.info("%s: deduped %d duplicate (stn,time) obs", provid, dropped)
        rec.n_obs_ground = len(ground_obs)
        # Refresh derived metadata after dedupe
        stn_list = ground_obs.stn.to_pylist()
        rec.n_stations = len(set(stn_list))
        times_mjd = ground_obs.obstime.mjd().to_numpy(zero_copy_only=False)
        rec.arc_days = float(times_mjd.max() - times_mjd.min())
        rec.mean_obs_epoch_mjd = float(np.mean(times_mjd))
        rec.station_codes = ",".join(sorted(set(stn_list)))

    # Cap obs to bound fit time on extremely well-observed objects (Apophis = 9500+).
    if rec.n_obs_ground > max_obs:
        # Keep latest max_obs observations.
        order = np.argsort(times_mjd)
        keep_idx = order[-max_obs:]
        keep_mask = np.zeros(rec.n_obs_ground, dtype=bool)
        keep_mask[keep_idx] = True
        ground_obs = ground_obs.apply_mask(pa.array(keep_mask))
        if ground_obs.fragmented():
            ground_obs = qv.concatenate([ground_obs])
        logger.info("%s: capped %d -> %d obs (latest)", provid, rec.n_obs_ground, max_obs)
        rec.n_obs_ground = len(ground_obs)

    # 3. Convert to OD observations
    try:
        od_obs = mpc_to_od_observations(ground_obs, prevent_nans=True)
    except Exception as e:
        rec.fit_status = "no_od_obs"
        rec.failure_reason = f"mpc_to_od_observations failed: {e}"
        return rec
    if od_obs is None or len(od_obs) == 0:
        rec.fit_status = "no_od_obs"
        rec.failure_reason = "mpc_to_od_observations returned None/empty"
        return rec

    # 4. Initial adam_fo fit (NO bias correction)
    try:
        fitted, _members = fitter.initial_fit(provid, od_obs)
    except Exception as e:
        rec.failure_reason = f"FindOrb initial_fit raised: {e}"
        return rec
    if len(fitted) == 0:
        rec.fit_status = "fit_no_solution"
        rec.failure_reason = "FindOrb returned empty FittedOrbits"
        return rec

    # Convert fitted (FittedOrbits) into a plain Orbits for propagation.
    our_orbit = Orbits.from_kwargs(
        orbit_id=fitted.orbit_id,
        object_id=fitted.object_id,
        coordinates=fitted.coordinates,
    )
    rec.our_reduced_chi2 = float(fitted.reduced_chi2[0].as_py())

    # Our Keplerian elements @ our epoch
    try:
        our_kep = our_orbit.coordinates.to_keplerian()
        rec.our_epoch_mjd = float(our_kep.time.mjd()[0].as_py())
        rec.our_a_au = float(our_kep.a[0].as_py())
        rec.our_e = float(our_kep.e[0].as_py())
        rec.our_i_deg = float(our_kep.i[0].as_py())
        rec.our_raan_deg = float(our_kep.raan[0].as_py())
        rec.our_argp_deg = float(our_kep.ap[0].as_py())
        rec.our_M_deg = float(our_kep.M[0].as_py())
    except Exception as e:
        rec.failure_reason = f"to_keplerian on our orbit failed: {e}"
        # continue — we can still produce gap if propagation works
        pass

    # 5. Fetch JPL/SBDB orbit
    try:
        jpl_orbit = query_sbdb_new([provid], allow_missing=True, orbit_id_from_input=True)
    except Exception as e:
        rec.fit_status = "converged"
        rec.failure_reason = f"converged but SBDB fetch failed: {e}"
        return rec
    if len(jpl_orbit) == 0:
        rec.fit_status = "converged"
        rec.failure_reason = "converged but SBDB returned no orbit"
        return rec

    rec.sbdb_object_id = str(jpl_orbit.object_id[0].as_py())
    rec.jpl_epoch_mjd = float(jpl_orbit.coordinates.time.mjd()[0].as_py())
    rec.jpl_pos_sigma_au = _jpl_pos_sigma_au(jpl_orbit)
    rec.jpl_has_covariance = bool(np.isfinite(rec.jpl_pos_sigma_au))

    # JPL Keplerian
    try:
        jpl_kep = jpl_orbit.coordinates.to_keplerian()
        rec.jpl_a_au = float(jpl_kep.a[0].as_py())
        rec.jpl_e = float(jpl_kep.e[0].as_py())
        rec.jpl_i_deg = float(jpl_kep.i[0].as_py())
        rec.jpl_raan_deg = float(jpl_kep.raan[0].as_py())
        rec.jpl_argp_deg = float(jpl_kep.ap[0].as_py())
        rec.jpl_M_deg = float(jpl_kep.M[0].as_py())
    except Exception:
        pass

    # Pull SBDB orbit class from the underlying API call. SBDB's classification
    # isn't carried on the Orbits table; we re-query just for the class code
    # (cheap, cached intra-session by requests connection pool).
    try:
        from adam_core.orbits.query.sbdb import _sbdb_api_get_json
        payload = _sbdb_api_get_json(provid, timeout_s=30.0, max_attempts=3)
        rec.primary_obs_class = str(
            (payload.get("object") or {}).get("orbit_class", {}).get("code", "")
        )
    except Exception:
        pass

    # 6. Propagate our orbit to JPL epoch using ASSIST and compute Cartesian gap
    common_epoch = Timestamp.from_mjd(
        [rec.jpl_epoch_mjd], scale="tdb"
    )
    rec.common_epoch_mjd = rec.jpl_epoch_mjd
    try:
        our_propagated = propagator.propagate_orbits(our_orbit, common_epoch)
    except Exception as e:
        rec.fit_status = "converged"
        rec.failure_reason = f"converged but ASSIST propagation failed: {e}"
        return rec

    our_cart = our_propagated.coordinates
    jpl_cart = jpl_orbit.coordinates
    try:
        dx = float(our_cart.x[0].as_py() - jpl_cart.x[0].as_py())
        dy = float(our_cart.y[0].as_py() - jpl_cart.y[0].as_py())
        dz = float(our_cart.z[0].as_py() - jpl_cart.z[0].as_py())
        dvx = float(our_cart.vx[0].as_py() - jpl_cart.vx[0].as_py())
        dvy = float(our_cart.vy[0].as_py() - jpl_cart.vy[0].as_py())
        dvz = float(our_cart.vz[0].as_py() - jpl_cart.vz[0].as_py())
    except Exception as e:
        rec.fit_status = "converged"
        rec.failure_reason = f"converged but Cartesian diff failed: {e}"
        return rec

    rec.delta_r_au = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    rec.delta_v_au_per_day = float(np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz))
    if rec.jpl_has_covariance and rec.jpl_pos_sigma_au > 0:
        rec.delta_r_over_sigma = rec.delta_r_au / rec.jpl_pos_sigma_au

    # Element gaps (handle angle wrap)
    if np.isfinite(rec.our_a_au) and np.isfinite(rec.jpl_a_au):
        rec.delta_a_au = rec.our_a_au - rec.jpl_a_au
        rec.delta_e = rec.our_e - rec.jpl_e
        rec.delta_i_deg = _wrap_deg(rec.our_i_deg - rec.jpl_i_deg)
        rec.delta_raan_deg = _wrap_deg(rec.our_raan_deg - rec.jpl_raan_deg)
        rec.delta_argp_deg = _wrap_deg(rec.our_argp_deg - rec.jpl_argp_deg)
        # Note: M is at different epochs unless we re-propagate. Useful only as
        # a smoke-test indicator; for true M gap one should propagate both
        # mean anomalies. We report the wrapped diff between elements as-fit.
        rec.delta_M_deg = _wrap_deg(rec.our_M_deg - rec.jpl_M_deg)

    rec.fit_status = "converged"
    return rec


def _classify_discrepant(rec: ObjectRecord, sigma_thresh: float, fallback_au: float) -> bool:
    if rec.fit_status != "converged":
        return False
    if not np.isfinite(rec.delta_r_au):
        return False
    if rec.jpl_has_covariance and np.isfinite(rec.delta_r_over_sigma):
        return rec.delta_r_over_sigma > sigma_thresh
    return rec.delta_r_au > fallback_au


# ────────────────────────────────────────────────────────────────────────────
# Aggregation & reporting
# ────────────────────────────────────────────────────────────────────────────


def _record_to_dict(rec: ObjectRecord) -> dict:
    out = {}
    for k, v in rec.__dict__.items():
        if isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def write_outputs(
    records: list[ObjectRecord],
    output_dir: Path,
    args: argparse.Namespace,
    high_confidence_stations: set[str] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([_record_to_dict(r) for r in records])
    parquet_path = output_dir / "discrepancy_ranking.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), parquet_path)

    # Failures: convenient JSON sidecar
    failures = [
        {"object_id": r.object_id, "fit_status": r.fit_status, "reason": r.failure_reason}
        for r in records
        if r.fit_status != "converged" or r.failure_reason
    ]
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))

    # ── REPORT.md ────────────────────────────────────────────────────────
    n_attempted = len(records)
    converged = [r for r in records if r.fit_status == "converged" and np.isfinite(r.delta_r_au)]
    n_converged = len(converged)
    discrepant = [r for r in converged if r.discrepant]
    n_discrepant = len(discrepant)

    # Top discrepant by σ-ratio (preferred) or by absolute Δr (fallback)
    def _rank_key(r: ObjectRecord) -> tuple:
        if r.jpl_has_covariance and np.isfinite(r.delta_r_over_sigma):
            return (0, -r.delta_r_over_sigma)
        return (1, -r.delta_r_au)
    top = sorted(converged, key=_rank_key)[:20]

    # Station mix of discrepant set
    stn_counter: dict[str, int] = {}
    for r in discrepant:
        for s in r.station_codes.split(","):
            if s:
                stn_counter[s] = stn_counter.get(s, 0) + 1
    top_stns = sorted(stn_counter.items(), key=lambda kv: -kv[1])[:15]

    if high_confidence_stations is not None and top_stns:
        coverage_flag = [
            f"{s} ({c} disc obj{'s' if c != 1 else ''}; "
            f"{'in' if s in high_confidence_stations else 'NOT in'} v1 HC)"
            for s, c in top_stns
        ]
    else:
        coverage_flag = [f"{s} ({c})" for s, c in top_stns]

    fit_status_counts: dict[str, int] = {}
    for r in records:
        fit_status_counts[r.fit_status] = fit_status_counts.get(r.fit_status, 0) + 1

    stratum_counts: dict[str, int] = {}
    for r in records:
        stratum_counts[r.stratum] = stratum_counts.get(r.stratum, 0) + 1

    md = []
    md.append("# NEO OD-discrepancy population — BEFORE bias correction")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"Bead: `od_experiments_setup-9d7`  ·  Branch: `kk/od-bias-experiments`")
    md.append("")
    md.append("## Population selection")
    md.append("")
    md.append(
        "Stratified sample combining: (a) explicit impact-monitor objects "
        "named in the bead — Apophis (2004 MN4), Bennu (1999 RQ36), 2017 BX, "
        "2022 AE1, 2019 PR2, 2024 YR4; (b) the priority well-known NEO list "
        "from `06_fetch_neo_sample.py`; (c) BQ-sourced strata across arc "
        "length and observation count (`public_mpc_orbits` with `q < 1.3`, "
        "asteroid designations only)."
    )
    md.append("")
    md.append("Stratum breakdown:")
    for k, v in stratum_counts.items():
        md.append(f"- `{k}`: {v}")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Total attempted: **{n_attempted}**")
    md.append(f"- Converged (adam_fo + Cartesian gap computable): **{n_converged}**")
    md.append(
        f"- Discrepant (Δr / σ > {args.discrepant_sigma_threshold} when σ available; "
        f"|Δr| > {args.discrepant_fallback_au:.1e} AU otherwise): **{n_discrepant}**"
    )
    md.append("")
    md.append("Fit status distribution:")
    for k, v in sorted(fit_status_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"- `{k}`: {v}")
    md.append("")
    md.append("## Top discrepant objects")
    md.append("")
    md.append("Ranked by Δr / σ_r (or absolute Δr if no JPL covariance).")
    md.append("")
    md.append("| Rank | object_id | n_obs | arc_d | n_stn | Δr (AU) | σ_r (AU) | Δr/σ | class |")
    md.append("|------|-----------|-------|-------|-------|---------|----------|------|-------|")
    for i, r in enumerate(top, 1):
        sigma_str = f"{r.jpl_pos_sigma_au:.2e}" if np.isfinite(r.jpl_pos_sigma_au) else "—"
        ratio_str = f"{r.delta_r_over_sigma:.1f}" if np.isfinite(r.delta_r_over_sigma) else "—"
        md.append(
            f"| {i} | `{r.object_id}` | {r.n_obs_ground} | "
            f"{r.arc_days:.0f} | {r.n_stations} | {r.delta_r_au:.2e} | "
            f"{sigma_str} | {ratio_str} | {r.primary_obs_class or '—'} |"
        )
    md.append("")
    md.append("## Station-mix profile of discrepant set")
    md.append("")
    md.append(
        "Top stations by frequency among discrepant objects "
        "(annotated with v1 high-confidence-table membership):"
    )
    md.append("")
    for line in coverage_flag:
        md.append(f"- {line}")
    md.append("")
    md.append("## Sanity checks")
    md.append("")
    md.append(
        "- Epoch alignment: all gap metrics computed at the JPL/SBDB epoch "
        "(`common_epoch_mjd == jpl_epoch_mjd`); the adam_fo orbit is "
        "propagated to that epoch with `ASSISTPropagator.propagate_orbits`."
    )
    if converged:
        chi2_arr = np.array(
            [r.our_reduced_chi2 for r in converged if np.isfinite(r.our_reduced_chi2)],
            dtype=np.float64,
        )
        if chi2_arr.size > 0:
            md.append(
                f"- Hold-in reduced χ²: median={np.median(chi2_arr):.2f}, "
                f"p95={np.percentile(chi2_arr, 95):.2f}, max={chi2_arr.max():.2f} "
                f"(over {chi2_arr.size} converged fits)."
            )
    md.append("")
    md.append("## Recommended input set for the bias-variant follow-on")
    md.append("")
    md.append(
        "The discrepant subset above is the natural input. Of those, "
        "prioritize objects whose dominant stations appear in the v1 "
        "high-confidence bias table — those are where the bias correction "
        "actually has a number to apply. Objects with `Δr/σ > 10` are "
        "candidates for diagnostic deep-dives (likely either truly biased "
        "stations or non-grav forces not in our 2-body/N-body propagator)."
    )
    md.append("")
    (output_dir / "REPORT.md").write_text("\n".join(md))
    logger.info("Wrote REPORT.md to %s", output_dir / "REPORT.md")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


_HIGH_CONFIDENCE_TABLE = Path(
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/data/"
    "mpc_scale_results_20260510/bias_catalog_published/high_confidence_bias_table.csv"
)


def _load_high_confidence_stations() -> set[str] | None:
    if not _HIGH_CONFIDENCE_TABLE.exists():
        logger.warning(
            "high_confidence_bias_table.csv not found at %s; "
            "station-mix coverage annotation will be skipped.",
            _HIGH_CONFIDENCE_TABLE,
        )
        return None
    df = pd.read_csv(_HIGH_CONFIDENCE_TABLE)
    # Column name varies historically; try the obvious candidates.
    for col in ("stn", "station", "obscode", "obs_code"):
        if col in df.columns:
            return set(df[col].astype(str).tolist())
    logger.warning("high_confidence table loaded but no station column found")
    return None


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Selecting population (cap=%d)", args.population_cap)
    selected, strata = select_population(args)

    high_conf = _load_high_confidence_stations()

    client = BigQueryMPCClient(
        dataset_id=args.dataset_id,
        views_dataset_id=args.views_dataset_id,
        project=args.project,
    )
    os.makedirs(args.fo_result_dir, exist_ok=True)
    fitter = FindOrbOrbitFitter(
        fo_result_dir=args.fo_result_dir,
        clean_up_fo_dir=True,
    )
    propagator = ASSISTPropagator()
    spacebased = set(get_spacebased_stns())

    records: list[ObjectRecord] = []
    start = time.time()
    for i, provid in enumerate(selected, 1):
        t0 = time.time()
        stratum = strata.get(provid, "unknown")
        logger.info("[%d/%d] %s (%s)", i, len(selected), provid, stratum)
        try:
            rec = _process_object(
                provid, stratum, client, fitter, propagator,
                spacebased, args.max_obs_per_object, args.min_stations,
            )
        except Exception as e:
            logger.exception("Unhandled exception on %s: %s", provid, e)
            rec = ObjectRecord(
                object_id=provid, stratum=stratum, fit_status="exception",
                failure_reason=f"unhandled exception: {e}\n{traceback.format_exc(limit=3)}",
            )
        rec.discrepant = _classify_discrepant(
            rec, args.discrepant_sigma_threshold, args.discrepant_fallback_au
        )
        records.append(rec)
        logger.info(
            "[%d/%d] %s status=%s Δr=%s discrepant=%s (%.1fs)",
            i, len(selected), provid, rec.fit_status,
            f"{rec.delta_r_au:.2e}" if np.isfinite(rec.delta_r_au) else "—",
            rec.discrepant, time.time() - t0,
        )

        # Incremental safety: re-write outputs every 10 objects so a crash
        # doesn't lose hours of progress.
        if i % 10 == 0:
            write_outputs(records, args.output_dir, args, high_conf)

    logger.info("Total runtime: %.1f min", (time.time() - start) / 60.0)
    write_outputs(records, args.output_dir, args, high_conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
