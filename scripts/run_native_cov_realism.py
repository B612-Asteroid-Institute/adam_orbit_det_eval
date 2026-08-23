#!/usr/bin/env python3
"""
run_native_cov_realism.py
=========================

Covariance-realism comparison vs JPL with the NATIVE adam_core fitter
(post covariance fix: 2N whitened residuals + analytic 2-body-STM Jacobian
with weak-direction validation), across the adam_core
ObservationUncertaintyModel interpreter variants.

Methodology follows run_jpl_cov_ip_comparison.py (2026-07-01, FindOrb): fit
each test-set object, propagate our orbit+covariance to JPL's SBDB epoch
(sigma-point), and compare:
  pos_sigma_ratio = sqrt(tr(C_our_pos)) / sqrt(tr(C_jpl_pos))  (1 = realistic)
  mahalanobis     = |r_our - r_jpl| in combined position covariance units

Differences from the July-01 run:
  * NATIVE fitter (Gauss IOD -> bounded ladder DC -> per-variant iterative_fit
    with the analytic Jacobian). No external orbit ever seeds a fit.
  * Variants are the adam_core interpreters on a BIAS_TABLE_SCHEMA table,
    plus: empirical-covar 'replace' mode and a high_confidence-gated table
    (question b), a Veres-floor + night-batch composite approximating agency
    weighting practice (question a), and the v1-subtract legacy anchor
    (position modification, reference only, via the eval-repo builder).
  * Fits are gravity-only (no nongrav estimation on this branch).

Run:
  cd adam_orbit_det_eval && PYTHONPATH=<adam_core_obs_uncertainty>/src \
    .venv/bin/python scripts/run_native_cov_realism.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv

from adam_assist import ASSISTPropagator
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.orbit_determination.differential_correction import (
    fit_least_squares,
    iterative_fit,
)
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.gauss import gaussIOD
from adam_core.orbit_determination.iod import select_observations
from adam_core.orbit_determination.observation_uncertainty import (
    CompositeModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
    ObservationUncertaintyModel,
    PerformanceWeightedModel,
    SigmaFloorModel,
)
from adam_core.orbits import Orbits
from adam_core.orbits.query.sbdb import query_sbdb
from mpcq.client import BigQueryMPCClient

from adam_orbit_det_eval.efcc18 import compute_efcc18_corrections, load_efcc18_biases
from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    get_veres2017_sigma,
    load_v2_bias_catalog,
    mpc_to_od_observations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("native_cov_realism")
logging.getLogger("adam_core").setLevel(logging.WARNING)

AU_KM = 1.495978707e8
ARCSEC_PER_DEG = 3600.0
V2_TABLE_PATH = "/Users/kathleenkiker/claude_cli/od_pin_data/v2_full.parquet"
V1_DICT_PATH = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
MANIFEST = Path(__file__).resolve().parent.parent / "data/ip_risklist/cov_ip_testset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data/native_cov_realism"

TIME_SORT = [
    ("coordinates.time.days", "ascending"),
    ("coordinates.time.nanos", "ascending"),
]


class VeresFloorModel(ObservationUncertaintyModel):
    """
    Floor each observation's angular sigmas at the Veres et al. (2017)
    per-(station, catalog) values: sigma_used = max(sigma_reported, veres).
    This approximates agency weighting practice (they do not trust reported
    rms below the station's demonstrated performance), in contrast to the
    'veres2017' baseline sigma model which only FILLS missing sigmas.

    Floors are per-observation (cos-dec-corrected arcsec for RA), keyed by
    observation id, computed from the raw MPC rows before conversion.
    """

    def __init__(self, floors: Dict[str, Tuple[float, float]]) -> None:
        self.floors = floors

    def apply(self, observations):
        if len(observations) == 0:
            return observations
        cov = observations.coordinates.covariance.to_matrix().copy()
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False)
        cos_dec = np.cos(np.radians(lat))
        ids = observations.id.to_pylist()
        changed = False
        for i, oid in enumerate(ids):
            floor = self.floors.get(oid)
            if floor is None or cos_dec[i] <= 0:
                continue
            floor_var_lon = (floor[0] / (ARCSEC_PER_DEG * cos_dec[i])) ** 2
            floor_var_lat = (floor[1] / ARCSEC_PER_DEG) ** 2
            if np.isfinite(cov[i, 1, 1]) and cov[i, 1, 1] < floor_var_lon:
                cov[i, 1, 1] = floor_var_lon
                changed = True
            if np.isfinite(cov[i, 2, 2]) and cov[i, 2, 2] < floor_var_lat:
                cov[i, 2, 2] = floor_var_lat
                changed = True
        if not changed:
            return observations
        return observations.set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(cov)
        )


def dedupe(obs):
    stns = obs.stn.to_pylist()
    t = obs.obstime.mjd().to_numpy(zero_copy_only=False) * 86400.0
    order = np.lexsort((t, np.asarray(stns, dtype=object)))
    keep = np.zeros(len(obs), dtype=bool)
    last: Dict[str, float] = {}
    for i in order:
        s = stns[i]
        ti = float(t[i])
        pv = last.get(s)
        if pv is None or abs(ti - pv) > 1.5:
            keep[i] = True
            last[s] = ti
    if keep.sum() == len(obs):
        return obs
    d = obs.apply_mask(pa.array(keep))
    return qv.concatenate([d]) if d.fragmented() else d


def native_iod(base: OrbitDeterminationObservations, propagator) -> Tuple[Optional[Orbits], str]:
    """
    Bounded native IOD ladder: Gauss on selected triplets, then differential
    correction on a widening window (triplet span -> x3 -> full arc), each
    rung bounded by max_nfev. Returns (orbit, note); orbit is None on failure.
    """
    mjd = base.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    attempts = []
    for method in ("thirds", "first+middle+last"):
        try:
            sel = select_observations(base, method=method)
        except Exception:
            continue
        for ids in sel[:2]:
            mask = pc.is_in(base.id, pa.array(list(ids), type=pa.large_string()))
            triplet = base.apply_mask(mask)
            t_trip = triplet.coordinates.time.mjd().to_numpy(zero_copy_only=False)
            try:
                sols = gaussIOD(
                    triplet.coordinates.values[:, 1:3],
                    t_trip,
                    triplet.observers.coordinates.r,
                    light_time=True,
                )
            except Exception:
                continue
            for i in range(len(sols)):
                r_helio = float(np.linalg.norm(sols[i : i + 1].coordinates.r[0]))
                if not (0.3 < r_helio < 8.0):
                    continue
                attempts.append((method, i, sols[i : i + 1], t_trip))

    for method, i, seed_orbits, t_trip in attempts:
        seed = seed_orbits
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for width in (1.0, 3.0, np.inf):
                    half = max((t_trip.max() - t_trip.min()) / 2.0 * width, 1.0)
                    m = pa.array(np.abs(mjd - t_trip.mean()) <= half)
                    sub = base.apply_mask(m)
                    if len(sub) < 6:
                        continue
                    fitted, _ = fit_least_squares(
                        seed,
                        sub,
                        propagator,
                        jacobian="analytic",
                        validate_covariance=False,
                        max_nfev=60,
                        xtol=1e-10,
                        ftol=1e-10,
                        gtol=1e-10,
                    )
                    seed = fitted.to_orbits()
                rchi2 = fitted.reduced_chi2[0].as_py()
                if rchi2 is not None and rchi2 < 200.0:
                    return seed, f"{method}/sol{i} rchi2={rchi2:.2f}"
        except Exception as e:
            logger.debug("ladder failed (%s/sol%d): %s", method, i, e)
            continue
    return None, "iod ladder failed"


def covariance_comparison(our_orbit, jpl_orbit, propagator, cov_samples=200):
    jpl_time = jpl_orbit.coordinates.time
    our_at = propagator.propagate_orbits(
        our_orbit, jpl_time, covariance=True,
        covariance_method="sigma-point", num_samples=cov_samples,
    )
    jpl_at = propagator.propagate_orbits(
        jpl_orbit, jpl_time, covariance=True,
        covariance_method="sigma-point", num_samples=cov_samples,
    )
    r_our = np.asarray(our_at.coordinates.values[0][:3], dtype=float)
    r_jpl = np.asarray(jpl_at.coordinates.values[0][:3], dtype=float)
    C_our = np.asarray(our_at.coordinates.covariance.to_matrix()[0][:3, :3], dtype=float)
    C_jpl = np.asarray(jpl_at.coordinates.covariance.to_matrix()[0][:3, :3], dtype=float)
    pos_sig_our = float(np.sqrt(np.trace(C_our)))
    pos_sig_jpl = float(np.sqrt(np.trace(C_jpl)))
    dr = r_our - r_jpl
    try:
        maha = float(np.sqrt(dr @ np.linalg.solve(C_our + C_jpl, dr)))
    except Exception:
        maha = float("nan")
    return dict(
        pos_sigma_ratio=pos_sig_our / pos_sig_jpl if pos_sig_jpl > 0 else float("nan"),
        pos_sigma_our_km=pos_sig_our * AU_KM,
        pos_sigma_jpl_km=pos_sig_jpl * AU_KM,
        mahalanobis=maha,
        dr_km=float(np.linalg.norm(dr)) * AU_KM,
    )


@dataclass
class Row:
    object_id: str
    method: str
    category: str = ""
    n_obs: int = 0
    arc_days: float = float("nan")
    iod_note: str = ""
    reduced_chi2: float = float("nan")
    n_outliers: int = 0
    fallback_fired: bool = False
    cov_warning: str = ""
    pos_sigma_ratio: float = float("nan")
    pos_sigma_our_km: float = float("nan")
    pos_sigma_jpl_km: float = float("nan")
    mahalanobis: float = float("nan")
    dr_km: float = float("nan")
    fit_seconds: float = float("nan")
    failure_reason: str = ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "comparison.parquet"

    man = pd.read_parquet(MANIFEST)
    if args.objects:
        man = man[man.provid.isin(args.objects)]
    if args.max_objects:
        man = man.head(args.max_objects)

    bias_table = pq.ParquetFile(V2_TABLE_PATH).read()
    if "program_code" in bias_table.column_names:
        bias_table = bias_table.drop_columns(["program_code"])
    hc_mask = pc.fill_null(bias_table["high_confidence"], False)
    bias_table_hc = bias_table.filter(hc_mask)
    logger.info(
        "bias table: %d rows (%d high-confidence)", len(bias_table), len(bias_table_hc)
    )

    v1cat = load_v2_bias_catalog(V1_DICT_PATH, rollup_only=True)
    v1_bias_dict = {
        code: (rec["bias_ra"], rec["bias_dec"])
        for code, rec in v1cat.items()
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"])
    }

    efcc18_bias_table = load_efcc18_biases()
    spacebased = set(get_spacebased_stns())
    client = BigQueryMPCClient(
        dataset_id="mpc_sbn_aurora",
        views_dataset_id="mpc_sbn_aurora_views",
        project="moeyens-thor-dev",
    )
    propagator = ASSISTPropagator()

    records: List[dict] = []
    done = set()
    if args.resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        records = prev.to_dict(orient="records")
        done = {(r["object_id"], r["method"]) for r in records}

    def flush():
        pd.DataFrame(records).to_parquet(out_path, index=False)

    start = time.time()
    for _, mrow in man.iterrows():
        provid = mrow["provid"]
        category = mrow.get("category", "")
        logger.info("[%.1fm] === %s (%s) ===", (time.time() - start) / 60, provid, category)

        # --- data ---
        try:
            raw = client.query_observations([provid])
            stns = raw.stn.to_pylist()
            gm = np.array(
                [(s is not None) and (s not in spacebased) for s in stns], dtype=bool
            )
            g = raw.apply_mask(pa.array(gm))
            if g.fragmented():
                g = qv.concatenate([g])
            obs_mpc = dedupe(g)
            times_jd = obs_mpc.obstime.mjd().to_numpy(zero_copy_only=False) + 2400000.5
            efcc = compute_efcc18_corrections(
                obs_mpc.ra.to_numpy(zero_copy_only=False),
                obs_mpc.dec.to_numpy(zero_copy_only=False),
                obs_mpc.astcat.to_pylist(),
                times_jd,
                bias_table=efcc18_bias_table,
            )
            jpl_orbit = query_sbdb([provid])
        except Exception as e:
            logger.warning("%s prep failed: %s", provid, e)
            records.append(asdict(Row(provid, "ALL", category, failure_reason=f"prep: {e}")))
            flush()
            continue

        # Per-observation Veres floors (cos-dec-corrected arcsec), keyed by
        # the ids mpc_to_od_observations assigns (obsid strings).
        base_veres = mpc_to_od_observations(
            obs_mpc, prevent_nans=True, sigma_model="veres2017"
        )
        base_efcc = mpc_to_od_observations(
            obs_mpc, prevent_nans=True, sigma_model="veres2017",
            catalog_debias_arcsec=efcc,
        )
        base_subtract = mpc_to_od_observations(
            obs_mpc, prevent_nans=True, sigma_model="veres2017",
            catalog_debias_arcsec=efcc,
            bias_application="subtract", bias_table=v1_bias_dict,
        )
        if base_veres is None or base_efcc is None or base_subtract is None:
            records.append(asdict(Row(provid, "ALL", category, failure_reason="od build failed")))
            flush()
            continue

        floors: Dict[str, Tuple[float, float]] = {}
        raw_ids_by_pos = base_veres.id.to_pylist()  # same row order as obs_mpc
        stns_list = obs_mpc.stn.to_pylist()
        cats_list = obs_mpc.astcat.to_pylist()
        for i, oid in enumerate(raw_ids_by_pos):
            floors[oid] = get_veres2017_sigma(stns_list[i], cats_list[i])

        base_veres = base_veres.sort_by(TIME_SORT)
        base_efcc = base_efcc.sort_by(TIME_SORT)
        base_subtract = base_subtract.sort_by(TIME_SORT)
        mjd_all = base_veres.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        arc_days = float(mjd_all.max() - mjd_all.min())
        n_obs = len(base_veres)

        variants: List[Tuple[str, OrbitDeterminationObservations, Optional[ObservationUncertaintyModel]]] = [
            ("no_bias", base_veres, None),
            ("v2_empirical_covar", base_efcc, EmpiricalCovarianceModel(bias_table)),
            ("v2_empcov_replace", base_efcc, EmpiricalCovarianceModel(bias_table, mode="replace")),
            ("v2_empcov_hc", base_efcc, EmpiricalCovarianceModel(bias_table_hc)),
            ("v2_performance_weighted", base_efcc, PerformanceWeightedModel(bias_table)),
            ("v2_sigma_floor", base_efcc, SigmaFloorModel(bias_table)),
            ("v2_empcov_nightbatch", base_efcc, CompositeModel(
                EmpiricalCovarianceModel(bias_table), NightBatchDeweightingModel(cap=4))),
            ("veres_floor_nightbatch", base_efcc, CompositeModel(
                VeresFloorModel(floors), NightBatchDeweightingModel(cap=4))),
            ("v1_subtract_reference", base_subtract, None),
        ]

        # --- IOD (once per object, from the efcc identity baseline) ---
        iod_start = time.time()
        iod_orbit, iod_note = native_iod(base_efcc, propagator)
        logger.info("%s IOD: %s (%.0fs)", provid, iod_note, time.time() - iod_start)
        if iod_orbit is None:
            for name, _, _ in variants:
                if (provid, name) not in done:
                    records.append(asdict(Row(
                        provid, name, category, n_obs, arc_days, iod_note,
                        failure_reason="iod failed")))
            flush()
            continue

        for name, base, model in variants:
            if (provid, name) in done:
                continue
            row = Row(provid, name, category, n_obs, arc_days, iod_note)
            fit_start = time.time()
            try:
                obs_v = model.apply(base) if model is not None else base
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted, members = iterative_fit(
                        iod_orbit, obs_v, propagator, max_nfev=200
                    )
                msgs = [str(w.message) for w in caught]
                row.fallback_fired = any("central-difference" in m for m in msgs)
                bad = [m for m in msgs if "weak-direction" in m and "Falling back" not in m]
                row.cov_warning = bad[0][:160] if bad else ""
                row.reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
                row.n_outliers = int(
                    pc.sum(pc.cast(members.outlier, "int64")).as_py() or 0
                )
                our = Orbits.from_kwargs(
                    orbit_id=fitted.orbit_id,
                    object_id=fitted.object_id,
                    coordinates=fitted.coordinates,
                )
                cc = covariance_comparison(our, jpl_orbit, propagator)
                for k, v in cc.items():
                    setattr(row, k, v)
            except Exception as e:
                row.failure_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
            row.fit_seconds = time.time() - fit_start
            records.append(asdict(row))
            logger.info(
                "[%.1fm] %s/%s rchi2=%.3f ratio=%s maha=%s fb=%s (%.0fs)",
                (time.time() - start) / 60, provid, name, row.reduced_chi2,
                f"{row.pos_sigma_ratio:.3f}" if np.isfinite(row.pos_sigma_ratio) else "NA",
                f"{row.mahalanobis:.2f}" if np.isfinite(row.mahalanobis) else "NA",
                row.fallback_fired, row.fit_seconds,
            )
            flush()

    flush()
    logger.info("DONE: %d rows in %.1f min", len(records), (time.time() - start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
