#!/usr/bin/env python3
"""
run_jpl_cov_ip_comparison.py
============================

Test OD bias methods vs JPL with TWO measures, to see which best discriminates:

  (1) DIRECT COVARIANCE COMPARISON at a common epoch (cheap, no long propagation):
      fit -> propagate our orbit+covariance to JPL's SBDB epoch -> compare our
      Cartesian covariance to JPL's:
        - pos_sigma_ratio = sqrt(tr(C_our_pos)) / sqrt(tr(C_jpl_pos))
          (~1 = realistic; <1 = we are OVERCONFIDENT / too tight)
        - mahalanobis = |r_our - r_jpl| in units of the combined position covariance
  (2) IMPACT PROBABILITY: sample our orbit+covariance, ASSIST-propagate to the
      impact window, our_ip = impacts/variants, vs JPL Sentry IP.

The hypothesis (from the no_bias screening): our covariance is too TIGHT, so IP
is too low; bias inflation (covar_inflation / empirical_covar) should widen the
covariance toward JPL (pos_sigma_ratio -> 1) and raise IP toward JPL.

REQUIRES the codex/nongrav-support branch of adam_core + adam-assist. Run with:
  PYTHONPATH=<adam_core_wt>/src:<adam-assist_wt>/src pdm run python scripts/run_jpl_cov_ip_comparison.py

Methods: no_bias, efcc18_only, v2_sigma_floor, v2_performance_weighted,
v2_covar_inflation, v2_empirical_covar.
Test set: data/ip_risklist/cov_ip_testset.parquet (good-fit discrepant + controls).
Output: data/jpl_cov_ip/ (comparison.parquet + REPORT.md)
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

from adam_assist import ASSISTPropagator
from adam_core.orbits import Orbits
from adam_core.orbits.query.sbdb import query_sbdb_new
from adam_core.dynamics.impacts import calculate_impacts, calculate_impact_probabilities
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.efcc18 import compute_efcc18_corrections, load_efcc18_biases
from adam_orbit_det_eval.utils import (
    get_spacebased_stns, load_v2_bias_catalog, mpc_to_od_observations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("jpl_cov_ip")
logging.getLogger("adam_core").setLevel(logging.WARNING)

AU_KM = 1.495978707e8
DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
RESID_COV_N_THRESHOLD = 30
NOW_YEAR = 2026


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_efcc18: bool = False
    use_station_chi2: bool = False
    use_resid_covar: bool = False


VARIANTS: List[VariantConfig] = [
    VariantConfig("no_bias"),
    VariantConfig("efcc18_only", use_efcc18=True),
    VariantConfig("v2_sigma_floor", bias_application="sigma_floor", use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_performance_weighted", bias_application="performance_weighted", use_bias_table=True, use_station_chi2=True, use_efcc18=True),
    VariantConfig("v2_covar_inflation", bias_application="covar_inflation", use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_empirical_covar", bias_application="empirical_covar", use_resid_covar=True, use_efcc18=True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/jpl_cov_ip"))
    p.add_argument("--manifest", type=Path, default=Path("data/ip_risklist/cov_ip_testset.parquet"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--cov-samples", type=int, default=200, help="sigma-point/MC samples for cov propagation")
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--project", default="moeyens-thor-dev")
    p.add_argument("--dataset-id", default="mpc_sbn_aurora")
    p.add_argument("--views-dataset-id", default="mpc_sbn_aurora_views")
    p.add_argument("--fo-result-dir", default="/tmp/fo_covip_runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--skip-ip", action="store_true", help="covariance comparison only (fast)")
    return p.parse_args()


@dataclass
class V2Catalogs:
    bias_table: Dict[str, Tuple[float, float]]
    station_chi2: Dict[str, float]
    station_resid_covar: Dict[str, Tuple[float, float, float, float]]


def build_v2_catalogs(path: str) -> V2Catalogs:
    cat = load_v2_bias_catalog(path, rollup_only=True)
    bt, ch, rc = {}, {}, {}
    for code, rec in cat.items():
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"]):
            bt[code] = (rec["bias_ra"], rec["bias_dec"])
        if np.isfinite(rec["chi2_per_obs"]):
            ch[code] = rec["chi2_per_obs"]
        rc[code] = (rec["resid_var_ra"], rec["resid_var_dec"], rec["resid_cov_ra_dec"], rec["resid_cov_n"])
    return V2Catalogs(bt, ch, rc)


def _dedupe(obs):
    stns = obs.stn.to_pylist()
    t = obs.obstime.mjd().to_numpy(zero_copy_only=False) * 86400.0
    order = np.lexsort((t, np.asarray(stns, dtype=object)))
    keep = np.zeros(len(obs), dtype=bool); last: Dict[str, float] = {}
    for i in order:
        s = stns[i]; ti = float(t[i]); pv = last.get(s)
        if pv is None or abs(ti - pv) > 1.5:
            keep[i] = True; last[s] = ti
    if keep.sum() == len(obs):
        return obs
    d = obs.apply_mask(pa.array(keep))
    return qv.concatenate([d]) if d.fragmented() else d


def prepare(provid, client, spacebased, efcc18_bias_table):
    raw = client.query_observations([provid])
    if raw is None or len(raw) == 0:
        return None
    stns = raw.stn.to_pylist()
    gm = np.array([(s is not None) and (s not in spacebased) for s in stns], dtype=bool)
    if not gm.any():
        return None
    g = raw.apply_mask(pa.array(gm))
    if g.fragmented():
        g = qv.concatenate([g])
    d = _dedupe(g)
    times = d.obstime.mjd().to_numpy(zero_copy_only=False)
    efcc = compute_efcc18_corrections(
        d.ra.to_numpy(zero_copy_only=False), d.dec.to_numpy(zero_copy_only=False),
        d.astcat.to_pylist(), times + 2400000.5, bias_table=efcc18_bias_table)
    return d, efcc, float(np.ptp(times))


def build_od(obs, efcc, v: VariantConfig, cats: V2Catalogs):
    kw: Dict[str, object] = {"prevent_nans": True, "sigma_model": "veres2017", "bias_application": v.bias_application}
    if v.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if v.use_efcc18:
        kw["catalog_debias_arcsec"] = efcc
    if v.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if v.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


def _cart_state_cov(orbit_at_epoch):
    """Return (position AU (3,), 6x6 Cartesian covariance) for a 1-orbit table."""
    coords = orbit_at_epoch.coordinates
    vals = coords.values[0]  # x,y,z,vx,vy,vz (au, au/day) — propagate_orbits returns cartesian
    cov = coords.covariance.to_matrix()[0]
    return np.asarray(vals[:3], dtype=float), np.asarray(cov, dtype=float)


def covariance_comparison(our_orbit, jpl_orbit, propagator, cov_samples):
    """Propagate both to JPL's epoch (cartesian, with covariance) and compare."""
    jpl_time = jpl_orbit.coordinates.time
    our_at = propagator.propagate_orbits(our_orbit, jpl_time, covariance=True,
                                         covariance_method="sigma-point", num_samples=cov_samples)
    jpl_at = propagator.propagate_orbits(jpl_orbit, jpl_time, covariance=True,
                                         covariance_method="sigma-point", num_samples=cov_samples)
    r_our, C_our = _cart_state_cov(our_at)
    r_jpl, C_jpl = _cart_state_cov(jpl_at)
    Cp_our, Cp_jpl = C_our[:3, :3], C_jpl[:3, :3]
    pos_sig_our = float(np.sqrt(np.trace(Cp_our)))
    pos_sig_jpl = float(np.sqrt(np.trace(Cp_jpl)))
    ratio = pos_sig_our / pos_sig_jpl if pos_sig_jpl > 0 else float("nan")
    dr = r_our - r_jpl
    try:
        maha = float(np.sqrt(dr @ np.linalg.solve(Cp_our + Cp_jpl, dr)))
    except Exception:
        maha = float("nan")
    return dict(pos_sigma_ratio=ratio, pos_sigma_our_km=pos_sig_our * AU_KM,
                pos_sigma_jpl_km=pos_sig_jpl * AU_KM, mahalanobis=maha,
                dr_km=float(np.linalg.norm(dr)) * AU_KM)


@dataclass
class Result:
    object_id: str
    method: str
    category: str = ""
    jpl_ip: float = float("nan")
    reduced_chi2: float = float("nan")
    n_obs: int = 0
    arc_days: float = float("nan")
    pos_sigma_ratio: float = float("nan")
    pos_sigma_our_km: float = float("nan")
    pos_sigma_jpl_km: float = float("nan")
    mahalanobis: float = float("nan")
    dr_km: float = float("nan")
    our_ip: float = float("nan")
    n_impacts: int = -1
    failure_reason: str = ""


def write_all(records, outdir, num_samples):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(outdir / "comparison.parquet", index=False)
    ok = df[df.pos_sigma_ratio.notna()]
    methods = [v.variant_id for v in VARIANTS if v.variant_id in set(df.method)]
    md = ["# OD methods vs JPL — covariance comparison & IP", "",
          f"Generated: {datetime.now(timezone.utc).isoformat()}. IP MC samples: {num_samples}.",
          "Covariance compared at JPL's SBDB epoch. pos_sigma_ratio = our/JPL position "
          "uncertainty (1=realistic, <1=overconfident). mahalanobis = mean offset in "
          "combined-covariance units. Nongrav branch.", ""]
    # which measure best: per method, aggregate both measures (discrepant objects only)
    disc = ok[ok.category == "discrepant"]
    md.append("## Method aggregates on DISCREPANT objects (the ones with headroom)")
    md.append("")
    md.append("| method | median pos_sigma_ratio | median mahalanobis | median our_IP | median IP/JPL |")
    md.append("|---|---|---|---|---|")
    for m in methods:
        s = disc[disc.method == m]
        if s.empty:
            continue
        ipr = (s.our_ip / s.jpl_ip).replace([np.inf, -np.inf], np.nan).dropna()
        md.append(f"| `{m}` | {s.pos_sigma_ratio.median():.3f} | {s.mahalanobis.median():.2f} | "
                  f"{s.our_ip.median():.2e} | {ipr.median():.2f} |")
    md.append("")
    md.append("Reading: pos_sigma_ratio→1 and mahalanobis→small = covariance matches JPL; "
              "IP/JPL→1 = IP matches JPL. Compare which measure moves most coherently "
              "from no_bias toward JPL as bias handling is added.")
    md.append("")
    # per-object detail
    md.append("## Per-object detail")
    md.append("")
    md.append("| object | cat | method | χ² | pos_σ ratio | maha | our_IP | JPL_IP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for oid in df.object_id.unique():
        for m in methods:
            r = df[(df.object_id == oid) & (df.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            md.append(f"| {oid} | {r.category} | {m} | {r.reduced_chi2:.2f} | "
                      f"{r.pos_sigma_ratio:.3f} | {r.mahalanobis:.2f} | "
                      f"{r.our_ip:.2e} | {r.jpl_ip:.2e} |")
    md.append("")
    (outdir / "REPORT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(args.manifest)
    if args.max_objects:
        man = man.head(args.max_objects)
    variants = [v for v in VARIANTS if (args.methods is None or v.variant_id in args.methods)]
    logger.info("cov+IP: %d objects × %d methods (skip_ip=%s)", len(man), len(variants), args.skip_ip)

    cats = build_v2_catalogs(str(args.v2_catalog_path))
    efcc18_bias_table = load_efcc18_biases()
    spacebased = set(get_spacebased_stns())
    client = BigQueryMPCClient(dataset_id=args.dataset_id, views_dataset_id=args.views_dataset_id, project=args.project)
    prop = ASSISTPropagator()
    fitter = FindOrbOrbitFitter(fo_result_dir=args.fo_result_dir, clean_up_fo_dir=True, propagator=prop)

    records: List[dict] = []
    done = set()
    if args.resume and (args.output_dir / "comparison.parquet").exists():
        prev = pd.read_parquet(args.output_dir / "comparison.parquet")
        records = prev.to_dict(orient="records")
        done = {(r["object_id"], r["method"]) for r in records}

    start = time.time()
    for _, row in man.iterrows():
        provid = row["provid"]; jpl_ip = float(row["jpl_ip"]); category = row.get("category", "")
        num_days = int((int(row["range_end"]) - NOW_YEAR) * 365.25)
        try:
            jpl_orbit = query_sbdb_new([provid])
        except Exception as e:
            logger.warning("%s SBDB failed: %s", provid, e); jpl_orbit = None
        try:
            prep = prepare(provid, client, spacebased, efcc18_bias_table)
        except Exception as e:
            logger.warning("%s prepare failed: %s", provid, e); prep = None
        if prep is None or jpl_orbit is None:
            for v in variants:
                records.append(asdict(Result(provid, v.variant_id, category, jpl_ip,
                                             failure_reason="prep/SBDB failed")))
            continue
        obs, efcc, arc_days = prep
        for v in variants:
            if (provid, v.variant_id) in done:
                continue
            rec = Result(provid, v.variant_id, category, jpl_ip, n_obs=len(obs), arc_days=arc_days)
            try:
                od = build_od(obs, efcc, v, cats)
                fitted, _ = fitter.initial_fit(provid, od)
                if len(fitted) == 0:
                    rec.failure_reason = "empty fit"; records.append(asdict(rec)); continue
                rec.reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
                orb = Orbits.from_kwargs(orbit_id=fitted.orbit_id, object_id=fitted.object_id,
                                         coordinates=fitted.coordinates)
                # (1) covariance comparison
                try:
                    cc = covariance_comparison(orb, jpl_orbit, prop, args.cov_samples)
                    for k, val in cc.items():
                        setattr(rec, k, val)
                except Exception as e:
                    rec.failure_reason += f"cov:{type(e).__name__}:{e}; "
                # (2) IP
                if not args.skip_ip:
                    try:
                        res, col = calculate_impacts(orb, num_days, prop, num_samples=args.num_samples,
                                                     processes=args.processes, seed=20260701)
                        ipr = calculate_impact_probabilities(res, col)
                        rec.our_ip = float(ipr.cumulative_probability[0].as_py())
                        rec.n_impacts = int(ipr.impacts[0].as_py())
                    except Exception as e:
                        rec.failure_reason += f"ip:{type(e).__name__}:{e}; "
            except Exception as e:
                rec.failure_reason += f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
            records.append(asdict(rec))
            logger.info("[%.1fm] %s/%s chi2=%.2f pos_ratio=%s maha=%s our_ip=%s jpl=%.2e",
                        (time.time()-start)/60, provid, v.variant_id, rec.reduced_chi2,
                        f"{rec.pos_sigma_ratio:.3f}" if np.isfinite(rec.pos_sigma_ratio) else "NA",
                        f"{rec.mahalanobis:.2f}" if np.isfinite(rec.mahalanobis) else "NA",
                        f"{rec.our_ip:.2e}" if np.isfinite(rec.our_ip) else "NA", jpl_ip)
        write_all(records, args.output_dir, args.num_samples)

    write_all(records, args.output_dir, args.num_samples)
    logger.info("Done: %d rows, %.1f min", len(records), (time.time()-start)/60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
