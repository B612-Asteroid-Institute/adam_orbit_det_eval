#!/usr/bin/env python3
"""
run_covariance_sweep.py
=======================

DEFINITIVE search for the ideal bias-handling method, using the measure that
works: direct same-epoch covariance comparison vs JPL. Large, wide pool (NEOs +
MBAs across arc/quality strata), all σ-modification methods, fit-quality gated.

For each (object, method): fit -> propagate our orbit+covariance to JPL's SBDB
epoch -> compare Cartesian position covariance to JPL's:
  pos_sigma_ratio = sqrt(tr(C_our_pos)) / sqrt(tr(C_jpl_pos))   (1 = realistic,
    <1 = overconfident/too tight, >1 = too loose)
  mahalanobis     = |r_our - r_jpl| in combined-covariance units
  reduced_chi2    = in-sample fit quality (target ~1)

The ideal method: brings pos_sigma_ratio toward 1 (fixes overconfidence) and
chi2 toward 1 on objects that need it, WITHOUT over-inflating objects already
well-calibrated. Ranked across the pool.

Fit-quality gate: objects whose no_bias fit is pathological (reduced_chi2 > GATE
or non-finite — gravity-only can't fit them; they need nongrav) are excluded
from the method comparison and flagged.

REQUIRES codex/nongrav-support branch (query_sbdb_new). Run with the worktrees on
PYTHONPATH; Ray workers inherit it via runtime_env.

Output: data/cov_sweep/ (covariance_sweep.parquet + REPORT.md)
"""

from __future__ import annotations

import argparse
import logging
import os
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
from adam_core.orbits.query.sbdb import query_sbdb_new
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.efcc18 import compute_efcc18_corrections, load_efcc18_biases
from adam_orbit_det_eval.utils import (
    get_spacebased_stns, load_v2_bias_catalog, mpc_to_od_observations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("cov_sweep")
logging.getLogger("adam_core").setLevel(logging.WARNING)

AU_KM = 1.495978707e8
DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
RESID_COV_N_THRESHOLD = 30
CHI2_GATE = 8.0   # no_bias reduced_chi2 above this => gravity-only fit fails => exclude
MAX_OBS_PER_OBJECT = 2000


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    bias_application: str = "sigma_floor"
    use_bias_table: bool = False
    use_efcc18: bool = False
    use_station_chi2: bool = False
    use_station_sem: bool = False
    use_resid_covar: bool = False


VARIANTS: List[VariantConfig] = [
    VariantConfig("no_bias"),
    VariantConfig("efcc18_only", use_efcc18=True),
    VariantConfig("v2_sigma_floor", bias_application="sigma_floor", use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_rss_additive", bias_application="rss_additive", use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_performance_weighted", bias_application="performance_weighted", use_bias_table=True, use_station_chi2=True, use_efcc18=True),
    VariantConfig("v2_bayes_shrinkage", bias_application="bayes_shrinkage", use_bias_table=True, use_station_sem=True, use_efcc18=True),
    VariantConfig("v2_covar_inflation", bias_application="covar_inflation", use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_empirical_covar", bias_application="empirical_covar", use_resid_covar=True, use_efcc18=True),
    VariantConfig("veres_v2_max_floor", bias_application="veres_v1_max_floor", use_bias_table=True, use_efcc18=True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/cov_sweep"))
    p.add_argument("--manifest", type=Path, default=Path("data/cov_sweep/pool_manifest.parquet"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--cov-samples", type=int, default=100)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--per-object-timeout", type=float, default=600.0)
    p.add_argument("--project", default="moeyens-thor-dev")
    p.add_argument("--dataset-id", default="mpc_sbn_aurora")
    p.add_argument("--views-dataset-id", default="mpc_sbn_aurora_views")
    p.add_argument("--fo-result-dir", default="/tmp/fo_cov_sweep")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None)
    return p.parse_args()


@dataclass
class V2Catalogs:
    bias_table: Dict[str, Tuple[float, float]]
    station_chi2: Dict[str, float]
    station_sem: Dict[str, Tuple[float, float]]
    station_resid_covar: Dict[str, Tuple[float, float, float, float]]


def build_v2_catalogs(path: str) -> V2Catalogs:
    cat = load_v2_bias_catalog(path, rollup_only=True)
    bt, ch, sem, rc = {}, {}, {}, {}
    for code, rec in cat.items():
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"]):
            bt[code] = (rec["bias_ra"], rec["bias_dec"])
        if np.isfinite(rec["chi2_per_obs"]):
            ch[code] = rec["chi2_per_obs"]
        if np.isfinite(rec["sem_ra"]) and np.isfinite(rec["sem_dec"]):
            sem[code] = (rec["sem_ra"], rec["sem_dec"])
        rc[code] = (rec["resid_var_ra"], rec["resid_var_dec"], rec["resid_cov_ra_dec"], rec["resid_cov_n"])
    return V2Catalogs(bt, ch, sem, rc)


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
    if len(d) > MAX_OBS_PER_OBJECT:
        t = d.obstime.mjd().to_numpy(zero_copy_only=False)
        keep = np.zeros(len(d), dtype=bool); keep[np.argsort(t)[-MAX_OBS_PER_OBJECT:]] = True
        d = d.apply_mask(pa.array(keep))
        if d.fragmented():
            d = qv.concatenate([d])
    times = d.obstime.mjd().to_numpy(zero_copy_only=False)
    efcc = compute_efcc18_corrections(
        d.ra.to_numpy(zero_copy_only=False), d.dec.to_numpy(zero_copy_only=False),
        d.astcat.to_pylist(), times + 2400000.5, bias_table=efcc18_bias_table)
    return d, efcc, float(np.ptp(times)), len(d)


def build_od(obs, efcc, v: VariantConfig, cats: V2Catalogs):
    kw: Dict[str, object] = {"prevent_nans": True, "sigma_model": "veres2017", "bias_application": v.bias_application}
    if v.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if v.use_efcc18:
        kw["catalog_debias_arcsec"] = efcc
    if v.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if v.use_station_sem:
        kw["station_sem_arcsec"] = cats.station_sem
    if v.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


def _cart(orbit_at):
    c = orbit_at.coordinates
    vals = np.asarray(c.values[0], dtype=float)
    cov = np.asarray(c.covariance.to_matrix()[0], dtype=float)
    return vals[:3], cov


def compare_to_jpl(our_orbit, jpl_r, jpl_Cpos, jpl_time, propagator, cov_samples):
    our_at = propagator.propagate_orbits(our_orbit, jpl_time, covariance=True,
                                         covariance_method="sigma-point", num_samples=cov_samples)
    r_our, C_our = _cart(our_at)
    Cp_our = C_our[:3, :3]
    ps_our = float(np.sqrt(np.trace(Cp_our))); ps_jpl = float(np.sqrt(np.trace(jpl_Cpos)))
    ratio = ps_our / ps_jpl if ps_jpl > 0 else float("nan")
    dr = r_our - jpl_r
    try:
        maha = float(np.sqrt(dr @ np.linalg.solve(Cp_our + jpl_Cpos, dr)))
    except Exception:
        maha = float("nan")
    return ratio, maha, ps_our * AU_KM, ps_jpl * AU_KM, float(np.linalg.norm(dr)) * AU_KM


@dataclass
class Result:
    object_id: str
    stratum: str
    method: str
    gated: bool = False
    reduced_chi2: float = float("nan")
    n_obs: int = 0
    arc_days: float = float("nan")
    pos_sigma_ratio: float = float("nan")
    pos_sigma_our_km: float = float("nan")
    pos_sigma_jpl_km: float = float("nan")
    mahalanobis: float = float("nan")
    dr_km: float = float("nan")
    failure_reason: str = ""


@ray.remote
class Worker:
    def __init__(self, idx, args_dict, cats, efcc18_bias_table):
        logging.basicConfig(level=logging.WARNING)
        self.cats = cats
        self.efcc18 = efcc18_bias_table
        self.spacebased = set(get_spacebased_stns())
        self.client = BigQueryMPCClient(dataset_id=args_dict["dataset_id"],
                                        views_dataset_id=args_dict["views_dataset_id"],
                                        project=args_dict["project"])
        self.prop = ASSISTPropagator()
        fo = f"{args_dict['fo_result_dir']}/actor_{idx}"; Path(fo).mkdir(parents=True, exist_ok=True)
        self.fitter = FindOrbOrbitFitter(fo_result_dir=fo, clean_up_fo_dir=True, propagator=self.prop)
        self.cov_samples = args_dict["cov_samples"]

    def process(self, row):
        provid = row["provid"]; stratum = row.get("stratum", "")
        def gated_rows(reason):
            return [asdict(Result(provid, stratum, v.variant_id, gated=True, failure_reason=reason))
                    for v in VARIANTS]
        try:
            jpl = query_sbdb_new([provid])
        except Exception as e:
            return gated_rows(f"sbdb:{type(e).__name__}:{str(e)[:60]}")
        try:
            prep = prepare(provid, self.client, self.spacebased, self.efcc18)
        except Exception as e:
            return gated_rows(f"prep:{type(e).__name__}:{str(e)[:60]}")
        if prep is None or jpl is None:
            return gated_rows("prep/sbdb None")
        obs, efcc, arc_days, n_obs = prep
        # propagate JPL to its own epoch once (cartesian + covariance)
        try:
            jpl_at = self.prop.propagate_orbits(jpl, jpl.coordinates.time, covariance=True,
                                                covariance_method="sigma-point", num_samples=self.cov_samples)
            jpl_r, jpl_C = _cart(jpl_at); jpl_Cpos = jpl_C[:3, :3]
            if not np.all(np.isfinite(jpl_Cpos)):
                return gated_rows("jpl cov non-finite")
        except Exception as e:
            return gated_rows(f"jpl_prop:{type(e).__name__}:{str(e)[:60]}")

        out = []
        # fit no_bias first for the gate
        try:
            od0 = build_od(obs, efcc, VARIANTS[0], self.cats)
            f0, _ = self.fitter.initial_fit(provid, od0)
            chi0 = float(f0.reduced_chi2[0].as_py()) if len(f0) else float("inf")
        except Exception as e:
            return gated_rows(f"no_bias_fit:{type(e).__name__}:{str(e)[:60]}")
        if len(f0) == 0 or not np.isfinite(chi0) or chi0 > CHI2_GATE or chi0 < 0.1:
            rows = gated_rows(f"no_bias chi2={chi0:.1f} (gate {CHI2_GATE})")
            for r in rows:
                r["reduced_chi2"] = chi0; r["n_obs"] = n_obs; r["arc_days"] = arc_days
            return rows

        for i, v in enumerate(VARIANTS):
            rec = Result(provid, stratum, v.variant_id, n_obs=n_obs, arc_days=arc_days)
            try:
                if i == 0:
                    fitted = f0; rec.reduced_chi2 = chi0
                else:
                    od = build_od(obs, efcc, v, self.cats)
                    fitted, _ = self.fitter.initial_fit(provid, od)
                    if len(fitted) == 0:
                        rec.failure_reason = "empty fit"; out.append(asdict(rec)); continue
                    rec.reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
                orb = Orbits.from_kwargs(orbit_id=fitted.orbit_id, object_id=fitted.object_id,
                                         coordinates=fitted.coordinates)
                ratio, maha, pso, psj, drkm = compare_to_jpl(orb, jpl_r, jpl_Cpos,
                                                             jpl.coordinates.time, self.prop, self.cov_samples)
                rec.pos_sigma_ratio = ratio; rec.mahalanobis = maha
                rec.pos_sigma_our_km = pso; rec.pos_sigma_jpl_km = psj; rec.dr_km = drkm
            except Exception as e:
                rec.failure_reason = f"{type(e).__name__}:{str(e)[:80]}"
            out.append(asdict(rec))
        return out


def write_all(records, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(outdir / "covariance_sweep.parquet", index=False)
    ok = df[(~df.gated) & df.pos_sigma_ratio.notna()]
    methods = [v.variant_id for v in VARIANTS]
    md = ["# Definitive covariance-vs-JPL sweep", "",
          f"Generated: {datetime.now(timezone.utc).isoformat()}",
          f"Pool: {df.object_id.nunique()} objects; well-fit (gate passed): "
          f"{ok.object_id.nunique()}; gated (bad fit): {df[df.gated].object_id.nunique()}.",
          "pos_sigma_ratio = our/JPL position-uncertainty at JPL epoch (1=realistic). "
          "|log10 ratio| = calibration error (0=perfect).", ""]
    if not ok.empty:
        md.append("## Method ranking — calibration to JPL (all well-fit objects)")
        md.append("")
        md.append("| method | median pos_σ ratio | median \\|log10 ratio\\| | median maha | median χ² | n |")
        md.append("|---|---|---|---|---|---|")
        rank = []
        for m in methods:
            s = ok[ok.method == m]
            if s.empty:
                continue
            lr = np.abs(np.log10(s.pos_sigma_ratio.replace(0, np.nan))).dropna()
            rank.append((m, lr.median(), s.pos_sigma_ratio.median(), s.mahalanobis.median(),
                         s.reduced_chi2.median(), len(s)))
        for m, lr, r, mh, c, n in sorted(rank, key=lambda x: x[1]):
            md.append(f"| `{m}` | {r:.3f} | {lr:.3f} | {mh:.2f} | {c:.2f} | {n} |")
        md.append("")
        md.append("Lower |log10 ratio| = better calibrated to JPL. Ideal method = "
                  "lowest |log10 ratio| with χ² near 1.")
        md.append("")
        # by stratum
        md.append("## Median pos_σ ratio by stratum")
        md.append("")
        strata = sorted(ok.stratum.dropna().unique())
        md.append("| method | " + " | ".join(strata) + " |")
        md.append("|" + "---|" * (len(strata) + 1))
        for m in methods:
            cells = []
            for st in strata:
                s = ok[(ok.method == m) & (ok.stratum == st)]
                cells.append(f"{s.pos_sigma_ratio.median():.2f}" if len(s) else "—")
            md.append(f"| `{m}` | " + " | ".join(cells) + " |")
        md.append("")
    (outdir / "REPORT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(args.manifest)
    if args.max_objects:
        man = man.head(args.max_objects)
    logger.info("cov sweep: %d objects × %d methods", len(man), len(VARIANTS))
    cats = build_v2_catalogs(str(args.v2_catalog_path))
    efcc18_bias_table = load_efcc18_biases()

    records: List[dict] = []
    done = set()
    if args.resume and (args.output_dir / "covariance_sweep.parquet").exists():
        prev = pd.read_parquet(args.output_dir / "covariance_sweep.parquet")
        records = prev.to_dict(orient="records"); done = set(prev.object_id.unique())

    targets = [r for r in man.to_dict(orient="records") if r["provid"] not in done]
    if not targets:
        write_all(records, args.output_dir); return 0

    ray.init(num_cpus=args.n_workers, ignore_reinit_error=True, include_dashboard=False,
             logging_level=logging.WARNING,
             runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    cats_ref = ray.put(cats); efcc_ref = ray.put(efcc18_bias_table)
    ad = {"project": args.project, "dataset_id": args.dataset_id, "views_dataset_id": args.views_dataset_id,
          "fo_result_dir": args.fo_result_dir, "cov_samples": args.cov_samples}
    workers = [Worker.remote(i, ad, cats_ref, efcc_ref) for i in range(args.n_workers)]
    start = time.time(); n_done = len(done); n_total = len(man); timeout_s = args.per_object_timeout
    queue = list(targets); inflight: Dict = {}

    def submit(w):
        if queue:
            row = queue.pop(0); ref = workers[w].process.remote(row); inflight[ref] = [w, row, time.time()]

    def record(rows):
        nonlocal n_done
        records.extend(rows); n_done += 1
        if rows and (n_done % 10 == 0):
            oid = rows[0]["object_id"]
            logger.info("[%d/%d] %s (cum %.1f min)", n_done, n_total, oid, (time.time()-start)/60)
            write_all(records, args.output_dir)

    for w in range(len(workers)):
        submit(w)
    while inflight:
        d, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=15.0); now = time.time()
        if d:
            ref = d[0]; w, row, _ = inflight.pop(ref)
            try:
                rows = ray.get(ref)
            except Exception as e:
                rows = [asdict(Result(row["provid"], row.get("stratum", ""), v.variant_id, gated=True,
                                      failure_reason=f"actor:{type(e).__name__}")) for v in VARIANTS]
            record(rows); submit(w)
        for ref in list(inflight.keys()):
            w, row, st = inflight[ref]
            if now - st > timeout_s:
                logger.warning("[timeout] %s — kill+replace", row["provid"])
                try: ray.kill(workers[w])
                except Exception: pass
                inflight.pop(ref)
                record([asdict(Result(row["provid"], row.get("stratum", ""), v.variant_id, gated=True,
                                      failure_reason="timeout")) for v in VARIANTS])
                workers[w] = Worker.remote(w, ad, cats_ref, efcc_ref); submit(w)

    write_all(records, args.output_dir)
    logger.info("Done: %d rows, %.1f min", len(records), (time.time()-start)/60)
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
