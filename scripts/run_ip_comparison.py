#!/usr/bin/env python3
"""
run_ip_comparison.py
====================

Impact-probability (IP) comparison vs JPL Sentry — the covariance-realism test.

The orbit-position and held-out tests showed bias handling barely moves the
orbit. But the documented benefit of bias measurements (FCCT14/EFCC18, Veres,
the rigorous error models) is in the realism of the *covariance* — which is what
drives impact probability (Apophis & 2024 YR4 IPs each moved ~an order of
magnitude under debiasing/reweighting). So we test where the signal lives:

For each risk-list object × method: fit the orbit + covariance, sample N
Monte-Carlo variants from that covariance, propagate with ASSIST to the impact
window, and compute **our IP = impacts / variants**. Compare to JPL's published
Sentry IP. The question: does any catalog lever make our IP track JPL's better
than no_bias / efcc18_only?

Methods (the "reasonable" OD ones): no_bias, efcc18_only, v2_sigma_floor,
v2_performance_weighted, v2_covar_inflation, v2_empirical_covar.

Population: data/ip_risklist/risklist_manifest.parquet (provid, jpl_ip, range_end).

Outputs (default data/ip_comparison/)
    ip_comparison.parquet   one row per (object, method)
    REPORT.md               our IP vs JPL per method

Usage
    pdm run python scripts/run_ip_comparison.py [--num-samples 5000] [--max-objects N]
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
from adam_core.dynamics.impacts import calculate_impacts, calculate_impact_probabilities
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.efcc18 import compute_efcc18_corrections, load_efcc18_biases
from adam_orbit_det_eval.utils import (
    get_spacebased_stns, load_v2_bias_catalog, mpc_to_od_observations,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ip_comparison")
logging.getLogger("adam_core").setLevel(logging.WARNING)

DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
MAX_OBS_PER_OBJECT = 2000
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
    VariantConfig("v2_sigma_floor", bias_application="sigma_floor",
                  use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_performance_weighted", bias_application="performance_weighted",
                  use_bias_table=True, use_station_chi2=True, use_efcc18=True),
    VariantConfig("v2_covar_inflation", bias_application="covar_inflation",
                  use_bias_table=True, use_efcc18=True),
    VariantConfig("v2_empirical_covar", bias_application="empirical_covar",
                  use_resid_covar=True, use_efcc18=True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/ip_comparison"))
    p.add_argument("--manifest", type=Path,
                   default=Path("data/ip_risklist/risklist_manifest.parquet"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--project", default="moeyens-thor-dev")
    p.add_argument("--dataset-id", default="mpc_sbn_aurora")
    p.add_argument("--views-dataset-id", default="mpc_sbn_aurora_views")
    p.add_argument("--fo-result-dir", default="/tmp/fo_ip_runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--methods", nargs="*", default=None,
                   help="subset of variant_ids to run (default: all)")
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
        rc[code] = (rec["resid_var_ra"], rec["resid_var_dec"],
                    rec["resid_cov_ra_dec"], rec["resid_cov_n"])
    return V2Catalogs(bt, ch, rc)


def _dedupe(obs: MPCObservations) -> MPCObservations:
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
    return d, efcc, float(np.ptp(times))


def build_od(obs, efcc, variant: VariantConfig, cats: V2Catalogs):
    kw: Dict[str, object] = {"prevent_nans": True, "sigma_model": "veres2017",
                             "bias_application": variant.bias_application}
    if variant.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if variant.use_efcc18:
        kw["catalog_debias_arcsec"] = efcc
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if variant.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


@dataclass
class IPResult:
    object_id: str
    method: str
    jpl_ip: float
    our_ip: float = float("nan")
    n_impacts: int = -1
    n_variants: int = -1
    num_days: int = -1
    reduced_chi2: float = float("nan")
    n_obs: int = 0
    arc_days: float = float("nan")
    failure_reason: str = ""


def write_out(records: List[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(outdir / "ip_comparison.parquet", index=False)


def write_report(records: List[dict], outdir: Path, num_samples: int) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        return
    md = ["# Impact-probability comparison vs JPL Sentry", "",
          f"Generated: {datetime.now(timezone.utc).isoformat()}",
          f"Monte-Carlo variants: {num_samples}. our_ip = impacts/variants, "
          "propagated with ASSIST to the impact-window end. Compared to JPL's "
          "published Sentry IP.", ""]
    ok = df[df.our_ip.notna()]
    # per-object table (no_bias + the levers vs JPL)
    md.append("## Per-object IP: JPL vs each method")
    md.append("")
    methods = [v.variant_id for v in VARIANTS if v.variant_id in set(df.method)]
    md.append("| object | JPL IP | " + " | ".join(methods) + " |")
    md.append("|" + "---|" * (len(methods) + 2))
    for oid in df.object_id.unique():
        sub = df[df.object_id == oid]
        jip = sub.jpl_ip.iloc[0]
        cells = []
        for m in methods:
            r = sub[sub.method == m]
            if len(r) and np.isfinite(r.our_ip.iloc[0]):
                cells.append(f"{r.our_ip.iloc[0]:.2e}")
            else:
                cells.append("—")
        md.append(f"| {oid} | {jip:.2e} | " + " | ".join(cells) + " |")
    md.append("")
    # which method tracks JPL best: median |log10(our_ip/jpl_ip)| (closer to 0 = better)
    md.append("## Which method tracks JPL best?")
    md.append("")
    md.append("Median |log10(our_IP / JPL_IP)| over objects (0 = perfect; lower = better). "
              "Only objects where our_ip>0 counted.")
    md.append("")
    md.append("| method | median \\|log10 ratio\\| | median our_IP/JPL | n |")
    md.append("|---|---|---|---|")
    for m in methods:
        s = ok[(ok.method == m) & (ok.our_ip > 0)]
        if s.empty:
            md.append(f"| `{m}` | — | — | 0 |"); continue
        lr = np.abs(np.log10(s.our_ip / s.jpl_ip))
        rr = s.our_ip / s.jpl_ip
        md.append(f"| `{m}` | {lr.median():.3f} | {rr.median():.2f} | {len(s)} |")
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
    logger.info("IP comparison: %d objects × %d methods, %d samples",
                len(man), len(variants), args.num_samples)

    cats = build_v2_catalogs(str(args.v2_catalog_path))
    efcc18_bias_table = load_efcc18_biases()
    spacebased = set(get_spacebased_stns())
    client = BigQueryMPCClient(dataset_id=args.dataset_id,
                               views_dataset_id=args.views_dataset_id, project=args.project)
    prop = ASSISTPropagator()
    fitter = FindOrbOrbitFitter(fo_result_dir=args.fo_result_dir,
                                clean_up_fo_dir=True, propagator=prop)

    records: List[dict] = []
    done = set()
    out_parquet = args.output_dir / "ip_comparison.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        records = prev.to_dict(orient="records")
        done = {(r["object_id"], r["method"]) for r in records}

    start = time.time()
    for _, row in man.iterrows():
        provid = row["provid"]; jpl_ip = float(row["jpl_ip"])
        num_days = int((int(row["range_end"]) - NOW_YEAR) * 365.25)
        try:
            prep = prepare(provid, client, spacebased, efcc18_bias_table)
        except Exception as e:
            logger.warning("%s prepare failed: %s", provid, e); prep = None
        if prep is None:
            for v in variants:
                records.append(asdict(IPResult(provid, v.variant_id, jpl_ip,
                                               failure_reason="prepare failed")))
            continue
        obs, efcc, arc_days = prep
        for v in variants:
            if (provid, v.variant_id) in done:
                continue
            rec = IPResult(provid, v.variant_id, jpl_ip, num_days=num_days,
                           n_obs=len(obs), arc_days=arc_days)
            try:
                od = build_od(obs, efcc, v, cats)
                fitted, _ = fitter.initial_fit(provid, od)
                if len(fitted) == 0:
                    rec.failure_reason = "empty fit"; records.append(asdict(rec)); continue
                rec.reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
                orb = Orbits.from_kwargs(orbit_id=fitted.orbit_id,
                                         object_id=fitted.object_id,
                                         coordinates=fitted.coordinates)
                results, collisions = calculate_impacts(
                    orb, num_days, prop, num_samples=args.num_samples,
                    processes=args.processes, seed=20260630)
                ip = calculate_impact_probabilities(results, collisions)
                rec.our_ip = float(ip.cumulative_probability[0].as_py())
                rec.n_impacts = int(ip.impacts[0].as_py())
                rec.n_variants = int(ip.variants[0].as_py())
            except Exception as e:
                rec.failure_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
            records.append(asdict(rec))
            logger.info("[%.1f min] %s/%s: our_ip=%s (%s/%s) jpl=%.2e",
                        (time.time() - start) / 60, provid, v.variant_id,
                        f"{rec.our_ip:.2e}" if np.isfinite(rec.our_ip) else "FAIL",
                        rec.n_impacts, rec.n_variants, jpl_ip)
        write_out(records, args.output_dir)
        try:
            write_report(records, args.output_dir, args.num_samples)
        except Exception as e:
            logger.warning("report failed: %s", e)

    write_out(records, args.output_dir)
    write_report(records, args.output_dir, args.num_samples)
    logger.info("Done: %d rows, %.1f min", len(records), (time.time() - start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
