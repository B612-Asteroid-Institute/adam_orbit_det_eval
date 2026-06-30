#!/usr/bin/env python3
"""
run_heldout_prediction_v2.py
============================

Held-out predictive-accuracy test for the v2 bias catalog (follow-on to bead
3bx; the test the metric diagnosis pointed to). Instead of comparing the fitted
orbit to JPL's orbit in JPL-formal-σ units — a metric that blows up on
well-determined objects and washes out per-station bias on long arcs — this
fits each object on an OPENING SUB-ARC and predicts the HELD-OUT observations,
scoring the **sky-plane angular residual (arcsec)** on data the fit never saw.

Why this is the right test
--------------------------
- The held-out *observations* are the truth target, so it is JPL-independent
  and needs no Horizons fetch.
- It is the regime where EFCC18 / per-station weighting demonstrably help
  OrbFit and Find_Orb: prediction / recovery / linkage.
- ``efcc18_only`` is a built-in validity check — EFCC18 debiasing is
  known-good, so if the harness is sound it MUST reduce held-out residuals.
  If even EFCC18 shows nothing, the harness is wrong, not the method.

Split modes
-----------
- ``frac`` (default): train on the first ``--train-frac`` of the arc by time,
  predict the rest. Robust; converges for most objects.
- ``opening_days``: train on observations within the first ``--train-days`` of
  the arc, predict the rest. The bias-sensitive short-arc regime (one or two
  stations dominate, bias does not average out).

Metric
------
Per held-out obs i: angular residual between the fitted orbit's predicted
(RA, Dec) and the observation, in arcsec. Per object: RMS over held-out obs.
Reported against BOTH the raw observed positions (operational: predicting the
actual telescope measurements) and the EFCC18-debiased positions (each method's
self-consistent frame). Headline comparison = per-object **ratio of RMS to
no_bias** (median over objects); < 1 means the method predicts better.

Methods: no_bias, efcc18_only, v2_sigma_floor, v2_performance_weighted,
v2_covar_inflation, v2_empirical_covar.

Outputs (default ``data/heldout_prediction_v2/``)
-------------------------------------------------
    heldout_results.parquet   one row per (object, method)
    REPORT.md                 per-method median RMS + paired ratio vs no_bias

Usage
-----
    pdm run python scripts/run_heldout_prediction_v2.py \\
        [--split-mode frac --train-frac 0.5] [--n-workers 8] [--max-objects N]
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
import ray

from adam_assist import ASSISTPropagator
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from mpcq.client import BigQueryMPCClient
from mpcq import MPCObservations

from adam_orbit_det_eval.efcc18 import compute_efcc18_corrections, load_efcc18_biases
from adam_orbit_det_eval.utils import (
    get_spacebased_stns,
    load_v2_bias_catalog,
    mpc_to_od_observations,
)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("heldout_prediction_v2")
logging.getLogger("adam_core.orbit_determination").setLevel(logging.WARNING)
logging.getLogger("adam_core").setLevel(logging.WARNING)


DEFAULT_PROJECT = "moeyens-thor-dev"
DEFAULT_DATASET = "mpc_sbn_aurora"
DEFAULT_VIEWS_DATASET = "mpc_sbn_aurora_views"
DEFAULT_V2_CATALOG = (
    "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
    "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
)
DEFAULT_POP_MANIFEST = Path("data/validation_sweep/cohort_manifest.parquet")

MAX_OBS_PER_OBJECT = 2000
MIN_TRAIN_OBS = 6
MIN_TEST_OBS = 6
RESID_COV_N_THRESHOLD = 30


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
    p.add_argument("--output-dir", type=Path, default=Path("data/heldout_prediction_v2"))
    p.add_argument("--v2-catalog-path", type=Path, default=Path(DEFAULT_V2_CATALOG))
    p.add_argument("--population-manifest", type=Path, default=DEFAULT_POP_MANIFEST)
    p.add_argument("--split-mode", choices=["frac", "opening_days"], default="frac")
    p.add_argument("--train-frac", type=float, default=0.5,
                   help="frac mode: fraction of the arc (by time) used for training")
    p.add_argument("--train-days", type=float, default=60.0,
                   help="opening_days mode: training window length in days")
    p.add_argument("--predict-horizon-days", type=float, default=0.0,
                   help="cap held-out obs to this many days after the training "
                        "arc ends (0 = no cap). Use a horizon (e.g. 730) for the "
                        "recovery regime; without it a short arc predicts the "
                        "whole multi-decade tail (chaotic, not bias-sensitive).")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--views-dataset-id", default=DEFAULT_VIEWS_DATASET)
    p.add_argument("--fo-result-dir", default="/tmp/fo_heldout_v2_runs")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--per-object-timeout", type=float, default=1800.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-objects", type=int, default=None)
    return p.parse_args()


# ── v2 catalog sub-dicts ──


@dataclass
class V2Catalogs:
    bias_table: Dict[str, Tuple[float, float]]
    station_chi2: Dict[str, float]
    station_resid_covar: Dict[str, Tuple[float, float, float, float]]
    n_stations: int


def build_v2_catalogs(path: str) -> V2Catalogs:
    cat = load_v2_bias_catalog(path, rollup_only=True)
    bias_table, station_chi2, resid = {}, {}, {}
    for code, rec in cat.items():
        if np.isfinite(rec["bias_ra"]) and np.isfinite(rec["bias_dec"]):
            bias_table[code] = (rec["bias_ra"], rec["bias_dec"])
        if np.isfinite(rec["chi2_per_obs"]):
            station_chi2[code] = rec["chi2_per_obs"]
        resid[code] = (rec["resid_var_ra"], rec["resid_var_dec"],
                       rec["resid_cov_ra_dec"], rec["resid_cov_n"])
    return V2Catalogs(bias_table, station_chi2, resid, len(cat))


# ── obs prep ──


def _dedupe_close_obs(obs: MPCObservations) -> MPCObservations:
    stns = obs.stn.to_pylist()
    times_sec = obs.obstime.mjd().to_numpy(zero_copy_only=False) * 86400.0
    order = np.lexsort((times_sec, np.asarray(stns, dtype=object)))
    keep = np.zeros(len(obs), dtype=bool)
    last: Dict[str, float] = {}
    for idx in order:
        s = stns[idx]; t = float(times_sec[idx])
        prev = last.get(s)
        if prev is None or abs(t - prev) > 1.5:
            keep[idx] = True; last[s] = t
    if keep.sum() == len(obs):
        return obs
    d = obs.apply_mask(pa.array(keep))
    return qv.concatenate([d]) if d.fragmented() else d


@dataclass
class Prepared:
    obs: MPCObservations
    efcc18: np.ndarray       # (N,2) cos(dec)-arcsec (bias_ra_cosdec, bias_dec)
    times_mjd: np.ndarray
    n_obs: int
    arc_days: float


def prepare_observations(provid, client, spacebased, efcc18_bias_table) -> Optional[Prepared]:
    raw = client.query_observations([provid])
    if raw is None or len(raw) == 0:
        return None
    stns = raw.stn.to_pylist()
    gm = np.array([(s is not None) and (s not in spacebased) for s in stns], dtype=bool)
    if not gm.any():
        return None
    ground = raw.apply_mask(pa.array(gm))
    if ground.fragmented():
        ground = qv.concatenate([ground])
    d = _dedupe_close_obs(ground)
    if len(d) > MAX_OBS_PER_OBJECT:
        t = d.obstime.mjd().to_numpy(zero_copy_only=False)
        keep = np.zeros(len(d), dtype=bool)
        keep[np.argsort(t)[-MAX_OBS_PER_OBJECT:]] = True
        d = d.apply_mask(pa.array(keep))
        if d.fragmented():
            d = qv.concatenate([d])
    times = d.obstime.mjd().to_numpy(zero_copy_only=False)
    efcc = compute_efcc18_corrections(
        d.ra.to_numpy(zero_copy_only=False), d.dec.to_numpy(zero_copy_only=False),
        d.astcat.to_pylist(), times + 2400000.5, bias_table=efcc18_bias_table)
    return Prepared(obs=d, efcc18=efcc, times_mjd=times, n_obs=len(d),
                    arc_days=float(times.max() - times.min()))


def split_indices(times: np.ndarray, mode: str, train_frac: float,
                  train_days: float, predict_horizon_days: float = 0.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask). Training is always the OPENING (earliest)
    sub-arc; the held-out set is later — a forward-prediction test.

    - ``opening_days``: train = obs within the first ``train_days`` of the arc
      (time-based; the short-arc / linkage regime where bias does not average
      out, but obs counts vary per object).
    - ``frac``: train = the earliest ``train_frac`` of obs by COUNT (chronological).
      Count-based keeps the held-out set populated for every object — a
      time-quantile split leaves back-loaded objects with too few held-out obs.

    ``predict_horizon_days`` > 0 caps the held-out set to obs within that many
    days after the training arc ends — the RECOVERY regime. Without a cap, a
    short opening arc predicts the whole multi-decade tail, which is so
    ill-posed the prediction is dominated by chaotic extrapolation, not bias.
    """
    train_mask = np.zeros(len(times), dtype=bool)
    if mode == "opening_days":
        train_mask = (times - times.min()) <= train_days
    else:  # frac — earliest train_frac of observations, chronological
        order = np.argsort(times, kind="stable")
        n_train = max(1, int(round(train_frac * len(times))))
        train_mask[order[:n_train]] = True
    test_mask = ~train_mask
    if predict_horizon_days and predict_horizon_days > 0 and train_mask.any():
        train_end = times[train_mask].max()
        test_mask = test_mask & (times <= train_end + predict_horizon_days)
    return train_mask, test_mask


# ── per (object, method) ──


def _build_train_od(obs, efcc_train, variant: VariantConfig, cats: V2Catalogs):
    kw: Dict[str, object] = {"prevent_nans": True, "sigma_model": "veres2017",
                             "bias_application": variant.bias_application}
    if variant.use_bias_table:
        kw["bias_table"] = cats.bias_table
    if variant.use_efcc18:
        kw["catalog_debias_arcsec"] = efcc_train
    if variant.use_station_chi2:
        kw["station_chi2_per_obs"] = cats.station_chi2
    if variant.use_resid_covar:
        kw["station_resid_covar"] = cats.station_resid_covar
        kw["resid_cov_n_threshold"] = RESID_COV_N_THRESHOLD
    return mpc_to_od_observations(obs, **kw)


def _angular_resid_arcsec(obs_ra, obs_dec, pred_ra, pred_dec) -> np.ndarray:
    cosd = np.cos(np.deg2rad(obs_dec))
    dra = (obs_ra - pred_ra + 180.0) % 360.0 - 180.0
    return np.sqrt((dra * cosd * 3600.0) ** 2 + ((obs_dec - pred_dec) * 3600.0) ** 2)


@dataclass
class HeldoutResult:
    object_id: str
    designation: str
    stratum: str
    variant: str
    split_mode: str
    n_obs_total: int = 0
    n_train: int = 0
    n_test: int = 0
    train_arc_days: float = float("nan")
    test_arc_days: float = float("nan")
    converged: bool = False
    failure_reason: str = ""
    train_reduced_chi2: float = float("nan")
    resid_rms_raw_arcsec: float = float("nan")
    resid_rms_deb_arcsec: float = float("nan")
    resid_median_raw_arcsec: float = float("nan")
    n_test_used: int = 0


def run_single(rec: HeldoutResult, variant, prepared: Prepared,
               train_mask, test_mask, *, cats, fitter, propagator) -> HeldoutResult:
    try:
        train_obs = prepared.obs.apply_mask(pa.array(train_mask))
        test_obs = prepared.obs.apply_mask(pa.array(test_mask))
        if train_obs.fragmented():
            train_obs = qv.concatenate([train_obs])
        if test_obs.fragmented():
            test_obs = qv.concatenate([test_obs])
        efcc_train = prepared.efcc18[train_mask]
        efcc_test = prepared.efcc18[test_mask]

        od_train = _build_train_od(train_obs, efcc_train, variant, cats)
        if od_train is None or len(od_train) == 0:
            rec.failure_reason = "mpc_to_od_observations returned None/empty"
            return rec

        fitted, _ = fitter.initial_fit(rec.object_id, od_train)
        if len(fitted) == 0:
            rec.failure_reason = "FindOrb returned empty FittedOrbits"
            return rec
        rec.train_reduced_chi2 = float(fitted.reduced_chi2[0].as_py())
        fit_orbit = Orbits.from_kwargs(orbit_id=fitted.orbit_id,
                                       object_id=fitted.object_id,
                                       coordinates=fitted.coordinates)

        # Predict the held-out obs and score the sky-plane residual.
        observers = Observers.from_codes(codes=test_obs.stn, times=test_obs.obstime)
        eph = propagator.generate_ephemeris(orbits=fit_orbit, observers=observers,
                                            max_processes=1)
        pred_ra = eph.coordinates.lon.to_numpy(zero_copy_only=False)
        pred_dec = eph.coordinates.lat.to_numpy(zero_copy_only=False)
        obs_ra = test_obs.ra.to_numpy(zero_copy_only=False)
        obs_dec = test_obs.dec.to_numpy(zero_copy_only=False)

        # raw frame (actual telescope measurements — same target for all methods)
        r_raw = _angular_resid_arcsec(obs_ra, obs_dec, pred_ra, pred_dec)
        # debiased frame (subtract EFCC18 from the held-out positions; == raw if no EFCC18)
        cosd = np.cos(np.deg2rad(obs_dec))
        with np.errstate(divide="ignore", invalid="ignore"):
            deb_ra = obs_ra - np.where(cosd != 0, (efcc_test[:, 0] / 3600.0) / cosd, 0.0)
        deb_dec = obs_dec - efcc_test[:, 1] / 3600.0
        r_deb = _angular_resid_arcsec(deb_ra, deb_dec, pred_ra, pred_dec)

        good = np.isfinite(r_raw)
        if good.sum() == 0:
            rec.failure_reason = "no finite held-out residuals"
            return rec
        rec.converged = True
        rec.n_test_used = int(good.sum())
        rec.resid_rms_raw_arcsec = float(np.sqrt(np.mean(r_raw[good] ** 2)))
        rec.resid_rms_deb_arcsec = float(np.sqrt(np.mean(r_deb[np.isfinite(r_deb)] ** 2)))
        rec.resid_median_raw_arcsec = float(np.median(r_raw[good]))
    except Exception as e:
        rec.failure_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    return rec


@ray.remote
class ObjectWorker:
    def __init__(self, idx: int, args_dict: dict, cats: V2Catalogs, efcc18_bias_table):
        logging.basicConfig(level=logging.WARNING)
        self.cats = cats
        self.args = args_dict
        self.spacebased = set(get_spacebased_stns())
        self.efcc18_bias_table = efcc18_bias_table
        self.client = BigQueryMPCClient(dataset_id=args_dict["dataset_id"],
                                        views_dataset_id=args_dict["views_dataset_id"],
                                        project=args_dict["project"])
        self.propagator = ASSISTPropagator()
        fo_dir = f"{args_dict['fo_result_dir']}/actor_{idx}"
        Path(fo_dir).mkdir(parents=True, exist_ok=True)
        self.fitter = FindOrbOrbitFitter(fo_result_dir=fo_dir, clean_up_fo_dir=True,
                                         propagator=self.propagator)

    def process(self, row: dict) -> List[dict]:
        provid = row["provid"]; stratum = row.get("stratum", "")
        a = self.args

        def fail(reason, **kw):
            return [asdict(HeldoutResult(
                object_id=provid, designation=provid, stratum=stratum,
                variant=v.variant_id, split_mode=a["split_mode"],
                failure_reason=reason, **kw)) for v in VARIANTS]

        try:
            prepared = prepare_observations(provid, self.client, self.spacebased,
                                            self.efcc18_bias_table)
        except Exception as e:
            logger.warning("%s: prep raised %s", provid, e); prepared = None
        if prepared is None:
            return fail("observation prep failed")

        train_mask, test_mask = split_indices(prepared.times_mjd, a["split_mode"],
                                              a["train_frac"], a["train_days"],
                                              a["predict_horizon_days"])
        n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
        if n_tr < MIN_TRAIN_OBS or n_te < MIN_TEST_OBS:
            return fail(f"insufficient split (train={n_tr}, test={n_te})",
                        n_obs_total=prepared.n_obs, n_train=n_tr, n_test=n_te)

        tr_days = float(np.ptp(prepared.times_mjd[train_mask]))
        te_days = float(np.ptp(prepared.times_mjd[test_mask]))
        out: List[dict] = []
        for v in VARIANTS:
            rec = HeldoutResult(object_id=provid, designation=provid, stratum=stratum,
                                variant=v.variant_id, split_mode=a["split_mode"],
                                n_obs_total=prepared.n_obs, n_train=n_tr, n_test=n_te,
                                train_arc_days=tr_days, test_arc_days=te_days)
            rec = run_single(rec, v, prepared, train_mask, test_mask,
                             cats=self.cats, fitter=self.fitter, propagator=self.propagator)
            out.append(asdict(rec))
        return out


# ── persistence + report ──


def write_parquet(records: List[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_dir / "heldout_results.parquet", index=False)
    return output_dir / "heldout_results.parquet"


def _failure_rows(row: dict, reason: str, split_mode: str) -> List[dict]:
    return [asdict(HeldoutResult(object_id=row["provid"], designation=row["provid"],
                                 stratum=row.get("stratum", ""), variant=v.variant_id,
                                 split_mode=split_mode, failure_reason=reason))
            for v in VARIANTS]


def write_report(records: List[dict], output_dir: Path, args, n_stations: int) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        (output_dir / "REPORT.md").write_text("# No results yet\n"); return
    ok = df[df.converged].copy()
    md: List[str] = []
    md.append("# Held-out predictive-accuracy test — v2 catalog")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"Split: `{args.split_mode}` (train-frac={args.train_frac}, "
              f"train-days={args.train_days}, predict-horizon-days="
              f"{args.predict_horizon_days or 'none'}); v2 stations={n_stations}")
    md.append("Metric: per-object RMS of held-out sky-plane residual (arcsec); "
              "lower = the fit predicts unseen observations better.")
    md.append("")
    md.append(f"Objects with a usable split: {ok.object_id.nunique()} / {df.object_id.nunique()}")
    md.append("")

    # paired ratio vs no_bias (per object), median over objects
    md.append("## Held-out residual RMS by method")
    md.append("")
    md.append("Paired ratio = median over objects of RMS(method)/RMS(no_bias); "
              "**< 1.00 means better prediction than no-bias.**")
    md.append("")
    md.append("| method | n | median RMS raw (\") | median RMS debiased (\") | "
              "paired ratio vs no_bias (raw) | % objects improved |")
    md.append("|---|---|---|---|---|---|")
    nb = ok[ok.variant == "no_bias"].set_index("object_id")["resid_rms_raw_arcsec"]
    for v in VARIANTS:
        s = ok[ok.variant == v.variant_id]
        if s.empty:
            continue
        paired = s.set_index("object_id")["resid_rms_raw_arcsec"]
        common = paired.index.intersection(nb.index)
        ratio = (paired[common] / nb[common]).replace([np.inf, -np.inf], np.nan).dropna()
        med_ratio = float(ratio.median()) if len(ratio) else float("nan")
        pct_better = float((ratio < 1.0).mean() * 100) if len(ratio) else float("nan")
        md.append(f"| `{v.variant_id}` | {len(s)} | {s.resid_rms_raw_arcsec.median():.4f} | "
                  f"{s.resid_rms_deb_arcsec.median():.4f} | {med_ratio:.3f} | {pct_better:.0f}% |")
    md.append("")
    md.append("**Validity check:** `efcc18_only` is known-good (EFCC18 debiasing helps "
              "OrbFit/Find_Orb). If its ratio is not < 1, the harness — not the method "
              "— is suspect.")
    md.append("")

    # by stratum
    md.append("## By stratum (median RMS raw, arcsec)")
    md.append("")
    strata = sorted(ok.stratum.dropna().unique())
    md.append("| method | " + " | ".join(strata) + " |")
    md.append("|" + "---|" * (len(strata) + 1))
    for v in VARIANTS:
        cells = []
        for st in strata:
            s = ok[(ok.variant == v.variant_id) & (ok.stratum == st)]
            cells.append(f"{s.resid_rms_raw_arcsec.median():.4f}" if len(s) else "—")
        md.append(f"| `{v.variant_id}` | " + " | ".join(cells) + " |")
    md.append("")

    # convergence
    md.append("## Convergence")
    md.append("")
    md.append("| method | converged / total |")
    md.append("|---|---|")
    for v in VARIANTS:
        s = df[df.variant == v.variant_id]
        md.append(f"| `{v.variant_id}` | {int(s.converged.sum())}/{len(s)} |")
    md.append("")
    (output_dir / "REPORT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.fo_result_dir).mkdir(parents=True, exist_ok=True)

    if not args.population_manifest.exists():
        logger.error("Population manifest not found: %s", args.population_manifest)
        return 1
    pop = pd.read_parquet(args.population_manifest)
    if "provid" not in pop.columns:
        logger.error("Manifest needs a 'provid' column"); return 1
    logger.info("Population: %d objects from %s", len(pop), args.population_manifest)
    if args.max_objects is not None:
        pop = pop.head(args.max_objects)

    cats = build_v2_catalogs(str(args.v2_catalog_path))
    logger.info("v2 catalog: %d rollup stations", cats.n_stations)

    existing: set = set()
    records: List[dict] = []
    out_parquet = args.output_dir / "heldout_results.parquet"
    if args.resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        records = prev.to_dict(orient="records")
        existing = set(prev.object_id.unique().tolist())
        logger.info("Resume: %d objects already done", len(existing))

    targets = [r for r in pop.to_dict(orient="records") if r["provid"] not in existing]
    logger.info("Objects to process: %d (split=%s)", len(targets), args.split_mode)

    if targets:
        efcc18_bias_table = load_efcc18_biases()
        logger.info("Loaded EFCC18 (%s)", efcc18_bias_table.shape)
        ray.init(num_cpus=args.n_workers, ignore_reinit_error=True,
                 include_dashboard=False, logging_level=logging.WARNING)
        cats_ref = ray.put(cats); efcc18_ref = ray.put(efcc18_bias_table)
        args_dict = {"project": args.project, "dataset_id": args.dataset_id,
                     "views_dataset_id": args.views_dataset_id,
                     "fo_result_dir": args.fo_result_dir, "split_mode": args.split_mode,
                     "train_frac": args.train_frac, "train_days": args.train_days,
                     "predict_horizon_days": args.predict_horizon_days}
        workers = [ObjectWorker.remote(i, args_dict, cats_ref, efcc18_ref)
                   for i in range(args.n_workers)]
        start = time.time(); n_done = len(existing); n_total = len(pop)
        timeout_s = args.per_object_timeout
        queue = list(targets); inflight: Dict = {}

        def submit(widx):
            if queue:
                row = queue.pop(0)
                ref = workers[widx].process.remote(row)
                inflight[ref] = [widx, row, time.time()]

        def record(rows):
            nonlocal n_done
            records.extend(rows); n_done += 1
            if rows:
                oid = rows[0]["object_id"]
                nconv = sum(1 for r in rows if r["converged"])
                logger.info("[%d/%d] %s (%d/%d methods predicted, cum %.1f min)",
                            n_done, n_total, oid, nconv, len(rows),
                            (time.time() - start) / 60.0)
            if n_done % 10 == 0:
                write_parquet(records, args.output_dir)

        for widx in range(len(workers)):
            submit(widx)
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=20.0)
            now = time.time()
            if done:
                ref = done[0]; widx, row, _ = inflight.pop(ref)
                try:
                    rows = ray.get(ref)
                except Exception as e:
                    rows = _failure_rows(row, f"actor error: {type(e).__name__}: {e}",
                                         args.split_mode)
                record(rows); submit(widx)
            for ref in list(inflight.keys()):
                widx, row, st = inflight[ref]
                if now - st > timeout_s:
                    logger.warning("[timeout] %s > %.0fs on actor %d — killing+replacing",
                                   row["provid"], timeout_s, widx)
                    try:
                        ray.kill(workers[widx])
                    except Exception:
                        pass
                    inflight.pop(ref)
                    record(_failure_rows(row, f"per-object timeout >{int(timeout_s)}s",
                                         args.split_mode))
                    workers[widx] = ObjectWorker.remote(widx, args_dict, cats_ref, efcc18_ref)
                    submit(widx)
        ray.shutdown()

    write_parquet(records, args.output_dir)
    write_report(records, args.output_dir, args, cats.n_stations)
    logger.info("Wrote %d rows + REPORT to %s", len(records), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
