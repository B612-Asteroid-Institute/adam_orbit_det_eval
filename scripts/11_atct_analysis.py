#!/usr/bin/env python3
"""
11_atct_analysis.py
===================
Post-process isolation study LOOO results to compute along-track / cross-track residuals.

DOES NOT modify any existing results.  Writes only new files:
  data/sim_products/isolation_study/atct_combined_recovery.csv

Along-track  (AT) = component of residual in the direction of the object's apparent motion.
Cross-track  (CT) = component perpendicular to the motion (90° CCW).

Motivation
----------
RA/Dec residuals mix the signal from different physical error sources.  AT/CT
separates them more cleanly:

  TimingBias      → pure AT  (clock error shifts position along track)
  TrailingBias    → pure AT  (PSF centroid error is along trailing direction)
  DCRBias         → mostly CT  (DCR shifts along altitude, ⊥ to diurnal motion)
  CatalogEpochBias→ mostly AT  (PM error is in RA, which for ecliptic objects ≈ AT)
  ConstantBias(RA)→ mixed, but projected consistently for eastward-moving objects

Join strategy
-------------
obs_id encodes ``{fake_code}_{object_id}_{i}`` where i is the 0-based rank of that
observation within the (object, fake_station) group sorted by time.  The truth
ephemeris cache stores rows for each (object, real_code) sorted by obstime_mjd.
So index i in the obs_id maps directly to row i in the sorted cache — no time
conversion required.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FAKE_TO_REAL = {
    "AA00": "F51", "AA01": "G96", "AA02": "F52", "AA03": "703",
    "AA04": "691", "AA05": "W84", "AA06": "W68", "AA07": "T09", "AA08": "V00",
}

# Regex to parse obs_id: "{FAKE}_{OBJECT}_{INDEX}"
# fake code is always 4 chars (AA\d\d), index is the trailing digits after last _
def parse_obs_id(obs_id: str):
    """
    Return (fake_code, index) from obs_id string.

    Format: {FAKE}_{object_with_underscores}_{index}
    e.g. AA04_1957_AA_0  →  fake_code=AA04, index=0

    We only parse fake_code (first 4 chars) and the trailing integer index;
    we rely on the LOOO result's object_id column for the actual object name
    (which preserves spaces, unlike the obs_id which uses underscores).
    """
    fake_code = obs_id[:4] if len(obs_id) >= 4 else None
    try:
        index = int(obs_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        index = None
    return fake_code, index


def load_velocity_cache(cache_dir: Path, object_id: str) -> dict:
    """
    Load truth ephemeris cache for one object and compute sky-plane velocities
    via finite differences.  Returns dict keyed by real_code, each value a
    numpy array of shape (N, 2): [(v_ra_cosdec, v_dec), ...] in arcsec/day,
    sorted by obstime_mjd.
    """
    cache_file = cache_dir / f"{object_id}_truth_ephemeris.parquet"
    if not cache_file.exists():
        return {}

    df = pq.read_table(cache_file).to_pandas()
    result = {}

    for real_code, grp in df.groupby("real_code"):
        grp = grp.sort_values("obstime_mjd").reset_index(drop=True)
        if len(grp) < 2:
            # Single obs: velocity undefined — use zero vector (no rotation applied)
            result[real_code] = np.zeros((len(grp), 2))
            continue

        ra_deg  = grp["ra_true_deg"].values
        dec_deg = grp["dec_true_deg"].values
        dec_rad = np.deg2rad(dec_deg)
        t_mjd   = grp["obstime_mjd"].values

        # Sky-plane RA: ra_sky = ra * cos(dec), in arcsec
        ra_sky_arcsec = ra_deg * np.cos(dec_rad) * 3600.0
        dec_arcsec    = dec_deg * 3600.0

        # Central finite differences (numpy.gradient handles endpoints gracefully)
        v_ra  = np.gradient(ra_sky_arcsec, t_mjd)   # arcsec/day
        v_dec = np.gradient(dec_arcsec,    t_mjd)    # arcsec/day

        result[real_code] = np.column_stack([v_ra, v_dec])

    return result


def rotate_atct(res_ra: np.ndarray, res_dec: np.ndarray,
                v_ra: np.ndarray, v_dec: np.ndarray):
    """
    Rotate residuals (res_ra in RA*cos(dec) direction, res_dec in Dec direction)
    into along-track / cross-track components.

    AT unit vector: (v_ra, v_dec) / speed
    CT unit vector: (-v_dec, v_ra) / speed   [90° CCW from AT]

    Returns arrays (at, ct) in arcsec.
    """
    speed = np.sqrt(v_ra**2 + v_dec**2)
    at = np.where(speed > 0,
                  (res_ra * v_ra + res_dec * v_dec) / speed, 0.0)
    ct = np.where(speed > 0,
                  (-res_ra * v_dec + res_dec * v_ra) / speed, 0.0)
    return at, ct


def sigma_atct(sigma_ra: np.ndarray, sigma_dec: np.ndarray,
               v_ra: np.ndarray, v_dec: np.ndarray):
    """
    Propagate per-observation noise sigmas into AT/CT frame.
    For isotropic noise (sigma_ra == sigma_dec) this is exact; for anisotropic
    noise this gives the 1-sigma extent along each axis.
    """
    speed = np.sqrt(v_ra**2 + v_dec**2)
    speed = np.where(speed > 0, speed, 1.0)
    u_ra  = v_ra  / speed   # unit AT vector component along RA
    u_dec = v_dec / speed   # unit AT vector component along Dec
    sigma_at = np.sqrt((sigma_ra * u_ra)**2  + (sigma_dec * u_dec)**2)
    sigma_ct = np.sqrt((sigma_ra * u_dec)**2 + (sigma_dec * u_ra)**2)
    return sigma_at, sigma_ct


def build_obstime_lookup(syn_obs_path: Path) -> dict:
    """
    Return dict mapping obs_id → obstime_mjd from the synthetic observations parquet.
    obstime struct: {days: int64, nanos: int64}, epoch = MJD 0 (days + nanos/86400e9).
    """
    df = pq.read_table(syn_obs_path, columns=["obsid", "obstime"]).to_pandas()
    lookup = {}
    for _, row in df.iterrows():
        t = row["obstime"]
        mjd = t["days"] + t["nanos"] / 86400e9 if isinstance(t, dict) else float(t)
        lookup[row["obsid"]] = mjd
    return lookup


def process_scenario(scenario_dir: Path, cache_dir: Path) -> pd.DataFrame | None:
    """
    Load LOOO results for one scenario, attach velocities, rotate to AT/CT,
    aggregate per station.  Returns a DataFrame with per-station AT/CT stats,
    or None on failure.
    """
    looo_path = scenario_dir / "looo_results" / "default" / "looo_results.parquet"
    if not looo_path.exists():
        return None

    df = pq.read_table(looo_path).to_pandas()
    if df.empty:
        return None

    # Parse fake_code from obs_id; use existing object_id column for object names
    df["fake_code"] = df["obs_id"].str[:4]
    df["real_code"] = df["fake_code"].map(FAKE_TO_REAL)

    # Get obstime_mjd per obs_id from the synthetic observations parquet
    syn_obs_path = scenario_dir / "datasets" / "default" / "mpc_observations.parquet"
    obstime_lookup = build_obstime_lookup(syn_obs_path)
    df["obstime_mjd"] = df["obs_id"].map(obstime_lookup)

    # Attach velocities: for each (object, real_code), match by obstime_mjd
    v_ra_list  = np.zeros(len(df))
    v_dec_list = np.zeros(len(df))

    for obj_id, obj_grp in df.groupby("object_id"):
        vel_cache = load_velocity_cache(cache_dir, obj_id)
        for real_code, rc_grp in obj_grp.groupby("real_code"):
            vels = vel_cache.get(real_code)
            if vels is None or len(vels) == 0:
                continue
            # Build time array for this (object, real_code) from the cache
            cache_file = cache_dir / f"{obj_id}_truth_ephemeris.parquet"
            cache_df = pq.read_table(cache_file).to_pandas()
            rc_cache = cache_df[cache_df["real_code"] == real_code].sort_values("obstime_mjd")
            cache_times = rc_cache["obstime_mjd"].values

            # Nearest-neighbour match by obstime (should be exact since same template)
            for idx_in_df, obs_mjd in zip(rc_grp.index, rc_grp["obstime_mjd"]):
                if pd.isna(obs_mjd):
                    continue
                j = np.searchsorted(cache_times, obs_mjd)
                j = min(j, len(cache_times) - 1)
                if j > 0 and abs(cache_times[j-1] - obs_mjd) < abs(cache_times[j] - obs_mjd):
                    j -= 1
                v_ra_list[idx_in_df]  = vels[j, 0]
                v_dec_list[idx_in_df] = vels[j, 1]

    df["v_ra_cosdec"] = v_ra_list
    df["v_dec"]       = v_dec_list

    # Rotate residuals
    at, ct = rotate_atct(
        df["residual_ra_arcsec"].values,
        df["residual_dec_arcsec"].values,
        df["v_ra_cosdec"].values,
        df["v_dec"].values,
    )
    df["residual_at_arcsec"] = at
    df["residual_ct_arcsec"] = ct

    # Propagate sigmas
    sig_at, sig_ct = sigma_atct(
        df["sigma_ra_cosdec_arcsec"].values,
        df["sigma_dec_arcsec"].values,
        df["v_ra_cosdec"].values,
        df["v_dec"].values,
    )
    df["sigma_at_arcsec"] = sig_at
    df["sigma_ct_arcsec"] = sig_ct

    # Per-station aggregation (same logic as compute_observatory_stats in the pipeline)
    rows = []
    for stn, grp in df.groupby("stn"):
        n = len(grp)
        if n == 0:
            continue
        mean_ra  = grp["residual_ra_arcsec"].mean()
        mean_dec = grp["residual_dec_arcsec"].mean()
        std_ra   = grp["residual_ra_arcsec"].std()
        std_dec  = grp["residual_dec_arcsec"].std()
        mean_at  = grp["residual_at_arcsec"].mean()
        mean_ct  = grp["residual_ct_arcsec"].mean()
        std_at   = grp["residual_at_arcsec"].std()
        std_ct   = grp["residual_ct_arcsec"].std()
        sem_ra   = std_ra  / np.sqrt(n)
        sem_dec  = std_dec / np.sqrt(n)
        sem_at   = std_at  / np.sqrt(n)
        sem_ct   = std_ct  / np.sqrt(n)
        # Mean speed for diagnostics
        speed    = np.sqrt(grp["v_ra_cosdec"]**2 + grp["v_dec"]**2)
        rows.append(dict(
            stn=stn,
            n_obs=n,
            mean_ra_arcsec=mean_ra,
            mean_dec_arcsec=mean_dec,
            std_ra_arcsec=std_ra,
            std_dec_arcsec=std_dec,
            sem_ra_arcsec=sem_ra,
            sem_dec_arcsec=sem_dec,
            snr_ra=abs(mean_ra) / sem_ra if sem_ra > 0 else 0.0,
            snr_dec=abs(mean_dec) / sem_dec if sem_dec > 0 else 0.0,
            mean_at_arcsec=mean_at,
            mean_ct_arcsec=mean_ct,
            std_at_arcsec=std_at,
            std_ct_arcsec=std_ct,
            sem_at_arcsec=sem_at,
            sem_ct_arcsec=sem_ct,
            snr_at=abs(mean_at) / sem_at if sem_at > 0 else 0.0,
            snr_ct=abs(mean_ct) / sem_ct if sem_ct > 0 else 0.0,
            mean_speed_arcsec_per_day=speed.mean(),
        ))

    return pd.DataFrame(rows)


def compare_detection(stats: pd.DataFrame, threshold_snr: float = 3.0) -> pd.DataFrame:
    """Add detected_* columns comparing RA/Dec vs AT/CT detection."""
    s = stats.copy()
    s["detected_ra"]  = s["snr_ra"]  >= threshold_snr
    s["detected_dec"] = s["snr_dec"] >= threshold_snr
    s["detected_at"]  = s["snr_at"]  >= threshold_snr
    s["detected_ct"]  = s["snr_ct"]  >= threshold_snr
    # Improvement flags
    s["at_better_than_ra"]  = s["snr_at"]  > s["snr_ra"]
    s["ct_better_than_dec"] = s["snr_ct"]  > s["snr_dec"]
    s["at_gains_detection"] = s["detected_at"]  & ~s["detected_ra"]
    s["at_loses_detection"] = ~s["detected_at"] & s["detected_ra"]
    s["ct_gains_detection"] = s["detected_ct"]  & ~s["detected_dec"]
    s["ct_loses_detection"] = ~s["detected_ct"] & s["detected_dec"]
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir",  type=Path,
                   default=Path("data/sim_products/isolation_study"))
    p.add_argument("--cache-dir",  type=Path,
                   default=Path("data/sim_products/cache"))
    p.add_argument("--threshold",  type=float, default=3.0,
                   help="SNR threshold for detection (default: 3.0)")
    p.add_argument("--station",    nargs="*", default=None)
    p.add_argument("--bias",       nargs="*", default=None)
    args = p.parse_args()

    scenario_dirs = sorted(
        d for d in args.study_dir.iterdir()
        if d.is_dir() and "_" in d.name and not d.name.startswith(".")
    )
    if args.station:
        scenario_dirs = [d for d in scenario_dirs
                         if any(d.name.startswith(s) for s in args.station)]
    if args.bias:
        scenario_dirs = [d for d in scenario_dirs
                         if any(d.name.endswith(b) for b in args.bias)]

    logger.info(f"Processing {len(scenario_dirs)} scenarios from {args.study_dir}")
    logger.info(f"Cache: {args.cache_dir}")

    all_rows = []
    for sd in scenario_dirs:
        parts = sd.name.split("_", 1)
        if len(parts) != 2:
            continue
        fake_code, bias_name = parts[0], parts[1]

        stats = process_scenario(sd, args.cache_dir)
        if stats is None:
            logger.warning(f"  {sd.name}: no LOOO results, skipping")
            continue

        stats = compare_detection(stats, threshold_snr=args.threshold)
        stats["scenario"]       = sd.name
        stats["target_station"] = fake_code
        stats["applied_bias"]   = bias_name
        # stn in LOOO results is the real code; map fake→real for comparison
        stats["is_target"]      = stats["stn"] == FAKE_TO_REAL.get(fake_code, "")
        all_rows.append(stats)
        logger.info(f"  {sd.name}: {len(stats)} stations")

    if not all_rows:
        logger.error("No results produced.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    out_csv = args.study_dir / "atct_combined_recovery.csv"
    combined.to_csv(out_csv, index=False)
    logger.info(f"\nWrote {len(combined)} rows → {out_csv}")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print_summary(combined, threshold_snr=args.threshold)


def print_summary(df: pd.DataFrame, threshold_snr: float):
    target = df[df["is_target"] & (df["applied_bias"] != "clean")].copy()

    print()
    print("=" * 100)
    print("  AT/CT vs RA/Dec RECOVERY COMPARISON  (target station rows, non-clean scenarios)")
    print(f"  Detection threshold: SNR ≥ {threshold_snr:.1f}")
    print("=" * 100)
    hdr = (f"{'Scenario':<22}  {'Bias':<12}  "
           f"{'SNR_RA':>7}  {'SNR_AT':>7}  {'ΔSNR_AT':>8}  "
           f"{'SNR_Dec':>8}  {'SNR_CT':>7}  {'ΔSNR_CT':>8}  "
           f"{'Det RA':>6}  {'Det AT':>6}  {'Det Dec':>7}  {'Det CT':>6}  "
           f"{'Speed\"':>7}")
    print(hdr)
    print("-" * 100)

    bias_order = ["constant","timing","mag_dep","epoch","seasonal","step","dcr","trailing"]
    target["bias_key"] = pd.Categorical(target["applied_bias"], categories=bias_order, ordered=True)
    target = target.sort_values(["bias_key", "target_station"])

    for _, r in target.iterrows():
        d_snr_at  = r["snr_at"]  - r["snr_ra"]
        d_snr_ct  = r["snr_ct"]  - r["snr_dec"]
        gain_at   = "↑" if r["at_gains_detection"]  else ("↓" if r["at_loses_detection"] else " ")
        gain_ct   = "↑" if r["ct_gains_detection"]  else ("↓" if r["ct_loses_detection"] else " ")
        det_ra    = "YES" if r["detected_ra"]  else "no"
        det_at    = "YES" if r["detected_at"]  else "no"
        det_dec   = "YES" if r["detected_dec"] else "no"
        det_ct    = "YES" if r["detected_ct"]  else "no"
        speed     = r["mean_speed_arcsec_per_day"]
        print(f"{r['scenario']:<22}  {r['applied_bias']:<12}  "
              f"{r['snr_ra']:7.2f}  {r['snr_at']:7.2f}  {d_snr_at:>+7.2f}{gain_at}  "
              f"{r['snr_dec']:8.2f}  {r['snr_ct']:7.2f}  {d_snr_ct:>+7.2f}{gain_ct}  "
              f"{det_ra:>6}  {det_at:>6}  {det_dec:>7}  {det_ct:>6}  "
              f"{speed:7.1f}")

    print()
    print("  SUMMARY — cases where AT/CT changes detection outcome:")
    gains = target[target["at_gains_detection"] | target["ct_gains_detection"] |
                   target["at_loses_detection"]  | target["ct_loses_detection"]]
    if gains.empty:
        print("  No detection changes at this threshold.")
    else:
        for _, r in gains.iterrows():
            changes = []
            if r["at_gains_detection"]:  changes.append("AT GAINS  (RA missed it)")
            if r["at_loses_detection"]:  changes.append("AT LOSES  (RA caught it)")
            if r["ct_gains_detection"]:  changes.append("CT GAINS  (Dec missed it)")
            if r["ct_loses_detection"]:  changes.append("CT LOSES  (Dec caught it)")
            print(f"  {r['scenario']:<25} {r['applied_bias']:<12}  {', '.join(changes)}")

    print()
    print("  SNR IMPROVEMENT SUMMARY (median ΔSNR across all non-clean target rows):")
    print(f"  {'Bias':<14} {'N':>3}  {'med ΔSNR_AT':>12}  {'med ΔSNR_CT':>12}  "
          f"{'AT gains':>9}  {'AT loses':>9}  {'CT gains':>9}  {'CT loses':>9}")
    print("  " + "-" * 88)
    for bias in bias_order:
        sub = target[target["applied_bias"] == bias]
        if sub.empty:
            continue
        print(f"  {bias:<14} {len(sub):>3}  "
              f"{sub['snr_at'].median() - sub['snr_ra'].median():>+12.2f}  "
              f"{sub['snr_ct'].median() - sub['snr_dec'].median():>+12.2f}  "
              f"{sub['at_gains_detection'].sum():>9}  "
              f"{sub['at_loses_detection'].sum():>9}  "
              f"{sub['ct_gains_detection'].sum():>9}  "
              f"{sub['ct_loses_detection'].sum():>9}")


if __name__ == "__main__":
    main()
