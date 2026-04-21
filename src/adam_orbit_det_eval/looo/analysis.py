"""
Statistical analysis of LOOO results.

The primary outputs are per-observatory statistics that answer:
  1. Bias: Does this observatory have a systematic RA or Dec offset?
  2. Scatter: Is the RMS residual consistent with the reported sigma?
  3. Chi2 calibration: Is sigma over/under-estimated?
     (chi2_per_obs == 1.0 means correctly calibrated)
  4. Sensitivity: How much do orbits change when this observatory is excluded?

All aggregation functions accept optional stratification parameters so you can
look at, e.g., only objects with arc_length_remaining > 30 days, or only
observations with n_obs_remaining >= 20, to control for confounds.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
from pathlib import Path
from typing import Optional

from .core import LOOOResult


class ObservatoryStats(qv.Table):
    """Aggregated per-observatory statistics from LOOO results."""

    stn = qv.LargeStringColumn()

    # Sample sizes
    n_obs = qv.Int64Column()          # total held-out observations
    n_objects = qv.Int64Column()      # distinct objects evaluated
    n_objects_filtered = qv.Int64Column()  # objects excluded by difficulty filter

    # Residual bias (arcseconds) — systematic offset
    mean_ra_arcsec = qv.Float64Column()
    mean_dec_arcsec = qv.Float64Column()

    # Residual scatter (arcseconds) — random error
    rms_ra_arcsec = qv.Float64Column()
    rms_dec_arcsec = qv.Float64Column()
    median_abs_ra_arcsec = qv.Float64Column()
    median_abs_dec_arcsec = qv.Float64Column()

    # Chi2 calibration: ratio of actual variance to reported variance
    # == 1.0 → correctly calibrated
    # >1.0  → under-estimated sigmas (observations weighted too highly)
    # <1.0  → over-estimated sigmas
    mean_chi2_per_obs = qv.Float64Column()
    median_chi2_per_obs = qv.Float64Column()

    # Orbit sensitivity: how much does excluding this stn change the orbit?
    mean_abs_delta_q_au = qv.Float64Column(nullable=True)
    mean_abs_delta_e = qv.Float64Column(nullable=True)
    mean_abs_delta_i_deg = qv.Float64Column(nullable=True)

    # Hold-in fit quality (sanity check — should be low for well-determined orbits)
    mean_hold_in_reduced_chi2 = qv.Float64Column(nullable=True)


class CatalogStats(qv.Table):
    """Aggregated statistics broken down by (stn, astcat) pair."""

    stn = qv.LargeStringColumn()
    astcat = qv.LargeStringColumn(nullable=True)
    n_obs = qv.Int64Column()
    mean_ra_arcsec = qv.Float64Column()
    mean_dec_arcsec = qv.Float64Column()
    rms_ra_arcsec = qv.Float64Column()
    rms_dec_arcsec = qv.Float64Column()
    mean_chi2_per_obs = qv.Float64Column()
    median_chi2_per_obs = qv.Float64Column()


class ProgramCodeStats(qv.Table):
    """Aggregated statistics broken down by (stn, program_code) pair."""

    stn = qv.LargeStringColumn()
    program_code = qv.LargeStringColumn(nullable=True)
    n_obs = qv.Int64Column()
    n_objects = qv.Int64Column()
    mean_ra_arcsec = qv.Float64Column()
    mean_dec_arcsec = qv.Float64Column()
    rms_ra_arcsec = qv.Float64Column()
    rms_dec_arcsec = qv.Float64Column()
    median_abs_ra_arcsec = qv.Float64Column()
    median_abs_dec_arcsec = qv.Float64Column()
    mean_chi2_per_obs = qv.Float64Column()
    median_chi2_per_obs = qv.Float64Column()


def _nanmean(arr: np.ndarray) -> float:
    v = np.nanmean(arr)
    return float(v) if np.isfinite(v) else np.nan


def _nanrms(arr: np.ndarray) -> float:
    v = np.sqrt(np.nanmean(arr**2))
    return float(v) if np.isfinite(v) else np.nan


def _nanmedian_abs(arr: np.ndarray) -> float:
    v = np.nanmedian(np.abs(arr))
    return float(v) if np.isfinite(v) else np.nan


def _flag_difficult_objects(
    tbl,
    max_object_mean_chi2: float,
) -> set:
    """
    Return set of object_ids where the mean held-out chi2 across ALL stations
    exceeds the threshold.

    These objects are likely subject to non-gravitational forces (comets,
    high-Yarkovsky asteroids) or propagator failure — the inflated chi2 is
    an object-level property, not a station-level one, and including them
    biases sigma estimates upward for every station that observed them.
    """
    from collections import defaultdict
    obj_chi2: dict = defaultdict(list)
    for obj, c in zip(tbl.column("object_id").to_pylist(), tbl.column("chi2").to_pylist()):
        if c is not None:
            obj_chi2[obj].append(c)
    return {
        obj_id
        for obj_id, vals in obj_chi2.items()
        if len(vals) > 0 and np.nanmean(vals) > max_object_mean_chi2
    }


def _filter_results(
    results: LOOOResult,
    min_obs_remaining: Optional[int] = None,
    min_arc_length_days: Optional[float] = None,
    max_held_out_fraction: Optional[float] = None,
    max_hold_in_reduced_chi2: Optional[float] = None,
) -> LOOOResult:
    """Apply additional stratification filters to the result set."""
    if len(results) == 0:
        return results
    mask = pa.array([True] * len(results), type=pa.bool_())
    if min_obs_remaining is not None:
        mask = pc.and_(mask, pc.greater_equal(results.n_obs_remaining, min_obs_remaining))
    if min_arc_length_days is not None:
        mask = pc.and_(
            mask,
            pc.greater_equal(results.arc_length_remaining_days, min_arc_length_days),
        )
    if max_held_out_fraction is not None:
        mask = pc.and_(
            mask,
            pc.less_equal(results.held_out_fraction, max_held_out_fraction),
        )
    if max_hold_in_reduced_chi2 is not None:
        valid_chi2 = pc.fill_null(
            pc.and_(
                pc.is_valid(results.hold_in_reduced_chi2),
                pc.less_equal(results.hold_in_reduced_chi2, max_hold_in_reduced_chi2),
            ),
            False,
        )
        mask = pc.and_(mask, valid_chi2)
    return results.apply_mask(mask)


def compute_observatory_stats(
    results: LOOOResult,
    min_obs_remaining: Optional[int] = None,
    min_arc_length_days: Optional[float] = None,
    max_held_out_fraction: Optional[float] = None,
    max_hold_in_reduced_chi2: Optional[float] = 100.0,
    min_obs_per_stn: int = 10,
    object_weighted: bool = False,
    max_object_mean_chi2: Optional[float] = None,
) -> ObservatoryStats:
    """
    Compute per-observatory summary statistics from LOOO results.

    Parameters
    ----------
    results : LOOOResult
        Output from run_looo_pipeline.
    min_obs_remaining : int, optional
        Only include rows where the hold-in fit had >= this many observations.
    min_arc_length_days : float, optional
        Only include rows where the hold-in arc was >= this many days.
    max_held_out_fraction : float, optional
        Only include rows where the held-out fraction <= this.
    max_hold_in_reduced_chi2 : float, optional
        Exclude rows where the hold-in fit was poor (chi2 > threshold).
    min_obs_per_stn : int
        Minimum observations per observatory to be reported.
    object_weighted : bool
        If True, compute two-level statistics: average per-object first, then
        average over objects (each object weighted equally regardless of how
        many observations it contributed). Prevents heavily-observed objects
        from dominating. Default False for backward compatibility.
    max_object_mean_chi2 : float, optional
        If set (and object_weighted=True), exclude objects where the mean
        held-out chi2 across ALL stations exceeds this threshold. These are
        likely non-gravitational-force objects (comets, high-Yarkovsky) whose
        inflated chi2 is an object-level property, not a station problem.
        Recommended: 50.0.

    Returns
    -------
    ObservatoryStats
    """
    filtered = _filter_results(
        results,
        min_obs_remaining=min_obs_remaining,
        min_arc_length_days=min_arc_length_days,
        max_held_out_fraction=max_held_out_fraction,
        max_hold_in_reduced_chi2=max_hold_in_reduced_chi2,
    )

    tbl = filtered.table

    # Identify and optionally remove difficult (likely non-grav) objects
    difficult_objects: set = set()
    if object_weighted and max_object_mean_chi2 is not None:
        difficult_objects = _flag_difficult_objects(tbl, max_object_mean_chi2)
        if difficult_objects:
            keep = pa.array(
                [obj not in difficult_objects for obj in tbl.column("object_id").to_pylist()]
            )
            tbl = tbl.filter(keep)

    stns = pc.unique(tbl.column("stn")).to_pylist()

    rows = []
    for stn in sorted(stns):
        stn_mask = pc.equal(tbl.column("stn"), stn)
        stn_tbl = tbl.filter(stn_mask)
        n_obs = len(stn_tbl)

        if n_obs < min_obs_per_stn:
            continue

        objects = stn_tbl.column("object_id")
        n_objects = int(len(pc.unique(objects)))

        # Count how many objects were filtered out for this station
        if difficult_objects:
            all_stn_mask = pc.equal(filtered.table.column("stn"), stn)
            all_objs = set(filtered.table.filter(all_stn_mask).column("object_id").to_pylist())
            n_objects_filtered = int(len(all_objs & difficult_objects))
        else:
            n_objects_filtered = 0

        dq = stn_tbl.column("delta_q_au").to_pylist()
        de = stn_tbl.column("delta_e").to_pylist()
        di = stn_tbl.column("delta_i_deg").to_pylist()
        rchi2 = stn_tbl.column("hold_in_reduced_chi2").to_pylist()
        dq_arr = np.array([x for x in dq if x is not None], dtype=float)
        de_arr = np.array([x for x in de if x is not None], dtype=float)
        di_arr = np.array([x for x in di if x is not None], dtype=float)
        rchi2_arr = np.array([x for x in rchi2 if x is not None], dtype=float)

        if object_weighted:
            # Two-level: compute per-object means first, then average over objects
            obj_ids = pc.unique(stn_tbl.column("object_id")).to_pylist()
            per_obj_ra, per_obj_dec, per_obj_rms_ra, per_obj_rms_dec, per_obj_chi2 = [], [], [], [], []
            for obj_id in obj_ids:
                obj_mask = pc.equal(stn_tbl.column("object_id"), obj_id)
                obj_tbl = stn_tbl.filter(obj_mask)
                ra_o = np.array([x for x in obj_tbl.column("residual_ra_arcsec").to_pylist() if x is not None], dtype=float)
                dec_o = np.array([x for x in obj_tbl.column("residual_dec_arcsec").to_pylist() if x is not None], dtype=float)
                chi2_o = np.array([x for x in obj_tbl.column("chi2").to_pylist() if x is not None], dtype=float)
                if len(ra_o) == 0:
                    continue
                per_obj_ra.append(_nanmean(ra_o))
                per_obj_dec.append(_nanmean(dec_o))
                per_obj_rms_ra.append(_nanrms(ra_o))
                per_obj_rms_dec.append(_nanrms(dec_o))
                per_obj_chi2.append(_nanmean(chi2_o))

            if not per_obj_ra:
                continue

            ra_arr = np.array(per_obj_ra)
            dec_arr = np.array(per_obj_dec)
            rms_ra_arr = np.array(per_obj_rms_ra)
            rms_dec_arr = np.array(per_obj_rms_dec)
            chi2_arr = np.array(per_obj_chi2)

            mean_ra = _nanmean(ra_arr)
            mean_dec = _nanmean(dec_arr)
            # RMS: average of per-object RMS (not RMS of per-object means, which would lose within-object scatter)
            rms_ra = _nanmean(rms_ra_arr)
            rms_dec = _nanmean(rms_dec_arr)
            med_abs_ra = float(np.nanmedian(np.abs(ra_arr)))
            med_abs_dec = float(np.nanmedian(np.abs(dec_arr)))
            mean_chi2 = _nanmean(chi2_arr)
            med_chi2 = float(np.nanmedian(chi2_arr))
        else:
            # Original: flat observation-weighted aggregation
            ra = stn_tbl.column("residual_ra_arcsec").to_pylist()
            dec = stn_tbl.column("residual_dec_arcsec").to_pylist()
            chi2 = stn_tbl.column("chi2").to_pylist()
            ra_arr = np.array([x for x in ra if x is not None], dtype=float)
            dec_arr = np.array([x for x in dec if x is not None], dtype=float)
            chi2_arr = np.array([x for x in chi2 if x is not None], dtype=float)
            mean_ra = _nanmean(ra_arr)
            mean_dec = _nanmean(dec_arr)
            rms_ra = _nanrms(ra_arr)
            rms_dec = _nanrms(dec_arr)
            med_abs_ra = _nanmedian_abs(ra_arr)
            med_abs_dec = _nanmedian_abs(dec_arr)
            mean_chi2 = _nanmean(chi2_arr)
            med_chi2 = float(np.nanmedian(chi2_arr)) if len(chi2_arr) > 0 else np.nan

        rows.append(
            dict(
                stn=stn,
                n_obs=n_obs,
                n_objects=n_objects,
                n_objects_filtered=n_objects_filtered,
                mean_ra_arcsec=mean_ra,
                mean_dec_arcsec=mean_dec,
                rms_ra_arcsec=rms_ra,
                rms_dec_arcsec=rms_dec,
                median_abs_ra_arcsec=med_abs_ra,
                median_abs_dec_arcsec=med_abs_dec,
                mean_chi2_per_obs=mean_chi2,
                median_chi2_per_obs=med_chi2,
                mean_abs_delta_q_au=_nanmean(np.abs(dq_arr)) if len(dq_arr) > 0 else np.nan,
                mean_abs_delta_e=_nanmean(np.abs(de_arr)) if len(de_arr) > 0 else np.nan,
                mean_abs_delta_i_deg=_nanmean(np.abs(di_arr)) if len(di_arr) > 0 else np.nan,
                mean_hold_in_reduced_chi2=_nanmean(rchi2_arr) if len(rchi2_arr) > 0 else np.nan,
            )
        )

    if not rows:
        return ObservatoryStats.empty()

    return ObservatoryStats.from_kwargs(
        stn=[r["stn"] for r in rows],
        n_obs=[r["n_obs"] for r in rows],
        n_objects=[r["n_objects"] for r in rows],
        n_objects_filtered=[r["n_objects_filtered"] for r in rows],
        mean_ra_arcsec=[r["mean_ra_arcsec"] for r in rows],
        mean_dec_arcsec=[r["mean_dec_arcsec"] for r in rows],
        rms_ra_arcsec=[r["rms_ra_arcsec"] for r in rows],
        rms_dec_arcsec=[r["rms_dec_arcsec"] for r in rows],
        median_abs_ra_arcsec=[r["median_abs_ra_arcsec"] for r in rows],
        median_abs_dec_arcsec=[r["median_abs_dec_arcsec"] for r in rows],
        mean_chi2_per_obs=[r["mean_chi2_per_obs"] for r in rows],
        median_chi2_per_obs=[r["median_chi2_per_obs"] for r in rows],
        mean_abs_delta_q_au=[r["mean_abs_delta_q_au"] for r in rows],
        mean_abs_delta_e=[r["mean_abs_delta_e"] for r in rows],
        mean_abs_delta_i_deg=[r["mean_abs_delta_i_deg"] for r in rows],
        mean_hold_in_reduced_chi2=[r["mean_hold_in_reduced_chi2"] for r in rows],
    )


def compute_catalog_stats(
    results: LOOOResult,
    min_obs_remaining: Optional[int] = None,
    min_arc_length_days: Optional[float] = None,
    max_hold_in_reduced_chi2: Optional[float] = 100.0,
    min_obs_per_group: int = 10,
    object_weighted: bool = False,
    max_object_mean_chi2: Optional[float] = None,
) -> CatalogStats:
    """
    Compute statistics broken down by (observatory, astrometric catalog) pair.

    This is the key comparison against Veres 2017 / FCCT14 weights,
    which are also defined at the (stn, catalog) level.
    """
    filtered = _filter_results(
        results,
        min_obs_remaining=min_obs_remaining,
        min_arc_length_days=min_arc_length_days,
        max_hold_in_reduced_chi2=max_hold_in_reduced_chi2,
    )

    tbl = filtered.table

    if object_weighted and max_object_mean_chi2 is not None:
        difficult_objects = _flag_difficult_objects(tbl, max_object_mean_chi2)
        if difficult_objects:
            keep = pa.array(
                [obj not in difficult_objects for obj in tbl.column("object_id").to_pylist()]
            )
            tbl = tbl.filter(keep)

    # Group by (stn, astcat)
    stn_col = tbl.column("stn").to_pylist()
    astcat_col = tbl.column("astcat").to_pylist()
    obj_col = tbl.column("object_id").to_pylist()
    groups = sorted(set(zip(stn_col, astcat_col)), key=lambda x: (x[0] or "", x[1] or ""))

    rows = []
    for stn, astcat in groups:
        grp_mask = pa.array(
            [(s == stn and a == astcat) for s, a in zip(stn_col, astcat_col)]
        )
        grp = tbl.filter(grp_mask)
        n_obs = len(grp)
        if n_obs < min_obs_per_group:
            continue

        if object_weighted:
            grp_obj_ids = pc.unique(grp.column("object_id")).to_pylist()
            per_obj_ra, per_obj_dec, per_obj_rms_ra, per_obj_rms_dec, per_obj_chi2 = [], [], [], [], []
            for obj_id in grp_obj_ids:
                obj_mask = pc.equal(grp.column("object_id"), obj_id)
                obj_grp = grp.filter(obj_mask)
                ra_o = np.array([x for x in obj_grp.column("residual_ra_arcsec").to_pylist() if x is not None], dtype=float)
                dec_o = np.array([x for x in obj_grp.column("residual_dec_arcsec").to_pylist() if x is not None], dtype=float)
                chi2_o = np.array([x for x in obj_grp.column("chi2").to_pylist() if x is not None], dtype=float)
                if len(ra_o) == 0:
                    continue
                per_obj_ra.append(_nanmean(ra_o))
                per_obj_dec.append(_nanmean(dec_o))
                per_obj_rms_ra.append(_nanrms(ra_o))
                per_obj_rms_dec.append(_nanrms(dec_o))
                per_obj_chi2.append(_nanmean(chi2_o))

            if not per_obj_ra:
                continue
            ra = np.array(per_obj_ra)
            dec = np.array(per_obj_dec)
            rms_ra = _nanmean(np.array(per_obj_rms_ra))
            rms_dec = _nanmean(np.array(per_obj_rms_dec))
            chi2 = np.array(per_obj_chi2)
            mean_ra = _nanmean(ra)
            mean_dec = _nanmean(dec)
            mean_chi2 = _nanmean(chi2)
            med_chi2 = float(np.nanmedian(chi2))
        else:
            ra = np.array([x for x in grp.column("residual_ra_arcsec").to_pylist() if x is not None], dtype=float)
            dec = np.array([x for x in grp.column("residual_dec_arcsec").to_pylist() if x is not None], dtype=float)
            chi2 = np.array([x for x in grp.column("chi2").to_pylist() if x is not None], dtype=float)
            mean_ra = _nanmean(ra)
            mean_dec = _nanmean(dec)
            rms_ra = _nanrms(ra)
            rms_dec = _nanrms(dec)
            mean_chi2 = _nanmean(chi2)
            med_chi2 = float(np.nanmedian(chi2)) if len(chi2) > 0 else np.nan

        rows.append(dict(
            stn=stn,
            astcat=astcat,
            n_obs=n_obs,
            mean_ra_arcsec=mean_ra,
            mean_dec_arcsec=mean_dec,
            rms_ra_arcsec=rms_ra,
            rms_dec_arcsec=rms_dec,
            mean_chi2_per_obs=mean_chi2,
            median_chi2_per_obs=med_chi2,
        ))

    if not rows:
        return CatalogStats.empty()

    return CatalogStats.from_kwargs(
        stn=[r["stn"] for r in rows],
        astcat=[r["astcat"] for r in rows],
        n_obs=[r["n_obs"] for r in rows],
        mean_ra_arcsec=[r["mean_ra_arcsec"] for r in rows],
        mean_dec_arcsec=[r["mean_dec_arcsec"] for r in rows],
        rms_ra_arcsec=[r["rms_ra_arcsec"] for r in rows],
        rms_dec_arcsec=[r["rms_dec_arcsec"] for r in rows],
        mean_chi2_per_obs=[r["mean_chi2_per_obs"] for r in rows],
        median_chi2_per_obs=[r["median_chi2_per_obs"] for r in rows],
    )


def compute_program_code_stats(
    results: LOOOResult,
    min_obs_remaining: Optional[int] = None,
    min_arc_length_days: Optional[float] = None,
    max_held_out_fraction: Optional[float] = None,
    max_hold_in_reduced_chi2: Optional[float] = 100.0,
    min_obs_per_group: int = 10,
) -> ProgramCodeStats:
    """
    Compute statistics broken down by (observatory, program_code) pair.

    program_code comes from the MPC `prog` field and indicates which
    submitting program / survey the observation belongs to.

    Parameters
    ----------
    results : LOOOResult
    min_obs_remaining, min_arc_length_days, max_held_out_fraction,
    max_hold_in_reduced_chi2 : see compute_observatory_stats
    min_obs_per_group : int
        Minimum observations per (stn, program_code) group to report.
    """
    filtered = _filter_results(
        results,
        min_obs_remaining=min_obs_remaining,
        min_arc_length_days=min_arc_length_days,
        max_held_out_fraction=max_held_out_fraction,
        max_hold_in_reduced_chi2=max_hold_in_reduced_chi2,
    )

    tbl = filtered.table
    if "program_code" not in tbl.schema.names:
        return ProgramCodeStats.empty()

    stn_col = tbl.column("stn").to_pylist()
    prog_col = tbl.column("program_code").to_pylist()
    obj_col = tbl.column("object_id").to_pylist()

    groups = sorted(
        set(zip(stn_col, prog_col)),
        key=lambda x: (x[0] or "", x[1] or ""),
    )

    rows = []
    for stn, prog in groups:
        grp_mask = pa.array(
            [(s == stn and p == prog) for s, p in zip(stn_col, prog_col)]
        )
        grp = tbl.filter(grp_mask)
        n_obs = len(grp)
        if n_obs < min_obs_per_group:
            continue

        n_objects = int(len(pc.unique(grp.column("object_id"))))

        ra = np.array(
            [x for x in grp.column("residual_ra_arcsec").to_pylist() if x is not None],
            dtype=float,
        )
        dec = np.array(
            [x for x in grp.column("residual_dec_arcsec").to_pylist() if x is not None],
            dtype=float,
        )
        chi2 = np.array(
            [x for x in grp.column("chi2").to_pylist() if x is not None],
            dtype=float,
        )

        rows.append(
            dict(
                stn=stn,
                program_code=prog,
                n_obs=n_obs,
                n_objects=n_objects,
                mean_ra_arcsec=_nanmean(ra),
                mean_dec_arcsec=_nanmean(dec),
                rms_ra_arcsec=_nanrms(ra),
                rms_dec_arcsec=_nanrms(dec),
                median_abs_ra_arcsec=_nanmedian_abs(ra),
                median_abs_dec_arcsec=_nanmedian_abs(dec),
                mean_chi2_per_obs=_nanmean(chi2),
                median_chi2_per_obs=(
                    float(np.nanmedian(chi2)) if len(chi2) > 0 else np.nan
                ),
            )
        )

    if not rows:
        return ProgramCodeStats.empty()

    return ProgramCodeStats.from_kwargs(
        stn=[r["stn"] for r in rows],
        program_code=[r["program_code"] for r in rows],
        n_obs=[r["n_obs"] for r in rows],
        n_objects=[r["n_objects"] for r in rows],
        mean_ra_arcsec=[r["mean_ra_arcsec"] for r in rows],
        mean_dec_arcsec=[r["mean_dec_arcsec"] for r in rows],
        rms_ra_arcsec=[r["rms_ra_arcsec"] for r in rows],
        rms_dec_arcsec=[r["rms_dec_arcsec"] for r in rows],
        median_abs_ra_arcsec=[r["median_abs_ra_arcsec"] for r in rows],
        median_abs_dec_arcsec=[r["median_abs_dec_arcsec"] for r in rows],
        mean_chi2_per_obs=[r["mean_chi2_per_obs"] for r in rows],
        median_chi2_per_obs=[r["median_chi2_per_obs"] for r in rows],
    )


def print_observatory_summary(stats: ObservatoryStats, top_n: int = 30) -> None:
    """Print a human-readable summary table of observatory statistics."""
    if len(stats) == 0:
        print("No results.")
        return

    # Sort by number of observations descending
    import pandas as pd
    df = stats.table.to_pandas().sort_values("n_obs", ascending=False).head(top_n)

    hdr_bias_ra = 'bias_RA"'
    hdr_bias_dec = 'bias_Dec"'
    hdr_rms_ra = 'RMS_RA"'
    hdr_rms_dec = 'RMS_Dec"'
    print(f"\n{'STN':<6} {'N_obs':>7} {'N_obj':>6} {hdr_bias_ra:>9} {hdr_bias_dec:>10} "
          f"{hdr_rms_ra:>8} {hdr_rms_dec:>9} {'chi2/obs':>9}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(
            f"{row.stn:<6} {int(row.n_obs):>7} {int(row.n_objects):>6} "
            f"{row.mean_ra_arcsec:>+9.3f} {row.mean_dec_arcsec:>+10.3f} "
            f"{row.rms_ra_arcsec:>8.3f} {row.rms_dec_arcsec:>9.3f} "
            f"{row.mean_chi2_per_obs:>9.2f}"
        )
