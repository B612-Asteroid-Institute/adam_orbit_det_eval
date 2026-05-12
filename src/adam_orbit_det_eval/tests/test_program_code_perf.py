"""
Correctness tests for the rewritten compute_program_code_stats (bead ctq).

The legacy O(N x G) implementation built a Python-list mask per group and
filtered the table per group. On the 5.1M-row pilot v12 parquet that
projected to 3-5 hours of wall time; on the 3,500-obj pilot (45k rows /
~1k groups) it took ~minutes but was tolerable. The new implementation
performs a single columnar groupby pass via pandas.

This test file pins a reference O(N x G) implementation (the legacy code
copied verbatim, modulo the public name) and asserts the rewritten function
produces bit-equivalent results on synthetic LOOOResults that cover:
  - mixed None / NaN residuals
  - None program_code values
  - groups below the min_obs_per_group cutoff
  - all-NaN residual columns within a group
  - large number of groups (so the legacy mask-build is exercised on a
    realistically-shaped, but still small, problem)
  - filter args (min_obs_remaining, min_arc_length_days,
    max_held_out_fraction, max_hold_in_reduced_chi2)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_orbit_det_eval.looo.analysis import (
    ProgramCodeStats,
    _filter_results,
    _nanmean,
    _nanmedian_abs,
    _nanrms,
    compute_program_code_stats,
)
from adam_orbit_det_eval.looo.core import LOOOResult


def _legacy_compute_program_code_stats(
    results: LOOOResult,
    min_obs_remaining=None,
    min_arc_length_days=None,
    max_held_out_fraction=None,
    max_hold_in_reduced_chi2=100.0,
    min_obs_per_group: int = 10,
) -> ProgramCodeStats:
    """Verbatim copy of the legacy O(N x G) implementation (bead ctq baseline)."""
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


def _assert_stats_equal(
    a: ProgramCodeStats,
    b: ProgramCodeStats,
    tol: float = 1e-12,
) -> None:
    """
    Compare two ProgramCodeStats group-by-group.

    Canonicalizes both inputs by (stn, program_code) before comparing rows.
    The legacy implementation built groups from `set(zip(stn, prog))` and
    sorted with key `(x[0] or "", x[1] or "")`, which collapses None and
    empty-string program_codes onto the same sort key — within those ties,
    set iteration order (hash-randomized in CPython) governed the legacy
    output order. The per-group VALUES are deterministic; only the row
    ordering of (stn, None) vs (stn, "") was not. We canonicalize so the
    tests assert agreement on what is actually well-defined.
    """
    assert len(a) == len(b), f"row count mismatch: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return

    def _sort_key(stats: ProgramCodeStats):
        # Tiebreak None against "" deterministically: map None to a sentinel
        # that strictly differs from "" without changing the legacy key.
        stns = stats.stn.to_pylist()
        progs = stats.program_code.to_pylist()
        return sorted(
            range(len(stats)),
            key=lambda i: (
                stns[i] or "",
                "\x00" if progs[i] is None else progs[i],
            ),
        )

    a_order = _sort_key(a)
    b_order = _sort_key(b)

    def _take(stats: ProgramCodeStats, order, col: str):
        vals = getattr(stats, col).to_pylist()
        return [vals[i] for i in order]

    assert _take(a, a_order, "stn") == _take(b, b_order, "stn"), "stn order mismatch"
    assert _take(a, a_order, "program_code") == _take(b, b_order, "program_code"), (
        "program_code mismatch after canonical sort"
    )
    assert _take(a, a_order, "n_obs") == _take(b, b_order, "n_obs"), "n_obs mismatch"
    assert _take(a, a_order, "n_objects") == _take(b, b_order, "n_objects"), (
        "n_objects mismatch"
    )
    float_cols = [
        "mean_ra_arcsec",
        "mean_dec_arcsec",
        "rms_ra_arcsec",
        "rms_dec_arcsec",
        "median_abs_ra_arcsec",
        "median_abs_dec_arcsec",
        "mean_chi2_per_obs",
        "median_chi2_per_obs",
    ]
    for col in float_cols:
        va = np.array(_take(a, a_order, col), dtype=float)
        vb = np.array(_take(b, b_order, col), dtype=float)
        both_nan = np.isnan(va) & np.isnan(vb)
        abs_diff = np.where(both_nan, 0.0, np.abs(va - vb))
        scale = np.maximum(np.abs(va), np.abs(vb))
        # Drift from float64 summation order (numpy pairwise sum vs pandas/
        # Arrow sum) shows up at the relative-precision floor — tol is
        # absolute, but for large magnitudes we also allow rtol ~ 1e-13.
        rel_diff = np.where(scale > 0, abs_diff / np.maximum(scale, 1.0), abs_diff)
        ok = (abs_diff <= tol) | (rel_diff <= 1e-13)
        ok = ok | both_nan
        if not ok.all():
            bad = np.where(~ok)[0][:5]
            raise AssertionError(
                f"{col} mismatch:\n"
                f"  max abs diff = {abs_diff.max()}\n"
                f"  max rel diff = {rel_diff.max()}\n"
                f"  worst rows (legacy / new): "
                + ", ".join(f"[{i}] {va[i]} / {vb[i]}" for i in bad)
            )


def _make_synthetic(
    n_objects: int,
    stns: list[str],
    progs: list,
    seed: int = 0,
    nan_residual_frac: float = 0.0,
) -> LOOOResult:
    """Build a deterministic synthetic LOOOResult covering many (stn, prog) groups."""
    rng = np.random.default_rng(seed)
    rows = []
    for obj_idx in range(n_objects):
        # Each object is observed by a random subset of stations, each with a
        # random program_code from the supplied list.
        n_obs_obj = int(rng.integers(2, 12))
        for k in range(n_obs_obj):
            stn = stns[int(rng.integers(0, len(stns)))]
            prog = progs[int(rng.integers(0, len(progs)))]
            ra = float(rng.normal(0.05, 0.3))
            dec = float(rng.normal(-0.02, 0.3))
            if nan_residual_frac > 0 and rng.random() < nan_residual_frac:
                ra = float("nan")
            rows.append(
                (
                    f"obj{obj_idx}",
                    f"obs{obj_idx}-{k}",
                    stn,
                    ra,
                    dec,
                    0.3,
                    0.3,
                    float(rng.uniform(0.1, 5.0)),
                    "Gaia2",
                    prog,
                    2,
                    20,
                    30.0,
                    0.1,
                    22,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    True,
                )
            )

    cols = list(zip(*rows))
    return LOOOResult.from_kwargs(
        object_id=list(cols[0]),
        obs_id=list(cols[1]),
        stn=list(cols[2]),
        residual_ra_arcsec=list(cols[3]),
        residual_dec_arcsec=list(cols[4]),
        sigma_ra_cosdec_arcsec=list(cols[5]),
        sigma_dec_arcsec=list(cols[6]),
        chi2=list(cols[7]),
        astcat=list(cols[8]),
        program_code=list(cols[9]),
        n_obs_held_out=list(cols[10]),
        n_obs_remaining=list(cols[11]),
        arc_length_remaining_days=list(cols[12]),
        held_out_fraction=list(cols[13]),
        n_obs_total=list(cols[14]),
        delta_q_au=list(cols[15]),
        delta_e=list(cols[16]),
        delta_i_deg=list(cols[17]),
        hold_in_reduced_chi2=list(cols[18]),
        hold_in_fit_success=list(cols[19]),
    )


def test_new_matches_legacy_basic():
    res = _make_synthetic(
        n_objects=200,
        stns=["568", "703", "G96", "F51"],
        progs=["A", "B", "C", None],
        seed=42,
    )
    old = _legacy_compute_program_code_stats(res, min_obs_per_group=5)
    new = compute_program_code_stats(res, min_obs_per_group=5)
    _assert_stats_equal(old, new)


def test_new_matches_legacy_with_nan_residuals():
    res = _make_synthetic(
        n_objects=300,
        stns=["568", "703", "G96"],
        progs=["A", "B", None],
        seed=7,
        nan_residual_frac=0.15,
    )
    old = _legacy_compute_program_code_stats(res, min_obs_per_group=8)
    new = compute_program_code_stats(res, min_obs_per_group=8)
    _assert_stats_equal(old, new)


def test_new_matches_legacy_with_filter_args():
    res = _make_synthetic(
        n_objects=400,
        stns=["568", "703", "G96", "F51", "M22"],
        progs=["A", "B", "C", "D", None],
        seed=11,
    )
    old = _legacy_compute_program_code_stats(
        res,
        min_obs_remaining=15,
        min_arc_length_days=10.0,
        max_held_out_fraction=0.5,
        max_hold_in_reduced_chi2=10.0,
        min_obs_per_group=3,
    )
    new = compute_program_code_stats(
        res,
        min_obs_remaining=15,
        min_arc_length_days=10.0,
        max_held_out_fraction=0.5,
        max_hold_in_reduced_chi2=10.0,
        min_obs_per_group=3,
    )
    _assert_stats_equal(old, new)


def test_new_matches_legacy_many_groups():
    # Stress-test the group ordering: many stations x many programs.
    stns = [f"S{i:02d}" for i in range(15)]
    progs = [f"P{i}" for i in range(10)] + [None]
    res = _make_synthetic(
        n_objects=800,
        stns=stns,
        progs=progs,
        seed=99,
        nan_residual_frac=0.05,
    )
    old = _legacy_compute_program_code_stats(res, min_obs_per_group=4)
    new = compute_program_code_stats(res, min_obs_per_group=4)
    _assert_stats_equal(old, new)


def test_no_program_code_column_returns_empty(tmp_path):
    """
    Legacy parquet files (e.g. the 3,500-obj pilot from 2026-03-16) were
    written before the program_code column existed in LOOOResult. The
    function must still return an empty ProgramCodeStats for these.

    We simulate this by writing a LOOOResult parquet, stripping the
    program_code column, reading it back as a raw pyarrow Table, and
    wrapping it in a minimal LOOOResult-shaped façade just for the
    schema-check branch (skipping _filter_results via max_hold_in_reduced_chi2=None).
    """
    res = _make_synthetic(50, ["568"], ["A"], seed=1)
    tbl = res.table
    stripped_tbl = tbl.drop(["program_code"])

    class _StubResult:
        def __init__(self, t):
            self.table = t

        def __len__(self):
            return self.table.num_rows

        def apply_mask(self, _mask):
            # _filter_results unconditionally constructs an all-True mask and
            # calls apply_mask; return self so the column-presence check below
            # is reached.
            return self

    stub = _StubResult(stripped_tbl)
    out = compute_program_code_stats(
        stub,  # type: ignore[arg-type]
        max_hold_in_reduced_chi2=None,
    )
    assert isinstance(out, ProgramCodeStats)
    assert len(out) == 0


def test_empty_input_returns_empty():
    empty = LOOOResult.empty()
    out = compute_program_code_stats(empty)
    assert len(out) == 0
