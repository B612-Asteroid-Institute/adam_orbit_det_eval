"""Tests for the aggregation-time bad-fit filter (bead 7bt)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pytest

from adam_orbit_det_eval.looo.bias_filter import (
    BiasFilterConfig,
    BiasFilterStats,
    apply_bias_filter,
    format_filter_audit,
)
from adam_orbit_det_eval.looo.core import LOOOResult


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_row(
    object_id: str,
    obs_id: str,
    stn: str,
    *,
    residual_ra_arcsec: float = 0.1,
    residual_dec_arcsec: float = 0.1,
    sigma_ra: float = 0.5,
    sigma_dec: float = 0.5,
    chi2: float = 1.0,
    astcat: Optional[str] = "Gaia2",
    program_code: Optional[str] = None,
    band: Optional[str] = None,
    n_obs_held_out: int = 5,
    n_obs_remaining: int = 50,
    arc_length_remaining_days: float = 365.0,
    held_out_fraction: float = 0.1,
    n_obs_total: int = 55,
    delta_q_au: Optional[float] = 0.0,
    delta_e: Optional[float] = 0.0,
    delta_i_deg: Optional[float] = 0.0,
    hold_in_reduced_chi2: Optional[float] = 1.5,
    hold_in_fit_success: Optional[bool] = True,
) -> Dict[str, Any]:
    return dict(
        object_id=object_id,
        obs_id=obs_id,
        stn=stn,
        residual_ra_arcsec=residual_ra_arcsec,
        residual_dec_arcsec=residual_dec_arcsec,
        sigma_ra_cosdec_arcsec=sigma_ra,
        sigma_dec_arcsec=sigma_dec,
        chi2=chi2,
        astcat=astcat,
        program_code=program_code,
        band=band,
        n_obs_held_out=n_obs_held_out,
        n_obs_remaining=n_obs_remaining,
        arc_length_remaining_days=arc_length_remaining_days,
        held_out_fraction=held_out_fraction,
        n_obs_total=n_obs_total,
        delta_q_au=delta_q_au,
        delta_e=delta_e,
        delta_i_deg=delta_i_deg,
        hold_in_reduced_chi2=hold_in_reduced_chi2,
        hold_in_fit_success=hold_in_fit_success,
    )


def _rows_to_table(rows: List[Dict[str, Any]]) -> pa.Table:
    """Build a pyarrow Table whose schema matches LOOOResult."""
    cols = list(rows[0].keys())
    arrays = {}
    for c in cols:
        vals = [r[c] for r in rows]
        if c in {"object_id", "obs_id", "stn", "astcat", "program_code", "band"}:
            arrays[c] = pa.array(vals, type=pa.large_string())
        elif c in {
            "residual_ra_arcsec",
            "residual_dec_arcsec",
            "sigma_ra_cosdec_arcsec",
            "sigma_dec_arcsec",
            "chi2",
            "arc_length_remaining_days",
            "held_out_fraction",
            "delta_q_au",
            "delta_e",
            "delta_i_deg",
            "hold_in_reduced_chi2",
        }:
            arrays[c] = pa.array(vals, type=pa.float64())
        elif c in {"n_obs_held_out", "n_obs_remaining", "n_obs_total"}:
            arrays[c] = pa.array(vals, type=pa.int64())
        elif c == "hold_in_fit_success":
            arrays[c] = pa.array(vals, type=pa.bool_())
        else:
            raise KeyError(c)
    return pa.table(arrays)


# ---------------------------------------------------------------------------
# Tier coverage
# ---------------------------------------------------------------------------


class TestTier1FailedFit:
    def test_drops_false_and_null(self):
        rows = [
            _make_row("o1", "a1", "500"),                    # keep
            _make_row("o1", "a2", "500", hold_in_fit_success=False),  # drop T1
            _make_row("o1", "a3", "500", hold_in_fit_success=None),   # drop T1
            _make_row("o2", "b1", "703"),                    # keep
        ]
        tbl = _rows_to_table(rows)
        filtered, stats = apply_bias_filter(tbl)
        assert stats.rows_in == 4
        assert stats.dropped_tier1_failed_fit == 2
        # Tier 2 sees only the surviving 2 rows; both have chi2=1.5 < 10
        assert stats.dropped_tier2_high_chi2 == 0
        assert stats.rows_out == 2


class TestTier2HighChi2:
    def test_drops_above_threshold(self):
        rows = [
            _make_row("o1", "a1", "500", hold_in_reduced_chi2=1.0),
            _make_row("o1", "a2", "500", hold_in_reduced_chi2=11.0),  # drop T2
            _make_row("o2", "b1", "703", hold_in_reduced_chi2=9.5),
            _make_row("o2", "b2", "703", hold_in_reduced_chi2=1000.0),  # drop T2
        ]
        tbl = _rows_to_table(rows)
        filtered, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier1_failed_fit == 0
        assert stats.dropped_tier2_high_chi2 == 2
        assert stats.rows_out == 2

    def test_drops_null_chi2(self):
        # On a "successful" row with null chi2, Tier 2 drops it: a null chi2
        # under success=True is exactly the v11 silent-failure mode.
        rows = [
            _make_row("o1", "a1", "500", hold_in_reduced_chi2=None),
            _make_row("o1", "a2", "500", hold_in_reduced_chi2=2.0),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier2_high_chi2 == 1
        assert stats.rows_out == 1

    def test_custom_threshold(self):
        rows = [
            _make_row("o1", "a1", "500", hold_in_reduced_chi2=15.0),
            _make_row("o2", "a1", "500", hold_in_reduced_chi2=5.0),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl, BiasFilterConfig(max_chi2=20.0))
        assert stats.dropped_tier2_high_chi2 == 0


class TestTier3OrbitDrift:
    def test_drops_all_rows_for_drifted_object(self):
        # Object 'bad' has one row with delta_q above 0.5; every row from
        # 'bad' should be dropped — including rows from other stations and
        # rows where this single row's delta_q is fine.
        rows = [
            _make_row("clean", "a1", "500", delta_q_au=0.0),
            _make_row("clean", "a2", "703", delta_q_au=0.0),
            _make_row("bad", "b1", "500", delta_q_au=0.7),    # triggers
            _make_row("bad", "b2", "703", delta_q_au=0.05),   # whole-obj cull
            _make_row("bad", "b3", "F52", delta_q_au=0.05),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.objects_dropped_tier3 == 1
        assert stats.dropped_tier3_orbit_drift == 3
        assert stats.rows_out == 2

    def test_delta_e_triggers_independently(self):
        rows = [
            _make_row("clean", "a1", "500", delta_e=0.0),
            _make_row("bad_e", "b1", "500", delta_e=0.4),
            _make_row("bad_e", "b2", "703", delta_e=0.0),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.objects_dropped_tier3 == 1
        assert stats.dropped_tier3_orbit_drift == 2

    def test_delta_i_triggers_independently(self):
        rows = [
            _make_row("clean", "a1", "500", delta_i_deg=0.0),
            _make_row("bad_i", "b1", "500", delta_i_deg=6.0),
            _make_row("bad_i", "b2", "703", delta_i_deg=0.0),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.objects_dropped_tier3 == 1
        assert stats.dropped_tier3_orbit_drift == 2

    def test_null_drift_columns_dont_indict(self):
        # Nulls in delta_* should not flag the object — non-finite is "no
        # evidence of drift" rather than "drift detected".
        rows = [
            _make_row("o1", "a1", "500", delta_q_au=None,
                      delta_e=None, delta_i_deg=None),
            _make_row("o2", "a1", "500"),
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.objects_dropped_tier3 == 0


class TestTier4MAD:
    def test_drops_residual_outlier(self):
        # 7 clean rows with tight residuals + 1 outlier at the same station.
        # Outlier should be dropped; 7 inliers retained.
        clean_residuals = [0.0, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01]
        rows = [
            _make_row(f"obj{i}", f"a{i}", "500", residual_ra_arcsec=r)
            for i, r in enumerate(clean_residuals)
        ]
        rows.append(_make_row("obj99", "a99", "500", residual_ra_arcsec=50.0))
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier4_mad == 1
        assert stats.per_station["500"]["rows_in"] == 8
        assert stats.per_station["500"]["rows_out"] == 7
        # MAD should be small, around the residual scatter
        assert stats.per_station["500"]["mad_ra_arcsec"] > 0
        assert stats.per_station["500"]["mad_ra_arcsec"] < 0.5

    def test_dec_outlier_also_drops(self):
        clean = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
        rows = [
            _make_row(f"obj{i}", f"a{i}", "703",
                      residual_ra_arcsec=0.0, residual_dec_arcsec=d)
            for i, d in enumerate(clean)
        ]
        rows.append(_make_row("obj99", "a99", "703",
                              residual_ra_arcsec=0.0,
                              residual_dec_arcsec=80.0))
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier4_mad == 1

    def test_independent_per_station_thresholds(self):
        # Station A is tight → flags a 5" point as outlier.
        # Station B is loose → 5" is normal.
        rows: List[Dict[str, Any]] = []
        for i, r in enumerate([0.0, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01]):
            rows.append(_make_row(f"a{i}", f"a{i}", "TIGHT",
                                  residual_ra_arcsec=r))
        rows.append(_make_row("a99", "a99", "TIGHT", residual_ra_arcsec=5.0))
        for i, r in enumerate([0.0, 5.0, -5.0, 2.0, -2.0, 1.0, -1.0]):
            rows.append(_make_row(f"b{i}", f"b{i}", "LOOSE",
                                  residual_ra_arcsec=r))
        rows.append(_make_row("b99", "b99", "LOOSE", residual_ra_arcsec=5.0))
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.per_station["TIGHT"]["rows_out"] == 7  # outlier dropped
        assert stats.per_station["LOOSE"]["rows_out"] == 8  # all kept

    def test_zero_mad_keeps_all_rows(self):
        # All identical residuals → MAD=0 → no rows dropped (degenerate).
        rows = [
            _make_row(f"o{i}", f"a{i}", "500",
                      residual_ra_arcsec=0.1, residual_dec_arcsec=0.1)
            for i in range(5)
        ]
        tbl = _rows_to_table(rows)
        _, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier4_mad == 0
        assert stats.per_station["500"]["mad_ra_arcsec"] == 0.0


class TestEmptyInput:
    def test_empty_table_returns_empty_with_zero_stats(self):
        rows = [_make_row("o1", "a1", "500")]
        tbl = _rows_to_table(rows).slice(0, 0)
        filtered, stats = apply_bias_filter(tbl)
        assert stats.rows_in == 0
        assert stats.rows_out == 0
        assert len(filtered) == 0


class TestQuivrInputOutput:
    def test_returns_looo_result_when_given_one(self):
        rows = [
            _make_row("o1", "a1", "500"),
            _make_row("o1", "a2", "500", hold_in_fit_success=False),
        ]
        tbl = _rows_to_table(rows)
        looo = LOOOResult.from_pyarrow(tbl)
        filtered, stats = apply_bias_filter(looo)
        assert isinstance(filtered, LOOOResult)
        assert len(filtered) == stats.rows_out


class TestStatsHelpers:
    def test_loss_fraction_zero_on_empty(self):
        s = BiasFilterStats()
        assert s.loss_fraction == 0.0

    def test_loss_fraction_value(self):
        s = BiasFilterStats(rows_in=100, rows_out=80)
        assert s.loss_fraction == pytest.approx(0.20)

    def test_high_mad_helper_respects_min_rows(self):
        s = BiasFilterStats(rows_in=10, rows_out=10)
        s.per_station = {
            "small": {"rows_in": 3, "rows_out": 2, "fraction_lost_tier4": 0.33},
            "big_clean": {"rows_in": 200, "rows_out": 198,
                          "fraction_lost_tier4": 0.01},
            "big_dirty": {"rows_in": 200, "rows_out": 180,
                          "fraction_lost_tier4": 0.10},
        }
        flagged = s.stations_with_high_mad_loss(threshold=0.05, min_rows=30)
        assert set(flagged.keys()) == {"big_dirty"}

    def test_format_filter_audit_runs(self):
        s = BiasFilterStats(
            rows_in=100, rows_out=80,
            dropped_tier1_failed_fit=1,
            dropped_tier2_high_chi2=10,
            dropped_tier3_orbit_drift=5,
            objects_dropped_tier3=1,
            dropped_tier4_mad=4,
        )
        text = format_filter_audit(s, BiasFilterConfig())
        assert "rows_in" in text
        assert "Tier 1" in text and "Tier 4" in text
        assert "loss 20.00%" in text


class TestEndToEndCombination:
    def test_all_tiers_together(self):
        rows: List[Dict[str, Any]] = []
        # 7 clean rows at station A
        for i, r in enumerate([0.0, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01]):
            rows.append(_make_row(f"clean{i}", f"a{i}", "A",
                                  residual_ra_arcsec=r))
        # Tier 1: failed fit
        rows.append(_make_row("t1", "x", "A", hold_in_fit_success=False))
        # Tier 2: high chi2
        rows.append(_make_row("t2", "x", "A", hold_in_reduced_chi2=999.0))
        # Tier 3: drifted object — TWO rows, both should drop
        rows.append(_make_row("t3", "y1", "A", delta_q_au=1.0))
        rows.append(_make_row("t3", "y2", "B", delta_q_au=0.0))
        # Tier 4: outlier at station A
        rows.append(_make_row("t4", "z", "A", residual_ra_arcsec=99.0))
        tbl = _rows_to_table(rows)
        filtered, stats = apply_bias_filter(tbl)
        assert stats.dropped_tier1_failed_fit == 1
        assert stats.dropped_tier2_high_chi2 == 1
        assert stats.dropped_tier3_orbit_drift == 2
        assert stats.objects_dropped_tier3 == 1
        assert stats.dropped_tier4_mad == 1
        assert stats.rows_out == 7  # only the 7 clean rows survive
        assert len(filtered) == 7
