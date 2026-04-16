"""
Tests for adam_orbit_det_eval.looo.bias_table.

Covers:
  - Bootstrap CI: width on synthetic Gaussian data matches the analytical
    sampling distribution of the mean within Monte-Carlo error.
  - Object-weighted aggregation: a heavily-observed object does not
    dominate the bias estimate.
  - Null program_code handling: mixed null / non-null inputs produce both
    observatory-level and per-program-code rows without crashing, and
    stations with 100% null program_code produce only an observatory row.

All synthetic; no reliance on the 3,500-object parquet or any cloud data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adam_orbit_det_eval.looo.bias_table import (
    BootstrapConfig,
    bootstrap_mean_ci,
    compute_bias_table,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_looo_df(rows):
    """Build a LOOO-shaped DataFrame from a list of dicts."""
    required = [
        "object_id", "obs_id", "stn",
        "residual_ra_arcsec", "residual_dec_arcsec",
        "chi2",
    ]
    df = pd.DataFrame(rows)
    for col in required:
        assert col in df.columns, f"missing required test column: {col}"
    return df


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_contains_true_mean_on_clean_data(self):
        """For N=1000 Gaussian samples with known mean, 95% CI must contain it."""
        rng = np.random.default_rng(0)
        true_mean = 0.25
        true_std = 0.5
        values = rng.normal(true_mean, true_std, size=1000)

        cfg = BootstrapConfig(n_resamples=2000, random_seed=42)
        ci_lo, ci_hi = bootstrap_mean_ci(values, cfg, np.random.default_rng(42))

        assert ci_lo < true_mean < ci_hi
        # The CI width should be roughly 4 * SEM (≈ 2 * 1.96 * sigma/sqrt(n)).
        expected_width = 2 * 1.96 * true_std / np.sqrt(len(values))
        observed_width = ci_hi - ci_lo
        # Allow 25% tolerance — bootstrap on 2000 draws has its own noise.
        assert 0.75 * expected_width < observed_width < 1.25 * expected_width, (
            f"CI width {observed_width:.4f} vs expected {expected_width:.4f}"
        )

    def test_ci_excludes_zero_for_biased_signal(self):
        """A mean clearly above zero should produce a CI that excludes zero."""
        rng = np.random.default_rng(1)
        values = rng.normal(0.5, 0.1, size=200)  # SEM ~0.007 → CI tight

        cfg = BootstrapConfig(n_resamples=2000, random_seed=42)
        ci_lo, ci_hi = bootstrap_mean_ci(values, cfg, np.random.default_rng(42))

        assert ci_lo > 0, f"expected CI to exclude zero, got [{ci_lo}, {ci_hi}]"

    def test_ci_spans_zero_for_null_signal(self):
        """Zero-mean data should yield a CI that contains zero."""
        rng = np.random.default_rng(2)
        values = rng.normal(0.0, 0.3, size=500)

        cfg = BootstrapConfig(n_resamples=2000, random_seed=42)
        ci_lo, ci_hi = bootstrap_mean_ci(values, cfg, np.random.default_rng(42))

        assert ci_lo <= 0 <= ci_hi, (
            f"expected CI to span zero, got [{ci_lo}, {ci_hi}]"
        )

    def test_empty_array_returns_nan_nan(self):
        cfg = BootstrapConfig(n_resamples=100)
        ci_lo, ci_hi = bootstrap_mean_ci(
            np.array([], dtype=float), cfg, np.random.default_rng(0)
        )
        assert np.isnan(ci_lo) and np.isnan(ci_hi)

    def test_all_nan_array_returns_nan_nan(self):
        cfg = BootstrapConfig(n_resamples=100)
        ci_lo, ci_hi = bootstrap_mean_ci(
            np.array([np.nan, np.nan, np.nan]), cfg, np.random.default_rng(0)
        )
        assert np.isnan(ci_lo) and np.isnan(ci_hi)

    def test_bootstrap_is_deterministic_with_seed(self):
        """Same seed → same CI (reproducibility guarantee)."""
        rng_vals = np.random.default_rng(99).normal(0.1, 0.5, 200)
        cfg = BootstrapConfig(n_resamples=500, random_seed=123)

        a_lo, a_hi = bootstrap_mean_ci(rng_vals, cfg, np.random.default_rng(123))
        b_lo, b_hi = bootstrap_mean_ci(rng_vals, cfg, np.random.default_rng(123))

        assert a_lo == b_lo and a_hi == b_hi


# ---------------------------------------------------------------------------
# Object-weighted aggregation tests
# ---------------------------------------------------------------------------


class TestObjectWeightedAggregation:
    def test_heavily_observed_object_does_not_dominate(self):
        """
        Object A contributes 1000 obs with bias +1.0".
        Object B contributes 10 obs with bias -1.0".
        Object-weighted mean should be near 0 (both objects equal weight),
        NOT near +1.0 (which is what observation-weighted would give).
        """
        rows = []
        # 1000 obs for object A, all biased at +1.0"
        for i in range(1000):
            rows.append(dict(
                object_id="A", obs_id=f"A{i:04d}", stn="XXX",
                residual_ra_arcsec=1.0, residual_dec_arcsec=1.0,
                chi2=1.0,
            ))
        # 10 obs for object B, all biased at -1.0"
        for i in range(10):
            rows.append(dict(
                object_id="B", obs_id=f"B{i:04d}", stn="XXX",
                residual_ra_arcsec=-1.0, residual_dec_arcsec=-1.0,
                chi2=1.0,
            ))
        # Add a third + fourth object so min_objects_per_group is satisfied
        for obj_id, sign in [("C", +0.5), ("D", -0.5)]:
            for i in range(20):
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}{i:04d}", stn="XXX",
                    residual_ra_arcsec=sign, residual_dec_arcsec=sign,
                    chi2=1.0,
                ))

        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=200, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=5, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )

        # Expect 1 observatory row (program_code null), no program-code rows
        assert len(table) == 1
        row = table.iloc[0]
        assert row["obs_code"] == "XXX"
        assert pd.isna(row["program_code"])

        # Object-weighted bias: (+1.0 + -1.0 + +0.5 + -0.5) / 4 = 0.0
        assert abs(row["bias_ra_arcsec"]) < 1e-9, (
            f"object-weighted bias should be 0, got {row['bias_ra_arcsec']}"
        )
        # Observation-weighted sanity check: if we had weighted by obs count
        # the bias would be dominated by the 1000-obs +1.0 signal, giving
        # roughly +0.97".  The fact that we get 0 confirms the weighting
        # rule is object-level.

        # n_obs should be the total obs, n_objects should be 4
        assert row["n_obs"] == 1000 + 10 + 20 + 20
        assert row["n_objects"] == 4

    def test_rms_reflects_within_object_scatter(self):
        """
        Objects with identical means but non-zero scatter should produce a
        non-zero RMS.  Specifically: if each object's residuals are ~N(0, s),
        the object-weighted RMS (mean-of-per-object-mean-square then sqrt)
        should be close to s for large samples.
        """
        rng = np.random.default_rng(7)
        s = 0.3
        rows = []
        for obj_idx in range(50):
            obj_id = f"obj{obj_idx}"
            for i in range(40):
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}-{i}", stn="YYY",
                    residual_ra_arcsec=float(rng.normal(0.0, s)),
                    residual_dec_arcsec=float(rng.normal(0.0, s)),
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=200, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        # Mean of per-object mean-square ≈ s^2 → sqrt ≈ s
        assert abs(row["rms_ra_arcsec"] - s) < 0.05


# ---------------------------------------------------------------------------
# Program-code handling
# ---------------------------------------------------------------------------


class TestProgramCodeHandling:
    def test_mixed_null_and_nonnull_produces_both_levels(self):
        """
        Station Z has:
          - 5 objects * 20 obs each with program_code = "P1"
          - 5 objects * 20 obs each with program_code = None
        Expected output rows:
          - Z, program_code=null (observatory-level aggregate over all obs)
          - Z, program_code="P1" (per-program-code row for the P1 subset)
        The null-program-code rows themselves should NOT produce a
        (stn, program_code=null) per-program row.
        """
        rows = []
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"p1_obj{obj_idx}", obs_id=f"p1_{obj_idx}_{i}",
                    stn="Z", residual_ra_arcsec=0.1, residual_dec_arcsec=0.0,
                    chi2=1.0, program_code="P1",
                ))
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"null_obj{obj_idx}",
                    obs_id=f"null_{obj_idx}_{i}",
                    stn="Z", residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0, program_code=None,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=200, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )

        # Expect exactly 2 rows: observatory-level + P1
        assert len(table) == 2

        obs_row = table[table["program_code"].isna()].iloc[0]
        prog_row = table[table["program_code"] == "P1"].iloc[0]

        assert obs_row["obs_code"] == "Z"
        assert obs_row["n_objects"] == 10
        assert obs_row["n_obs"] == 200
        # Observatory-level bias: mean over 10 objects (5 @ +0.1, 5 @ 0) = 0.05
        assert abs(obs_row["bias_ra_arcsec"] - 0.05) < 1e-9

        assert prog_row["obs_code"] == "Z"
        assert prog_row["program_code"] == "P1"
        assert prog_row["n_objects"] == 5
        assert prog_row["n_obs"] == 100
        assert abs(prog_row["bias_ra_arcsec"] - 0.1) < 1e-9

    def test_all_null_program_code_produces_only_observatory_row(self):
        """Station with 100% null program_code should not crash and should
        produce exactly one observatory-level row."""
        rows = []
        for obj_idx in range(4):
            for i in range(15):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"obj{obj_idx}_{i}",
                    stn="ALLNULL",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0, program_code=None,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        assert len(table) == 1
        assert table.iloc[0]["obs_code"] == "ALLNULL"
        assert pd.isna(table.iloc[0]["program_code"])

    def test_missing_program_code_column_is_handled(self):
        """Pre-trksub LOOO parquets have no program_code column at all."""
        rows = []
        for obj_idx in range(4):
            for i in range(15):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"obj{obj_idx}_{i}",
                    stn="NOCOL",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        assert "program_code" not in df.columns

        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        assert len(table) == 1
        assert table.iloc[0]["obs_code"] == "NOCOL"
        assert pd.isna(table.iloc[0]["program_code"])


# ---------------------------------------------------------------------------
# Optional-column handling (AT/CT absent)
# ---------------------------------------------------------------------------


class TestOptionalATCT:
    def test_at_ct_nan_when_columns_absent(self):
        """Input without residual_at_arcsec / residual_ct_arcsec → NaN AT/CT outputs."""
        rows = []
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"{obj_idx}_{i}",
                    stn="S",
                    residual_ra_arcsec=0.2, residual_dec_arcsec=0.1,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert np.isnan(row["bias_at_arcsec"])
        assert np.isnan(row["bias_ct_arcsec"])
        assert np.isnan(row["rms_at_arcsec"])
        # RA/Dec are still populated
        assert abs(row["bias_ra_arcsec"] - 0.2) < 1e-9
        assert abs(row["bias_dec_arcsec"] - 0.1) < 1e-9

    def test_at_ct_populated_when_columns_present(self):
        """Input with AT/CT columns → non-NaN AT/CT outputs."""
        rows = []
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"{obj_idx}_{i}",
                    stn="S",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    residual_at_arcsec=0.3, residual_ct_arcsec=-0.15,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert abs(row["bias_at_arcsec"] - 0.3) < 1e-9
        assert abs(row["bias_ct_arcsec"] - (-0.15)) < 1e-9


# ---------------------------------------------------------------------------
# Minimum-size filtering
# ---------------------------------------------------------------------------


class TestMinSizeFilter:
    def test_groups_below_threshold_are_dropped(self):
        rows = []
        # Station "BIG" — 5 objects, 20 obs each → passes
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"big{obj_idx}", obs_id=f"big_{obj_idx}_{i}",
                    stn="BIG",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0,
                ))
        # Station "TINY" — 2 objects, 3 obs each → fails on both n_obs and n_objects
        for obj_idx in range(2):
            for i in range(3):
                rows.append(dict(
                    object_id=f"tiny{obj_idx}", obs_id=f"tiny_{obj_idx}_{i}",
                    stn="TINY",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)

        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        assert set(table["obs_code"]) == {"BIG"}
