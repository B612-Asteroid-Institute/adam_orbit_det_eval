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

from adam_orbit_det_eval.looo.bias_table import (
    BootstrapConfig,
    bootstrap_mean_ci,
    compute_bias_table,
    compute_residual_covariance,
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


class TestResidualCovariance:
    """Per-station 2×2 residual covariance (RA/Dec frame)."""

    # New column family must always be present in the output schema.
    _RADEC_COV_COLS = (
        "resid_var_ra", "resid_var_dec", "resid_cov_ra_dec", "resid_cov_n",
    )

    @staticmethod
    def _station_rows(stn, ra_pattern, dec_pattern, n_objects=4):
        """Replicate a fixed (RA, Dec) residual pattern across `n_objects`.

        Every object gets the identical multiset of residuals, with zero
        per-object mean, so the object-weighted RMS and the pooled per-obs
        variance coincide exactly — the regime in which
        `resid_var ≈ rms²` is an exact identity.
        """
        rows = []
        for obj_idx in range(n_objects):
            obj_id = f"{stn}_obj{obj_idx}"
            for i, (ra, dec) in enumerate(zip(ra_pattern, dec_pattern)):
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}_{i}", stn=stn,
                    residual_ra_arcsec=float(ra),
                    residual_dec_arcsec=float(dec),
                    chi2=1.0,
                ))
        return rows

    def test_columns_present(self):
        df = _make_looo_df(self._station_rows(
            "S", [0.3, -0.3], [0.2, -0.2], n_objects=6,
        ))
        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        for col in self._RADEC_COV_COLS:
            assert col in table.columns, f"missing covariance column {col}"

    def test_resid_var_matches_rms_squared(self):
        """resid_var_ra ≈ rms_ra² and resid_var_dec ≈ rms_dec² (cross-check)."""
        df = _make_looo_df(self._station_rows(
            "S", [0.3, -0.3, 0.3, -0.3], [0.2, -0.2, 0.2, -0.2], n_objects=5,
        ))
        cfg = BootstrapConfig(n_resamples=100, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert abs(row["resid_var_ra"] - row["rms_ra_arcsec"] ** 2) < 1e-12
        assert abs(row["resid_var_dec"] - row["rms_dec_arcsec"] ** 2) < 1e-12
        # Sanity on the absolute values too.
        assert abs(row["resid_var_ra"] - 0.09) < 1e-12
        assert abs(row["resid_var_dec"] - 0.04) < 1e-12

    def test_resid_cov_n_equals_n_obs(self):
        df = _make_looo_df(self._station_rows(
            "S", [0.3, -0.3], [0.2, -0.2], n_objects=6,
        ))
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert row["resid_cov_n"] == row["n_obs"] == 12

    def test_cov_sign_positive_for_correlated_station(self):
        """RA and Dec moving together → positive covariance."""
        df = _make_looo_df(self._station_rows(
            "POS", [1, 1, -1, -1], [1, 1, -1, -1], n_objects=4,
        ))
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert row["resid_cov_ra_dec"] > 0
        assert abs(row["resid_cov_ra_dec"] - 1.0) < 1e-12
        # Diagonal cross-check still holds.
        assert abs(row["resid_var_ra"] - row["rms_ra_arcsec"] ** 2) < 1e-12

    def test_cov_sign_negative_for_anticorrelated_station(self):
        df = _make_looo_df(self._station_rows(
            "NEG", [1, 1, -1, -1], [-1, -1, 1, 1], n_objects=4,
        ))
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert row["resid_cov_ra_dec"] < 0
        assert abs(row["resid_cov_ra_dec"] - (-1.0)) < 1e-12

    def test_cov_zero_for_independent_station(self):
        """Orthogonal RA/Dec pattern → covariance exactly zero."""
        df = _make_looo_df(self._station_rows(
            "IND", [1, 1, -1, -1], [1, -1, 1, -1], n_objects=4,
        ))
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert abs(row["resid_cov_ra_dec"]) < 1e-12

    def test_helper_drops_nan_pairs(self):
        """compute_residual_covariance counts only finite (RA, Dec) pairs."""
        rows = self._station_rows("S", [0.3, -0.3], [0.2, -0.2], n_objects=4)
        # Inject one NaN-RA observation; it must be excluded from resid_cov_n.
        rows.append(dict(
            object_id="S_obj0", obs_id="S_nan", stn="S",
            residual_ra_arcsec=float("nan"), residual_dec_arcsec=0.2,
            chi2=1.0,
        ))
        df = _make_looo_df(rows)
        cov = compute_residual_covariance(df, ["stn"], ["ra", "dec"])
        assert int(cov.iloc[0]["resid_cov_n"]) == 8  # 4 objects * 2 obs


class TestATCTCovariance:
    """AT/CT covariance mirrors the RA/Dec family when AT/CT are present."""

    _ATCT_COV_COLS = ("resid_var_at", "resid_var_ct", "resid_cov_at_ct")

    def test_atct_cov_null_when_columns_absent(self):
        rows = []
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"{obj_idx}_{i}",
                    stn="S", residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        for col in self._ATCT_COV_COLS:
            assert col in table.columns
            assert np.isnan(row[col])

    def test_atct_cov_populated_and_sign_checked(self):
        """AT/CT present → variances match rms² and covariance sign is right."""
        rows = []
        # Per object: AT/CT anti-correlated, zero per-object mean.
        at_pat = [0.4, 0.4, -0.4, -0.4]
        ct_pat = [-0.2, -0.2, 0.2, 0.2]
        for obj_idx in range(5):
            obj_id = f"obj{obj_idx}"
            for i, (at, ct) in enumerate(zip(at_pat, ct_pat)):
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}_{i}", stn="S",
                    residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    residual_at_arcsec=float(at), residual_ct_arcsec=float(ct),
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert abs(row["resid_var_at"] - 0.16) < 1e-12
        assert abs(row["resid_var_ct"] - 0.04) < 1e-12
        assert abs(row["resid_var_at"] - row["rms_at_arcsec"] ** 2) < 1e-12
        assert abs(row["resid_var_ct"] - row["rms_ct_arcsec"] ** 2) < 1e-12
        # AT up while CT down → negative covariance: -0.4 * 0.2 = -0.08.
        assert row["resid_cov_at_ct"] < 0
        assert abs(row["resid_cov_at_ct"] - (-0.08)) < 1e-12


class TestConfidenceScore:
    """Graded continuous confidence column + legacy boolean label."""

    def test_confidence_columns_present(self):
        rows = []
        for obj_idx in range(5):
            for i in range(20):
                rows.append(dict(
                    object_id=f"obj{obj_idx}", obs_id=f"{obj_idx}_{i}",
                    stn="S", residual_ra_arcsec=0.0, residual_dec_arcsec=0.0,
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=50, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        assert "confidence_score" in table.columns
        assert "high_confidence" in table.columns
        score = table.iloc[0]["confidence_score"]
        assert 0.0 <= score <= 1.0

    def test_high_confidence_station_scores_high(self):
        """Many objects, many obs, tight scatter → confidence_score >= 0.9."""
        rng = np.random.default_rng(11)
        rows = []
        for obj_idx in range(50):          # 50 objects → n_objects large
            obj_id = f"obj{obj_idx}"
            for i in range(40):            # 2000 obs total → size term → 1
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}_{i}", stn="HI",
                    residual_ra_arcsec=float(rng.normal(0.0, 0.1)),
                    residual_dec_arcsec=float(rng.normal(0.0, 0.1)),
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=500, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert row["confidence_score"] >= 0.9, row["confidence_score"]
        assert bool(row["high_confidence"]) is True

    def test_low_confidence_station_scores_low(self):
        """n_obs < 20 → confidence_score <= 0.1 and high_confidence False."""
        rng = np.random.default_rng(13)
        rows = []
        for obj_idx in range(3):           # 3 objects * 4 obs = 12 obs (< 20)
            obj_id = f"obj{obj_idx}"
            for i in range(4):
                rows.append(dict(
                    object_id=obj_id, obs_id=f"{obj_id}_{i}", stn="LO",
                    residual_ra_arcsec=float(rng.normal(0.0, 0.3)),
                    residual_dec_arcsec=float(rng.normal(0.0, 0.3)),
                    chi2=1.0,
                ))
        df = _make_looo_df(rows)
        cfg = BootstrapConfig(n_resamples=200, random_seed=42)
        table = compute_bias_table(
            df, min_obs_per_group=10, min_objects_per_group=3,
            max_hold_in_reduced_chi2=None, bootstrap=cfg,
        )
        row = table.iloc[0]
        assert row["confidence_score"] <= 0.1, row["confidence_score"]
        assert bool(row["high_confidence"]) is False
        # Critically, the low-confidence station is NOT dropped — graded
        # confidence is a label, not a filter.
        assert row["obs_code"] == "LO"


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
