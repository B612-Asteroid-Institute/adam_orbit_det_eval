"""
Per-observatory bias table with bootstrap confidence intervals.

This module produces the production bias catalogue for the MPC observatory
bias study.  It is designed so the same aggregation runs identically on the
development-scale 3,500-object LOOO output and on the full MPC-cloud output
(~200,000+ objects) without changes.

Design
------
The estimator is object-weighted: we first compute a per-object mean residual
at each (observatory, program_code) group, then average across objects with
equal weight.  This matches the convention already used in
`analysis.compute_observatory_stats(object_weighted=True)` and prevents
heavily-observed objects from dominating a station's statistic.

Confidence intervals are computed by a non-parametric bootstrap that resamples
*objects* with replacement, not observations.  Object-level resampling is the
correct unit for these statistics because:
  - individual observations from the same object are correlated (shared orbit
    error, same epoch);
  - the per-object mean is what enters the outer aggregation, so uncertainty
    in the outer aggregation is driven by the scatter of object means.

The bootstrap indices are shared across the RA / Dec / AT / CT dimensions so
the four CIs for a given group are drawn from a consistent set of
resamples.

Output schema
-------------
One row per (observatory) and one row per (observatory, program_code) with:

  obs_code, program_code       # program_code null for observatory-level rows
  n_objects, n_obs
  # Mean bias (point estimate = simple object-weighted mean)
  bias_ra_arcsec, bias_dec_arcsec, bias_at_arcsec, bias_ct_arcsec
  bias_ra_ci_low, bias_ra_ci_high      # 2.5 / 97.5 percentile bootstrap
  bias_dec_ci_low, bias_dec_ci_high
  bias_at_ci_low, bias_at_ci_high
  bias_ct_ci_low, bias_ct_ci_high
  # Median bias (point estimate = median of per-object means)
  bias_ra_median_arcsec, bias_dec_median_arcsec,
  bias_at_median_arcsec, bias_ct_median_arcsec
  bias_ra_median_ci_low, bias_ra_median_ci_high
  bias_dec_median_ci_low, bias_dec_median_ci_high
  bias_at_median_ci_low, bias_at_median_ci_high
  bias_ct_median_ci_low, bias_ct_median_ci_high
  # RMS scatter (point estimate + bootstrap CI)
  rms_ra_arcsec, rms_dec_arcsec, rms_at_arcsec, rms_ct_arcsec
  rms_ra_ci_low, rms_ra_ci_high
  rms_dec_ci_low, rms_dec_ci_high
  rms_at_ci_low, rms_at_ci_high
  rms_ct_ci_low, rms_ct_ci_high
  sem_ra_arcsec, sem_dec_arcsec, sem_at_arcsec, sem_ct_arcsec
  # Per-station 2×2 residual covariance — pooled per-observation, population
  # (ddof=0) sample covariance about the group mean.  The genuine residual
  # scatter (noise covariance) an OD weighting consumes; the off-diagonal
  # measures the RA/Dec (resp. AT/CT) error correlation directly.
  resid_var_ra, resid_var_dec, resid_cov_ra_dec, resid_cov_n
  resid_var_at, resid_var_ct, resid_cov_at_ct   # null when AT/CT absent
  chi2_per_obs
  # True when the 95% CI on mean bias_ra OR bias_dec excludes zero — i.e.
  # the station shows a statistically resolvable bias in at least one axis.
  bias_significant
  # Graded per-station confidence in [0, 1] (geometric mean of a sample-volume
  # ramp and a CI-tightness term) and the legacy-compatible boolean label
  # (confidence_score >= 0.5).  A publication-layer relaxation of the old hard
  # high-confidence cutoff — no rows are dropped.
  confidence_score, high_confidence
  sigma_model_source            # fraction of obs with empirical rmsra/rmsdec
  obs_epoch_start, obs_epoch_end  # MJD (TAI/UTC whatever LOOO used)

AT/CT columns are NaN when the input does not contain `residual_at_arcsec`
and `residual_ct_arcsec` (e.g. the 3,500-object real-data parquet and the
2026-05-10 MPC-scale parquet, which pre-date the AT/CT augmentation).

Provenance columns (`sigma_model_source`, `obs_epoch_*`) are NaN when the
caller does not provide the source observations DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

#: Residual column suffixes tracked in the output table, in a fixed order so
#: the CI columns pair up reliably with the bias columns.
_RESIDUAL_DIMS: Tuple[str, ...] = ("ra", "dec", "at", "ct")

#: Source column in the LOOO parquet for each output dimension.
_SOURCE_COL: Dict[str, str] = {
    "ra": "residual_ra_arcsec",
    "dec": "residual_dec_arcsec",
    "at": "residual_at_arcsec",
    "ct": "residual_ct_arcsec",
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for object-level bootstrap resampling."""

    n_resamples: int = 2000
    random_seed: int = 42
    ci_lower_pct: float = 2.5
    ci_upper_pct: float = 97.5
    #: Max (n_objects * n_resamples) cells to hold in memory at once.
    #: When exceeded, the bootstrap iterates in chunks.  ~20M cells ≈ 160 MB
    #: of float64 per residual dimension, which is comfortable on a laptop.
    max_cells_per_chunk: int = 20_000_000


def bootstrap_mean_ci(
    per_object_values: np.ndarray,
    config: BootstrapConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Compute a bootstrap 95% CI for the mean of a 1-D array.

    Resamples with replacement from the input array and returns the
    (lower, upper) percentiles of the resampled means.  NaN values are
    treated as missing: each resample computes a nan-mean so groups with
    some NaNs still yield a finite CI (this matters for AT/CT where some
    observations are near-stationary and projected residuals are NaN).

    Parameters
    ----------
    per_object_values : ndarray of float64, shape (n_objects,)
        The per-object means for one (group, dimension) combination.
    config : BootstrapConfig
    rng : numpy Generator

    Returns
    -------
    (ci_low, ci_high) : tuple of floats
        2.5 / 97.5 percentile of bootstrap means.  Both NaN if every input
        is NaN or the array is empty.
    """
    n = per_object_values.size
    if n == 0:
        return float("nan"), float("nan")
    if np.all(~np.isfinite(per_object_values)):
        return float("nan"), float("nan")

    cells_per_iter = n
    if cells_per_iter == 0:
        return float("nan"), float("nan")
    chunk_size = max(1, min(config.n_resamples,
                            config.max_cells_per_chunk // max(cells_per_iter, 1)))

    bootstrap_means = np.empty(config.n_resamples, dtype=np.float64)
    offset = 0
    while offset < config.n_resamples:
        this_chunk = min(chunk_size, config.n_resamples - offset)
        idx = rng.integers(0, n, size=(this_chunk, n))
        resamples = per_object_values[idx]  # (this_chunk, n)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(resamples, axis=1)
        bootstrap_means[offset:offset + this_chunk] = means
        offset += this_chunk

    finite = bootstrap_means[np.isfinite(bootstrap_means)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, config.ci_lower_pct))
    hi = float(np.percentile(finite, config.ci_upper_pct))
    return lo, hi


def bootstrap_stats_ci(
    per_object_values: np.ndarray,
    per_object_rms: Optional[np.ndarray],
    config: BootstrapConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap 95% CIs for mean, median, and (object-weighted) RMS
    using a single shared set of resample indices.

    Resamples objects with replacement. Each resample gives (mean, median,
    mean-of-per-object-rms) so the three CIs come from a consistent set
    of bootstrap draws.

    Parameters
    ----------
    per_object_values : ndarray of float64, shape (n_objects,)
        Per-object mean residuals (signed).
    per_object_rms : ndarray of float64, shape (n_objects,) or None
        Per-object RMS values (sqrt of per-object mean-square residual).
        When None, the RMS CI returned is (nan, nan).
    config : BootstrapConfig
    rng : numpy Generator

    Returns
    -------
    Dict with keys ``mean_lo``/``mean_hi``, ``median_lo``/``median_hi``,
    ``rms_lo``/``rms_hi``. Any key may be NaN when there is insufficient
    finite input data.
    """
    nan_pair = (float("nan"), float("nan"))
    out_keys = ("mean", "median", "rms")
    nan_out = {f"{k}_lo": float("nan") for k in out_keys}
    nan_out.update({f"{k}_hi": float("nan") for k in out_keys})

    n = per_object_values.size
    if n == 0 or np.all(~np.isfinite(per_object_values)):
        return nan_out

    has_rms = per_object_rms is not None and per_object_rms.size == n

    cells_per_iter = n
    chunk_size = max(1, min(
        config.n_resamples,
        config.max_cells_per_chunk // max(cells_per_iter, 1),
    ))

    mean_boot = np.empty(config.n_resamples, dtype=np.float64)
    median_boot = np.empty(config.n_resamples, dtype=np.float64)
    rms_boot = np.empty(config.n_resamples, dtype=np.float64) if has_rms else None

    offset = 0
    while offset < config.n_resamples:
        this_chunk = min(chunk_size, config.n_resamples - offset)
        idx = rng.integers(0, n, size=(this_chunk, n))
        resamples = per_object_values[idx]  # (this_chunk, n)
        with np.errstate(invalid="ignore"):
            mean_boot[offset:offset + this_chunk] = np.nanmean(resamples, axis=1)
            median_boot[offset:offset + this_chunk] = np.nanmedian(resamples, axis=1)
            if has_rms:
                rms_resamples = per_object_rms[idx]
                rms_boot[offset:offset + this_chunk] = np.nanmean(
                    rms_resamples, axis=1
                )
        offset += this_chunk

    def _pct(arr: np.ndarray) -> Tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return nan_pair
        return (
            float(np.percentile(finite, config.ci_lower_pct)),
            float(np.percentile(finite, config.ci_upper_pct)),
        )

    mean_lo, mean_hi = _pct(mean_boot)
    median_lo, median_hi = _pct(median_boot)
    rms_lo, rms_hi = _pct(rms_boot) if has_rms else nan_pair

    return {
        "mean_lo": mean_lo, "mean_hi": mean_hi,
        "median_lo": median_lo, "median_hi": median_hi,
        "rms_lo": rms_lo, "rms_hi": rms_hi,
    }


# ---------------------------------------------------------------------------
# Object-weighted aggregation
# ---------------------------------------------------------------------------


def _compute_per_object_means(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    dim_cols: Iterable[str],
) -> pd.DataFrame:
    """
    For each (group, object), compute the mean of every residual dimension
    and the mean of chi2 / squared residual.

    The caller feeds the result of this function into group-level
    aggregation — each row here is one (group, object) cell with the
    per-object mean values that will enter the outer aggregation.

    Parameters
    ----------
    df : DataFrame
        Filtered LOOO results, with at least `object_id` and the columns
        in `group_cols` and `dim_cols`.
    group_cols : iterable of str
        Group keys (e.g. ["stn"] or ["stn", "program_code"]).
    dim_cols : iterable of str
        Residual column names to aggregate (e.g. ["residual_ra_arcsec"]).

    Returns
    -------
    DataFrame indexed by the group key plus object_id.
    """
    keys = list(group_cols) + ["object_id"]
    # Keep dropna=False so rows with NaN program_code still group together.
    grouped = df.groupby(keys, dropna=False, sort=False)

    # Prepare squared-residual columns for per-object mean-square (→ RMS)
    for col in dim_cols:
        if col in df.columns:
            df[f"_sq_{col}"] = df[col] ** 2

    primary_specs = {
        "n_obs": ("object_id", "size"),
    }
    for col in dim_cols:
        if col in df.columns:
            primary_specs[f"{col}_mean"] = (col, "mean")
            primary_specs[f"{col}_sq_mean"] = (f"_sq_{col}", "mean")
    if "chi2" in df.columns:
        primary_specs["chi2_mean"] = ("chi2", "mean")

    per_obj = grouped.agg(**primary_specs).reset_index()

    # Per-object RMS = sqrt(mean(residual^2)) — matches analysis.py convention
    # where outer aggregation takes mean-of-per-object-RMS rather than
    # sqrt-of-mean-of-per-object-MS.  The two differ when per-object scatter
    # is heterogeneous; we match the baseline for exact reproducibility.
    for col in dim_cols:
        if f"{col}_sq_mean" in per_obj.columns:
            per_obj[f"{col}_rms_obj"] = np.sqrt(per_obj[f"{col}_sq_mean"])

    # Clean up the temp squared columns on the input frame
    for col in dim_cols:
        sq_col = f"_sq_{col}"
        if sq_col in df.columns:
            del df[sq_col]

    return per_obj


def _nanmean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        v = np.nanmean(arr)
    return float(v) if np.isfinite(v) else float("nan")


def _nansqrt_mean(arr: np.ndarray) -> float:
    """Object-weighted RMS: mean of per-object mean-square, then sqrt."""
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        v = np.sqrt(np.nanmean(arr))
    return float(v) if np.isfinite(v) else float("nan")


def _nanmedian(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        v = np.nanmedian(arr)
    return float(v) if np.isfinite(v) else float("nan")


def _nanstd_sem(arr: np.ndarray) -> float:
    """Standard error of the mean from the per-object scatter."""
    finite = arr[np.isfinite(arr)]
    n = finite.size
    if n < 2:
        return float("nan")
    return float(np.std(finite, ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Observation provenance helpers
# ---------------------------------------------------------------------------


def compute_provenance(
    looo_df: pd.DataFrame,
    observations_df: Optional[pd.DataFrame],
    group_cols: Iterable[str],
) -> Optional[pd.DataFrame]:
    """
    Compute per-group `sigma_model_source` and `obs_epoch_start/end`.

    `sigma_model_source` is the fraction of held-out observations (after
    filtering) whose reported `rmsra` is non-null — i.e. the empirical MPC
    sigma was used instead of a Veres 2017 fill-in.

    Parameters
    ----------
    looo_df : DataFrame
        Filtered LOOO results.  Must contain `obs_id` and the group columns.
    observations_df : DataFrame or None
        Source observations with at least `obsid`, `rmsra`, `obstime_mjd`.
        When None, returns None and the caller should leave provenance
        columns NaN.
    group_cols : iterable of str
        Group keys.

    Returns
    -------
    DataFrame with columns group_cols + [sigma_model_source, obs_epoch_start,
    obs_epoch_end], or None if `observations_df` is None.
    """
    if observations_df is None:
        return None

    needed = {"obsid"}
    missing = needed - set(observations_df.columns)
    if missing:
        logger.warning(
            "observations_df missing columns %s — skipping provenance join",
            sorted(missing),
        )
        return None

    # Join LOOO rows to the source observations by obs_id == obsid
    obs_slice_cols = ["obsid"]
    if "rmsra" in observations_df.columns:
        obs_slice_cols.append("rmsra")
    if "obstime_mjd" in observations_df.columns:
        obs_slice_cols.append("obstime_mjd")

    merged = looo_df.merge(
        observations_df[obs_slice_cols].rename(columns={"obsid": "obs_id"}),
        on="obs_id",
        how="left",
    )

    group_keys = list(group_cols)
    grp = merged.groupby(group_keys, dropna=False, sort=False)

    pieces: Dict[str, pd.Series] = {}
    if "rmsra" in merged.columns:
        pieces["sigma_model_source"] = grp["rmsra"].apply(
            lambda s: float(s.notna().mean()) if len(s) > 0 else float("nan")
        )
    if "obstime_mjd" in merged.columns:
        pieces["obs_epoch_start"] = grp["obstime_mjd"].min()
        pieces["obs_epoch_end"] = grp["obstime_mjd"].max()

    if not pieces:
        return None

    out = pd.concat(pieces, axis=1).reset_index()
    return out


# ---------------------------------------------------------------------------
# Per-station residual covariance (RA/Dec and AT/CT frames)
# ---------------------------------------------------------------------------


def _pairwise_cov(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, int]:
    """
    Population (ddof=0) 2-variable sample covariance over finite (x, y) pairs.

    Returns ``(var_x, var_y, cov_xy, n_pairs)`` where the variances and the
    covariance are taken about the sample means of the finite pairs.  Rows in
    which either coordinate is non-finite are dropped (so AT/CT, where some
    near-stationary observations have NaN projected residuals, still yields a
    finite covariance).

    The population (divide-by-N) estimator is used rather than the unbiased
    (divide-by-N-1) one for two reasons:
      * it is the maximum-likelihood noise covariance an OD weighting consumes;
      * under the homogeneous, zero-mean sampling the cross-check fixtures are
        built in, the diagonal then equals the object-weighted RMS² exactly
        (see ``test_resid_var_matches_rms_squared``).
    """
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(finite.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    xf = x[finite]
    yf = y[finite]
    dx = xf - xf.mean()
    dy = yf - yf.mean()
    var_x = float(np.mean(dx * dx))
    var_y = float(np.mean(dy * dy))
    cov_xy = float(np.mean(dx * dy))
    return var_x, var_y, cov_xy, n


def compute_residual_covariance(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    dims_available: Iterable[str],
) -> pd.DataFrame:
    """
    Per-group 2×2 residual covariance in the RA/Dec frame (and the AT/CT frame
    when those residuals are present).

    Unlike the bias / RMS estimators in this module — which are *object*-
    weighted (per-object mean first, then equal-weight across objects) — the
    residual covariance is a *pooled per-observation* statistic: every held-out
    residual in the group contributes one (RA, Dec) point and the 2×2 matrix is
    the population (ddof=0) sample covariance over those points, taken about the
    group mean.  This is the genuine scatter of the residuals — the noise
    covariance an OD weighting needs — with the off-diagonal measuring the
    RA/Dec error correlation directly (near zero when the two axes are
    independent, signed and non-zero when they covary).

    Relationship to the published RMS columns
    ------------------------------------------
    ``resid_var_ra`` equals ``rms_ra_arcsec²`` *exactly* only in the zero-bias,
    homogeneous-sampling limit (equal observations per object, identical
    per-object scatter).  On real biased data the two differ for two reasons:
    this estimator is central (about the mean) whereas RMS is about zero, and
    this estimator is observation-pooled whereas RMS is object-weighted.  The
    cross-check fixtures are constructed in that limit so the equality is exact
    there (asserted in the tests, not at runtime).

    Parameters
    ----------
    df : DataFrame
        Filtered LOOO results with the residual source columns.
    group_cols : iterable of str
        Group keys (e.g. ``["stn"]`` or ``["stn", "program_code"]``).
    dims_available : iterable of str
        Residual dimensions present in ``df`` (subset of ``_RESIDUAL_DIMS``).

    Returns
    -------
    DataFrame keyed by ``group_cols`` with columns ``resid_var_ra``,
    ``resid_var_dec``, ``resid_cov_ra_dec``, ``resid_cov_n`` and — only when
    both AT and CT residuals are present — ``resid_var_at``, ``resid_var_ct``,
    ``resid_cov_at_ct``.  ``resid_cov_n`` is the number of finite (RA, Dec)
    pairs and equals the group's ``n_obs`` except where some residuals are NaN.
    """
    group_cols = list(group_cols)
    dims_available = list(dims_available)
    has_radec = "ra" in dims_available and "dec" in dims_available
    has_atct = "at" in dims_available and "ct" in dims_available

    out_rows = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, object] = dict(zip(group_cols, key_tuple))

        if has_radec:
            ra = sub[_SOURCE_COL["ra"]].to_numpy(dtype=np.float64)
            dec = sub[_SOURCE_COL["dec"]].to_numpy(dtype=np.float64)
            var_ra, var_dec, cov_rd, n_rd = _pairwise_cov(ra, dec)
        else:
            var_ra = var_dec = cov_rd = float("nan")
            n_rd = 0
        row["resid_var_ra"] = var_ra
        row["resid_var_dec"] = var_dec
        row["resid_cov_ra_dec"] = cov_rd
        row["resid_cov_n"] = n_rd

        if has_atct:
            at = sub[_SOURCE_COL["at"]].to_numpy(dtype=np.float64)
            ct = sub[_SOURCE_COL["ct"]].to_numpy(dtype=np.float64)
            var_at, var_ct, cov_ac, _ = _pairwise_cov(at, ct)
            row["resid_var_at"] = var_at
            row["resid_var_ct"] = var_ct
            row["resid_cov_at_ct"] = cov_ac

        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=group_cols)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Bias table construction
# ---------------------------------------------------------------------------


def _aggregate_one_group_key(
    per_obj: pd.DataFrame,
    group_cols: Iterable[str],
    dims: Iterable[str],
    bootstrap: BootstrapConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Collapse a per-(group, object) table to per-group rows with bootstrap CIs.
    """
    group_cols = list(group_cols)
    dims = list(dims)

    grouped = per_obj.groupby(group_cols, dropna=False, sort=False)

    rows = []
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, object] = dict(zip(group_cols, key_tuple))

        n_objects = len(sub)
        n_obs = int(sub["n_obs"].sum())
        row["n_objects"] = int(n_objects)
        row["n_obs"] = n_obs

        chi2_mean = (
            _nanmean(sub["chi2_mean"].to_numpy(dtype=np.float64))
            if "chi2_mean" in sub.columns
            else float("nan")
        )
        row["chi2_per_obs"] = chi2_mean

        # For each residual dimension: bias (mean + median), RMS, SEM,
        # bootstrap CIs for mean, median, and RMS.
        for dim in dims:
            src = _SOURCE_COL[dim]
            mean_col = f"{src}_mean"
            rms_obj_col = f"{src}_rms_obj"
            if mean_col not in sub.columns:
                row[f"bias_{dim}_arcsec"] = float("nan")
                row[f"bias_{dim}_ci_low"] = float("nan")
                row[f"bias_{dim}_ci_high"] = float("nan")
                row[f"bias_{dim}_median_arcsec"] = float("nan")
                row[f"bias_{dim}_median_ci_low"] = float("nan")
                row[f"bias_{dim}_median_ci_high"] = float("nan")
                row[f"rms_{dim}_arcsec"] = float("nan")
                row[f"rms_{dim}_ci_low"] = float("nan")
                row[f"rms_{dim}_ci_high"] = float("nan")
                row[f"sem_{dim}_arcsec"] = float("nan")
                continue

            vals = sub[mean_col].to_numpy(dtype=np.float64)
            per_obj_rms = (
                sub[rms_obj_col].to_numpy(dtype=np.float64)
                if rms_obj_col in sub.columns
                else None
            )

            row[f"bias_{dim}_arcsec"] = _nanmean(vals)
            row[f"bias_{dim}_median_arcsec"] = _nanmedian(vals)
            # Mean-of-per-object-RMS matches analysis.compute_observatory_stats
            # when object_weighted=True.  Reproduces the existing baseline
            # observatory_stats.parquet for exact cross-validation.
            row[f"rms_{dim}_arcsec"] = (
                _nanmean(per_obj_rms)
                if per_obj_rms is not None
                else float("nan")
            )
            row[f"sem_{dim}_arcsec"] = _nanstd_sem(vals)

            cis = bootstrap_stats_ci(vals, per_obj_rms, bootstrap, rng)
            row[f"bias_{dim}_ci_low"] = cis["mean_lo"]
            row[f"bias_{dim}_ci_high"] = cis["mean_hi"]
            row[f"bias_{dim}_median_ci_low"] = cis["median_lo"]
            row[f"bias_{dim}_median_ci_high"] = cis["median_hi"]
            row[f"rms_{dim}_ci_low"] = cis["rms_lo"]
            row[f"rms_{dim}_ci_high"] = cis["rms_hi"]

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=group_cols)
    return pd.DataFrame(rows)


def compute_bias_table(
    looo_df: pd.DataFrame,
    *,
    observations_df: Optional[pd.DataFrame] = None,
    min_obs_per_group: int = 10,
    min_objects_per_group: int = 3,
    max_hold_in_reduced_chi2: Optional[float] = 100.0,
    max_object_mean_chi2: Optional[float] = None,
    bootstrap: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """
    Compute the per-observatory bias table with bootstrap CIs.

    Produces one row per observatory (with program_code null) and one row per
    (observatory, program_code) where program_code is non-null.  Groups below
    `min_obs_per_group` or `min_objects_per_group` are dropped.

    Parameters
    ----------
    looo_df : DataFrame
        LOOO results with at least: object_id, obs_id, stn,
        residual_ra_arcsec, residual_dec_arcsec, chi2.  Optional:
        program_code, residual_at_arcsec, residual_ct_arcsec,
        hold_in_reduced_chi2.
    observations_df : DataFrame, optional
        Source observations DataFrame used to compute `sigma_model_source`
        and `obs_epoch_start/end`.  Must have columns obsid, and optionally
        rmsra and obstime_mjd.  When None, provenance columns are NaN.
    min_obs_per_group : int
        Groups with fewer held-out observations are dropped.
    min_objects_per_group : int
        Groups with fewer distinct objects are dropped (bootstrap CIs are
        meaningless with too few objects).
    max_hold_in_reduced_chi2 : float, optional
        If the LOOO DataFrame has `hold_in_reduced_chi2`, rows whose
        reduced chi2 exceeds this threshold are dropped before aggregation.
        Set to None to skip the filter.
    max_object_mean_chi2 : float, optional
        Drop entire objects whose mean held-out chi2 across ALL stations
        exceeds this threshold.  These are likely non-gravitational-force
        targets (comets, high-Yarkovsky NEOs) whose orbit fits are biased
        at the object level; leaving them in inflates RMS for every station
        that observed them.  Matches the existing `analysis.py` convention.
        Set to None to skip.
    bootstrap : BootstrapConfig, optional
        Bootstrap parameters.  Default is 2000 resamples with seed=42.

    Returns
    -------
    DataFrame with schema documented in this module's docstring.  Rows are
    sorted by `obs_code`, then by `program_code` (nulls last).
    """
    if bootstrap is None:
        bootstrap = BootstrapConfig()

    if len(looo_df) == 0:
        logger.warning("Empty LOOO input — returning empty bias table.")
        return _empty_bias_table()

    df = looo_df.copy()

    # Optional fit-quality filter
    if max_hold_in_reduced_chi2 is not None and "hold_in_reduced_chi2" in df.columns:
        before = len(df)
        df = df[
            df["hold_in_reduced_chi2"].isna()
            | (df["hold_in_reduced_chi2"] <= max_hold_in_reduced_chi2)
        ].copy()
        logger.info(
            "Filter hold_in_reduced_chi2<=%s: %d → %d rows",
            max_hold_in_reduced_chi2, before, len(df),
        )

    # Object-level filter: drop objects with inflated chi2 across all stations
    # (non-gravitational-force contamination)
    n_objects_filtered = 0
    if max_object_mean_chi2 is not None and "chi2" in df.columns:
        obj_mean_chi2 = df.groupby("object_id", dropna=False)["chi2"].mean()
        bad_objects = set(obj_mean_chi2[obj_mean_chi2 > max_object_mean_chi2].index)
        if bad_objects:
            before = len(df)
            df = df[~df["object_id"].isin(bad_objects)].copy()
            n_objects_filtered = len(bad_objects)
            logger.info(
                "Filter max_object_mean_chi2<=%s: dropped %d objects, %d → %d rows",
                max_object_mean_chi2, n_objects_filtered, before, len(df),
            )

    # Determine which residual dimensions are available
    dims_available = [d for d in _RESIDUAL_DIMS if _SOURCE_COL[d] in df.columns]
    dim_cols = [_SOURCE_COL[d] for d in dims_available]
    logger.info("Residual dimensions available: %s", dims_available)

    # Normalize program_code: if absent, create a null column so downstream
    # logic has a uniform shape.
    if "program_code" not in df.columns:
        df["program_code"] = pd.NA

    # -----------------------------------------------------------------
    # Pass 1: per-observatory rows
    # -----------------------------------------------------------------
    per_obj_obs = _compute_per_object_means(df, ["stn"], dim_cols)
    rng_obs = np.random.default_rng(bootstrap.random_seed)
    obs_rows = _aggregate_one_group_key(
        per_obj_obs,
        group_cols=["stn"],
        dims=dims_available,
        bootstrap=bootstrap,
        rng=rng_obs,
    )
    obs_rows["program_code"] = pd.NA

    # -----------------------------------------------------------------
    # Pass 2: per-(stn, program_code) rows, only for non-null program_code
    # -----------------------------------------------------------------
    prog_df = df[df["program_code"].notna()].copy()
    if len(prog_df) > 0:
        # Pre-filter (stn, program_code) groups that cannot pass the
        # final min_obs_per_group / min_objects_per_group threshold.
        # On MPC-scale input there are ~1.1M (stn, program_code) combos
        # but only ~300 pass both filters; computing per-object stats and
        # bootstrap CIs for the rest wastes ~15 minutes of wall time.
        prog_sizes = (
            prog_df.groupby(["stn", "program_code"], dropna=False, sort=False)
            .agg(_n_obj=("object_id", "nunique"), _n_obs=("object_id", "size"))
            .reset_index()
        )
        keep = prog_sizes[
            (prog_sizes["_n_obj"] >= min_objects_per_group)
            & (prog_sizes["_n_obs"] >= min_obs_per_group)
        ][["stn", "program_code"]]
        before = len(prog_df)
        prog_df = prog_df.merge(keep, on=["stn", "program_code"], how="inner")
        logger.info(
            "Pre-filter (stn, program_code) groups: %d / %d kept, %d → %d rows",
            len(keep), len(prog_sizes), before, len(prog_df),
        )
    if len(prog_df) > 0:
        per_obj_prog = _compute_per_object_means(
            prog_df, ["stn", "program_code"], dim_cols,
        )
        # Use an offset seed so program-code bootstrap noise is independent
        # of the observatory-level bootstrap.
        rng_prog = np.random.default_rng(bootstrap.random_seed + 1)
        prog_rows = _aggregate_one_group_key(
            per_obj_prog,
            group_cols=["stn", "program_code"],
            dims=dims_available,
            bootstrap=bootstrap,
            rng=rng_prog,
        )
    else:
        prog_rows = pd.DataFrame(columns=["stn", "program_code"])

    # -----------------------------------------------------------------
    # Combine + filter small groups + provenance + final schema
    # -----------------------------------------------------------------
    table = pd.concat([obs_rows, prog_rows], ignore_index=True, sort=False)

    # Drop small groups
    before = len(table)
    table = table[
        (table["n_obs"] >= min_obs_per_group)
        & (table["n_objects"] >= min_objects_per_group)
    ].reset_index(drop=True)
    logger.info(
        "Min-size filter (n_obs>=%d, n_objects>=%d): %d → %d groups",
        min_obs_per_group, min_objects_per_group, before, len(table),
    )

    # Provenance join — need two passes (observatory-level and program-level)
    prov_obs = compute_provenance(df, observations_df, ["stn"])
    prov_prog = (
        compute_provenance(df[df["program_code"].notna()], observations_df,
                           ["stn", "program_code"])
        if len(prog_df) > 0
        else None
    )

    for col in ("sigma_model_source", "obs_epoch_start", "obs_epoch_end"):
        if col not in table.columns:
            table[col] = np.nan

    if prov_obs is not None:
        _fill_provenance(table, prov_obs, ["stn"], program_code_null=True)
    if prov_prog is not None:
        _fill_provenance(table, prov_prog, ["stn", "program_code"],
                         program_code_null=False)

    # -----------------------------------------------------------------
    # Per-station 2×2 residual covariance (RA/Dec, and AT/CT when present)
    # -----------------------------------------------------------------
    has_atct = "at" in dims_available and "ct" in dims_available
    if not has_atct:
        logger.warning(
            "AT/CT residuals absent — resid_var_at/resid_var_ct/"
            "resid_cov_at_ct left null."
        )
    radec_cov_cols = (
        "resid_var_ra", "resid_var_dec", "resid_cov_ra_dec", "resid_cov_n",
    )
    atct_cov_cols = ("resid_var_at", "resid_var_ct", "resid_cov_at_ct")
    cov_cols = radec_cov_cols + atct_cov_cols
    for col in cov_cols:
        if col not in table.columns:
            table[col] = np.nan

    cov_obs = compute_residual_covariance(df, ["stn"], dims_available)
    _fill_group_columns(table, cov_obs, ["stn"], cov_cols,
                        program_code_null=True)
    if len(prog_df) > 0:
        cov_prog = compute_residual_covariance(
            prog_df, ["stn", "program_code"], dims_available,
        )
        _fill_group_columns(table, cov_prog, ["stn", "program_code"], cov_cols,
                            program_code_null=False)

    # Rename stn → obs_code for the final catalogue schema
    table = table.rename(columns={"stn": "obs_code"})

    # bias_significant: True when the 95% CI on the mean bias_ra OR
    # bias_dec excludes zero. False when both CIs straddle zero (i.e. no
    # statistically resolvable bias in either axis). NaN-aware: when a CI
    # bound is NaN that side is treated as not-excluding-zero so the flag
    # never spuriously fires on missing data.
    if "bias_ra_ci_low" in table.columns and "bias_ra_ci_high" in table.columns:
        ra_lo = table["bias_ra_ci_low"]
        ra_hi = table["bias_ra_ci_high"]
        ra_excl = (ra_lo > 0) | (ra_hi < 0)
    else:
        ra_excl = pd.Series(False, index=table.index)
    if "bias_dec_ci_low" in table.columns and "bias_dec_ci_high" in table.columns:
        dec_lo = table["bias_dec_ci_low"]
        dec_hi = table["bias_dec_ci_high"]
        dec_excl = (dec_lo > 0) | (dec_hi < 0)
    else:
        dec_excl = pd.Series(False, index=table.index)
    table["bias_significant"] = (ra_excl | dec_excl).fillna(False).astype(bool)

    # -----------------------------------------------------------------
    # Graded per-station confidence (continuous, in [0, 1])
    # -----------------------------------------------------------------
    # Replaces the legacy hard high-confidence cutoff (n_obs >= 100 AND
    # n_objects >= 20), which dropped exactly the long-tail stations that
    # observe short-arc NEOs.  No rows are dropped here — this is a
    # publication-layer label, not a filter.
    #
    # The score is the geometric mean of two ingredients, each in [0, 1], so
    # the product stays in [0, 1] and a weakness in either term pulls it down:
    #
    #   size  — sample volume, a log ramp (smoothstep) between N_CONF_LOW
    #           (=20 obs, the low-confidence floor: at/below it size→0) and
    #           N_CONF_FULL (=500 obs: at/above it size→1).  n_obs is the
    #           natural volume axis and equals resid_cov_n.
    #   prec  — precision of the mean-bias estimate: the RA/Dec residual scale
    #           relative to the mean-bias bootstrap CI half-width.  A CI tight
    #           relative to the scatter → ~1; a CI as wide as the scatter
    #           → ~0.5; wider → →0.  This is where the object count enters: the
    #           object-level bootstrap CI widens when a station is carried by
    #           few objects, so a many-obs / few-object station is correctly
    #           penalised even though its n_obs is large.
    #
    # high_confidence = confidence_score >= HIGH_CONF_THRESHOLD (0.5), chosen
    # so a station near the legacy cutoff lands on the True side — rough
    # backward compatibility with the dropped boolean filter.
    N_CONF_LOW = 20.0
    N_CONF_FULL = 500.0
    PREC_FLOOR = 0.05  # arcsec; floors the scatter scale so a near-zero-RMS
    #                    station's precision term stays well-defined.
    HIGH_CONF_THRESHOLD = 0.5

    n_obs_arr = table["n_obs"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_size = (
            np.log10(np.maximum(n_obs_arr, 1.0)) - np.log10(N_CONF_LOW)
        ) / (np.log10(N_CONF_FULL) - np.log10(N_CONF_LOW))
    t_size = np.clip(t_size, 0.0, 1.0)
    size = t_size * t_size * (3.0 - 2.0 * t_size)  # smoothstep

    def _col(name: str) -> np.ndarray:
        if name in table.columns:
            return table[name].to_numpy(dtype=np.float64)
        return np.full(len(table), np.nan)

    hw_ra = (_col("bias_ra_ci_high") - _col("bias_ra_ci_low")) / 2.0
    hw_dec = (_col("bias_dec_ci_high") - _col("bias_dec_ci_low")) / 2.0
    with np.errstate(invalid="ignore"):
        hw = np.nanmean(np.vstack([hw_ra, hw_dec]), axis=0)
        scale = np.nanmean(
            np.vstack([_col("rms_ra_arcsec"), _col("rms_dec_arcsec")]), axis=0
        ) + PREC_FLOOR
        prec = scale / (scale + hw)
    # When the CI is unavailable, fall back to the size term alone.
    prec = np.where(np.isfinite(prec), prec, 1.0)

    confidence = np.sqrt(
        np.clip(size, 0.0, 1.0) * np.clip(prec, 0.0, 1.0)
    )
    table["confidence_score"] = confidence
    table["high_confidence"] = confidence >= HIGH_CONF_THRESHOLD

    # Column ordering
    ordered = [
        "obs_code", "program_code", "n_objects", "n_obs",
        "bias_ra_arcsec", "bias_ra_ci_low", "bias_ra_ci_high",
        "bias_dec_arcsec", "bias_dec_ci_low", "bias_dec_ci_high",
        "bias_at_arcsec", "bias_at_ci_low", "bias_at_ci_high",
        "bias_ct_arcsec", "bias_ct_ci_low", "bias_ct_ci_high",
        "bias_ra_median_arcsec",
        "bias_ra_median_ci_low", "bias_ra_median_ci_high",
        "bias_dec_median_arcsec",
        "bias_dec_median_ci_low", "bias_dec_median_ci_high",
        "bias_at_median_arcsec",
        "bias_at_median_ci_low", "bias_at_median_ci_high",
        "bias_ct_median_arcsec",
        "bias_ct_median_ci_low", "bias_ct_median_ci_high",
        "rms_ra_arcsec", "rms_ra_ci_low", "rms_ra_ci_high",
        "rms_dec_arcsec", "rms_dec_ci_low", "rms_dec_ci_high",
        "rms_at_arcsec", "rms_at_ci_low", "rms_at_ci_high",
        "rms_ct_arcsec", "rms_ct_ci_low", "rms_ct_ci_high",
        "sem_ra_arcsec", "sem_dec_arcsec", "sem_at_arcsec", "sem_ct_arcsec",
        # Per-station 2×2 residual covariance (pooled per-obs, ddof=0)
        "resid_var_ra", "resid_var_dec", "resid_cov_ra_dec", "resid_cov_n",
        "resid_var_at", "resid_var_ct", "resid_cov_at_ct",
        "chi2_per_obs",
        "bias_significant",
        "confidence_score", "high_confidence",
        "sigma_model_source", "obs_epoch_start", "obs_epoch_end",
    ]
    # Ensure every expected column exists, even when AT/CT is unavailable
    for col in ordered:
        if col not in table.columns:
            table[col] = np.nan
    table = table[ordered]

    # Sort: obs_code ascending, then program_code with nulls first (the
    # observatory-level aggregate comes before its per-program rows).
    table["_prog_sort"] = table["program_code"].isna().map({True: 0, False: 1})
    table = table.sort_values(
        ["obs_code", "_prog_sort", "program_code"],
        na_position="first",
    ).drop(columns=["_prog_sort"]).reset_index(drop=True)

    return table


def _fill_provenance(
    table: pd.DataFrame,
    prov: pd.DataFrame,
    key_cols: Iterable[str],
    program_code_null: bool,
) -> None:
    """
    In-place: copy provenance columns from `prov` into `table` for rows
    whose keys match and whose program_code is/isn't null.
    """
    key_cols = list(key_cols)
    prov_indexed = prov.set_index(key_cols)

    if program_code_null:
        mask = table["program_code"].isna()
    else:
        mask = table["program_code"].notna()

    for i in table.index[mask]:
        key = tuple(table.loc[i, k] for k in key_cols)
        if len(key) == 1:
            key = key[0]
        if key in prov_indexed.index:
            src = prov_indexed.loc[key]
            for col in ("sigma_model_source", "obs_epoch_start", "obs_epoch_end"):
                if col in prov.columns:
                    val = src[col]
                    # `src` may itself be a DataFrame if non-unique keys —
                    # in that case take the first row.
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    table.at[i, col] = val


def _fill_group_columns(
    table: pd.DataFrame,
    src: pd.DataFrame,
    key_cols: Iterable[str],
    value_cols: Iterable[str],
    program_code_null: bool,
) -> None:
    """
    In-place: copy ``value_cols`` from ``src`` into ``table`` for rows whose
    keys match and whose program_code is/isn't null.

    A generic version of :func:`_fill_provenance` used to graft the residual
    covariance columns onto the assembled table.  Only columns present in
    ``src`` are copied (so AT/CT covariance is skipped when absent).
    """
    key_cols = list(key_cols)
    if src is None or len(src) == 0:
        return
    present = [c for c in value_cols if c in src.columns]
    if not present:
        return
    src_indexed = src.set_index(key_cols)

    mask = (
        table["program_code"].isna()
        if program_code_null
        else table["program_code"].notna()
    )

    for i in table.index[mask]:
        key = tuple(table.loc[i, k] for k in key_cols)
        if len(key) == 1:
            key = key[0]
        if key in src_indexed.index:
            row = src_indexed.loc[key]
            for col in present:
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                table.at[i, col] = val


def _empty_bias_table() -> pd.DataFrame:
    cols = [
        "obs_code", "program_code", "n_objects", "n_obs",
        "bias_ra_arcsec", "bias_ra_ci_low", "bias_ra_ci_high",
        "bias_dec_arcsec", "bias_dec_ci_low", "bias_dec_ci_high",
        "bias_at_arcsec", "bias_at_ci_low", "bias_at_ci_high",
        "bias_ct_arcsec", "bias_ct_ci_low", "bias_ct_ci_high",
        "bias_ra_median_arcsec",
        "bias_ra_median_ci_low", "bias_ra_median_ci_high",
        "bias_dec_median_arcsec",
        "bias_dec_median_ci_low", "bias_dec_median_ci_high",
        "bias_at_median_arcsec",
        "bias_at_median_ci_low", "bias_at_median_ci_high",
        "bias_ct_median_arcsec",
        "bias_ct_median_ci_low", "bias_ct_median_ci_high",
        "rms_ra_arcsec", "rms_ra_ci_low", "rms_ra_ci_high",
        "rms_dec_arcsec", "rms_dec_ci_low", "rms_dec_ci_high",
        "rms_at_arcsec", "rms_at_ci_low", "rms_at_ci_high",
        "rms_ct_arcsec", "rms_ct_ci_low", "rms_ct_ci_high",
        "sem_ra_arcsec", "sem_dec_arcsec", "sem_at_arcsec", "sem_ct_arcsec",
        "resid_var_ra", "resid_var_dec", "resid_cov_ra_dec", "resid_cov_n",
        "resid_var_at", "resid_var_ct", "resid_cov_at_ct",
        "chi2_per_obs",
        "bias_significant",
        "confidence_score", "high_confidence",
        "sigma_model_source", "obs_epoch_start", "obs_epoch_end",
    ]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@dataclass
class AnchorStation:
    """Reference entry for validating the bias table against published values.

    A "well-calibrated" station is defined as |bias| below a small threshold
    rather than "CI spans zero": at the 3,500-object scale many stations have
    SEMs below 0.01" so even small biases are statistically resolvable from
    zero — that's scientifically correct, not a validation failure.
    """

    obs_code: str
    expected_bias_ra_arcsec: Optional[float] = None
    expected_bias_dec_arcsec: Optional[float] = None
    #: If True, the CI must exclude zero for both RA and Dec.
    ci_excludes_zero: bool = False
    #: If True, this station is claimed well-calibrated:  |bias_ra| and
    #: |bias_dec| must both fall under `well_calibrated_max_bias_arcsec`.
    well_calibrated: bool = False
    well_calibrated_max_bias_arcsec: float = 0.10
    tolerance_arcsec: float = 0.05


def validate_against_anchors(
    bias_table: pd.DataFrame,
    anchors: Iterable[AnchorStation],
) -> pd.DataFrame:
    """
    Compare the observatory-level rows of a bias table against a set of
    `AnchorStation` expectations.  Used to sanity-check the 3,500-object
    output against the published values in runs/run_3500obj/README.md.

    Returns a DataFrame with one row per anchor, indicating pass/fail for
    bias magnitude and CI behaviour.
    """
    obs_only = bias_table[bias_table["program_code"].isna()].set_index("obs_code")

    rows = []
    for anchor in anchors:
        if anchor.obs_code not in obs_only.index:
            rows.append({
                "obs_code": anchor.obs_code,
                "present": False,
                "bias_ra_ok": False,
                "bias_dec_ok": False,
                "ci_ra_ok": False,
                "ci_dec_ok": False,
            })
            continue

        row = obs_only.loc[anchor.obs_code]
        bias_ra = float(row["bias_ra_arcsec"])
        bias_dec = float(row["bias_dec_arcsec"])
        ci_ra = (float(row["bias_ra_ci_low"]), float(row["bias_ra_ci_high"]))
        ci_dec = (float(row["bias_dec_ci_low"]), float(row["bias_dec_ci_high"]))

        bias_ra_ok = (
            anchor.expected_bias_ra_arcsec is None
            or abs(bias_ra - anchor.expected_bias_ra_arcsec) <= anchor.tolerance_arcsec
        )
        bias_dec_ok = (
            anchor.expected_bias_dec_arcsec is None
            or abs(bias_dec - anchor.expected_bias_dec_arcsec) <= anchor.tolerance_arcsec
        )

        ci_ra_excludes_zero = ci_ra[0] > 0 or ci_ra[1] < 0
        ci_dec_excludes_zero = ci_dec[0] > 0 or ci_dec[1] < 0

        if anchor.ci_excludes_zero:
            ci_ra_ok = ci_ra_excludes_zero
            ci_dec_ok = ci_dec_excludes_zero
        elif anchor.well_calibrated:
            # Well-calibrated means |bias| small, not necessarily CI-spans-zero.
            # At N=3500 objects many SEMs are <0.01" so tiny real biases are
            # resolvable from zero — that's science, not failure.
            ci_ra_ok = abs(bias_ra) < anchor.well_calibrated_max_bias_arcsec
            ci_dec_ok = abs(bias_dec) < anchor.well_calibrated_max_bias_arcsec
        else:
            ci_ra_ok = True
            ci_dec_ok = True

        rows.append({
            "obs_code": anchor.obs_code,
            "present": True,
            "bias_ra_arcsec": bias_ra,
            "bias_dec_arcsec": bias_dec,
            "ci_ra": ci_ra,
            "ci_dec": ci_dec,
            "bias_ra_ok": bias_ra_ok,
            "bias_dec_ok": bias_dec_ok,
            "ci_ra_ok": ci_ra_ok,
            "ci_dec_ok": ci_dec_ok,
        })

    return pd.DataFrame(rows)
