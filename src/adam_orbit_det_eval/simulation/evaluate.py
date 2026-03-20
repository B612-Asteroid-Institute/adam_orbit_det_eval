"""
Recovery evaluation for simulation validation.

Compares the biases recovered by the LOOO pipeline against the known injected
truth biases to measure detection power and recovery accuracy.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StationRecovery:
    """
    Per-station recovery result comparing injected vs recovered bias.

    Attributes
    ----------
    fake_code : str
        Synthetic station code.
    real_code : str
        Corresponding real MPC station code.
    bias_type : str
        Name(s) of injected bias class(es).
    bias_params : dict
        Serialised bias parameters.
    injected_ra_arcsec : float
        Expected mean RA offset the LOOO should recover (arcsec).
    injected_dec_arcsec : float
        Expected mean Dec offset the LOOO should recover (arcsec).
    recovered_mean_ra_arcsec : float
        Measured mean RA residual from LOOO observatory_stats.
    recovered_mean_dec_arcsec : float
        Measured mean Dec residual from LOOO observatory_stats.
    recovered_rms_ra_arcsec : float
        Measured RMS of RA residuals.
    recovered_rms_dec_arcsec : float
        Measured RMS of Dec residuals.
    n_obs : int
        Number of held-out observations used for statistics.
    n_objects : int
        Number of distinct objects contributing.
    recovery_error_ra : float
        ``injected_ra_arcsec - recovered_mean_ra_arcsec``.
    recovery_error_dec : float
        ``injected_dec_arcsec - recovered_mean_dec_arcsec``.
    detection_snr_ra : float
        ``|injected_ra_arcsec| / (recovered_rms_ra_arcsec / sqrt(n_obs))``.
    detection_snr_dec : float
        ``|injected_dec_arcsec| / (recovered_rms_dec_arcsec / sqrt(n_obs))``.
    detected_ra : bool
        ``|recovery_error_ra| < threshold_arcsec``.
    detected_dec : bool
        ``|recovery_error_dec| < threshold_arcsec``.
    """

    fake_code: str
    real_code: str
    bias_type: str
    bias_params: dict
    injected_ra_arcsec: float
    injected_dec_arcsec: float
    recovered_mean_ra_arcsec: float
    recovered_mean_dec_arcsec: float
    recovered_rms_ra_arcsec: float
    recovered_rms_dec_arcsec: float
    n_obs: int
    n_objects: int
    recovery_error_ra: float
    recovery_error_dec: float
    detection_snr_ra: float
    detection_snr_dec: float
    detected_ra: bool
    detected_dec: bool


def evaluate_recovery(
    observatory_stats_parquet: Path,
    truth_biases_csv: Path,
    threshold_arcsec: float = 0.05,
) -> pd.DataFrame:
    """
    Join LOOO-recovered statistics against the injected truth table.

    Parameters
    ----------
    observatory_stats_parquet : Path
        ``observatory_stats.parquet`` produced by ``03_analyze.py``.
    truth_biases_csv : Path
        ``truth_biases.csv`` produced by ``07_generate_sim_dataset.py``.
    threshold_arcsec : float
        Maximum allowable recovery error (arcsec) for a bias to be considered
        successfully detected.

    Returns
    -------
    pd.DataFrame
        One row per fake station with all ``StationRecovery`` fields as columns.
    """
    import pyarrow.parquet as pq

    # Load observatory statistics
    obs_stats = pq.read_table(observatory_stats_parquet).to_pandas()
    # Load truth biases
    truth = pd.read_csv(truth_biases_csv)

    # The observatory_stats are keyed by 'stn', which for synthetic data is the fake code.
    # truth_biases are keyed by 'fake_code'.
    merged = truth.merge(
        obs_stats.rename(columns={"stn": "fake_code"}),
        on="fake_code",
        how="left",
    )

    rows = []
    for _, row in merged.iterrows():
        fake_code = str(row["fake_code"])
        real_code = str(row["real_code"])
        bias_type = str(row.get("bias_type", "Unknown"))

        try:
            import json as _json
            bias_params = _json.loads(row.get("bias_params_json", "{}") or "{}")
        except Exception:
            bias_params = {}

        injected_ra = float(row.get("expected_mean_ra_arcsec", 0.0) or 0.0)
        injected_dec = float(row.get("expected_mean_dec_arcsec", 0.0) or 0.0)

        # Recovered values from LOOO stats (NaN if station not present in results)
        rec_ra = _float_or_nan(row.get("mean_ra_arcsec"))
        rec_dec = _float_or_nan(row.get("mean_dec_arcsec"))
        rms_ra = _float_or_nan(row.get("rms_ra_arcsec"))
        rms_dec = _float_or_nan(row.get("rms_dec_arcsec"))
        n_obs = int(row.get("n_obs", 0) or 0)
        n_objects = int(row.get("n_objects", 0) or 0)

        err_ra = injected_ra - rec_ra
        err_dec = injected_dec - rec_dec

        # Detection SNR: injected / (rms / sqrt(n))
        sqrt_n = np.sqrt(max(n_obs, 1))
        snr_ra = (
            abs(injected_ra) / (rms_ra / sqrt_n)
            if np.isfinite(rms_ra) and rms_ra > 0
            else 0.0
        )
        snr_dec = (
            abs(injected_dec) / (rms_dec / sqrt_n)
            if np.isfinite(rms_dec) and rms_dec > 0
            else 0.0
        )

        detected_ra = np.isfinite(err_ra) and abs(err_ra) < threshold_arcsec
        detected_dec = np.isfinite(err_dec) and abs(err_dec) < threshold_arcsec

        rows.append(
            {
                "fake_code": fake_code,
                "real_code": real_code,
                "bias_type": bias_type,
                "bias_params": bias_params,
                "injected_ra_arcsec": injected_ra,
                "injected_dec_arcsec": injected_dec,
                "recovered_mean_ra_arcsec": rec_ra,
                "recovered_mean_dec_arcsec": rec_dec,
                "recovered_rms_ra_arcsec": rms_ra,
                "recovered_rms_dec_arcsec": rms_dec,
                "n_obs": n_obs,
                "n_objects": n_objects,
                "recovery_error_ra": err_ra,
                "recovery_error_dec": err_dec,
                "detection_snr_ra": snr_ra,
                "detection_snr_dec": snr_dec,
                "detected_ra": detected_ra,
                "detected_dec": detected_dec,
            }
        )

    return pd.DataFrame(rows)


def print_recovery_summary(recovery_df: pd.DataFrame) -> None:
    """
    Print a human-readable recovery report to stdout.

    Parameters
    ----------
    recovery_df : pd.DataFrame
        Output of ``evaluate_recovery()``.
    """
    if recovery_df.empty:
        print("No recovery results to display.")
        return

    header = (
        f"{'Station':<8}  {'RealCode':<8}  {'BiasType':<28}  "
        f"{'Inj_RA':>7}  {'Inj_Dec':>7}  "
        f"{'Rec_RA':>7}  {'Rec_Dec':>7}  "
        f"{'Err_RA':>7}  {'Err_Dec':>7}  "
        f"{'SNR_RA':>7}  {'SNR_Dec':>7}  "
        f"{'N_obs':>6}  {'Det_RA':>6}  {'Det_Dec':>7}"
    )
    print()
    print("=" * len(header))
    print("  SIMULATION RECOVERY REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for _, row in recovery_df.iterrows():
        def _fmt(v, digits=3):
            if isinstance(v, float) and not np.isfinite(v):
                return "   N/A "
            return f"{v:+7.{digits}f}"

        print(
            f"{str(row['fake_code']):<8}  "
            f"{str(row['real_code']):<8}  "
            f"{str(row['bias_type']):<28}  "
            f"{_fmt(row['injected_ra_arcsec'])}  "
            f"{_fmt(row['injected_dec_arcsec'])}  "
            f"{_fmt(row['recovered_mean_ra_arcsec'])}  "
            f"{_fmt(row['recovered_mean_dec_arcsec'])}  "
            f"{_fmt(row['recovery_error_ra'])}  "
            f"{_fmt(row['recovery_error_dec'])}  "
            f"{_fmt(row['detection_snr_ra'], 1)}  "
            f"{_fmt(row['detection_snr_dec'], 1)}  "
            f"{int(row['n_obs']):>6}  "
            f"{'YES' if row['detected_ra'] else ' NO':>6}  "
            f"{'YES' if row['detected_dec'] else ' NO':>7}"
        )

    print("-" * len(header))
    n_total = len(recovery_df)
    n_det_ra = recovery_df["detected_ra"].sum()
    n_det_dec = recovery_df["detected_dec"].sum()
    print(
        f"Detection rate: RA={n_det_ra}/{n_total} "
        f"({100 * n_det_ra / max(n_total, 1):.0f}%)  "
        f"Dec={n_det_dec}/{n_total} "
        f"({100 * n_det_dec / max(n_total, 1):.0f}%)"
    )
    print()


def _float_or_nan(val) -> float:
    """Convert a value to float, returning NaN if missing or non-finite."""
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return float("nan")
