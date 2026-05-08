"""
Aggregation-time bad-fit filter for the per-station bias catalog.

Multi-tier filter applied to a merged LOOO results table BEFORE per-station
aggregation. Catches the residual failure modes that bypass the eligibility
checks and the per-fit success flag:

  Tier 1 — silent fitter failures (``hold_in_fit_success`` is False or null).
  Tier 2 — poor hold-in fits (``hold_in_reduced_chi2`` above ``max_chi2``).
  Tier 3 — orbit drift: drop ALL rows for objects whose hold-in orbit landed
           in a wrong basin, detected via abnormally large
           ``|delta_q_au|``, ``|delta_e|``, or ``|delta_i_deg|``. Catches
           "fit converged numerically but orbit is junk" — chi² may look fine
           but the residuals at held-out times are garbage.
  Tier 4 — per-station MAD-based outlier rejection on held-out residuals.
           Self-tuning: each station's median and MAD are computed after
           Tiers 1-3, and rows beyond ``mad_factor`` MADs from the median in
           either RA or Dec are dropped. No per-station threshold tuning.

The pre-Tier-4 row set is what the bias estimator should treat as the
"plausible LOOO outcomes" universe: Tiers 1-3 cull whole-fit failures, and
Tier 4 then trims the per-station tail without distorting the central
location.

Usage
-----
    from adam_orbit_det_eval.looo.bias_filter import (
        BiasFilterConfig, apply_bias_filter,
    )
    filtered, stats = apply_bias_filter(looo_results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import numpy as np
import pyarrow as pa

from .core import LOOOResult

logger = logging.getLogger(__name__)


# Conversion from MAD to a robust 1-sigma estimate (Gaussian assumption).
_MAD_SCALE = 1.4826


@dataclass
class BiasFilterConfig:
    """Thresholds for the aggregation-time bad-fit filter.

    Defaults are tuned against the 3,500-object reference run (see bead 7bt
    validation): ~12% total loss, no anchor station regresses, Tier 3 catches
    nothing on clean data and starts catching on cloud pilots that exhibit
    orbit-drift failure modes.
    """

    #: Maximum hold-in reduced chi² before the row is dropped.
    max_chi2: float = 10.0
    #: Object-level orbit-drift cap on |delta_q| in AU.
    max_delta_q: float = 0.5
    #: Object-level orbit-drift cap on |delta_e|.
    max_delta_e: float = 0.3
    #: Object-level orbit-drift cap on |delta_i| in degrees.
    max_delta_i_deg: float = 5.0
    #: Per-station MAD multiplier; rows beyond this many MADs from the
    #: station median in either RA or Dec are dropped.
    mad_factor: float = 5.0


@dataclass
class BiasFilterStats:
    """Audit stats for ``apply_bias_filter``.

    Mirrors the ``ExclusionStats`` pattern in ``eligibility.py``: counts
    drops at every tier so a downstream report can quantify how much of the
    catastrophic tail each tier carved off, and which stations leaned hardest
    on Tier 4 (a signal for follow-up — a station that loses >5% to MAD is
    probably hiding a station-scale failure mode).
    """

    rows_in: int = 0
    rows_out: int = 0
    dropped_tier1_failed_fit: int = 0
    dropped_tier2_high_chi2: int = 0
    #: Rows dropped because their object failed Tier 3 (whole-object cull).
    dropped_tier3_orbit_drift: int = 0
    #: Distinct objects rejected by Tier 3.
    objects_dropped_tier3: int = 0
    dropped_tier4_mad: int = 0
    #: Per-station audit: stn -> {"rows_in", "rows_out",
    #: "fraction_lost_tier4", "median_ra_arcsec", "median_dec_arcsec",
    #: "mad_ra_arcsec", "mad_dec_arcsec"}.
    per_station: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def loss_fraction(self) -> float:
        if self.rows_in == 0:
            return 0.0
        return (self.rows_in - self.rows_out) / self.rows_in

    def summary(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "loss_fraction": self.loss_fraction,
            "dropped_tier1_failed_fit": self.dropped_tier1_failed_fit,
            "dropped_tier2_high_chi2": self.dropped_tier2_high_chi2,
            "dropped_tier3_orbit_drift": self.dropped_tier3_orbit_drift,
            "objects_dropped_tier3": self.objects_dropped_tier3,
            "dropped_tier4_mad": self.dropped_tier4_mad,
            "per_station": self.per_station,
        }

    def stations_with_high_mad_loss(
        self, threshold: float = 0.05, min_rows: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """Return stations where Tier 4 dropped at least ``threshold`` of rows.

        ``min_rows`` filters out small-N stations whose MAD loss is dominated
        by counting noise (e.g. a station with 3 rows that loses 1 reads as
        33% lost). At ~30 rows the loss fraction is at least somewhat
        diagnostic.

        These are not automatically excluded — they're flagged so the
        validation report surfaces them as "investigate". A station that
        loses >5% of its rows to MAD-based rejection above the small-N floor
        is probably hiding a station-scale failure mode (consistent
        calibration error, broken astrometric reduction at one telescope,
        etc.) rather than the per-object outliers Tier 4 is designed to
        clean up.
        """
        return {
            s: info
            for s, info in self.per_station.items()
            if info.get("rows_in", 0) >= min_rows
            and info.get("fraction_lost_tier4", 0.0) >= threshold
        }


# Type accepted by apply_bias_filter — either a quivr LOOOResult or a raw
# pyarrow Table (the cloud collector loads parquet directly to pa.Table).
LOOOInput = Union[LOOOResult, pa.Table]


def _as_float_array(col: pa.Array) -> np.ndarray:
    """Materialise a pyarrow column as float64, mapping nulls to NaN."""
    return col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def apply_bias_filter(
    looo_results: LOOOInput,
    config: BiasFilterConfig = BiasFilterConfig(),
) -> Tuple[LOOOInput, BiasFilterStats]:
    """Run the multi-tier bad-fit filter on a merged LOOO results table.

    Parameters
    ----------
    looo_results : LOOOResult or pyarrow.Table
        Merged per-shard results (output of the cloud collector or a local
        run). Must have at least: ``object_id``, ``stn``, ``residual_ra_arcsec``,
        ``residual_dec_arcsec``. The ``hold_in_fit_success``,
        ``hold_in_reduced_chi2``, and ``delta_*`` columns are used when present;
        when absent (older parquets) the corresponding tier is a no-op.
    config : BiasFilterConfig

    Returns
    -------
    (filtered, stats)
        Filtered table of the same type as the input and a populated
        ``BiasFilterStats``.
    """
    is_qv = isinstance(looo_results, LOOOResult)
    tbl: pa.Table = looo_results.table if is_qv else looo_results

    stats = BiasFilterStats()
    stats.rows_in = len(tbl)
    if stats.rows_in == 0:
        stats.rows_out = 0
        return looo_results, stats

    schema_names = set(tbl.schema.names)
    n = len(tbl)

    # ------------------------------------------------------------------
    # Tier 1 — drop rows where the fitter signalled failure (or returned
    # null, which on the v11 cloud regression silently masqueraded as success
    # in the analysis filter that used fill_null(False)).
    # ------------------------------------------------------------------
    if "hold_in_fit_success" in schema_names:
        success = tbl.column("hold_in_fit_success").to_pylist()
        keep_t1 = np.array([v is True for v in success], dtype=bool)
    else:
        keep_t1 = np.ones(n, dtype=bool)
    stats.dropped_tier1_failed_fit = int(n - keep_t1.sum())

    # ------------------------------------------------------------------
    # Tier 2 — drop poor hold-in fits.  We require both finite chi² and
    # ``<= max_chi2``; a null/NaN chi² on a "successful" fit is itself
    # suspicious (this was the v11 silent-failure signature).
    # ------------------------------------------------------------------
    if "hold_in_reduced_chi2" in schema_names:
        chi2 = _as_float_array(tbl.column("hold_in_reduced_chi2"))
        with np.errstate(invalid="ignore"):
            keep_chi2 = np.isfinite(chi2) & (chi2 <= config.max_chi2)
        keep_t2 = keep_t1 & keep_chi2
    else:
        keep_t2 = keep_t1
    stats.dropped_tier2_high_chi2 = int(keep_t1.sum() - keep_t2.sum())

    # ------------------------------------------------------------------
    # Tier 3 — drop ALL rows for objects whose hold-in orbit drifted to a
    # wrong basin. Detection is on |delta_q|, |delta_e|, |delta_i| against
    # the reference orbit. We use the FULL row set (not just keep_t2) to
    # decide which objects are bad, because evidence from any of an object's
    # hold-out fits is enough to indict it.
    # ------------------------------------------------------------------
    object_id_arr = np.array(tbl.column("object_id").to_pylist())
    bad_objects: set = set()
    drift_columns = (
        ("delta_q_au", config.max_delta_q),
        ("delta_e", config.max_delta_e),
        ("delta_i_deg", config.max_delta_i_deg),
    )
    for col_name, max_v in drift_columns:
        if col_name not in schema_names:
            continue
        arr = _as_float_array(tbl.column(col_name))
        with np.errstate(invalid="ignore"):
            drift = np.isfinite(arr) & (np.abs(arr) > max_v)
        if drift.any():
            bad_objects.update(object_id_arr[drift].tolist())

    if bad_objects:
        keep_obj = ~np.isin(object_id_arr, np.array(sorted(bad_objects)))
    else:
        keep_obj = np.ones(n, dtype=bool)
    keep_t3 = keep_t2 & keep_obj
    stats.dropped_tier3_orbit_drift = int(keep_t2.sum() - keep_t3.sum())
    stats.objects_dropped_tier3 = len(bad_objects)

    # ------------------------------------------------------------------
    # Tier 4 — per-station MAD-based outlier rejection on held-out
    # residuals. Operates on the post-Tier-3 row set so the per-station
    # location/scale aren't poisoned by the failure modes Tiers 1-3 already
    # carved off. A row is kept iff |residual - station_median| <=
    # mad_factor * station_MAD in BOTH RA and Dec; if a station's MAD is 0
    # (degenerate — too few rows or all residuals identical), that
    # dimension's check is a no-op for that station.
    # ------------------------------------------------------------------
    ra = _as_float_array(tbl.column("residual_ra_arcsec"))
    dec = _as_float_array(tbl.column("residual_dec_arcsec"))
    stn = np.array(tbl.column("stn").to_pylist())

    keep_t4 = keep_t3.copy()
    sub_idx = np.where(keep_t3)[0]

    if sub_idx.size > 0:
        sub_stn = stn[sub_idx]
        sub_ra = ra[sub_idx]
        sub_dec = dec[sub_idx]

        unique_stns = np.unique(sub_stn)
        for s in unique_stns:
            m = sub_stn == s
            global_idx = sub_idx[m]
            sra = sub_ra[m]
            sdec = sub_dec[m]

            sra_finite = sra[np.isfinite(sra)]
            sdec_finite = sdec[np.isfinite(sdec)]

            n_in = int(m.sum())
            if sra_finite.size == 0 and sdec_finite.size == 0:
                # No usable residuals at all — keep the rows and let
                # downstream NaN handling take care of them.
                stats.per_station[str(s)] = {
                    "rows_in": n_in,
                    "rows_out": n_in,
                    "fraction_lost_tier4": 0.0,
                    "median_ra_arcsec": float("nan"),
                    "median_dec_arcsec": float("nan"),
                    "mad_ra_arcsec": float("nan"),
                    "mad_dec_arcsec": float("nan"),
                }
                continue

            med_ra = float(np.median(sra_finite)) if sra_finite.size else 0.0
            med_dec = float(np.median(sdec_finite)) if sdec_finite.size else 0.0
            mad_ra = (
                float(np.median(np.abs(sra_finite - med_ra))) * _MAD_SCALE
                if sra_finite.size
                else 0.0
            )
            mad_dec = (
                float(np.median(np.abs(sdec_finite - med_dec))) * _MAD_SCALE
                if sdec_finite.size
                else 0.0
            )

            with np.errstate(invalid="ignore"):
                if mad_ra > 0:
                    ok_ra = np.isfinite(sra) & (
                        np.abs(sra - med_ra) <= config.mad_factor * mad_ra
                    )
                    # Rows whose RA residual is non-finite are also dropped —
                    # we cannot validate them against the station tail.
                else:
                    ok_ra = np.ones_like(sra, dtype=bool)

                if mad_dec > 0:
                    ok_dec = np.isfinite(sdec) & (
                        np.abs(sdec - med_dec) <= config.mad_factor * mad_dec
                    )
                else:
                    ok_dec = np.ones_like(sdec, dtype=bool)

            ok = ok_ra & ok_dec
            drop_global = global_idx[~ok]
            if drop_global.size:
                keep_t4[drop_global] = False

            n_lost = int(n_in - ok.sum())
            stats.per_station[str(s)] = {
                "rows_in": n_in,
                "rows_out": int(ok.sum()),
                "fraction_lost_tier4": (n_lost / n_in) if n_in > 0 else 0.0,
                "median_ra_arcsec": med_ra,
                "median_dec_arcsec": med_dec,
                "mad_ra_arcsec": mad_ra,
                "mad_dec_arcsec": mad_dec,
            }

    stats.dropped_tier4_mad = int(keep_t3.sum() - keep_t4.sum())
    stats.rows_out = int(keep_t4.sum())

    mask = pa.array(keep_t4)
    filtered_tbl = tbl.filter(mask)

    logger.info(
        "bias_filter: %d -> %d rows (loss %.2f%%) — "
        "T1=%d T2=%d T3=%d (objs=%d) T4=%d",
        stats.rows_in,
        stats.rows_out,
        stats.loss_fraction * 100.0,
        stats.dropped_tier1_failed_fit,
        stats.dropped_tier2_high_chi2,
        stats.dropped_tier3_orbit_drift,
        stats.objects_dropped_tier3,
        stats.dropped_tier4_mad,
    )

    if is_qv:
        return LOOOResult.from_pyarrow(filtered_tbl), stats
    return filtered_tbl, stats


def format_filter_audit(
    stats: BiasFilterStats,
    config: BiasFilterConfig,
    high_mad_threshold: float = 0.05,
    high_mad_min_rows: int = 30,
) -> str:
    """Format a human-readable audit block for ``validation_report.txt``.

    Lists per-tier counts/fractions and flags any station whose Tier-4 loss
    exceeds ``high_mad_threshold`` (default 5%).
    """
    lines = []
    lines.append("Aggregation-time bad-fit filter")
    lines.append("=" * 72)
    lines.append(
        f"Config: max_chi2={config.max_chi2}, "
        f"max_delta_q={config.max_delta_q} AU, "
        f"max_delta_e={config.max_delta_e}, "
        f"max_delta_i_deg={config.max_delta_i_deg}, "
        f"mad_factor={config.mad_factor}"
    )
    lines.append("")
    lines.append(
        f"  rows_in:                    {stats.rows_in}"
    )
    lines.append(
        f"  rows_out:                   {stats.rows_out}  "
        f"(loss {stats.loss_fraction * 100:.2f}%)"
    )

    def _frac(n: int) -> str:
        if stats.rows_in == 0:
            return "0.0%"
        return f"{n / stats.rows_in * 100:.2f}%"

    lines.append(
        f"  Tier 1 (failed fit):        "
        f"{stats.dropped_tier1_failed_fit}  ({_frac(stats.dropped_tier1_failed_fit)})"
    )
    lines.append(
        f"  Tier 2 (high chi2):         "
        f"{stats.dropped_tier2_high_chi2}  ({_frac(stats.dropped_tier2_high_chi2)})"
    )
    lines.append(
        f"  Tier 3 (orbit drift):       "
        f"{stats.dropped_tier3_orbit_drift}  ({_frac(stats.dropped_tier3_orbit_drift)}) "
        f"[{stats.objects_dropped_tier3} object(s)]"
    )
    lines.append(
        f"  Tier 4 (per-station MAD):   "
        f"{stats.dropped_tier4_mad}  ({_frac(stats.dropped_tier4_mad)})"
    )

    high = stats.stations_with_high_mad_loss(
        threshold=high_mad_threshold, min_rows=high_mad_min_rows
    )
    if high:
        lines.append("")
        lines.append(
            f"  Stations (n>={high_mad_min_rows}) losing "
            f">={high_mad_threshold * 100:.0f}% to Tier 4 "
            f"(investigate, not auto-cut):"
        )
        lines.append(
            f"    {'stn':<6} {'rows_in':>8} {'rows_out':>9} {'lost':>10}"
        )
        for s, info in sorted(high.items(), key=lambda kv: -kv[1]["fraction_lost_tier4"]):
            lines.append(
                f"    {s:<6} {info['rows_in']:>8} {info['rows_out']:>9} "
                f"{info['fraction_lost_tier4'] * 100:>9.2f}%"
            )

    return "\n".join(lines)
