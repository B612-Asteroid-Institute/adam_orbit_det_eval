"""
Publication-time hygiene filters for the per-station LOOO bias catalog.

Three filters are applied before a catalog is considered publishable:

1. mode='OCC' drop — occultation-timing observations are a fundamentally
   different astrometric measurement mode and must not contribute to a
   per-station optical-residual bias catalog. The LOOO pipeline already
   excludes space-based stations (which is where every MPC OCC obs in
   the v12 fetch comes from), so this filter is typically a no-op on
   real-data merged outputs; it is kept as a hard guard against future
   inputs that route occultation-timing rows through the pipeline.

2. Unknown observatory codes — codes that the version of adam_core in use
   does not recognise. adam_core.observers.get_observer_state raises on
   such codes at OD-prep time, so they should never reach a merged LOOO
   output; if they do, it means the catalog was assembled with a different
   mpc_obscodes version than the one we can verify against locally. We
   drop those rows so the published catalog matches the local resolver,
   and document the disposition.

3. Small-sample cutoff — per-station statistics with very small n_obs or
   n_objects are noise. Applied to the aggregated observatory_stats table
   (NOT to the row-level merged_looo_results parquet, which keeps every
   row that survives 1 and 2 so downstream consumers can re-aggregate
   with their own thresholds).

These filters are deliberately separate from the row-level aggregation-time
bad-fit filter (`bias_filter.py`, bead 7bt), which removes individual rows
based on fit-quality tiers. Publication hygiene is about *what to publish*,
not *what is a valid observation*.

This module deliberately DOES NOT touch reported sigmas (`sigma_ra_cosdec_arcsec`,
`sigma_dec_arcsec`). The LOOO catalog is built specifically to bypass
reported sigmas in weighting and bias derivation; reduced chi² >> 1 against
reported sigmas is the *signal* of station-level error misstatement, not
data corruption. See `docs/mpc-bias-catalog-interpretation.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .analysis import ObservatoryStats
from .core import LOOOResult

#: Default thresholds for the small-sample cutoff applied to observatory_stats.
DEFAULT_MIN_OBS = 100
DEFAULT_MIN_OBJECTS = 20


@dataclass
class HygieneStats:
    """Row-count audit trail for a publication_hygiene pass over a merged catalog."""

    rows_in: int
    rows_after_occ_drop: int
    rows_after_unknown_stn_drop: int

    occ_obs_ids_in_source: int
    occ_rows_dropped: int

    unknown_stn_codes: list[str] = field(default_factory=list)
    unknown_stn_rows_dropped: int = 0

    def summary(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_after_occ_drop": self.rows_after_occ_drop,
            "rows_after_unknown_stn_drop": self.rows_after_unknown_stn_drop,
            "rows_out": self.rows_after_unknown_stn_drop,
            "occ_obs_ids_in_source": self.occ_obs_ids_in_source,
            "occ_rows_dropped": self.occ_rows_dropped,
            "unknown_stn_codes": list(self.unknown_stn_codes),
            "unknown_stn_rows_dropped": self.unknown_stn_rows_dropped,
        }


def load_occ_obs_ids(source_obs_dir: Path) -> set[str]:
    """
    Scan the source MPC observation shards under `source_obs_dir` and return
    the set of obs_ids with mode='OCC'. Each shard must contain a file named
    `mpc_observations.parquet` with `obsid` and `mode` columns.

    Returns an empty set if no shards are found or if the `mode` column is
    absent (older fetches predate the column).
    """
    source_obs_dir = Path(source_obs_dir)
    occ_ids: set[str] = set()
    if not source_obs_dir.exists():
        return occ_ids
    shards = sorted(d for d in os.listdir(source_obs_dir) if d.startswith("shard_"))
    for shard in shards:
        path = source_obs_dir / shard / "mpc_observations.parquet"
        if not path.exists():
            continue
        schema = pq.read_schema(path)
        if "mode" not in schema.names or "obsid" not in schema.names:
            continue
        tbl = pq.read_table(path, columns=["obsid", "mode"])
        mask = pc.equal(tbl.column("mode"), "OCC")
        for oid in tbl.filter(mask).column("obsid").to_pylist():
            if oid is not None:
                occ_ids.add(oid)
    return occ_ids


def find_unknown_stations(stns: Iterable[str]) -> list[str]:
    """
    Return the subset of `stns` not recognised by the installed adam_core
    observatory code table. These are codes for which `Observers.from_codes`
    would raise — and so should not normally appear in a merged LOOO output.
    """
    from adam_core.observers.observers import OBSERVATORY_CODES

    seen: set[str] = set()
    unknown: list[str] = []
    for s in stns:
        if s is None or s in seen:
            continue
        seen.add(s)
        if s not in OBSERVATORY_CODES:
            unknown.append(s)
    return sorted(unknown)


def apply_publication_hygiene(
    merged: LOOOResult,
    source_obs_dir: Optional[Path] = None,
    occ_obs_ids: Optional[Iterable[str]] = None,
) -> tuple[LOOOResult, HygieneStats]:
    """
    Apply row-level publication hygiene to a merged LOOO catalog.

    Parameters
    ----------
    merged : LOOOResult
        Row-level merged catalog (typically the output of bias_filter).
    source_obs_dir : Path, optional
        Directory of source `mpc_observations.parquet` shards used to identify
        mode='OCC' obs_ids. Mutually exclusive with `occ_obs_ids`. If neither
        is provided, the OCC drop is skipped (rows_after_occ_drop == rows_in).
    occ_obs_ids : iterable of str, optional
        Pre-computed set of mode='OCC' obs_ids. Use this when the caller
        already has the list (avoids re-scanning shards).

    Returns
    -------
    (cleaned, stats) — cleaned LOOOResult and the per-filter row-count audit.
    """
    rows_in = len(merged)

    if occ_obs_ids is None and source_obs_dir is not None:
        occ_set = load_occ_obs_ids(Path(source_obs_dir))
    elif occ_obs_ids is not None:
        occ_set = set(occ_obs_ids)
    else:
        occ_set = set()

    if occ_set:
        keep_mask = pc.invert(
            pc.is_in(
                merged.obs_id,
                value_set=pa.array(sorted(occ_set), type=pa.large_string()),
            )
        )
        after_occ = merged.apply_mask(keep_mask)
    else:
        after_occ = merged
    rows_after_occ = len(after_occ)
    occ_dropped = rows_in - rows_after_occ

    stns_present = pc.unique(after_occ.stn).to_pylist()
    unknown = find_unknown_stations(stns_present)

    if unknown:
        keep_mask = pc.invert(
            pc.is_in(
                after_occ.stn,
                value_set=pa.array(unknown, type=pa.large_string()),
            )
        )
        cleaned = after_occ.apply_mask(keep_mask)
    else:
        cleaned = after_occ
    rows_after_unknown = len(cleaned)
    unknown_dropped = rows_after_occ - rows_after_unknown

    stats = HygieneStats(
        rows_in=rows_in,
        rows_after_occ_drop=rows_after_occ,
        rows_after_unknown_stn_drop=rows_after_unknown,
        occ_obs_ids_in_source=len(occ_set),
        occ_rows_dropped=occ_dropped,
        unknown_stn_codes=unknown,
        unknown_stn_rows_dropped=unknown_dropped,
    )
    return cleaned, stats


def filter_observatory_stats_small_sample(
    stats: ObservatoryStats,
    min_obs: int = DEFAULT_MIN_OBS,
    min_objects: int = DEFAULT_MIN_OBJECTS,
) -> tuple[ObservatoryStats, dict]:
    """
    Apply the publication small-sample cutoff to an `ObservatoryStats` table.

    Rows are dropped (not flagged) when either `n_obs < min_obs` or
    `n_objects < min_objects`. The convention is filter-out rather than
    mark `published=False` because the consumer of the published catalog
    should not have to learn a flag semantics; what is published is the
    catalog.

    Returns
    -------
    (filtered, audit) — filtered ObservatoryStats and an audit dict with
    row counts and the dropped stations.
    """
    n_in = len(stats)
    if n_in == 0:
        return stats, {
            "min_obs": min_obs,
            "min_objects": min_objects,
            "stations_in": 0,
            "stations_out": 0,
            "stations_dropped": 0,
            "dropped_stations": [],
        }

    n_obs_arr = np.asarray(stats.n_obs.to_pylist(), dtype=np.int64)
    n_obj_arr = np.asarray(stats.n_objects.to_pylist(), dtype=np.int64)
    stn_arr = stats.stn.to_pylist()

    keep_idx = np.where((n_obs_arr >= min_obs) & (n_obj_arr >= min_objects))[0]
    dropped_idx = np.where((n_obs_arr < min_obs) | (n_obj_arr < min_objects))[0]

    dropped_stations = [
        {
            "stn": stn_arr[i],
            "n_obs": int(n_obs_arr[i]),
            "n_objects": int(n_obj_arr[i]),
        }
        for i in dropped_idx
    ]

    keep_mask = pa.array(
        [bool((n_obs_arr[i] >= min_obs) and (n_obj_arr[i] >= min_objects)) for i in range(n_in)],
        type=pa.bool_(),
    )
    filtered = stats.apply_mask(keep_mask)

    audit = {
        "min_obs": min_obs,
        "min_objects": min_objects,
        "stations_in": int(n_in),
        "stations_out": int(len(filtered)),
        "stations_dropped": int(len(dropped_stations)),
        "dropped_stations": dropped_stations,
    }
    return filtered, audit


def format_hygiene_audit(stats: HygieneStats, stats_audit: Optional[dict] = None) -> str:
    """Return a human-readable audit of the hygiene pass for the validation report."""
    lines = []
    lines.append("Publication-hygiene audit")
    lines.append("=" * 72)
    lines.append(f"Rows in (merged_looo_results_filtered):      {stats.rows_in:>12d}")
    lines.append(f"  After mode='OCC' drop ({stats.occ_rows_dropped} dropped, "
                 f"{stats.occ_obs_ids_in_source} OCC obs_ids in source): "
                 f"{stats.rows_after_occ_drop:>12d}")
    if stats.unknown_stn_codes:
        codes_str = ", ".join(stats.unknown_stn_codes)
        lines.append(
            f"  After unknown-code drop ({stats.unknown_stn_rows_dropped} dropped, "
            f"codes: {codes_str}): {stats.rows_after_unknown_stn_drop:>12d}"
        )
    else:
        lines.append(
            f"  After unknown-code drop (0 unknown codes found): "
            f"{stats.rows_after_unknown_stn_drop:>12d}"
        )
    lines.append("")
    if stats_audit is not None:
        lines.append(
            f"Small-sample cutoff (n_obs >= {stats_audit['min_obs']} AND "
            f"n_objects >= {stats_audit['min_objects']}): "
            f"{stats_audit['stations_in']} -> {stats_audit['stations_out']} stations "
            f"({stats_audit['stations_dropped']} dropped)"
        )
    return "\n".join(lines)
