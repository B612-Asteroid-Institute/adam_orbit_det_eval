"""Tests for `looo.publication_hygiene` (bead fqv)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adam_orbit_det_eval.looo.analysis import ObservatoryStats
from adam_orbit_det_eval.looo.core import LOOOResult
from adam_orbit_det_eval.looo.publication_hygiene import (
    DEFAULT_MIN_OBJECTS,
    DEFAULT_MIN_OBS,
    apply_publication_hygiene,
    filter_observatory_stats_small_sample,
    find_unknown_stations,
    load_occ_obs_ids,
)


def _make_looo(rows: list[dict]) -> LOOOResult:
    n = len(rows)

    def col(name, default=None):
        return [r.get(name, default) for r in rows]

    return LOOOResult.from_kwargs(
        object_id=col("object_id"),
        obs_id=col("obs_id"),
        stn=col("stn"),
        residual_ra_arcsec=col("residual_ra_arcsec", 0.0),
        residual_dec_arcsec=col("residual_dec_arcsec", 0.0),
        sigma_ra_cosdec_arcsec=col("sigma_ra_cosdec_arcsec", 0.5),
        sigma_dec_arcsec=col("sigma_dec_arcsec", 0.5),
        chi2=col("chi2", 1.0),
        astcat=col("astcat"),
        program_code=col("program_code"),
        n_obs_held_out=col("n_obs_held_out", 1),
        n_obs_remaining=col("n_obs_remaining", 50),
        arc_length_remaining_days=col("arc_length_remaining_days", 365.0),
        held_out_fraction=col("held_out_fraction", 0.02),
        n_obs_total=col("n_obs_total", 51),
        delta_q_au=col("delta_q_au", 0.0),
        delta_e=col("delta_e", 0.0),
        delta_i_deg=col("delta_i_deg", 0.0),
        hold_in_reduced_chi2=col("hold_in_reduced_chi2", 1.0),
        hold_in_fit_success=col("hold_in_fit_success", True),
    )


def test_find_unknown_stations_filters_against_adam_core():
    # '500' is the geocenter and is in OBSERVATORY_CODES; 'ZZZ_unlikely_code'
    # should not be.
    unknown = find_unknown_stations(["500", "ZZZ_unlikely_code", "500"])
    assert "500" not in unknown
    assert "ZZZ_unlikely_code" in unknown
    # No duplicates returned.
    assert unknown == sorted(set(unknown))


def test_apply_publication_hygiene_drops_occ_and_unknown(tmp_path):
    rows = [
        {"object_id": "A", "obs_id": "good-1", "stn": "703"},
        {"object_id": "A", "obs_id": "occ-1", "stn": "703"},
        {"object_id": "B", "obs_id": "good-2", "stn": "ZZZ_unknown"},
        {"object_id": "C", "obs_id": "good-3", "stn": "F52"},
    ]
    merged = _make_looo(rows)

    cleaned, stats = apply_publication_hygiene(
        merged, occ_obs_ids={"occ-1"}
    )

    out_obs_ids = set(cleaned.obs_id.to_pylist())
    assert out_obs_ids == {"good-1", "good-3"}
    assert stats.rows_in == 4
    assert stats.occ_rows_dropped == 1
    assert stats.rows_after_occ_drop == 3
    assert "ZZZ_unknown" in stats.unknown_stn_codes
    assert stats.unknown_stn_rows_dropped == 1
    assert stats.rows_after_unknown_stn_drop == 2


def test_apply_publication_hygiene_no_op_when_nothing_to_drop():
    rows = [
        {"object_id": "A", "obs_id": "x", "stn": "703"},
        {"object_id": "B", "obs_id": "y", "stn": "F52"},
    ]
    merged = _make_looo(rows)
    cleaned, stats = apply_publication_hygiene(merged, occ_obs_ids=set())
    assert len(cleaned) == 2
    assert stats.occ_rows_dropped == 0
    assert stats.unknown_stn_rows_dropped == 0
    assert stats.unknown_stn_codes == []


def test_load_occ_obs_ids_reads_shards(tmp_path):
    # Build a tiny source-obs shard layout and confirm OCC obs_ids are
    # collected; rows without `mode` or `obsid` are skipped gracefully.
    shard_dir = tmp_path / "shard_000"
    shard_dir.mkdir()
    tbl = pa.table(
        {
            "obsid": pa.array(["a", "b", "c"], type=pa.large_string()),
            "mode": pa.array(["CCD", "OCC", "CMO"], type=pa.large_string()),
        }
    )
    pq.write_table(tbl, shard_dir / "mpc_observations.parquet")

    shard_dir2 = tmp_path / "shard_001"
    shard_dir2.mkdir()
    tbl2 = pa.table(
        {
            "obsid": pa.array(["d", "e"], type=pa.large_string()),
            "mode": pa.array(["OCC", "OCC"], type=pa.large_string()),
        }
    )
    pq.write_table(tbl2, shard_dir2 / "mpc_observations.parquet")

    occ_ids = load_occ_obs_ids(tmp_path)
    assert occ_ids == {"b", "d", "e"}


def test_load_occ_obs_ids_missing_dir_returns_empty():
    assert load_occ_obs_ids(Path("/nonexistent/path/xyz")) == set()


def _make_stats(rows: list[tuple[str, int, int]]) -> ObservatoryStats:
    return ObservatoryStats.from_kwargs(
        stn=[r[0] for r in rows],
        n_obs=[r[1] for r in rows],
        n_objects=[r[2] for r in rows],
        n_objects_filtered=[0] * len(rows),
        mean_ra_arcsec=[0.0] * len(rows),
        mean_dec_arcsec=[0.0] * len(rows),
        rms_ra_arcsec=[0.3] * len(rows),
        rms_dec_arcsec=[0.3] * len(rows),
        median_abs_ra_arcsec=[0.2] * len(rows),
        median_abs_dec_arcsec=[0.2] * len(rows),
        mean_chi2_per_obs=[1.0] * len(rows),
        median_chi2_per_obs=[1.0] * len(rows),
        mean_abs_delta_q_au=[0.0] * len(rows),
        mean_abs_delta_e=[0.0] * len(rows),
        mean_abs_delta_i_deg=[0.0] * len(rows),
        mean_hold_in_reduced_chi2=[1.0] * len(rows),
    )


def test_small_sample_cutoff_applies_AND_semantics():
    stats = _make_stats(
        [
            ("KEEP", 200, 50),     # >= both
            ("FEW_OBS", 50, 50),   # n_obs too small
            ("FEW_OBJ", 200, 5),   # n_objects too small
            ("BOTH_FAIL", 10, 2),  # both fail
        ]
    )
    filtered, audit = filter_observatory_stats_small_sample(
        stats, min_obs=100, min_objects=20
    )
    assert filtered.stn.to_pylist() == ["KEEP"]
    assert audit["stations_in"] == 4
    assert audit["stations_out"] == 1
    assert audit["stations_dropped"] == 3
    dropped_codes = {d["stn"] for d in audit["dropped_stations"]}
    assert dropped_codes == {"FEW_OBS", "FEW_OBJ", "BOTH_FAIL"}


def test_small_sample_cutoff_defaults_match_documented_thresholds():
    assert DEFAULT_MIN_OBS == 100
    assert DEFAULT_MIN_OBJECTS == 20


def test_small_sample_cutoff_empty_input():
    stats = ObservatoryStats.empty()
    filtered, audit = filter_observatory_stats_small_sample(stats)
    assert len(filtered) == 0
    assert audit["stations_in"] == 0
    assert audit["stations_out"] == 0
