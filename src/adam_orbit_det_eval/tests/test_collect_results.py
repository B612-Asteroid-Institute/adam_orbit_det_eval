"""
Tests for scripts/13_collect_cloud_results.py — results collector and merger.

Tests the GCS scanning, shard classification, deduplication, merge logic,
and CLI argument parsing using mocked GCS.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import the script as a module
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _import_script():
    """Import 13_collect_cloud_results.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "collect_results",
        SCRIPTS_DIR / "13_collect_cloud_results.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collector = _import_script()


# ---------------------------------------------------------------------------
# GCS URI parsing
# ---------------------------------------------------------------------------


class TestParseGcsUri:
    def test_simple(self):
        bucket, prefix = collector._parse_gcs_uri("gs://my-bucket/output/path")
        assert bucket == "my-bucket"
        assert prefix == "output/path"

    def test_bucket_only(self):
        bucket, prefix = collector._parse_gcs_uri("gs://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""


# ---------------------------------------------------------------------------
# Shard scanning
# ---------------------------------------------------------------------------


class TestScanShards:
    def _make_mock_blobs(self, blob_names: list[str], prefix: str) -> list[MagicMock]:
        blobs = []
        for name in blob_names:
            blob = MagicMock()
            blob.name = name
            blobs.append(blob)
        return blobs

    @patch.object(collector, "_get_gcs_client")
    def test_scan_classifies_correctly(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client

        blob_names = [
            "output/shard_000/SUCCESS",
            "output/shard_000/looo_results.parquet",
            "output/shard_001/FAILED",
            "output/shard_002/looo_results.parquet",  # no marker = unknown
            "output/shard_003/SUCCESS",
            "output/shard_003/looo_results.parquet",
        ]
        client.list_blobs.return_value = self._make_mock_blobs(blob_names, "output/")

        succeeded, failed, unknown = collector.scan_shards("gs://bucket/output")

        assert succeeded == ["shard_000", "shard_003"]
        assert failed == ["shard_001"]
        assert unknown == ["shard_002"]

    @patch.object(collector, "_get_gcs_client")
    def test_scan_empty_returns_empty(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.list_blobs.return_value = []

        succeeded, failed, unknown = collector.scan_shards("gs://bucket/output")

        assert succeeded == []
        assert failed == []
        assert unknown == []

    @patch.object(collector, "_get_gcs_client")
    def test_success_overrides_failed(self, mock_get_client):
        """If both SUCCESS and FAILED exist (retry), SUCCESS wins."""
        client = MagicMock()
        mock_get_client.return_value = client

        blob_names = [
            "output/shard_000/FAILED",
            "output/shard_000/SUCCESS",
            "output/shard_000/looo_results.parquet",
        ]
        client.list_blobs.return_value = self._make_mock_blobs(blob_names, "output/")

        succeeded, failed, unknown = collector.scan_shards("gs://bucket/output")
        assert succeeded == ["shard_000"]
        assert failed == []


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def _make_looo_parquet(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    """Create a minimal LOOO-like parquet file for merge testing."""
    tbl = pa.table({
        "object_id": pa.array([r["object_id"] for r in rows], type=pa.large_string()),
        "obs_id": pa.array([r.get("obs_id", "obs_1") for r in rows], type=pa.large_string()),
        "stn": pa.array([r.get("stn", "500") for r in rows], type=pa.large_string()),
        "residual_ra_arcsec": pa.array([r.get("ra", 0.1) for r in rows], type=pa.float64()),
        "residual_dec_arcsec": pa.array([r.get("dec", -0.1) for r in rows], type=pa.float64()),
    })
    path = tmp_path / name
    pq.write_table(tbl, path)
    return path


class TestMergeResults:
    def test_merge_single_shard(self, tmp_path):
        p = _make_looo_parquet(tmp_path, "shard_0.parquet", [
            {"object_id": "obj1", "obs_id": "o1", "stn": "500"},
            {"object_id": "obj1", "obs_id": "o2", "stn": "500"},
        ])
        out = tmp_path / "merged.parquet"
        n = collector.merge_looo_results([p], out)
        assert n == 2
        assert out.exists()

    def test_merge_multiple_shards(self, tmp_path):
        p1 = _make_looo_parquet(tmp_path, "s1.parquet", [
            {"object_id": "obj1", "obs_id": "o1", "stn": "500"},
        ])
        p2 = _make_looo_parquet(tmp_path, "s2.parquet", [
            {"object_id": "obj2", "obs_id": "o1", "stn": "703"},
        ])
        out = tmp_path / "merged.parquet"
        n = collector.merge_looo_results([p1, p2], out)
        assert n == 2

    def test_merge_deduplicates(self, tmp_path):
        """Duplicate (object_id, obs_id, stn) from retried shards gets deduplicated."""
        p1 = _make_looo_parquet(tmp_path, "s1.parquet", [
            {"object_id": "obj1", "obs_id": "o1", "stn": "500", "ra": 0.1},
        ])
        p2 = _make_looo_parquet(tmp_path, "s2.parquet", [
            {"object_id": "obj1", "obs_id": "o1", "stn": "500", "ra": 0.2},
        ])
        out = tmp_path / "merged.parquet"
        n = collector.merge_looo_results([p1, p2], out)
        assert n == 1

    def test_merge_empty_returns_zero(self, tmp_path):
        out = tmp_path / "merged.parquet"
        n = collector.merge_looo_results([], out)
        assert n == 0

    def test_merge_skips_unreadable_file(self, tmp_path):
        good = _make_looo_parquet(tmp_path, "good.parquet", [
            {"object_id": "obj1", "obs_id": "o1", "stn": "500"},
        ])
        bad = tmp_path / "bad.parquet"
        bad.write_text("not a parquet file")

        out = tmp_path / "merged.parquet"
        n = collector.merge_looo_results([bad, good], out)
        assert n == 1


# ---------------------------------------------------------------------------
# Download shard results (mocked GCS)
# ---------------------------------------------------------------------------


class TestDownloadShardResults:
    @patch.object(collector, "_get_gcs_client")
    def test_download_existing_files(self, mock_get_client, tmp_path):
        client = MagicMock()
        mock_get_client.return_value = client
        bucket = MagicMock()
        client.bucket.return_value = bucket

        blob = MagicMock()
        blob.exists.return_value = True
        bucket.blob.return_value = blob

        result = collector.download_shard_results(
            "gs://bucket/output", "shard_005", tmp_path
        )

        assert len(result) == 3
        assert "looo_results.parquet" in result

    @patch.object(collector, "_get_gcs_client")
    def test_missing_file_logged_as_corrupt(self, mock_get_client, tmp_path):
        client = MagicMock()
        mock_get_client.return_value = client
        bucket = MagicMock()
        client.bucket.return_value = bucket

        blob = MagicMock()
        blob.exists.return_value = False  # file missing
        bucket.blob.return_value = blob

        result = collector.download_shard_results(
            "gs://bucket/output", "shard_005", tmp_path
        )

        assert len(result) == 0  # nothing downloaded


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self):
        args = collector.parse_args([
            "--gcs-output-prefix", "gs://bucket/output",
        ])
        assert args.gcs_output_prefix == "gs://bucket/output"
        assert args.output_dir == Path("data/mpc_scale_results")
        assert args.min_shards_pct == 90.0

    def test_all_args(self):
        args = collector.parse_args([
            "--gcs-output-prefix", "gs://bucket/output",
            "--output-dir", "/tmp/results",
            "--min-shards-pct", "75",
        ])
        assert args.output_dir == Path("/tmp/results")
        assert args.min_shards_pct == 75.0
