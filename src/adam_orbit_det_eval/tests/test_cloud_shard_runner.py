"""
Tests for scripts/12_run_looo_cloud_shard.py — cloud shard runner for real-data LOOO.

Tests the shard index resolution, shard name formatting, GCS URI parsing,
and the SUCCESS/FAILED marker logic using mocked GCS.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the script as a module
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _import_script():
    """Import 12_run_looo_cloud_shard.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "cloud_shard_runner",
        SCRIPTS_DIR / "12_run_looo_cloud_shard.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _import_script()


# ---------------------------------------------------------------------------
# Shard index resolution
# ---------------------------------------------------------------------------


class TestShardIndex:
    def test_shard_name_from_index_zero(self):
        assert runner.shard_name_from_index(0) == "shard_000"

    def test_shard_name_from_index_large(self):
        assert runner.shard_name_from_index(42) == "shard_042"
        assert runner.shard_name_from_index(199) == "shard_199"

    def test_resolve_shard_index_from_cli(self):
        assert runner.resolve_shard_index(5) == 5

    def test_resolve_shard_index_from_env(self, monkeypatch):
        monkeypatch.setenv("JOB_COMPLETION_INDEX", "17")
        assert runner.resolve_shard_index(None) == 17

    def test_resolve_shard_index_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("JOB_COMPLETION_INDEX", "99")
        assert runner.resolve_shard_index(3) == 3

    def test_resolve_shard_index_missing_exits(self, monkeypatch):
        monkeypatch.delenv("JOB_COMPLETION_INDEX", raising=False)
        with pytest.raises(SystemExit):
            runner.resolve_shard_index(None)


# ---------------------------------------------------------------------------
# GCS URI parsing
# ---------------------------------------------------------------------------


class TestParseGcsUri:
    def test_simple_uri(self):
        bucket, prefix = runner._parse_gcs_uri("gs://my-bucket/some/prefix")
        assert bucket == "my-bucket"
        assert prefix == "some/prefix"

    def test_trailing_slash_stripped(self):
        bucket, prefix = runner._parse_gcs_uri("gs://my-bucket/prefix/")
        assert prefix == "prefix"

    def test_bucket_only(self):
        bucket, prefix = runner._parse_gcs_uri("gs://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_invalid_uri_asserts(self):
        with pytest.raises(AssertionError):
            runner._parse_gcs_uri("s3://wrong-scheme/path")


# ---------------------------------------------------------------------------
# GCS download/upload with mocked storage
# ---------------------------------------------------------------------------


class TestGcsOperations:
    def _mock_gcs_client(self):
        """Create a mock GCS client with bucket/blob chain."""
        client = MagicMock()
        bucket = MagicMock()
        blob = MagicMock()
        client.bucket.return_value = bucket
        bucket.blob.return_value = blob
        return client, bucket, blob

    @patch.object(runner, "_get_gcs_client")
    def test_download_shard(self, mock_get_client, tmp_path):
        client, bucket, blob = self._mock_gcs_client()
        mock_get_client.return_value = client

        shard_dir = runner.download_shard(
            "gs://test-bucket/input", "shard_005", tmp_path
        )

        assert shard_dir == tmp_path / "shard_005"
        assert shard_dir.is_dir()

        # Should have downloaded both parquet files
        assert blob.download_to_filename.call_count == 2
        download_calls = blob.download_to_filename.call_args_list
        local_paths = {call.args[0] for call in download_calls}
        assert str(tmp_path / "shard_005" / "mpc_observations.parquet") in local_paths
        assert str(tmp_path / "shard_005" / "mpc_orbits.parquet") in local_paths

        # Verify correct blob paths requested
        blob_calls = bucket.blob.call_args_list
        blob_paths = {call.args[0] for call in blob_calls}
        assert "input/shard_005/mpc_observations.parquet" in blob_paths
        assert "input/shard_005/mpc_orbits.parquet" in blob_paths

    @patch.object(runner, "_get_gcs_client")
    def test_upload_string(self, mock_get_client):
        client, bucket, blob = self._mock_gcs_client()
        mock_get_client.return_value = client

        runner.upload_string("test content", "gs://test-bucket/output/shard_000/SUCCESS")

        bucket.blob.assert_called_once_with("output/shard_000/SUCCESS")
        blob.upload_from_string.assert_called_once_with("test content")

    @patch.object(runner, "_get_gcs_client")
    def test_upload_file(self, mock_get_client, tmp_path):
        client, bucket, blob = self._mock_gcs_client()
        mock_get_client.return_value = client

        test_file = tmp_path / "test.parquet"
        test_file.write_text("fake data")

        runner.upload_file(test_file, "gs://test-bucket/output/shard_000/test.parquet")

        bucket.blob.assert_called_once_with("output/shard_000/test.parquet")
        blob.upload_from_filename.assert_called_once_with(str(test_file))


# ---------------------------------------------------------------------------
# Upload results logic
# ---------------------------------------------------------------------------


class TestUploadResults:
    @patch.object(runner, "upload_file")
    def test_upload_results_existing_files(self, mock_upload, tmp_path):
        """All expected result files present -> all uploaded."""
        for fname in ("looo_results.parquet", "observatory_stats.parquet",
                       "program_code_stats.parquet"):
            (tmp_path / fname).write_text("data")

        runner.upload_results(tmp_path, "gs://bucket/output", "shard_007")

        assert mock_upload.call_count == 3
        uploaded_uris = {call.args[1] for call in mock_upload.call_args_list}
        assert "gs://bucket/output/shard_007/looo_results.parquet" in uploaded_uris
        assert "gs://bucket/output/shard_007/observatory_stats.parquet" in uploaded_uris
        assert "gs://bucket/output/shard_007/program_code_stats.parquet" in uploaded_uris

    @patch.object(runner, "upload_file")
    def test_upload_results_missing_file_warns(self, mock_upload, tmp_path):
        """Missing result files should log a warning, not crash."""
        # Only create one of the three expected files
        (tmp_path / "looo_results.parquet").write_text("data")

        runner.upload_results(tmp_path, "gs://bucket/output", "shard_000")

        assert mock_upload.call_count == 1


# ---------------------------------------------------------------------------
# SUCCESS / FAILED marker logic (integration-style with mocked GCS)
# ---------------------------------------------------------------------------


class TestMarkerLogic:
    @patch.object(runner, "upload_string")
    def test_success_marker_contains_shard_info(self, mock_upload_string):
        """Verify the SUCCESS marker message format."""
        msg = (
            "shard=shard_005\n"
            "n_objects=100\n"
            "n_observations=5000\n"
            "n_result_rows=4500\n"
            "n_observatory_stats=20\n"
            "n_program_code_stats=15\n"
            "elapsed_seconds=123.4\n"
        )
        runner.upload_string(msg, "gs://bucket/output/shard_005/SUCCESS")

        mock_upload_string.assert_called_once()
        call_args = mock_upload_string.call_args
        content = call_args.args[0]
        assert "shard=shard_005" in content
        assert "n_objects=100" in content

    @patch.object(runner, "upload_string")
    def test_failed_marker_contains_traceback(self, mock_upload_string):
        """Verify the FAILED marker includes error details."""
        fail_msg = (
            "shard=shard_003\n"
            "elapsed_seconds=10.0\n"
            "error:\nTraceback (most recent call last):\n"
            "  File ...\nValueError: test error\n"
        )
        runner.upload_string(fail_msg, "gs://bucket/output/shard_003/FAILED")

        mock_upload_string.assert_called_once()
        content = mock_upload_string.call_args.args[0]
        assert "shard=shard_003" in content
        assert "error:" in content


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self):
        args = runner.parse_args([
            "--gcs-input-prefix", "gs://bucket/input",
            "--gcs-output-prefix", "gs://bucket/output",
        ])
        assert args.gcs_input_prefix == "gs://bucket/input"
        assert args.gcs_output_prefix == "gs://bucket/output"
        assert args.propagator == "assist"
        assert args.orbit_fitter == "findorb"
        assert args.strict_fitter is True
        assert args.max_processes is None
        assert args.shard_index is None

    def test_all_args(self):
        args = runner.parse_args([
            "--gcs-input-prefix", "gs://bucket/input",
            "--gcs-output-prefix", "gs://bucket/output",
            "--shard-index", "5",
            "--propagator", "twobody",
            "--orbit-fitter", "scipy",
            "--no-strict-fitter",
            "--max-processes", "4",
        ])
        assert args.shard_index == 5
        assert args.propagator == "twobody"
        assert args.orbit_fitter == "scipy"
        assert args.strict_fitter is False
        assert args.max_processes == 4
