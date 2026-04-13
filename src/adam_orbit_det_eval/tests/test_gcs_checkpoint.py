"""Tests for GCSCheckpointStore.

Uses a fake in-memory GCS client so we don't depend on google-cloud-storage
or network access.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adam_orbit_det_eval.looo.gcs_checkpoint import (
    GCSCheckpointStore,
    _parse_gcs_uri,
    _safe_id,
)


class FakeBlob:
    def __init__(self, name: str, storage: dict):
        self.name = name
        self._storage = storage

    def upload_from_filename(self, path: str) -> None:
        self._storage[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self._storage[self.name])


class FakeBucket:
    def __init__(self, name: str, storage: dict):
        self.name = name
        self._storage = storage

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self._storage)


class FakeGCSClient:
    def __init__(self):
        self._storage: dict[str, bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(name, self._storage)

    def list_blobs(self, bucket_name: str, prefix: str | None = None):
        for key in list(self._storage.keys()):
            if prefix is None or key.startswith(prefix):
                yield FakeBlob(key, self._storage)


def test_parse_gcs_uri():
    assert _parse_gcs_uri("gs://mybucket/foo/bar/") == ("mybucket", "foo/bar")
    assert _parse_gcs_uri("gs://mybucket/foo/bar") == ("mybucket", "foo/bar")
    assert _parse_gcs_uri("gs://mybucket") == ("mybucket", "")
    assert _parse_gcs_uri("gs://mybucket/") == ("mybucket", "")


def test_parse_gcs_uri_invalid():
    with pytest.raises(ValueError):
        _parse_gcs_uri("s3://wrong-scheme/x")
    with pytest.raises(ValueError):
        _parse_gcs_uri("gs:///no-bucket")


def test_safe_id():
    assert _safe_id("2020 AV2") == "2020_AV2"
    assert _safe_id("C/2020 F3") == "C_2020_F3"


def test_upload_and_sync_from_gcs(tmp_path: Path):
    local = tmp_path / "checkpoints"
    local.mkdir()
    (local / "obj1.parquet").write_bytes(b"PARQUET_DATA_1")
    (local / "obj2.parquet").write_bytes(b"PARQUET_DATA_2")

    fake = FakeGCSClient()
    store = GCSCheckpointStore(local, "gs://bucket/runs/abc/", client=fake)

    assert store.upload_checkpoint("obj1") is True
    assert store.upload_checkpoint("obj2") is True
    assert store.upload_checkpoint("missing") is False

    # Downloading into a fresh local dir should recover both files
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    store2 = GCSCheckpointStore(fresh, "gs://bucket/runs/abc/", client=fake)
    n = store2.sync_from_gcs()
    assert n == 2
    assert (fresh / "obj1.parquet").read_bytes() == b"PARQUET_DATA_1"
    assert (fresh / "obj2.parquet").read_bytes() == b"PARQUET_DATA_2"


def test_sync_from_gcs_does_not_clobber_local(tmp_path: Path):
    local = tmp_path / "checkpoints"
    local.mkdir()
    (local / "obj1.parquet").write_bytes(b"NEWER_LOCAL")

    fake = FakeGCSClient()
    # Seed cloud storage with an older version of obj1.
    store_remote = GCSCheckpointStore(
        tmp_path / "other", "gs://bucket/runs/abc/", client=fake
    )
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "obj1.parquet").write_bytes(b"OLDER_CLOUD")
    store_remote.upload_checkpoint("obj1")

    # Sync should NOT clobber the newer local file.
    store = GCSCheckpointStore(local, "gs://bucket/runs/abc/", client=fake)
    store.sync_from_gcs()
    assert (local / "obj1.parquet").read_bytes() == b"NEWER_LOCAL"


def test_sync_to_gcs_bulk(tmp_path: Path):
    local = tmp_path / "checkpoints"
    local.mkdir()
    (local / "a.parquet").write_bytes(b"A")
    (local / "b.parquet").write_bytes(b"B")
    (local / "c.parquet").write_bytes(b"C")

    fake = FakeGCSClient()
    store = GCSCheckpointStore(local, "gs://bucket/runs/abc/", client=fake)
    n = store.sync_to_gcs()
    assert n == 3
    assert set(fake._storage.keys()) == {
        "runs/abc/a.parquet",
        "runs/abc/b.parquet",
        "runs/abc/c.parquet",
    }


def test_register_sigterm_handler_does_not_raise(tmp_path: Path):
    """The handler should install without error; actual SIGTERM isn't sent."""
    import signal

    local = tmp_path / "checkpoints"
    local.mkdir()
    store = GCSCheckpointStore(local, "gs://b/p/", client=FakeGCSClient())

    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        store.register_sigterm_handler()
        assert signal.getsignal(signal.SIGTERM) is not old_handler
    finally:
        signal.signal(signal.SIGTERM, old_handler)
