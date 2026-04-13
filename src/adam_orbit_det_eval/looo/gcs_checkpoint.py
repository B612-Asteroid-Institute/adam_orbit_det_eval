"""
GCS-backed checkpoint store for LOOO pipeline spot-preemption recovery.

Wraps the existing local per-object checkpoint mechanism so that on cloud
runs (particularly spot/preemptible instances), checkpoints survive pod
eviction and a restarted job resumes from where it left off.

Design
------
- The local checkpoint directory is still the source of truth during a run.
  Workers write per-object parquet files to `local_dir` exactly as before.
- `GCSCheckpointStore` provides thin sync operations:
    * `sync_from_gcs()` — at startup, mirror existing cloud checkpoints into
      `local_dir` so the local resume logic picks them up.
    * `upload_checkpoint(object_id)` — after a single checkpoint is written,
      push it to GCS (call this from the worker).
    * `sync_to_gcs()` — bulk upload everything in `local_dir` to GCS.
    * `register_sigterm_handler()` — on SIGTERM (spot preemption), flush
      local state to GCS before exit.
- Fully optional: if no store is passed to `run_looo_pipeline`, behavior is
  unchanged (pure local checkpoints).
- `google-cloud-storage` is imported lazily so local-only runs don't need
  the dependency installed.
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)


def _import_gcs():
    """Import google.cloud.storage lazily; raise with a helpful message."""
    try:
        from google.cloud import storage  # type: ignore
        return storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-storage is required for GCSCheckpointStore. "
            "Install it in your cloud image (pip install google-cloud-storage)."
        ) from e


def _parse_gcs_uri(gcs_prefix: str) -> tuple[str, str]:
    """Parse a gs://bucket/prefix/ URI into (bucket, prefix).

    Trailing slash on the prefix is stripped; prefix may be empty.
    """
    if not gcs_prefix.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_prefix!r}")
    rest = gcs_prefix[len("gs://"):]
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    prefix = prefix.rstrip("/")
    if not bucket:
        raise ValueError(f"Invalid gs:// URI (missing bucket): {gcs_prefix!r}")
    return bucket, prefix


def _safe_id(object_id: str) -> str:
    """Match pipeline._checkpoint_path sanitization."""
    return object_id.replace("/", "_").replace(" ", "_")


class GCSCheckpointStore:
    """Sync per-object LOOO checkpoints between a local dir and GCS.

    Parameters
    ----------
    local_dir : Path
        Existing local checkpoint directory (same one the pipeline uses).
    gcs_prefix : str
        GCS destination as ``gs://bucket/path/checkpoints/``.
    client : google.cloud.storage.Client, optional
        Pre-built GCS client. If not given, one is constructed on first use.
    """

    def __init__(
        self,
        local_dir: Path,
        gcs_prefix: str,
        client: Optional["gcs_storage.Client"] = None,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.gcs_prefix = gcs_prefix
        self.bucket_name, self.prefix = _parse_gcs_uri(gcs_prefix)
        self._client = client
        self._bucket = None

    @property
    def client(self) -> "gcs_storage.Client":
        if self._client is None:
            storage = _import_gcs()
            self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> "gcs_storage.Bucket":
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def _blob_name(self, filename: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{filename}"
        return filename

    def sync_from_gcs(self) -> int:
        """Download all existing GCS checkpoint parquets into ``local_dir``.

        Called once at pipeline startup so that after a restart the local
        resume logic in ``_load_completed_ids`` picks them up automatically.

        Returns the number of files downloaded.
        """
        self.local_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        prefix_filter = f"{self.prefix}/" if self.prefix else None
        for blob in self.client.list_blobs(self.bucket_name, prefix=prefix_filter):
            if not blob.name.endswith(".parquet"):
                continue
            filename = blob.name.split("/")[-1]
            local_path = self.local_dir / filename
            # Skip if local file already exists and is non-empty (don't clobber
            # fresher local state with an older GCS copy).
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            blob.download_to_filename(str(local_path))
            n += 1
        logger.info(f"sync_from_gcs: downloaded {n} checkpoint(s) from {self.gcs_prefix}")
        return n

    def upload_checkpoint(self, object_id: str) -> bool:
        """Upload a single checkpoint parquet to GCS.

        Returns True if a file was uploaded, False if the local file is
        missing (worker produced no checkpoint).
        """
        filename = f"{_safe_id(object_id)}.parquet"
        local_path = self.local_dir / filename
        if not local_path.exists():
            logger.debug(f"upload_checkpoint: no local file for {object_id}")
            return False
        blob = self.bucket.blob(self._blob_name(filename))
        blob.upload_from_filename(str(local_path))
        return True

    def sync_to_gcs(self) -> int:
        """Bulk upload all local checkpoint parquets to GCS.

        Safe to call repeatedly; re-uploads overwrite existing blobs.
        """
        if not self.local_dir.exists():
            return 0
        n = 0
        for f in sorted(self.local_dir.glob("*.parquet")):
            blob = self.bucket.blob(self._blob_name(f.name))
            blob.upload_from_filename(str(f))
            n += 1
        logger.info(f"sync_to_gcs: uploaded {n} checkpoint(s) to {self.gcs_prefix}")
        return n

    def register_sigterm_handler(self) -> None:
        """Install a SIGTERM handler that flushes local checkpoints to GCS.

        Spot/preemptible nodes receive SIGTERM roughly 30s before eviction.
        The handler re-raises SystemExit after flushing so the interpreter
        exits cleanly.
        """
        def _handler(signum, frame):  # pragma: no cover - process-level
            logger.warning(
                f"SIGTERM received (signum={signum}); flushing checkpoints to GCS..."
            )
            try:
                self.sync_to_gcs()
            except Exception as e:  # best-effort
                logger.error(f"sync_to_gcs during SIGTERM failed: {e}", exc_info=True)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _handler)
        logger.info("Registered SIGTERM handler for GCS checkpoint flush")
