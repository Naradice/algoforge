"""
Write-through artifact store.

After writing any parquet or metadata file locally, call ``upload(local_path)``
to push it to remote object storage.  If ``ARTIFACT_REMOTE_URL`` is not set the
call is a no-op and local-only behaviour is preserved.

Supported URL schemes
---------------------
``file:///mount/path``
    Local filesystem or NAS mount — no extra dependencies. Ideal for local dev.
    The NAS must be mounted and visible inside the container (bind-mount or NFS volume).
    Example: ``ARTIFACT_REMOTE_URL=file:///mnt/nas/algoforge``

``s3://bucket/prefix``
    Amazon S3 — credentials via the standard AWS chain (env vars, ~/.aws, IAM role).
    Required env: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or an IAM role.
    Optional env: AWS_DEFAULT_REGION (default: us-east-1)

``gs://bucket/prefix``
    Google Cloud Storage — credentials via ADC or GOOGLE_APPLICATION_CREDENTIALS.

``az://container/prefix``
    Azure Blob Storage — requires AZURE_STORAGE_CONNECTION_STRING.

Upload failures are non-fatal: the error is logged and the simulation continues.
The local copy is always the authoritative source of truth.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("artifact_store")

_REMOTE_URL: str = os.getenv("ARTIFACT_REMOTE_URL", "").rstrip("/")
_LOCAL_BASE: Path = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()


def is_remote_enabled() -> bool:
    return bool(_REMOTE_URL)


def upload(local_path: Path) -> None:
    """Upload *local_path* to remote storage.

    The remote key is derived by stripping ``ARTIFACT_STORE_PATH`` from the
    front of *local_path*, so the directory tree is mirrored exactly.

    No-op when ``ARTIFACT_REMOTE_URL`` is not set.
    """
    if not _REMOTE_URL:
        return

    try:
        relative = local_path.resolve().relative_to(_LOCAL_BASE)
    except ValueError:
        log.warning("upload: %s is not under %s — skipping", local_path, _LOCAL_BASE)
        return

    remote_key = relative.as_posix()

    try:
        if _REMOTE_URL.startswith("s3://"):
            _upload_s3(local_path, _REMOTE_URL, remote_key)
        elif _REMOTE_URL.startswith("gs://"):
            _upload_gcs(local_path, _REMOTE_URL, remote_key)
        elif _REMOTE_URL.startswith("az://"):
            _upload_azure(local_path, _REMOTE_URL, remote_key)
        elif _REMOTE_URL.startswith("file://"):
            _upload_local(local_path, _REMOTE_URL, remote_key)
        else:
            log.warning("upload: unsupported scheme in ARTIFACT_REMOTE_URL=%r", _REMOTE_URL)
    except Exception as exc:
        # Non-fatal — local copy is authoritative; remote is durability backup.
        log.error("upload failed for %s: %s", remote_key, exc)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _upload_local(local_path: Path, remote_url: str, key: str) -> None:
    """Copy to a local/NAS path — file:///mnt/nas/algoforge or file://nas-host/share."""
    import shutil

    # file:///abs/path  →  /abs/path
    # file://host/share →  //host/share  (UNC on Windows)
    dest_base = Path(remote_url[7:])   # strip "file://"
    dest = dest_base / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest)
    log.debug("copied → %s", dest)


def _upload_s3(local_path: Path, remote_url: str, key: str) -> None:
    import boto3

    # remote_url = "s3://bucket" or "s3://bucket/prefix"
    without_scheme = remote_url[5:]                    # strip "s3://"
    bucket, _, prefix = without_scheme.partition("/")
    full_key = f"{prefix}/{key}" if prefix else key

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, full_key)
    log.debug("uploaded → s3://%s/%s", bucket, full_key)


def _upload_gcs(local_path: Path, remote_url: str, key: str) -> None:
    from google.cloud import storage as gcs

    without_scheme = remote_url[5:]                    # strip "gs://"
    bucket_name, _, prefix = without_scheme.partition("/")
    blob_name = f"{prefix}/{key}" if prefix else key

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    log.debug("uploaded → gs://%s/%s", bucket_name, blob_name)


def _upload_azure(local_path: Path, remote_url: str, key: str) -> None:
    from azure.storage.blob import BlobServiceClient

    without_scheme = remote_url[5:]                    # strip "az://"
    container, _, prefix = without_scheme.partition("/")
    blob_name = f"{prefix}/{key}" if prefix else key

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")

    client = BlobServiceClient.from_connection_string(conn_str)
    with open(local_path, "rb") as fh:
        client.get_container_client(container).upload_blob(
            blob_name, fh, overwrite=True
        )
    log.debug("uploaded → az://%s/%s", container, blob_name)
