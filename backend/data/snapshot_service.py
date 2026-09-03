"""
Dataset snapshot export — an immutable, single-file, hash-stamped copy of a dataset's current
contents, for handing to a disconnected external execution environment (e.g. a Colab notebook)
that can't reach the live dataset directly, and which incremental collection keeps mutating
anyway (see docs/data-layer.md). See scripts/export_dataset_snapshot.py for the CLI entry point
and gdrive_export.py for the optional upload step.

Deliberately a plain module (not a Celery task) — the dev Celery worker runs a single
--pool=solo process across all queues, so adding a task there would require restarting it to
pick up the code, which would kill whatever training job that process is mid-run on. This runs
out-of-process from both the backend and the worker.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import DatasetSnapshot
from data.service import data_service

log = logging.getLogger("snapshot_service")

_SNAPSHOT_SUBDIR = "dataset_snapshots"


async def create_snapshot(db: AsyncSession, dataset_id: int) -> DatasetSnapshot:
    """Read the dataset's current artifact (partitioned directory or single file) into one
    Parquet file under ARTIFACT_STORE_PATH/dataset_snapshots/{dataset_id}/, hash it, and record
    a DatasetSnapshot row. Only reads the dataset's own artifact and writes a new file plus one
    new row in a table nothing else writes to — safe to run alongside active collection/training
    jobs.
    """
    dataset, artifact_path = await data_service.resolve_dataset_artifact(db, dataset_id)

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
    out_dir = store / _SNAPSHOT_SUBDIR / str(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp_path = out_dir / f".tmp_{timestamp}.parquet"

    if artifact_path.is_dir():
        table = pa_dataset.dataset(str(artifact_path), format="parquet", partitioning="hive").to_table()
    else:
        table = pq.read_table(str(artifact_path))
    pq.write_table(table, tmp_path)
    row_count = table.num_rows

    digest = hashlib.sha256()
    with tmp_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    hex_digest = digest.hexdigest()

    final_path = out_dir / f"{timestamp}_{hex_digest[:8]}.parquet"
    tmp_path.rename(final_path)
    size_bytes = final_path.stat().st_size

    # .as_posix() — this backend also runs in Linux containers (docker-compose.dev.yml) that
    # resolve artifact_path as Path(ARTIFACT_STORE_PATH) / dataset.artifact_path; a Windows-style
    # backslash path stored here would not resolve there.
    snapshot = DatasetSnapshot(
        dataset_id=dataset_id,
        artifact_path=final_path.relative_to(store).as_posix(),
        sha256=hex_digest,
        row_count=row_count,
        size_bytes=size_bytes,
        status="local",
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    log.info(
        f"dataset {dataset_id}: snapshot {snapshot.id} created at {snapshot.artifact_path} "
        f"({row_count} rows, sha256={hex_digest[:8]})"
    )
    return snapshot


async def upload_snapshot(db: AsyncSession, snapshot_id: int, provider: str = "gdrive") -> DatasetSnapshot:
    """Upload an already-created snapshot to an external provider and record where it landed.
    Only "gdrive" is implemented today — see gdrive_export.py's module docstring for setup."""
    snapshot = (
        await db.execute(select(DatasetSnapshot).where(DatasetSnapshot.id == snapshot_id))
    ).scalar_one_or_none()
    if snapshot is None:
        raise ValueError(f"DatasetSnapshot {snapshot_id} not found")
    if provider != "gdrive":
        raise ValueError(f"Unsupported snapshot export provider: {provider!r}")

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
    local_path = store / snapshot.artifact_path

    from data import gdrive_export
    name = f"dataset_{snapshot.dataset_id}_{Path(snapshot.artifact_path).name}"
    try:
        ref = gdrive_export.upload(local_path, name=name)
    except Exception as exc:
        snapshot.status = "error"
        snapshot.export_provider = provider
        snapshot.export_ref = {"error": str(exc)}
        await db.flush()
        raise

    snapshot.status = "uploaded"
    snapshot.export_provider = provider
    snapshot.export_ref = {**ref, "uploaded_at": datetime.now(timezone.utc).isoformat()}
    await db.flush()
    await db.refresh(snapshot)
    log.info(f"snapshot {snapshot.id}: uploaded to {provider} -> {ref.get('url')}")
    return snapshot
