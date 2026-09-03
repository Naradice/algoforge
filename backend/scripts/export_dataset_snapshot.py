"""Export an immutable snapshot of a dataset, optionally uploading it to Google Drive.

Runs independently of the backend and Celery processes — connects to Postgres directly and
touches no table any running job writes to, so it's safe to use while collection/training jobs
are active. See data/snapshot_service.py's module docstring for why this isn't a Celery task.

Usage (from backend/):
    python -m scripts.export_dataset_snapshot --dataset-id 5
    python -m scripts.export_dataset_snapshot --dataset-id 5 --upload-gdrive

Google Drive upload requires GDRIVE_SERVICE_ACCOUNT_JSON and GDRIVE_SNAPSHOT_FOLDER_ID (and the
google-api-python-client package) — see data/gdrive_export.py's module docstring for one-time
setup. Without --upload-gdrive, the snapshot is written locally only and can be uploaded later
via data.snapshot_service.upload_snapshot(db, snapshot_id).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


async def _run(dataset_id: int, upload_gdrive: bool) -> None:
    from database import async_session_factory
    from data import snapshot_service

    async with async_session_factory() as db:
        snapshot = await snapshot_service.create_snapshot(db, dataset_id)
        await db.commit()
        print(f"Created snapshot {snapshot.id} for dataset {dataset_id}:")
        print(f"  artifact_path: {snapshot.artifact_path}")
        print(f"  rows:          {snapshot.row_count}")
        print(f"  sha256:        {snapshot.sha256}")
        print(f"  size_bytes:    {snapshot.size_bytes}")

        if upload_gdrive:
            snapshot = await snapshot_service.upload_snapshot(db, snapshot.id, provider="gdrive")
            await db.commit()
            print("Uploaded to Google Drive:")
            print(f"  file_id: {snapshot.export_ref.get('file_id')}")
            print(f"  url:     {snapshot.export_ref.get('url')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--upload-gdrive", action="store_true", help="Also upload the snapshot to Google Drive")
    args = parser.parse_args()
    asyncio.run(_run(args.dataset_id, args.upload_gdrive))


if __name__ == "__main__":
    main()
