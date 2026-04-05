"""Data Management layer — business logic."""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Datasource, DatasourceCreate, DatasourceUpdate, Dataset, CollectionJobCreate
from data.repository import data_repo

_PANDAS_OFFSET = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}


def _resample_ticks(tick_series: "pd.Series", timeframe: str) -> "pd.DataFrame":
    """Resample a tick price series to OHLC candles."""
    freq = _PANDAS_OFFSET.get(timeframe, "1min")
    ohlc = tick_series.resample(freq).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = tick_series.resample(freq).count()
    return ohlc.dropna()

DATASOURCE_NOT_FOUND = "DATASOURCE_NOT_FOUND"
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
DATASET_NOT_READY = "DATASET_NOT_READY"
COLLECTION_JOB_NOT_FOUND = "COLLECTION_JOB_NOT_FOUND"


def _revoke_collection_task(job_id: int) -> None:
    """Clear the dedup lock for a collection job so it can be re-enqueued.

    Celery task IDs are now UUIDs (not deterministic), so we cannot revoke the
    old task by ID.  We only delete the Redis lock key so enqueue() won't raise
    AlreadyRunningError.  The old worker process will keep running until the
    collection container is restarted, but the new run will write fresh data
    after _clear_artifact_dir() removes the stale partitions.
    Best-effort: failures are silently ignored.
    """
    import os
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.delete(f"algoforge:enqueued:run_collection_job:{job_id}")
    except Exception:
        pass


class DataService:
    async def list_datasources(self, db: AsyncSession, offset: int = 0, limit: int = 20) -> tuple[list[Datasource], int]:
        return await data_repo.get_datasources(db, offset=offset, limit=limit)

    async def get_datasource(self, db: AsyncSession, datasource_id: int) -> Datasource:
        obj = await data_repo.get_datasource(db, datasource_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=DATASOURCE_NOT_FOUND)
        return obj

    async def create_datasource(self, db: AsyncSession, body: DatasourceCreate) -> Datasource:
        return await data_repo.create_datasource(db, name=body.name, type=body.type, config=body.config)

    async def update_datasource(self, db: AsyncSession, datasource_id: int, body: DatasourceUpdate) -> Datasource:
        await self.get_datasource(db, datasource_id)
        updates = body.model_dump(exclude_none=True)
        return await data_repo.update_datasource(db, datasource_id, **updates)

    async def delete_datasource(self, db: AsyncSession, datasource_id: int) -> None:
        await self.get_datasource(db, datasource_id)
        await data_repo.delete_datasource(db, datasource_id)

    async def list_datasets(self, db: AsyncSession, symbol: str | None = None, timeframe: str | None = None, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list[Dataset], int]:
        return await data_repo.get_datasets(db, symbol=symbol, timeframe=timeframe, datasource_id=datasource_id, offset=offset, limit=limit)

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Dataset:
        obj = await data_repo.get_dataset(db, dataset_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=DATASET_NOT_FOUND)
        return obj

    async def list_collection_jobs(self, db: AsyncSession, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list, int]:
        return await data_repo.get_collection_jobs(db, datasource_id=datasource_id, offset=offset, limit=limit)

    async def get_collection_job(self, db: AsyncSession, job_id: int):
        obj = await data_repo.get_collection_job(db, job_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=COLLECTION_JOB_NOT_FOUND)
        return obj

    async def create_collection_job(self, db: AsyncSession, body: CollectionJobCreate):
        await self.get_datasource(db, body.datasource_id)
        return await data_repo.create_collection_job(db, datasource_id=body.datasource_id, schedule_cron=body.schedule_cron)

    async def get_characteristics(self, db: AsyncSession, dataset_id: int):
        await self.get_dataset(db, dataset_id)
        return await data_repo.get_characteristics(db, dataset_id)

    async def trigger_collection(self, db: AsyncSession, job_id: int):
        """Enqueue a run_collection_job Celery task and return the job.

        If the job is already running (status=running or dedup lock held), the
        existing Celery task is revoked first so the new run starts cleanly
        without a concurrent instance writing to the same artifact directory.
        """
        job = await self.get_collection_job(db, job_id)
        _revoke_collection_task(job_id)
        from celery_app import enqueue
        await enqueue("run_collection_job", job_id)
        return job

    async def trigger_analysis(self, db: AsyncSession, dataset_id: int):
        """Enqueue a compute_characteristics Celery task and return the dataset."""
        dataset = await self.get_dataset(db, dataset_id)
        if dataset.status not in ("ready", "running"):
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)
        from celery_app import enqueue
        await enqueue("compute_characteristics", dataset_id)
        return dataset

    async def update_collection_job(self, db: AsyncSession, job_id: int, body):
        job = await self.get_collection_job(db, job_id)
        updates = body.model_dump(exclude_none=True)
        if not updates:
            return job
        return await data_repo.update_collection_job(db, job_id, **updates)

    async def delete_collection_job(self, db: AsyncSession, job_id: int):
        await self.get_collection_job(db, job_id)
        await data_repo.delete_collection_job(db, job_id)

    async def list_job_runs(self, db: AsyncSession, job_id: int) -> list:
        await self.get_collection_job(db, job_id)
        return await data_repo.get_job_runs(db, job_id)

    async def delete_dataset(self, db: AsyncSession, dataset_id: int):
        import os
        from pathlib import Path
        from strategy.models import StrategyRun
        from model.models import TrainingRun
        from sqlalchemy import select

        dataset = await self.get_dataset(db, dataset_id)
        # Check strategy run references
        ref_run = (await db.execute(select(StrategyRun).where(StrategyRun.dataset_id == dataset_id).limit(1))).scalar_one_or_none()
        if ref_run:
            raise HTTPException(status_code=409, detail={"code": "DATASET_IN_USE", "message": f"Dataset is referenced by strategy run {ref_run.id}"})
        ref_tr = (await db.execute(select(TrainingRun).where(TrainingRun.dataset_id == dataset_id).limit(1))).scalar_one_or_none()
        if ref_tr:
            raise HTTPException(status_code=409, detail={"code": "DATASET_IN_USE", "message": f"Dataset is referenced by training run {ref_tr.id}"})
        await data_repo.delete_dataset(db, dataset_id)
        # Remove artifact file from disk
        if dataset.artifact_path:
            import shutil
            store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
            artifact = store / dataset.artifact_path
            if artifact.is_dir():
                shutil.rmtree(artifact)
            elif artifact.exists():
                artifact.unlink()

    async def create_dataset_from_upload(self, db: AsyncSession, file, datasource_id=None, symbol=None, timeframe=None, col_map: dict | None = None):
        import os
        import uuid
        from pathlib import Path
        import pandas as pd
        from data.models import Dataset

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        dataset_dir = store / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Read CSV
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))

        # Resolve column names: explicit mapping → case-insensitive auto-detect
        col_lower = {c.lower(): c for c in df.columns}
        _ALIASES = {
            "close":    ["close", "adj close", "adjusted close", "price"],
            "open":     ["open"],
            "high":     ["high"],
            "low":      ["low"],
            "volume":   ["volume", "vol"],
            "datetime": ["datetime", "date", "time", "timestamp"],
        }

        def _resolve(field: str) -> str | None:
            """Return the actual DataFrame column name for a logical field."""
            explicit = (col_map or {}).get(field)
            if explicit:
                if explicit in df.columns:
                    return explicit
                if explicit.lower() in col_lower:
                    return col_lower[explicit.lower()]
            for alias in _ALIASES[field]:
                if alias in col_lower:
                    return col_lower[alias]
            return None

        close_src = _resolve("close")
        if close_src is None:
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_CSV",
                "message": f"Could not find a 'close' column. Available columns: {list(df.columns)}",
            })

        rename: dict[str, str] = {}
        for field in ("close", "open", "high", "low", "volume", "datetime"):
            src = _resolve(field)
            if src and src not in rename:
                rename[src] = field

        df = df.rename(columns=rename)
        keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        # Use datetime column as index so characteristics (seasonality etc.) work correctly
        if "datetime" in df.columns:
            df = df.set_index("datetime")
            df.index = pd.to_datetime(df.index)
            df.index.name = "datetime"

        # Save as parquet
        artifact_name = f"datasets/upload_{uuid.uuid4().hex}.parquet"
        full_path = store / artifact_name
        df.to_parquet(full_path)

        # Create DB record
        ds = Dataset(
            datasource_id=datasource_id,
            name=file.filename or "uploaded_dataset",
            symbol=symbol,
            timeframe=timeframe,
            row_count=len(df),
            artifact_path=artifact_name,
            status="ready",
        )
        db.add(ds)
        await db.flush()
        await db.refresh(ds)
        return ds

    async def trigger_datasource_collection(self, db: AsyncSession, datasource_id: int, body=None):
        """Find or create a collection job for this datasource, then enqueue it."""
        from data.models import CollectionJob
        from sqlalchemy import select
        datasource = await self.get_datasource(db, datasource_id)

        # Look for an existing job
        result = await db.execute(select(CollectionJob).where(CollectionJob.datasource_id == datasource_id).limit(1))
        job = result.scalar_one_or_none()

        if job is None:
            job = CollectionJob(datasource_id=datasource_id)
            db.add(job)
            await db.flush()
            await db.refresh(job)

        from celery_app import enqueue
        await enqueue("run_collection_job", job.id)
        return job

    async def get_dataset_preview(self, db: AsyncSession, dataset_id: int, rows: int = 100, timeframe: str | None = None) -> list[dict]:
        """Return up to `rows` OHLC rows of a dataset as a list of dicts.

        DDM tick directories: reads the most-recent batch files, resamples to
        OHLC, and returns the most-recent `rows` candles so an ongoing endless
        simulation always shows the current state.

        File-based OHLC datasets: returns the first `rows` rows.
        """
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd

        dataset = await self.get_dataset(db, dataset_id)
        if dataset.artifact_path is None or dataset.status not in ("ready", "running"):
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        full_path = store / dataset.artifact_path

        is_tick_dir = full_path.is_dir()

        if is_tick_dir:
            # Read only the most-recent fragment files to keep load bounded.
            # 20 files × ~10 000 ticks ≈ 4-5 days of M1 data — enough for preview.
            from data.parquet_reader import load_ddm_ticks_recent
            try:
                tick_df = load_ddm_ticks_recent(full_path, n_files=20)
            except Exception:
                return []

            if tick_df.empty or "price" not in tick_df.columns:
                return []

            tf = timeframe or dataset.timeframe or "M1"
            df = _resample_ticks(tick_df["price"], tf)
            # Return the most-recent candles to reflect current simulation state.
            df = df.tail(rows)

        else:
            df = pd.read_parquet(full_path)

            if "price" in df.columns:
                # Legacy tick file (not a directory) — normalise index then resample.
                if "datetime" in df.columns:
                    df = df.set_index("datetime")
                df.index = pd.to_datetime(df.index, utc=True)
                tf = timeframe or dataset.timeframe or "M1"
                df = _resample_ticks(df["price"], tf)

            df = df.head(rows)

        # Sanitise ±inf so they don't become null in the JSON response.
        df = df.replace([np.inf, -np.inf], np.nan)
        df.index = df.index.astype(str)
        records = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
        return records.to_dict(orient="records")


data_service = DataService()
