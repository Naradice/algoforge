"""Data Management layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Datasource, DatasourceCreate, DatasourceUpdate, Dataset, CollectionJobCreate
from data.repository import data_repo

DATASOURCE_NOT_FOUND = "DATASOURCE_NOT_FOUND"
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
DATASET_NOT_READY = "DATASET_NOT_READY"
COLLECTION_JOB_NOT_FOUND = "COLLECTION_JOB_NOT_FOUND"


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

    async def list_datasets(self, db: AsyncSession, symbol: str | None = None, timeframe: str | None = None, offset: int = 0, limit: int = 20) -> tuple[list[Dataset], int]:
        return await data_repo.get_datasets(db, symbol=symbol, timeframe=timeframe, offset=offset, limit=limit)

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
        """Enqueue a run_collection_job arq task and return the job."""
        job = await self.get_collection_job(db, job_id)
        from arq_pool import enqueue
        await enqueue("run_collection_job", job_id)
        return job

    async def trigger_analysis(self, db: AsyncSession, dataset_id: int):
        """Enqueue a compute_characteristics arq task and return the dataset."""
        dataset = await self.get_dataset(db, dataset_id)
        if dataset.status != "ready":
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)
        from arq_pool import enqueue
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

    async def create_dataset_from_upload(self, db: AsyncSession, file, datasource_id=None, symbol=None, timeframe=None):
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
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            raise HTTPException(status_code=422, detail={"code": "INVALID_CSV", "message": "CSV must contain 'close' column"})

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

        from arq_pool import enqueue
        await enqueue("run_collection_job", job.id)
        return job

    async def get_dataset_preview(self, db: AsyncSession, dataset_id: int, rows: int = 100) -> list[dict]:
        """Return the first `rows` rows of a dataset as a list of dicts."""
        import os
        from pathlib import Path
        import pandas as pd

        dataset = await self.get_dataset(db, dataset_id)
        if dataset.artifact_path is None or dataset.status != "ready":
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        df = pd.read_parquet(store / dataset.artifact_path).head(rows)
        df.index = df.index.astype(str)
        return df.reset_index().rename(columns={df.index.name or "index": "datetime"}).to_dict(orient="records")


data_service = DataService()
