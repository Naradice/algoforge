"""Data Management layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Datasource, DatasourceCreate, DatasourceUpdate, Dataset, CollectionJobCreate
from .repository import data_repo

DATASOURCE_NOT_FOUND = "DATASOURCE_NOT_FOUND"
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
DATASET_NOT_READY = "DATASET_NOT_READY"
COLLECTION_JOB_NOT_FOUND = "COLLECTION_JOB_NOT_FOUND"


class DataService:
    async def list_datasources(self, db: AsyncSession) -> list[Datasource]:
        return await data_repo.get_datasources(db)

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

    async def list_datasets(self, db: AsyncSession, symbol: str | None = None, timeframe: str | None = None) -> list[Dataset]:
        return await data_repo.get_datasets(db, symbol=symbol, timeframe=timeframe)

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Dataset:
        obj = await data_repo.get_dataset(db, dataset_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=DATASET_NOT_FOUND)
        return obj

    async def list_collection_jobs(self, db: AsyncSession, datasource_id: int | None = None) -> list:
        return await data_repo.get_collection_jobs(db, datasource_id=datasource_id)

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


data_service = DataService()
