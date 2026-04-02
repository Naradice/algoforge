"""Data Management layer — database queries."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Datasource, Dataset, CollectionJob, DataCharacteristics


class DataRepository:
    async def get_datasources(self, db: AsyncSession, offset: int = 0, limit: int = 20) -> tuple[list[Datasource], int]:
        q = select(Datasource)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(Datasource.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def get_datasource(self, db: AsyncSession, datasource_id: int) -> Datasource | None:
        result = await db.execute(select(Datasource).where(Datasource.id == datasource_id))
        return result.scalar_one_or_none()

    async def create_datasource(self, db: AsyncSession, **kwargs) -> Datasource:
        obj = Datasource(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update_datasource(self, db: AsyncSession, datasource_id: int, **kwargs) -> Datasource | None:
        await db.execute(update(Datasource).where(Datasource.id == datasource_id).values(**kwargs))
        return await self.get_datasource(db, datasource_id)

    async def delete_datasource(self, db: AsyncSession, datasource_id: int) -> None:
        obj = await self.get_datasource(db, datasource_id)
        if obj:
            await db.delete(obj)

    async def get_datasets(self, db: AsyncSession, symbol: str | None = None, timeframe: str | None = None, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list[Dataset], int]:
        q = select(Dataset)
        if symbol:
            q = q.where(Dataset.symbol == symbol)
        if timeframe:
            q = q.where(Dataset.timeframe == timeframe)
        if datasource_id is not None:
            q = q.where(Dataset.datasource_id == datasource_id)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(Dataset.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Dataset | None:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalar_one_or_none()

    async def create_dataset(self, db: AsyncSession, **kwargs) -> Dataset:
        obj = Dataset(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update_dataset(self, db: AsyncSession, dataset_id: int, **kwargs) -> Dataset | None:
        await db.execute(update(Dataset).where(Dataset.id == dataset_id).values(**kwargs))
        return await self.get_dataset(db, dataset_id)

    async def get_collection_jobs(self, db: AsyncSession, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list[CollectionJob], int]:
        q = select(CollectionJob)
        if datasource_id:
            q = q.where(CollectionJob.datasource_id == datasource_id)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(CollectionJob.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def get_collection_job(self, db: AsyncSession, job_id: int) -> CollectionJob | None:
        result = await db.execute(select(CollectionJob).where(CollectionJob.id == job_id))
        return result.scalar_one_or_none()

    async def create_collection_job(self, db: AsyncSession, **kwargs) -> CollectionJob:
        obj = CollectionJob(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def get_characteristics(self, db: AsyncSession, dataset_id: int) -> DataCharacteristics | None:
        result = await db.execute(
            select(DataCharacteristics).where(DataCharacteristics.dataset_id == dataset_id).order_by(DataCharacteristics.computed_at.desc())
        )
        return result.scalars().first()

    async def update_collection_job(self, db: AsyncSession, job_id: int, **kwargs):
        from data.models import CollectionJob
        await db.execute(update(CollectionJob).where(CollectionJob.id == job_id).values(**kwargs))
        return await self.get_collection_job(db, job_id)

    async def delete_collection_job(self, db: AsyncSession, job_id: int):
        from data.models import CollectionJob
        obj = await self.get_collection_job(db, job_id)
        if obj:
            await db.delete(obj)

    async def get_job_runs(self, db: AsyncSession, job_id: int) -> list:
        from data.models import CollectionJobRun
        result = await db.execute(
            select(CollectionJobRun).where(CollectionJobRun.job_id == job_id).order_by(CollectionJobRun.started_at.desc())
        )
        return list(result.scalars().all())

    async def delete_dataset(self, db: AsyncSession, dataset_id: int):
        from data.models import Dataset
        obj = await self.get_dataset(db, dataset_id)
        if obj:
            await db.delete(obj)


data_repo = DataRepository()
