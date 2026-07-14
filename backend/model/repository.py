"""ML Model layer — database queries."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from model.models import MLModel, TrainingRun, TrainingCheckpoint, ModelValidation, PreprocessedDataset


class ModelRepository:
    async def get_all(
        self,
        db: AsyncSession,
        status: str | None = None,
        architecture: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[MLModel], int]:
        q = select(MLModel)
        if status:
            q = q.where(MLModel.status == status)
        if architecture:
            q = q.where(MLModel.architecture == architecture)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(MLModel.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def get_by_id(self, db: AsyncSession, model_id: int) -> MLModel | None:
        result = await db.execute(select(MLModel).where(MLModel.id == model_id))
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, **kwargs) -> MLModel:
        obj = MLModel(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update(self, db: AsyncSession, model_id: int, **kwargs) -> MLModel | None:
        await db.execute(update(MLModel).where(MLModel.id == model_id).values(**kwargs))
        return await self.get_by_id(db, model_id)

    async def get_training_run(self, db: AsyncSession, run_id: int) -> TrainingRun | None:
        result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
        return result.scalar_one_or_none()

    async def get_training_runs(
        self,
        db: AsyncSession,
        model_id: int,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[TrainingRun], int]:
        q = select(TrainingRun).where(TrainingRun.model_id == model_id)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(TrainingRun.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def create_training_run(self, db: AsyncSession, **kwargs) -> TrainingRun:
        obj = TrainingRun(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update_training_run(self, db: AsyncSession, run_id: int, **kwargs) -> TrainingRun | None:
        await db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(**kwargs))
        return await self.get_training_run(db, run_id)

    async def get_validations(self, db: AsyncSession, model_id: int) -> list[ModelValidation]:
        result = await db.execute(
            select(ModelValidation).where(ModelValidation.model_id == model_id).order_by(ModelValidation.computed_at.desc())
        )
        return list(result.scalars().all())

    async def get_training_run_by_id(self, db: AsyncSession, run_id: int):
        from model.models import TrainingRun
        result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
        return result.scalar_one_or_none()

    async def get_latest_validation_for_run(self, db: AsyncSession, training_run_id: int) -> ModelValidation | None:
        result = await db.execute(
            select(ModelValidation)
            .where(ModelValidation.training_run_id == training_run_id)
            .order_by(ModelValidation.computed_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def get_epoch_metrics(self, db: AsyncSession, training_run_id: int) -> list:
        from model.models import TrainingRunMetric
        result = await db.execute(
            select(TrainingRunMetric)
            .where(TrainingRunMetric.training_run_id == training_run_id)
            .order_by(TrainingRunMetric.epoch)
        )
        return list(result.scalars().all())

    async def get_preprocessed_datasets(
        self,
        db: AsyncSession,
        dataset_id: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[PreprocessedDataset], int]:
        q = select(PreprocessedDataset)
        if dataset_id is not None:
            q = q.where(PreprocessedDataset.dataset_id == dataset_id)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(PreprocessedDataset.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

    async def get_preprocessed_dataset_by_id(self, db: AsyncSession, preprocessed_dataset_id: int) -> PreprocessedDataset | None:
        result = await db.execute(select(PreprocessedDataset).where(PreprocessedDataset.id == preprocessed_dataset_id))
        return result.scalar_one_or_none()

    async def create_preprocessed_dataset(self, db: AsyncSession, **kwargs) -> PreprocessedDataset:
        obj = PreprocessedDataset(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update_preprocessed_dataset(self, db: AsyncSession, preprocessed_dataset_id: int, **kwargs) -> PreprocessedDataset | None:
        await db.execute(update(PreprocessedDataset).where(PreprocessedDataset.id == preprocessed_dataset_id).values(**kwargs))
        return await self.get_preprocessed_dataset_by_id(db, preprocessed_dataset_id)

    async def delete_preprocessed_dataset(self, db: AsyncSession, preprocessed_dataset_id: int) -> None:
        obj = await self.get_preprocessed_dataset_by_id(db, preprocessed_dataset_id)
        if obj is not None:
            await db.delete(obj)


model_repo = ModelRepository()
