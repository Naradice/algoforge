"""ML Model layer — database queries."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import MLModel, TrainingRun, TrainingCheckpoint, ModelValidation


class ModelRepository:
    async def get_all(self, db: AsyncSession, status: str | None = None, architecture: str | None = None) -> list[MLModel]:
        q = select(MLModel)
        if status:
            q = q.where(MLModel.status == status)
        if architecture:
            q = q.where(MLModel.architecture == architecture)
        result = await db.execute(q.order_by(MLModel.created_at.desc()))
        return list(result.scalars().all())

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

    async def get_training_runs(self, db: AsyncSession, model_id: int) -> list[TrainingRun]:
        result = await db.execute(
            select(TrainingRun).where(TrainingRun.model_id == model_id).order_by(TrainingRun.created_at.desc())
        )
        return list(result.scalars().all())

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


model_repo = ModelRepository()
