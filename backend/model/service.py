"""ML Model layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from model.models import MLModel, MLModelCreate, MLModelUpdate, TrainingRun, TrainingRunCreate
from model.repository import model_repo

MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
TRAINING_RUN_NOT_FOUND = "TRAINING_RUN_NOT_FOUND"
MODEL_NOT_DEPLOYED = "MODEL_NOT_DEPLOYED"


class ModelService:
    async def list_models(self, db: AsyncSession, status: str | None = None, architecture: str | None = None) -> list[MLModel]:
        return await model_repo.get_all(db, status=status, architecture=architecture)

    async def get_model(self, db: AsyncSession, model_id: int) -> MLModel:
        obj = await model_repo.get_by_id(db, model_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=MODEL_NOT_FOUND)
        return obj

    async def create_model(self, db: AsyncSession, body: MLModelCreate) -> MLModel:
        return await model_repo.create(db, name=body.name, architecture=body.architecture, config=body.config)

    async def update_model(self, db: AsyncSession, model_id: int, body: MLModelUpdate) -> MLModel:
        await self.get_model(db, model_id)
        updates = body.model_dump(exclude_none=True)
        return await model_repo.update(db, model_id, **updates)

    async def delete_model(self, db: AsyncSession, model_id: int) -> None:
        obj = await self.get_model(db, model_id)
        await db.delete(obj)

    async def deploy_model(self, db: AsyncSession, model_id: int, training_run_id: int) -> MLModel:
        model = await self.get_model(db, model_id)
        training_run = await model_repo.get_training_run(db, training_run_id)
        if training_run is None or training_run.model_id != model_id:
            raise HTTPException(status_code=404, detail=TRAINING_RUN_NOT_FOUND)
        if training_run.status != "completed":
            raise HTTPException(status_code=422, detail="Training run is not completed")
        return await model_repo.update(db, model_id, status="deployed", artifact_path=training_run.artifact_path)

    async def list_training_runs(self, db: AsyncSession, model_id: int) -> list[TrainingRun]:
        await self.get_model(db, model_id)
        return await model_repo.get_training_runs(db, model_id)

    async def get_training_run(self, db: AsyncSession, model_id: int, run_id: int) -> TrainingRun:
        run = await model_repo.get_training_run(db, run_id)
        if run is None or run.model_id != model_id:
            raise HTTPException(status_code=404, detail=TRAINING_RUN_NOT_FOUND)
        return run

    async def create_training_run(self, db: AsyncSession, model_id: int, body: TrainingRunCreate) -> TrainingRun:
        await self.get_model(db, model_id)
        return await model_repo.create_training_run(db, model_id=model_id, dataset_id=body.dataset_id, hyperparams=body.hyperparams)

    async def get_validations(self, db: AsyncSession, model_id: int) -> list:
        await self.get_model(db, model_id)
        return await model_repo.get_validations(db, model_id)


model_service = ModelService()
