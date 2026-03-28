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
    async def list_models(self, db: AsyncSession, status: str | None = None, architecture: str | None = None, offset: int = 0, limit: int = 20) -> tuple[list[MLModel], int]:
        return await model_repo.get_all(db, status=status, architecture=architecture, offset=offset, limit=limit)

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

    async def list_training_runs(self, db: AsyncSession, model_id: int, offset: int = 0, limit: int = 20) -> tuple[list[TrainingRun], int]:
        await self.get_model(db, model_id)
        return await model_repo.get_training_runs(db, model_id, offset=offset, limit=limit)

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

    async def get_training_run_by_id(self, db: AsyncSession, run_id: int):
        run = await model_repo.get_training_run_by_id(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=TRAINING_RUN_NOT_FOUND)
        return run

    async def list_epoch_metrics(self, db: AsyncSession, run_id: int) -> list:
        await self.get_training_run_by_id(db, run_id)
        return await model_repo.get_epoch_metrics(db, run_id)

    async def stop_training_run(self, db: AsyncSession, run_id: int):
        run = await self.get_training_run_by_id(db, run_id)
        if run.status not in ("pending", "running"):
            raise HTTPException(status_code=422, detail="Training run is not active")
        return await model_repo.update_training_run(db, run_id, stop_requested=True)

    async def compare_training_runs(self, db: AsyncSession, run_ids: list[int]) -> list:
        result = []
        for rid in run_ids:
            try:
                run = await self.get_training_run_by_id(db, rid)
                result.append({
                    "run_id": run.id,
                    "model_id": run.model_id,
                    "dataset_id": run.dataset_id,
                    "hyperparams": run.hyperparams,
                    "status": run.status,
                    "best_epoch": run.best_epoch,
                    "val_loss": run.val_loss,
                    "artifact_path": run.artifact_path,
                })
            except HTTPException:
                result.append({"run_id": rid, "error": "not found"})
        return result

    async def create_search_runs(self, db: AsyncSession, body) -> list[int]:
        import itertools
        keys = list(body.search_grid.keys())
        values = list(body.search_grid.values())
        run_ids = []
        for combo in itertools.product(*values):
            hyperparams = dict(zip(keys, combo))
            from model.models import TrainingRunCreate
            run = await self.create_training_run(db, body.model_id, TrainingRunCreate(dataset_id=body.dataset_id, hyperparams=hyperparams))
            run_ids.append(run.id)
        return run_ids

    async def create_validation_job(self, db: AsyncSession, model_id: int, body):
        from model.models import ModelValidation
        model = await self.get_model(db, model_id)
        val = ModelValidation(model_id=model_id, training_run_id=body.training_run_id, dataset_id=body.dataset_id, metrics={})
        db.add(val)
        await db.flush()
        await db.refresh(val)
        return val


model_service = ModelService()
