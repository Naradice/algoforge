"""ML Model layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from model.models import (
    MLModel, MLModelCreate, MLModelUpdate, TrainingRun, TrainingRunCreate,
    PreprocessedDataset, PreprocessedDatasetCreate, PreprocessedDatasetUpdate,
    TrainingRunImportCreate,
)
from model.repository import model_repo

MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
TRAINING_RUN_NOT_FOUND = "TRAINING_RUN_NOT_FOUND"
MODEL_NOT_DEPLOYED = "MODEL_NOT_DEPLOYED"
PREPROCESSED_DATASET_NOT_FOUND = "PREPROCESSED_DATASET_NOT_FOUND"
PREPROCESSED_DATASET_IN_USE = "PREPROCESSED_DATASET_IN_USE"
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"


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
        model = await self.get_model(db, model_id)
        dataset_id = body.dataset_id
        if body.preprocessed_dataset_id is not None:
            # The recipe is the single source of truth for which dataset it was built on —
            # override whatever (if anything) the caller sent as dataset_id.
            pd = await self.get_preprocessed_dataset(db, body.preprocessed_dataset_id)
            dataset_id = pd.dataset_id
        if dataset_id is None:
            raise HTTPException(status_code=422, detail="dataset_id or preprocessed_dataset_id is required")

        if body.execution_target not in ("local", "colab"):
            raise HTTPException(
                status_code=422,
                detail=f"execution_target must be 'local' or 'colab', got {body.execution_target!r}",
            )
        if body.execution_target == "colab":
            from model.colab_trainer import check_colab_supported
            check_colab_supported(model.architecture, body.preprocessed_dataset_id, body.hyperparams)

        return await model_repo.create_training_run(
            db, model_id=model_id, dataset_id=dataset_id,
            preprocessed_dataset_id=body.preprocessed_dataset_id, hyperparams=body.hyperparams,
            execution_target=body.execution_target,
        )

    async def import_external_training_run(
        self, db: AsyncSession, model_id: int, checkpoint: UploadFile, body: TrainingRunImportCreate,
    ) -> TrainingRun:
        """Register a training run that was executed outside this backend (e.g. a Colab
        notebook) as a first-class TrainingRun, so it appears in /model/compare and the model
        detail page like any run the celery worker trained itself.

        Mirrors the artifact layout celery_worker.py's _train_model uses
        (store/models/{model_id}/training_{run_id}/best.<ext>) so downstream code (deploy,
        predict) that resolves artifact_path relative to ARTIFACT_STORE_PATH needs no changes.
        """
        import os
        from datetime import datetime, timezone
        from pathlib import Path
        from sqlalchemy import select
        from data.models import Dataset
        from model.models import TrainingCheckpoint, TrainingRunMetric

        await self.get_model(db, model_id)

        ds = (await db.execute(select(Dataset).where(Dataset.id == body.dataset_id))).scalar_one_or_none()
        if ds is None:
            raise HTTPException(status_code=404, detail=DATASET_NOT_FOUND)
        if body.preprocessed_dataset_id is not None:
            await self.get_preprocessed_dataset(db, body.preprocessed_dataset_id)

        hyperparams = dict(body.hyperparams)
        if body.external_ref:
            hyperparams["_external_ref"] = body.external_ref
        if body.notes:
            hyperparams["_external_notes"] = body.notes

        run = await model_repo.create_training_run(
            db,
            model_id=model_id,
            dataset_id=body.dataset_id,
            preprocessed_dataset_id=body.preprocessed_dataset_id,
            hyperparams=hyperparams,
            status="completed",
            source="external",
            current_epoch=body.best_epoch,
            best_epoch=body.best_epoch,
            val_loss=body.val_loss,
            num_params=body.num_params,
            started_at=body.started_at,
            ended_at=body.ended_at or datetime.now(timezone.utc),
        )

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
        checkpoint_dir = store / "models" / str(model_id) / f"training_{run.id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(checkpoint.filename or "best.pt").suffix or ".pt"
        dest = checkpoint_dir / f"best{suffix}"
        with dest.open("wb") as f:
            while chunk := await checkpoint.read(1024 * 1024):
                f.write(chunk)
        # .as_posix() -- this backend also runs in Linux containers (docker-compose.dev.yml)
        # that resolve artifact_path as Path(ARTIFACT_STORE_PATH) / training_run.artifact_path;
        # a Windows-style backslash path stored here would not resolve there (same fix as
        # data/snapshot_service.py's create_snapshot).
        artifact_rel = dest.relative_to(store).as_posix()

        db.add(TrainingCheckpoint(
            training_run_id=run.id,
            epoch=body.best_epoch,
            metrics={"val_loss": body.val_loss, "source": "external"},
            artifact_path=artifact_rel,
        ))
        for m in body.epoch_metrics:
            db.add(TrainingRunMetric(
                training_run_id=run.id, epoch=m.epoch,
                train_loss=m.train_loss, val_loss=m.val_loss, lr=m.lr,
            ))

        return await model_repo.update_training_run(db, run.id, artifact_path=artifact_rel)

    async def get_validations(self, db: AsyncSession, model_id: int) -> list:
        await self.get_model(db, model_id)
        return await model_repo.get_validations(db, model_id)

    async def get_training_run_by_id(self, db: AsyncSession, run_id: int):
        run = await model_repo.get_training_run_by_id(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=TRAINING_RUN_NOT_FOUND)
        return run

    async def get_training_progress(self, db: AsyncSession, run_id: int) -> dict:
        """Single source of truth for "how is this run doing right now" -- used by both the
        REST status endpoint (training_runs_router.get_training_status) and the MCP tool of the
        same name, so an AI agent polling over MCP sees exactly what a human watching the UI
        sees. Epoch-rate ETA math works the same for a "local" and a "colab" run: both update
        started_at/current_epoch the same way (see celery_worker._train_model and
        colab_trainer._poll_and_maybe_stop), so no execution_target branch is needed for that
        part.

        For a "colab" run, also surfaces colab_timeout_seconds (the value passed to `colab exec
        --timeout`, stored in hyperparams -- see colab_trainer.py's _DEFAULT_TIMEOUT_SECONDS
        comment) alongside how much of it is left and whether the epoch-rate ETA fits inside
        that remainder, so a caller can tell "will this finish before Colab kicks it off for
        idling/quota" without separately knowing the timeout value it was started with -- there
        is no API (colab-cli or Google's) that reports remaining Colab compute quota directly,
        so this time-budget comparison is the closest available proxy.
        """
        from datetime import datetime, timezone

        run = await self.get_training_run_by_id(db, run_id)
        hyperparams = run.hyperparams or {}
        total_epochs = hyperparams.get("epochs")

        elapsed = None
        eta = None
        if run.started_at:
            elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
            if run.current_epoch and total_epochs:
                rate = elapsed / max(run.current_epoch, 1)
                eta = rate * (total_epochs - run.current_epoch)

        result = {
            "status": run.status,
            "execution_target": run.execution_target,
            "current_epoch": run.current_epoch,
            "total_epochs": total_epochs,
            "best_epoch": run.best_epoch,
            "val_loss": run.val_loss,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "stop_requested": run.stop_requested,
        }

        if run.execution_target == "colab":
            timeout_seconds = hyperparams.get("colab_timeout_seconds")
            timeout_remaining = (
                timeout_seconds - elapsed if timeout_seconds is not None and elapsed is not None else None
            )
            result["colab_timeout_seconds"] = timeout_seconds
            result["colab_timeout_remaining_seconds"] = timeout_remaining
            result["likely_to_finish_before_timeout"] = (
                eta <= timeout_remaining if eta is not None and timeout_remaining is not None else None
            )

        return result

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
                model_rec = await model_repo.get_by_id(db, run.model_id)
                validation = await model_repo.get_latest_validation_for_run(db, run.id)
                result.append({
                    "run_id": run.id,
                    "model_id": run.model_id,
                    "model_name": model_rec.name if model_rec else None,
                    "architecture": model_rec.architecture if model_rec else None,
                    "dataset_id": run.dataset_id,
                    "preprocessed_dataset_id": run.preprocessed_dataset_id,
                    "hyperparams": run.hyperparams,
                    "status": run.status,
                    "best_epoch": run.best_epoch,
                    "val_loss": run.val_loss,
                    "num_params": run.num_params,
                    "preprocessed_characteristics": run.preprocessed_characteristics,
                    "artifact_path": run.artifact_path,
                    "validation": validation.metrics if validation else None,
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
            run = await self.create_training_run(
                db, body.model_id,
                TrainingRunCreate(
                    dataset_id=body.dataset_id, hyperparams=hyperparams,
                    execution_target=body.execution_target,
                ),
            )
            run_ids.append(run.id)
        return run_ids

    async def list_preprocessed_datasets(
        self, db: AsyncSession, dataset_id: int | None = None, offset: int = 0, limit: int = 20,
    ) -> tuple[list[PreprocessedDataset], int]:
        return await model_repo.get_preprocessed_datasets(db, dataset_id=dataset_id, offset=offset, limit=limit)

    async def get_preprocessed_dataset(self, db: AsyncSession, preprocessed_dataset_id: int) -> PreprocessedDataset:
        obj = await model_repo.get_preprocessed_dataset_by_id(db, preprocessed_dataset_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=PREPROCESSED_DATASET_NOT_FOUND)
        return obj

    async def create_preprocessed_dataset(self, db: AsyncSession, body: PreprocessedDatasetCreate) -> PreprocessedDataset:
        from sqlalchemy import select
        from data.models import Dataset

        ds = (await db.execute(select(Dataset).where(Dataset.id == body.dataset_id))).scalar_one_or_none()
        if ds is None:
            raise HTTPException(status_code=404, detail=DATASET_NOT_FOUND)
        obj = await model_repo.create_preprocessed_dataset(
            db,
            name=body.name,
            dataset_id=body.dataset_id,
            preprocessing=body.preprocessing,
            feature_cols=body.feature_cols,
            normalize=body.normalize,
            status="pending",
        )
        from celery_app import enqueue
        await enqueue("compute_preprocessed_characteristics", obj.id)
        return obj

    async def update_preprocessed_dataset(
        self, db: AsyncSession, preprocessed_dataset_id: int, body: PreprocessedDatasetUpdate,
    ) -> PreprocessedDataset:
        await self.get_preprocessed_dataset(db, preprocessed_dataset_id)
        return await model_repo.update_preprocessed_dataset(db, preprocessed_dataset_id, name=body.name)

    async def delete_preprocessed_dataset(self, db: AsyncSession, preprocessed_dataset_id: int) -> None:
        from sqlalchemy import select

        await self.get_preprocessed_dataset(db, preprocessed_dataset_id)
        ref = (await db.execute(
            select(TrainingRun).where(TrainingRun.preprocessed_dataset_id == preprocessed_dataset_id).limit(1)
        )).scalar_one_or_none()
        if ref is not None:
            raise HTTPException(status_code=409, detail={
                "code": PREPROCESSED_DATASET_IN_USE,
                "message": f"Referenced by training run {ref.id}",
            })
        await model_repo.delete_preprocessed_dataset(db, preprocessed_dataset_id)

    async def recompute_preprocessed_characteristics(self, db: AsyncSession, preprocessed_dataset_id: int) -> PreprocessedDataset:
        obj = await self.get_preprocessed_dataset(db, preprocessed_dataset_id)
        from celery_app import enqueue
        await enqueue("compute_preprocessed_characteristics", preprocessed_dataset_id)
        return obj

    async def create_validation_job(self, db: AsyncSession, model_id: int, body):
        from model.models import ModelValidation
        model = await self.get_model(db, model_id)
        val = ModelValidation(model_id=model_id, training_run_id=body.training_run_id, dataset_id=body.dataset_id, metrics={})
        db.add(val)
        await db.flush()
        await db.refresh(val)
        return val


model_service = ModelService()
