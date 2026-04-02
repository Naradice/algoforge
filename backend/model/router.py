"""ML Model layer — HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from pagination import Pagination
from schemas import DataResponse, Meta
from model.models import (
    MLModelCreate, MLModelUpdate, MLModelRead,
    TrainingRunCreate, TrainingRunRead,
    ModelValidationRead, PredictRequest, PredictResponse,
    ValidationCreate,
)
from model.service import model_service
from celery_app import enqueue

router = APIRouter(prefix="/models", tags=["model"])


# ── Model CRUD ─────────────────────────────────────────────────────────────────

@router.get("", response_model=DataResponse[list[MLModelRead]])
async def list_models(
    status: str | None = None,
    architecture: str | None = None,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await model_service.list_models(db, status=status, architecture=architecture, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("", response_model=DataResponse[MLModelRead], status_code=201)
async def create_model(body: MLModelCreate, db: AsyncSession = Depends(get_db)):
    item = await model_service.create_model(db, body)
    return DataResponse(data=item)


@router.get("/{model_id}", response_model=DataResponse[MLModelRead])
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    item = await model_service.get_model(db, model_id)
    return DataResponse(data=item)


@router.patch("/{model_id}", response_model=DataResponse[MLModelRead])
async def update_model(model_id: int, body: MLModelUpdate, db: AsyncSession = Depends(get_db)):
    item = await model_service.update_model(db, model_id, body)
    return DataResponse(data=item)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: int, db: AsyncSession = Depends(get_db)):
    await model_service.delete_model(db, model_id)


@router.post("/{model_id}/deploy", response_model=DataResponse[MLModelRead])
async def deploy_model(model_id: int, training_run_id: int, db: AsyncSession = Depends(get_db)):
    item = await model_service.deploy_model(db, model_id, training_run_id)
    return DataResponse(data=item)


@router.post("/{model_id}/predict", response_model=DataResponse[PredictResponse])
async def predict(model_id: int, body: PredictRequest, db: AsyncSession = Depends(get_db)):
    import asyncio
    from fastapi import HTTPException
    from model.inference import predict as run_predict

    model_rec = await model_service.get_model(db, model_id)
    if model_rec.status != "deployed" or not model_rec.artifact_path:
        raise HTTPException(status_code=422, detail={"code": "MODEL_NOT_DEPLOYED", "message": "Model is not deployed"})

    loop = asyncio.get_event_loop()
    training_runs, _ = await model_service.list_training_runs(db, model_id, limit=100)
    deployed_run = next((r for r in training_runs if r.artifact_path == model_rec.artifact_path), None)
    hyperparams = deployed_run.hyperparams if deployed_run else {}

    preds = await loop.run_in_executor(
        None,
        run_predict,
        model_id,
        model_rec.architecture,
        model_rec.config,
        model_rec.artifact_path,
        hyperparams,
        body.features,
        body.feature_names,
    )
    return DataResponse(data=PredictResponse(predictions=preds))


# ── Training Runs ──────────────────────────────────────────────────────────────

@router.get("/{model_id}/training-runs", response_model=DataResponse[list[TrainingRunRead]])
async def list_training_runs(
    model_id: int,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await model_service.list_training_runs(db, model_id, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("/{model_id}/training-runs", response_model=DataResponse[TrainingRunRead], status_code=202)
async def start_training_run(model_id: int, body: TrainingRunCreate, db: AsyncSession = Depends(get_db)):
    run = await model_service.create_training_run(db, model_id, body)
    await enqueue("train_model", run.id)
    return DataResponse(data=run)


@router.get("/{model_id}/training-runs/{run_id}", response_model=DataResponse[TrainingRunRead])
async def get_training_run(model_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    item = await model_service.get_training_run(db, model_id, run_id)
    return DataResponse(data=item)


# ── Validations ────────────────────────────────────────────────────────────────

@router.get("/{model_id}/validations", response_model=DataResponse[list[ModelValidationRead]])
async def get_validations(
    model_id: int,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items = await model_service.get_validations(db, model_id)
    total = len(items)
    page_items = items[pagination.offset: pagination.offset + pagination.page_size]
    return DataResponse(data=page_items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("/{model_id}/validations", status_code=202)
async def trigger_validation(model_id: int, body: ValidationCreate, db: AsyncSession = Depends(get_db)):
    val = await model_service.create_validation_job(db, model_id, body)
    await enqueue("validate_model", val.id)
    return DataResponse(data={"id": val.id, "status": "pending"})
