"""ML Model layer — HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from model.models import (
    MLModelCreate, MLModelUpdate, MLModelRead,
    TrainingRunCreate, TrainingRunRead,
    ModelValidationRead, PredictRequest, PredictResponse,
)
from model.service import model_service

router = APIRouter(prefix="/models", tags=["model"])


# ── Model CRUD ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[MLModelRead])
async def list_models(status: str | None = None, architecture: str | None = None, db: AsyncSession = Depends(get_db)):
    return await model_service.list_models(db, status=status, architecture=architecture)


@router.post("", response_model=MLModelRead, status_code=201)
async def create_model(body: MLModelCreate, db: AsyncSession = Depends(get_db)):
    return await model_service.create_model(db, body)


@router.get("/{model_id}", response_model=MLModelRead)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    return await model_service.get_model(db, model_id)


@router.patch("/{model_id}", response_model=MLModelRead)
async def update_model(model_id: int, body: MLModelUpdate, db: AsyncSession = Depends(get_db)):
    return await model_service.update_model(db, model_id, body)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: int, db: AsyncSession = Depends(get_db)):
    await model_service.delete_model(db, model_id)


@router.post("/{model_id}/deploy", response_model=MLModelRead)
async def deploy_model(model_id: int, training_run_id: int, db: AsyncSession = Depends(get_db)):
    return await model_service.deploy_model(db, model_id, training_run_id)


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id: int, body: PredictRequest, db: AsyncSession = Depends(get_db)):
    # TODO Phase 2: load deployed model artifact and run inference
    raise NotImplementedError("Inference not yet implemented — Phase 2")


# ── Training Runs ──────────────────────────────────────────────────────────────

@router.get("/{model_id}/training-runs", response_model=list[TrainingRunRead])
async def list_training_runs(model_id: int, db: AsyncSession = Depends(get_db)):
    return await model_service.list_training_runs(db, model_id)


@router.post("/{model_id}/training-runs", response_model=TrainingRunRead, status_code=202)
async def start_training_run(model_id: int, body: TrainingRunCreate, db: AsyncSession = Depends(get_db)):
    run = await model_service.create_training_run(db, model_id, body)
    # TODO Phase 2: enqueue train_model arq job
    return run


@router.get("/{model_id}/training-runs/{run_id}", response_model=TrainingRunRead)
async def get_training_run(model_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    return await model_service.get_training_run(db, model_id, run_id)


# ── Validations ────────────────────────────────────────────────────────────────

@router.get("/{model_id}/validations", response_model=list[ModelValidationRead])
async def get_validations(model_id: int, db: AsyncSession = Depends(get_db)):
    return await model_service.get_validations(db, model_id)
