"""Training Runs — top-level /training-runs endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from events import event_bus
from schemas import DataResponse
from model.models import TrainingRunMetricRead, HyperparamSearchCreate, ValidationCreate
from model.service import model_service
from celery_app import enqueue

tr_router = APIRouter(prefix="/training-runs", tags=["training-runs"])


@tr_router.get("/compare")
async def compare_training_runs(run_ids: str, db: AsyncSession = Depends(get_db)):
    ids = [int(x.strip()) for x in run_ids.split(",")]
    comparison = await model_service.compare_training_runs(db, ids)
    return DataResponse(data=comparison)


@tr_router.post("/search", status_code=202)
async def start_hyperparameter_search(body: HyperparamSearchCreate, db: AsyncSession = Depends(get_db)):
    run_ids = await model_service.create_search_runs(db, body)
    for run_id in run_ids:
        await enqueue("train_model", run_id)
    return DataResponse(data={"run_ids": run_ids})


@tr_router.get("/{run_id}/status")
async def get_training_status(run_id: int, db: AsyncSession = Depends(get_db)):
    from datetime import datetime, timezone
    run = await model_service.get_training_run_by_id(db, run_id)
    elapsed = None
    eta = None
    if run.started_at:
        elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
        total_epochs = run.hyperparams.get("epochs") if run.hyperparams else None
        if elapsed and run.current_epoch and total_epochs:
            rate = elapsed / max(run.current_epoch, 1)
            remaining = total_epochs - run.current_epoch
            eta = rate * remaining
    return DataResponse(data={
        "status": run.status,
        "current_epoch": run.current_epoch,
        "total_epochs": run.hyperparams.get("epochs") if run.hyperparams else None,
        "best_epoch": run.best_epoch,
        "val_loss": run.val_loss,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "stop_requested": run.stop_requested,
    })


@tr_router.get("/{run_id}/metrics", response_model=DataResponse[list[TrainingRunMetricRead]])
async def get_epoch_metrics(run_id: int, db: AsyncSession = Depends(get_db)):
    rows = await model_service.list_epoch_metrics(db, run_id)
    return DataResponse(data=rows)


@tr_router.post("/{run_id}/stop")
async def stop_training_run(run_id: int, db: AsyncSession = Depends(get_db)):
    run = await model_service.stop_training_run(db, run_id)
    return DataResponse(data={"id": run.id, "status": run.status, "stop_requested": run.stop_requested})


@tr_router.get("/{run_id}/events")
async def stream_training_events(run_id: int):
    async def generator():
        import json
        async with event_bus.subscribe(f"training:{run_id}") as queue:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")
