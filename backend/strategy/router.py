"""Strategy layer — HTTP endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from pagination import Pagination
from schemas import DataResponse, Meta
from strategy.models import (
    StrategyCreate, StrategyUpdate, StrategyRead,
    StrategyRunCreate, StrategyRunRead,
    ChatMessageCreate, ChatMessageRead,
)
from strategy.service import strategy_service
from celery_app import enqueue
from events import event_bus

router = APIRouter(prefix="/strategies", tags=["strategy"])


# ── Strategy CRUD ──────────────────────────────────────────────────────────────

@router.get("", response_model=DataResponse[list[StrategyRead]])
async def list_strategies(
    status: str | None = None,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await strategy_service.list_strategies(db, status=status, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("", response_model=DataResponse[StrategyRead], status_code=201)
async def create_strategy(body: StrategyCreate, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.create_strategy(db, body)
    return DataResponse(data=item)


@router.get("/{strategy_id}", response_model=DataResponse[StrategyRead])
async def get_strategy(strategy_id: int, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.get_strategy(db, strategy_id)
    return DataResponse(data=item)


@router.patch("/{strategy_id}", response_model=DataResponse[StrategyRead])
async def update_strategy(strategy_id: int, body: StrategyUpdate, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.update_strategy(db, strategy_id, body)
    return DataResponse(data=item)


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(strategy_id: int, db: AsyncSession = Depends(get_db)):
    await strategy_service.delete_strategy(db, strategy_id)


# ── Strategy Runs ──────────────────────────────────────────────────────────────

@router.get("/{strategy_id}/runs", response_model=DataResponse[list[StrategyRunRead]])
async def list_runs(
    strategy_id: int,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await strategy_service.list_runs(db, strategy_id, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("/{strategy_id}/runs", response_model=DataResponse[StrategyRunRead], status_code=202)
async def start_run(strategy_id: int, body: StrategyRunCreate, db: AsyncSession = Depends(get_db)):
    run = await strategy_service.create_run(db, strategy_id, body)
    await enqueue("execute_strategy_run", run.id)
    return DataResponse(data=run)


@router.get("/{strategy_id}/runs/compare")
async def compare_runs(
    strategy_id: int,
    run_ids: str,
    db: AsyncSession = Depends(get_db),
):
    ids = [int(x.strip()) for x in run_ids.split(",")]
    comparison = await strategy_service.compare_runs(db, strategy_id, ids)
    return DataResponse(data=comparison)


@router.get("/{strategy_id}/runs/{run_id}", response_model=DataResponse[StrategyRunRead])
async def get_run(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.get_run(db, strategy_id, run_id)
    return DataResponse(data=item)


@router.post("/{strategy_id}/runs/{run_id}/stop", response_model=DataResponse[StrategyRunRead])
async def stop_run(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.stop_run(db, strategy_id, run_id)
    return DataResponse(data=item)


@router.delete("/{strategy_id}/runs/{run_id}", status_code=204)
async def delete_run(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    await strategy_service.delete_run(db, strategy_id, run_id)


@router.get("/{strategy_id}/runs/{run_id}/metrics", response_model=DataResponse[dict])
async def get_run_metrics(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    metrics = await strategy_service.get_metrics(db, strategy_id, run_id)
    return DataResponse(data=metrics)


@router.get("/{strategy_id}/runs/{run_id}/trades", response_model=DataResponse[list])
async def get_run_trades(
    strategy_id: int,
    run_id: int,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    trades, total = await strategy_service.get_trades(
        db, strategy_id, run_id,
        offset=pagination.offset,
        limit=pagination.page_size,
    )
    return DataResponse(data=trades, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.get("/{strategy_id}/runs/{run_id}/chat", response_model=DataResponse[list[ChatMessageRead]])
async def get_chat_history(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    items = await strategy_service.get_chat_history(db, strategy_id, run_id)
    return DataResponse(data=items)


@router.post("/{strategy_id}/runs/{run_id}/chat", response_model=DataResponse[ChatMessageRead], status_code=201)
async def send_chat_message(strategy_id: int, run_id: int, body: ChatMessageCreate, db: AsyncSession = Depends(get_db)):
    item = await strategy_service.send_chat_message(db, strategy_id, run_id, body)
    return DataResponse(data=item)


@router.get("/{strategy_id}/runs/{run_id}/status")
async def get_run_status(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    run = await strategy_service.get_run(db, strategy_id, run_id)
    return DataResponse(data={"status": run.status, "progress_pct": run.progress_pct, "message": run.message})


@router.get("/{strategy_id}/runs/{run_id}/equity")
async def get_equity_curve(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    curve = await strategy_service.get_equity_curve(db, strategy_id, run_id)
    return DataResponse(data=curve)


@router.get("/{strategy_id}/runs/{run_id}/chart-data")
async def get_chart_data(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    """Return OHLC candles + indicator series + trade markers for the run chart."""
    data = await strategy_service.get_chart_data(db, strategy_id, run_id)
    return DataResponse(data=data)


@router.get("/{strategy_id}/runs/{run_id}/trades/{trade_id}")
async def get_trade_detail(
    strategy_id: int, run_id: int, trade_id: int,
    db: AsyncSession = Depends(get_db),
):
    detail = await strategy_service.get_trade_detail(db, strategy_id, run_id, trade_id)
    return DataResponse(data=detail)


@router.get("/{strategy_id}/runs/{run_id}/events")
async def stream_run_events(strategy_id: int, run_id: int):
    async def generator():
        import json
        async with event_bus.subscribe(f"run:{run_id}") as queue:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")


@router.get("/{strategy_id}/versions")
async def list_versions(strategy_id: int, db: AsyncSession = Depends(get_db)):
    versions = await strategy_service.list_versions(db, strategy_id)
    from strategy.models import StrategyVersionRead
    return DataResponse(data=[StrategyVersionRead.model_validate(v) for v in versions])
