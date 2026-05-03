"""Strategy layer — HTTP endpoints."""

from __future__ import annotations

import asyncio
import functools
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select as sa_select

from database import get_db
from pagination import Pagination
from schemas import DataResponse, Meta
from strategy.models import (
    StrategyCreate, StrategyUpdate, StrategyRead,
    StrategyRunCreate, StrategyRunRead,
    TradeRead,
    ChatMessageCreate, ChatMessageRead,
)
from strategy.service import strategy_service
from celery_app import enqueue
from events import event_bus

router = APIRouter(prefix="/strategies", tags=["strategy"])


# ── Condition frequency stats ─────────────────────────────────────────────────

class ConditionStatsRequest(BaseModel):
    dataset_id: int
    indicators: list[dict]
    conditions: list[dict]


@router.post("/condition-stats", response_model=DataResponse[dict])
async def condition_stats(body: ConditionStatsRequest, db: AsyncSession = Depends(get_db)):
    """Evaluate how often each condition fires across a dataset — vectorised, no full backtest."""
    from data.models import Dataset
    from strategy.engine.backtest import _load_df
    from strategy.engine.indicators import apply_indicators
    from strategy.engine.conditions import eval_condition_series

    result = await db.execute(sa_select(Dataset).where(Dataset.id == body.dataset_id))
    ds = result.scalar_one_or_none()
    if not ds or not ds.artifact_path:
        raise HTTPException(status_code=404, detail="Dataset not found or has no artifact")

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

    def compute() -> dict:
        df = _load_df(store / ds.artifact_path)
        df = apply_indicators(df, body.indicators)
        total = len(df)
        items = []
        mask_all = None

        for i, cond in enumerate(body.conditions):
            series = eval_condition_series(df, cond)
            if series is None:
                items.append({"index": i, "matches": None, "total": total, "pct": None})
            else:
                matches = int(series.sum())
                items.append({"index": i, "matches": matches, "total": total, "pct": round(matches / total * 100, 1) if total else 0.0})
                mask_all = series if mask_all is None else (mask_all & series)

        combined = int(mask_all.sum()) if mask_all is not None else 0
        return {
            "total_bars": total,
            "items": items,
            "combined_matches": combined,
            "combined_pct": round(combined / total * 100, 1) if total else 0.0,
        }

    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, functools.partial(compute))
    return DataResponse(data=stats)


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


@router.post("/{strategy_id}/copy", response_model=DataResponse[StrategyRead], status_code=201)
async def copy_strategy(strategy_id: int, db: AsyncSession = Depends(get_db)):
    src = await strategy_service.get_strategy(db, strategy_id)
    body = StrategyCreate(
        name=f"{src.name} (copy)",
        description=src.description,
        definition=src.definition,
    )
    item = await strategy_service.create_strategy(db, body)
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


@router.get("/{strategy_id}/runs/{run_id}/trades", response_model=DataResponse[list[TradeRead]])
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


@router.post("/{strategy_id}/validate/stream")
async def validate_strategy_stream(
    strategy_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Stream a bar-by-bar validation backtest as SSE events.

    Body: { "dataset_id": <int>, "definition": <optional override dict> }
    Events: init, marker, progress, done, error
    """
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    definition_override = body.get("definition") or None
    limit_bars = int(body["limit_bars"]) if body.get("limit_bars") else None
    gen = strategy_service.validate_strategy_stream(db, strategy_id, int(dataset_id), definition_override, limit_bars=limit_bars)
    return StreamingResponse(gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{strategy_id}/validate")
async def validate_strategy(
    strategy_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Run a synchronous backtest against the given dataset and return chart data.

    Body: { "dataset_id": <int> }
    Returns the same shape as chart-data: { candles, indicators, markers, trade_count }.
    No DB writes — purely ephemeral validation.
    """
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="dataset_id is required")
    definition_override = body.get("definition") or None
    data = await strategy_service.validate_strategy(db, strategy_id, int(dataset_id), definition_override)
    return DataResponse(data=data)


@router.get("/{strategy_id}/runs/{run_id}/chart-data")
async def get_chart_data(
    strategy_id: int,
    run_id: int,
    from_ts: int | None = None,
    to_ts: int | None = None,
    limit: int = 2_000,
    db: AsyncSession = Depends(get_db),
):
    """Return OHLC candles + indicator series + trade markers for the run chart.

    Optional from_ts / to_ts (Unix seconds) restrict the time window.
    limit controls how many bars are returned (default 2000).
    Response includes has_more=true when older data exists before the window.
    """
    data = await strategy_service.get_chart_data(db, strategy_id, run_id, from_ts=from_ts, to_ts=to_ts, limit=limit)
    return DataResponse(data=data)


@router.get("/{strategy_id}/runs/{run_id}/trades/{trade_id}")
async def get_trade_detail(
    strategy_id: int, run_id: int, trade_id: int,
    db: AsyncSession = Depends(get_db),
):
    detail = await strategy_service.get_trade_detail(db, strategy_id, run_id, trade_id)
    return DataResponse(data=detail)


class CostOverlayRequest(BaseModel):
    slippage_pct: float = 0.0
    commission_pct: float = 0.0


@router.post("/{strategy_id}/runs/{run_id}/cost-overlay")
async def cost_overlay(
    strategy_id: int,
    run_id: int,
    body: CostOverlayRequest,
    db: AsyncSession = Depends(get_db),
):
    """Recompute run metrics with different cost assumptions — no re-run needed.

    Returns adjusted metrics plus the original costs used in the run so the
    frontend can initialise sliders to the correct starting values.
    """
    from strategy.engine.backtest import _metrics_for

    run = await strategy_service.get_run(db, strategy_id, run_id)
    strategy = await strategy_service.get_strategy(db, strategy_id)

    risk = {**((strategy.definition or {}).get("risk", {}))}
    if run.risk_override:
        risk.update(run.risk_override)
    orig_slip = float(risk.get("slippage_pct", 0.0005))
    orig_comm = float(risk.get("commission_pct", 0.001))

    trades, _ = await strategy_service.get_trades(db, strategy_id, run_id, limit=100_000)
    if not trades:
        return DataResponse(data={
            "metrics": {}, "orig_metrics": {},
            "orig_slip": orig_slip, "orig_comm": orig_comm,
        })

    # Approximate cost delta: treat slippage as a round-trip cost applied once
    # per trade (entry + exit, same direction), commission once per trade.
    adjusted, orig_dicts = [], []
    for t in trades:
        vol = float(t.volume or 1.0)
        delta = ((orig_comm - body.commission_pct) + 2.0 * (orig_slip - body.slippage_pct)) * vol
        adjusted.append({"profit": float(t.profit or 0.0) + delta})
        orig_dicts.append({"profit": float(t.profit or 0.0)})

    return DataResponse(data={
        "metrics": _metrics_for(adjusted),
        "orig_metrics": _metrics_for(orig_dicts),
        "orig_slip": orig_slip,
        "orig_comm": orig_comm,
    })


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


@router.post("/{strategy_id}/investigate")
async def investigate_strategy(strategy_id: int, body: dict):
    """Stream an AI-driven investigation of the strategy as SSE events.

    Body: { "dataset_id": <int> }
    Events: start | thinking | tool_call | tool_result | done | error
    """
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    from strategy.investigation import stream_investigation
    gen = stream_investigation(strategy_id, int(dataset_id))
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
