"""Strategy layer — HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from .models import (
    StrategyCreate, StrategyUpdate, StrategyRead,
    StrategyRunCreate, StrategyRunRead,
    ChatMessageCreate, ChatMessageRead,
)
from .service import strategy_service

router = APIRouter(prefix="/strategies", tags=["strategy"])


# ── Strategy CRUD ──────────────────────────────────────────────────────────────

@router.get("", response_model=list[StrategyRead])
async def list_strategies(status: str | None = None, db: AsyncSession = Depends(get_db)):
    return await strategy_service.list_strategies(db, status=status)


@router.post("", response_model=StrategyRead, status_code=201)
async def create_strategy(body: StrategyCreate, db: AsyncSession = Depends(get_db)):
    return await strategy_service.create_strategy(db, body)


@router.get("/{strategy_id}", response_model=StrategyRead)
async def get_strategy(strategy_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.get_strategy(db, strategy_id)


@router.patch("/{strategy_id}", response_model=StrategyRead)
async def update_strategy(strategy_id: int, body: StrategyUpdate, db: AsyncSession = Depends(get_db)):
    return await strategy_service.update_strategy(db, strategy_id, body)


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(strategy_id: int, db: AsyncSession = Depends(get_db)):
    await strategy_service.delete_strategy(db, strategy_id)


# ── Strategy Runs ──────────────────────────────────────────────────────────────

@router.get("/{strategy_id}/runs", response_model=list[StrategyRunRead])
async def list_runs(strategy_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.list_runs(db, strategy_id)


@router.post("/{strategy_id}/runs", response_model=StrategyRunRead, status_code=202)
async def start_run(strategy_id: int, body: StrategyRunCreate, db: AsyncSession = Depends(get_db)):
    run = await strategy_service.create_run(db, strategy_id, body)
    # TODO Phase 3: enqueue execute_strategy_run arq job
    return run


@router.get("/{strategy_id}/runs/{run_id}", response_model=StrategyRunRead)
async def get_run(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.get_run(db, strategy_id, run_id)


@router.get("/{strategy_id}/runs/{run_id}/metrics")
async def get_run_metrics(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.get_metrics(db, strategy_id, run_id)


@router.get("/{strategy_id}/runs/{run_id}/trades")
async def get_run_trades(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.get_trades(db, strategy_id, run_id)


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.get("/{strategy_id}/runs/{run_id}/chat", response_model=list[ChatMessageRead])
async def get_chat_history(strategy_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    return await strategy_service.get_chat_history(db, strategy_id, run_id)


@router.post("/{strategy_id}/runs/{run_id}/chat", response_model=ChatMessageRead, status_code=201)
async def send_chat_message(strategy_id: int, run_id: int, body: ChatMessageCreate, db: AsyncSession = Depends(get_db)):
    return await strategy_service.send_chat_message(db, strategy_id, run_id, body)
