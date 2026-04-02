"""Strategy layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from strategy.models import (
    Strategy, StrategyCreate, StrategyUpdate,
    StrategyRun, StrategyRunCreate, StrategyRunRead,
    ChatMessageCreate, ChatMessageRead,
)
from strategy.repository import strategy_repo

# Error codes (used in HTTPException detail)
STRATEGY_NOT_FOUND = "STRATEGY_NOT_FOUND"
STRATEGY_RUN_NOT_FOUND = "STRATEGY_RUN_NOT_FOUND"
STRATEGY_RUN_ACTIVE = "STRATEGY_RUN_ACTIVE"


class StrategyService:
    async def list_strategies(self, db: AsyncSession, status: str | None = None, offset: int = 0, limit: int = 20) -> tuple[list[Strategy], int]:
        return await strategy_repo.get_all(db, status=status, offset=offset, limit=limit)

    async def get_strategy(self, db: AsyncSession, strategy_id: int) -> Strategy:
        obj = await strategy_repo.get_by_id(db, strategy_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=STRATEGY_NOT_FOUND)
        return obj

    async def create_strategy(self, db: AsyncSession, body: StrategyCreate) -> Strategy:
        return await strategy_repo.create(db, name=body.name, description=body.description, definition=body.definition)

    async def update_strategy(self, db: AsyncSession, strategy_id: int, body: StrategyUpdate) -> Strategy:
        return await self.update_strategy_with_version(db, strategy_id, body)

    async def delete_strategy(self, db: AsyncSession, strategy_id: int) -> None:
        obj = await self.get_strategy(db, strategy_id)
        await db.delete(obj)

    async def list_runs(self, db: AsyncSession, strategy_id: int, offset: int = 0, limit: int = 20) -> tuple[list[StrategyRun], int]:
        await self.get_strategy(db, strategy_id)
        return await strategy_repo.get_runs(db, strategy_id, offset=offset, limit=limit)

    async def get_run(self, db: AsyncSession, strategy_id: int, run_id: int) -> StrategyRun:
        run = await strategy_repo.get_run(db, run_id)
        if run is None or run.strategy_id != strategy_id:
            raise HTTPException(status_code=404, detail=STRATEGY_RUN_NOT_FOUND)
        return run

    async def create_run(self, db: AsyncSession, strategy_id: int, body: StrategyRunCreate) -> StrategyRun:
        await self.get_strategy(db, strategy_id)
        return await strategy_repo.create_run(
            db,
            strategy_id=strategy_id,
            mode=body.mode,
            dataset_id=body.dataset_id,
            broker_client=body.broker_client,
            from_ts=body.from_ts,
            to_ts=body.to_ts,
        )

    async def get_metrics(self, db: AsyncSession, strategy_id: int, run_id: int) -> dict:
        await self.get_run(db, strategy_id, run_id)
        return await strategy_repo.get_metrics(db, run_id)

    async def get_trades(self, db: AsyncSession, strategy_id: int, run_id: int, offset: int = 0, limit: int = 100) -> tuple[list, int]:
        await self.get_run(db, strategy_id, run_id)
        return await strategy_repo.get_trades(db, run_id, offset=offset, limit=limit)

    async def get_chat_history(self, db: AsyncSession, strategy_id: int, run_id: int) -> list:
        await self.get_run(db, strategy_id, run_id)
        return await strategy_repo.get_chat_history(db, run_id)

    async def send_chat_message(self, db: AsyncSession, strategy_id: int, run_id: int, body: ChatMessageCreate) -> ChatMessageRead:
        await self.get_run(db, strategy_id, run_id)
        msg = await strategy_repo.add_chat_message(db, run_id=run_id, role="user", message=body.message)
        return ChatMessageRead.model_validate(msg)

    async def stop_run(self, db: AsyncSession, strategy_id: int, run_id: int) -> StrategyRun:
        run = await self.get_run(db, strategy_id, run_id)
        if run.status not in ("pending", "running"):
            raise HTTPException(status_code=422, detail="Run is not active")
        return await strategy_repo.update_run(db, run_id, status="stopped")

    async def get_equity_curve(self, db: AsyncSession, strategy_id: int, run_id: int) -> list:
        run = await self.get_run(db, strategy_id, run_id)
        return run.equity_curve or []

    async def compare_runs(self, db: AsyncSession, strategy_id: int, run_ids: list[int]) -> dict:
        # Verify all runs belong to this strategy
        for rid in run_ids:
            await self.get_run(db, strategy_id, rid)
        metrics_map = await strategy_repo.get_run_metrics_multi(db, run_ids)
        return {str(rid): metrics_map.get(rid, {}) for rid in run_ids}

    async def get_trade_detail(self, db: AsyncSession, strategy_id: int, run_id: int, trade_id: int) -> dict:
        await self.get_run(db, strategy_id, run_id)
        trade = await strategy_repo.get_trade_by_id(db, trade_id)
        if trade is None or trade.run_id != run_id:
            raise HTTPException(status_code=404, detail="TRADE_NOT_FOUND")
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "volume": trade.volume,
            "sl_price": trade.sl_price,
            "tp_price": trade.tp_price,
            "profit": trade.profit,
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
        }

    async def list_versions(self, db: AsyncSession, strategy_id: int) -> list:
        await self.get_strategy(db, strategy_id)
        return await strategy_repo.get_versions(db, strategy_id)

    async def update_strategy_with_version(self, db: AsyncSession, strategy_id: int, body) -> "Strategy":
        existing = await self.get_strategy(db, strategy_id)
        updates = body.model_dump(exclude_none=True)
        if not updates:
            return existing
        # If definition changed, save a version snapshot
        if "definition" in updates:
            count = await strategy_repo.get_version_count(db, strategy_id)
            await strategy_repo.create_version(db, strategy_id, count + 1, existing.definition)
        return await strategy_repo.update(db, strategy_id, **updates)


strategy_service = StrategyService()
