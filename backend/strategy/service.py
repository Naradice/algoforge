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
    async def list_strategies(self, db: AsyncSession, status: str | None = None) -> list[Strategy]:
        return await strategy_repo.get_all(db, status=status)

    async def get_strategy(self, db: AsyncSession, strategy_id: int) -> Strategy:
        obj = await strategy_repo.get_by_id(db, strategy_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=STRATEGY_NOT_FOUND)
        return obj

    async def create_strategy(self, db: AsyncSession, body: StrategyCreate) -> Strategy:
        return await strategy_repo.create(db, name=body.name, description=body.description, definition=body.definition)

    async def update_strategy(self, db: AsyncSession, strategy_id: int, body: StrategyUpdate) -> Strategy:
        await self.get_strategy(db, strategy_id)  # raises 404 if not found
        updates = body.model_dump(exclude_none=True)
        if not updates:
            return await self.get_strategy(db, strategy_id)
        return await strategy_repo.update(db, strategy_id, **updates)

    async def delete_strategy(self, db: AsyncSession, strategy_id: int) -> None:
        obj = await self.get_strategy(db, strategy_id)
        await db.delete(obj)

    async def list_runs(self, db: AsyncSession, strategy_id: int) -> list[StrategyRun]:
        await self.get_strategy(db, strategy_id)
        return await strategy_repo.get_runs(db, strategy_id)

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

    async def get_trades(self, db: AsyncSession, strategy_id: int, run_id: int) -> list:
        await self.get_run(db, strategy_id, run_id)
        return await strategy_repo.get_trades(db, run_id)

    async def get_chat_history(self, db: AsyncSession, strategy_id: int, run_id: int) -> list:
        await self.get_run(db, strategy_id, run_id)
        return await strategy_repo.get_chat_history(db, run_id)

    async def send_chat_message(self, db: AsyncSession, strategy_id: int, run_id: int, body: ChatMessageCreate) -> ChatMessageRead:
        await self.get_run(db, strategy_id, run_id)
        msg = await strategy_repo.add_chat_message(db, run_id=run_id, role="user", message=body.message)
        return ChatMessageRead.model_validate(msg)


strategy_service = StrategyService()
