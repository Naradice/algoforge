"""Strategy layer — database queries (SQLAlchemy only, no business logic)."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from strategy.models import Strategy, StrategyRun, StrategyRunCreate, Trade, RunMetric, StrategyRunChat, StrategyEvent


class StrategyRepository:
    async def get_all(self, db: AsyncSession, status: str | None = None) -> list[Strategy]:
        q = select(Strategy)
        if status:
            q = q.where(Strategy.status == status)
        result = await db.execute(q.order_by(Strategy.created_at.desc()))
        return list(result.scalars().all())

    async def get_by_id(self, db: AsyncSession, strategy_id: int) -> Strategy | None:
        result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, **kwargs) -> Strategy:
        obj = Strategy(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update(self, db: AsyncSession, strategy_id: int, **kwargs) -> Strategy | None:
        await db.execute(update(Strategy).where(Strategy.id == strategy_id).values(**kwargs))
        return await self.get_by_id(db, strategy_id)

    async def get_run(self, db: AsyncSession, run_id: int) -> StrategyRun | None:
        result = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))
        return result.scalar_one_or_none()

    async def get_runs(self, db: AsyncSession, strategy_id: int) -> list[StrategyRun]:
        result = await db.execute(
            select(StrategyRun).where(StrategyRun.strategy_id == strategy_id).order_by(StrategyRun.created_at.desc())
        )
        return list(result.scalars().all())

    async def create_run(self, db: AsyncSession, **kwargs) -> StrategyRun:
        obj = StrategyRun(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update_run(self, db: AsyncSession, run_id: int, **kwargs) -> StrategyRun | None:
        await db.execute(update(StrategyRun).where(StrategyRun.id == run_id).values(**kwargs))
        return await self.get_run(db, run_id)

    async def get_trades(self, db: AsyncSession, run_id: int) -> list[Trade]:
        result = await db.execute(select(Trade).where(Trade.run_id == run_id).order_by(Trade.opened_at))
        return list(result.scalars().all())

    async def get_metrics(self, db: AsyncSession, run_id: int) -> dict[str, float]:
        result = await db.execute(select(RunMetric).where(RunMetric.run_id == run_id))
        return {m.key: m.value for m in result.scalars().all()}

    async def get_chat_history(self, db: AsyncSession, run_id: int, limit: int = 200) -> list[StrategyRunChat]:
        result = await db.execute(
            select(StrategyRunChat).where(StrategyRunChat.run_id == run_id).order_by(StrategyRunChat.created_at).limit(limit)
        )
        return list(result.scalars().all())

    async def add_chat_message(self, db: AsyncSession, **kwargs) -> StrategyRunChat:
        obj = StrategyRunChat(**kwargs)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj


strategy_repo = StrategyRepository()
