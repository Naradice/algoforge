"""Strategy layer — database queries (SQLAlchemy only, no business logic)."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from strategy.models import Strategy, StrategyRun, StrategyRunCreate, Trade, RunMetric, StrategyRunChat, StrategyEvent


class StrategyRepository:
    async def get_all(
        self,
        db: AsyncSession,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Strategy], int]:
        q = select(Strategy)
        if status:
            q = q.where(Strategy.status == status)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(Strategy.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

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

    async def get_runs(
        self,
        db: AsyncSession,
        strategy_id: int,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[StrategyRun], int]:
        q = select(StrategyRun).where(StrategyRun.strategy_id == strategy_id)
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.order_by(StrategyRun.created_at.desc()).offset(offset).limit(limit))).scalars().all()
        return list(items), total

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

    async def get_trade_by_id(self, db: AsyncSession, trade_id: int) -> Trade | None:
        result = await db.execute(select(Trade).where(Trade.id == trade_id))
        return result.scalar_one_or_none()

    async def get_run_metrics_multi(self, db: AsyncSession, run_ids: list[int]) -> dict[int, dict]:
        result = await db.execute(select(RunMetric).where(RunMetric.run_id.in_(run_ids)))
        out: dict[int, dict] = {}
        for m in result.scalars().all():
            out.setdefault(m.run_id, {})[m.key] = m.value
        return out

    async def create_version(self, db: AsyncSession, strategy_id: int, version: int, definition: dict):
        from strategy.models import StrategyVersion
        obj = StrategyVersion(strategy_id=strategy_id, version=version, definition=definition)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def get_versions(self, db: AsyncSession, strategy_id: int) -> list:
        from strategy.models import StrategyVersion
        result = await db.execute(
            select(StrategyVersion).where(StrategyVersion.strategy_id == strategy_id).order_by(StrategyVersion.version.desc())
        )
        return list(result.scalars().all())

    async def get_version_count(self, db: AsyncSession, strategy_id: int) -> int:
        from strategy.models import StrategyVersion
        result = await db.execute(select(func.count()).where(StrategyVersion.strategy_id == strategy_id))
        return result.scalar_one()


strategy_repo = StrategyRepository()
