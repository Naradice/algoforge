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

    async def delete_run(self, db: AsyncSession, strategy_id: int, run_id: int) -> None:
        run = await self.get_run(db, strategy_id, run_id)
        if run.status in ("pending", "running"):
            raise HTTPException(status_code=409, detail="Cannot delete an active run. Stop it first.")
        await db.delete(run)

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
            walk_forward_ratio=body.walk_forward_ratio,
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
        run = await self.get_run(db, strategy_id, run_id)
        strategy = await self.get_strategy(db, strategy_id)

        # Persist user message
        await strategy_repo.add_chat_message(db, run_id=run_id, role="user", message=body.message)

        # Gather context for AI
        metrics = await strategy_repo.get_metrics(db, run_id)
        recent_trades, _ = await strategy_repo.get_trades(db, run_id, offset=0, limit=20)
        history = await strategy_repo.get_chat_history(db, run_id, limit=20)

        # Call AI — collect streamed chunks (currently yields in one shot)
        from strategy.chat_agent import stream_response
        chunks: list[str] = []
        async for chunk in stream_response(
            strategy_definition=strategy.definition,
            metrics=metrics,
            recent_trades=recent_trades,
            chat_history=history,
            user_message=body.message,
        ):
            chunks.append(chunk)

        ai_text = "".join(chunks).strip() or "No response."

        # Persist AI reply and return it
        ai_msg = await strategy_repo.add_chat_message(db, run_id=run_id, role="assistant", message=ai_text)
        return ChatMessageRead.model_validate(ai_msg)

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

    async def get_chart_data(self, db: AsyncSession, strategy_id: int, run_id: int) -> dict:
        """
        Return OHLC candles + indicator series + trade markers + economic events
        for the run chart. Candle times are Unix seconds (int) as required by
        lightweight-charts.

        Economic calendar datasets that overlap the OHLC time window are
        auto-discovered and merged — no manual linking required.
        """
        import asyncio
        import functools

        run = await self.get_run(db, strategy_id, run_id)
        strategy = await self.get_strategy(db, run.strategy_id)

        # Need dataset artifact path
        artifact_path: str | None = None
        if run.dataset_id:
            from data.models import Dataset
            from sqlalchemy import select as sa_select
            result = await db.execute(sa_select(Dataset).where(Dataset.id == run.dataset_id))
            ds_rec = result.scalar_one_or_none()
            if ds_rec:
                artifact_path = ds_rec.artifact_path

        indicator_specs: list = (strategy.definition or {}).get("indicators", [])

        # Build candle + indicator data in a thread (CPU-bound I/O)
        loop = asyncio.get_event_loop()
        if artifact_path:
            base = await loop.run_in_executor(
                None,
                functools.partial(_build_chart_base, artifact_path, indicator_specs),
            )
        else:
            base = {"candles": [], "indicators": {}}

        # Derive the OHLC time window from the candles themselves
        candle_times: list[int] = [c["time"] for c in base["candles"]]
        ohlc_from: int | None = min(candle_times) if candle_times else None
        ohlc_to: int | None = max(candle_times) if candle_times else None

        # Trade markers from DB
        trades, _ = await strategy_repo.get_trades(db, run_id, offset=0, limit=10_000)
        markers: list[dict] = []
        for t in trades:
            if t.opened_at:
                markers.append({
                    "time": int(t.opened_at.timestamp()),
                    "position": "belowBar" if t.direction == "buy" else "aboveBar",
                    "color": "#22c55e" if t.direction == "buy" else "#ef4444",
                    "shape": "arrowUp" if t.direction == "buy" else "arrowDown",
                    "text": f"{t.direction.upper()} {t.entry_price:.4f}",
                })
            if t.closed_at and t.exit_price is not None:
                markers.append({
                    "time": int(t.closed_at.timestamp()),
                    "position": "aboveBar" if t.direction == "buy" else "belowBar",
                    "color": "#f59e0b",
                    "shape": "arrowDown" if t.direction == "buy" else "arrowUp",
                    "text": f"EXIT {t.exit_price:.4f}",
                })
        markers.sort(key=lambda m: m["time"])
        base["markers"] = markers

        # Auto-discover economic calendar datasets that overlap the OHLC window
        events: list[dict] = []
        if ohlc_from is not None and ohlc_to is not None:
            from data.models import Dataset, Datasource
            from sqlalchemy import select as sa_select

            eco_result = await db.execute(
                sa_select(Dataset)
                .join(Datasource, Dataset.datasource_id == Datasource.id)
                .where(Datasource.type == "economic_calendar")
                .where(Dataset.artifact_path.is_not(None))
                .where(Dataset.status == "ready")
            )
            eco_datasets = eco_result.scalars().all()

            for eco_ds in eco_datasets:
                # Quick overlap check using dataset-level from/to metadata (if available)
                if eco_ds.to_ts and int(eco_ds.to_ts.timestamp()) < ohlc_from:
                    continue
                if eco_ds.from_ts and int(eco_ds.from_ts.timestamp()) > ohlc_to:
                    continue

                batch = await loop.run_in_executor(
                    None,
                    functools.partial(
                        _load_economic_events,
                        eco_ds.artifact_path,
                        ohlc_from,
                        ohlc_to,
                    ),
                )
                events.extend(batch)

        events.sort(key=lambda e: e["time"])
        base["events"] = events
        return base

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


# ---------------------------------------------------------------------------
# Sync helpers (run in thread executor)
# ---------------------------------------------------------------------------

# Indicator types that are drawn on the price axis (same scale as OHLC)
_OVERLAY_TYPES = {"ema", "sma", "bb"}
# Which grouped prefix maps to "separate" pane label
_PANE_GROUPS = {
    "macd_line": "macd",
    "macd_signal": "macd",
    "macd_hist": "macd",
    "bb_upper": "bb",
    "bb_middle": "bb",
    "bb_lower": "bb",
}

_INDICATOR_COLORS = {
    "ema": "#f59e0b",
    "sma": "#a78bfa",
    "bb_upper": "#64748b",
    "bb_middle": "#94a3b8",
    "bb_lower": "#64748b",
    "macd_line": "#0ea5e9",
    "macd_signal": "#f97316",
    "macd_hist": "#6b7280",
    "rsi": "#a855f7",
    "atr": "#84cc16",
    "slope": "#22d3ee",
}

_MAX_BARS = 2_000


def _build_chart_base(artifact_path: str, indicator_specs: list) -> dict:
    """Load parquet, apply indicators, downsample, return candles + indicator series."""
    import math
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from strategy.engine.backtest import _load_df
    from strategy.engine.indicators import apply_indicators

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = _load_df(store / artifact_path)

    if indicator_specs:
        df = apply_indicators(df, indicator_specs)

    # Downsample to keep the payload manageable
    if len(df) > _MAX_BARS:
        step = math.ceil(len(df) / _MAX_BARS)
        df = df.iloc[::step]

    # Build candle list
    candles: list[dict] = []
    for ts, row in df.iterrows():
        if not isinstance(ts, pd.Timestamp):
            continue
        t = int(ts.timestamp())
        candles.append({
            "time": t,
            "open": float(row.get("open", row["close"])),
            "high": float(row.get("high", row["close"])),
            "low": float(row.get("low", row["close"])),
            "close": float(row["close"]),
        })

    # Build indicator series from spec (preserves original bar resolution)
    indicators: dict[str, dict] = {}
    for spec in indicator_specs:
        itype = spec["type"].lower()
        iid = spec.get("id", itype)
        pane = "overlay" if itype in _OVERLAY_TYPES else "separate"

        if itype == "macd":
            for suffix, stype in [("_line", "line"), ("_signal", "line"), ("_hist", "histogram")]:
                col = f"{iid}{suffix}"
                if col in df.columns:
                    indicators[col] = _series_from_col(df, col, stype, pane)
        elif itype == "bb":
            for suffix in ("_upper", "_middle", "_lower"):
                col = f"{iid}{suffix}"
                if col in df.columns:
                    indicators[col] = _series_from_col(df, col, "line", pane)
        else:
            if iid in df.columns:
                indicators[iid] = _series_from_col(df, iid, "line", pane)

    # Attach display metadata (color, pane group)
    for name, series in indicators.items():
        base_key = next((k for k in _INDICATOR_COLORS if name.startswith(k)), name)
        series["color"] = _INDICATOR_COLORS.get(base_key, "#6b7280")
        series["group"] = _PANE_GROUPS.get(name, name)

    return {"candles": candles, "indicators": indicators}


def _load_economic_events(artifact_path: str, from_unix: int, to_unix: int) -> list[dict]:
    """
    Read an economic_calendar parquet directory and return events within the
    given Unix-second window as a list of dicts ready for the chart response.

    Each event: {"time": int, "indicator": str, "value": float, "unit": str}
    """
    import math
    import os
    from pathlib import Path

    import pandas as pd
    import pyarrow.dataset as pad

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    path = store / artifact_path

    if not path.exists():
        return []

    try:
        ds = pad.dataset(str(path), format="parquet", partitioning="hive")
        table = ds.to_table(columns=["datetime", "indicator", "value", "unit"])
        df = table.to_pandas()
    except Exception:
        return []

    if df.empty:
        return []

    # Normalise the datetime column (may come back as object from partition-less read)
    if "datetime" not in df.columns:
        return []

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    from_ts = pd.Timestamp(from_unix, unit="s", tz="UTC")
    to_ts = pd.Timestamp(to_unix, unit="s", tz="UTC")
    df = df[(df["datetime"] >= from_ts) & (df["datetime"] <= to_ts)]

    events = []
    for _, row in df.iterrows():
        v = row.get("value")
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        events.append({
            "time": int(row["datetime"].timestamp()),
            "indicator": str(row.get("indicator", "")),
            "value": round(float(v), 6),
            "unit": str(row.get("unit", "")),
        })

    return events


def _series_from_col(df, col: str, stype: str, pane: str) -> dict:
    import pandas as pd

    data = []
    for ts, v in df[col].items():
        if not isinstance(ts, pd.Timestamp):
            continue
        if v is None or (hasattr(v, "__float__") and not __import__("math").isfinite(float(v))):
            continue
        data.append({"time": int(ts.timestamp()), "value": round(float(v), 6)})
    return {"type": stype, "pane": pane, "data": data}
