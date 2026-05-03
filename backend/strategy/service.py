"""Strategy layer — business logic."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from strategy.models import (
    Strategy, StrategyCreate, StrategyUpdate,
    StrategyRun, StrategyRunCreate,
    ChatMessageCreate, ChatMessageRead,
)
from strategy.repository import strategy_repo

# Error codes (used in HTTPException detail)
STRATEGY_NOT_FOUND = "STRATEGY_NOT_FOUND"
STRATEGY_RUN_NOT_FOUND = "STRATEGY_RUN_NOT_FOUND"
STRATEGY_RUN_ACTIVE = "STRATEGY_RUN_ACTIVE"

_INITIAL_CHART_BARS = 2_000  # default bars returned per chart-data request


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
            risk_override=body.risk_override,
            window_size=body.window_size,
            starting_capital=body.starting_capital,
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

    async def get_chart_data(
        self,
        db: AsyncSession,
        strategy_id: int,
        run_id: int,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int = _INITIAL_CHART_BARS,
    ) -> dict:
        """Return OHLC candles + indicator series + trade markers + economic events.

        from_ts / to_ts (Unix seconds) restrict the time window.
        limit controls how many bars are returned (default _INITIAL_CHART_BARS).
        Response includes has_more/bar_count for incremental loading support.
        """
        import asyncio
        import functools
        import logging

        _log = logging.getLogger("strategy.chart_data")

        # Load only the columns needed — skip equity_curve (can be 40+ MB of Python
        # objects for long backtests) since we only need dataset_id / strategy_id here.
        from sqlalchemy import select as _sa_select
        from sqlalchemy.orm import load_only as _load_only
        _run_result = await db.execute(
            _sa_select(StrategyRun)
            .options(_load_only(
                StrategyRun.id,
                StrategyRun.strategy_id,
                StrategyRun.dataset_id,
            ))
            .where(StrategyRun.id == run_id)
        )
        run = _run_result.scalar_one_or_none()
        if run is None or run.strategy_id != strategy_id:
            raise HTTPException(status_code=404, detail=STRATEGY_RUN_NOT_FOUND)

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

        # Build candle + indicator data in a thread (CPU-bound I/O).
        # Use get_running_loop() — get_event_loop() is deprecated in Python 3.12.
        loop = asyncio.get_running_loop()
        if artifact_path:
            try:
                base = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            _build_chart_base, artifact_path, indicator_specs, from_ts, to_ts,
                            limit=limit,
                            definition=strategy.definition or {},
                        ),
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                _log.error("chart-data timed out after 120s for run %s", run_id)
                raise HTTPException(status_code=504, detail="Chart data took too long to build")
            except Exception as exc:
                _log.exception("chart-data failed for run %s: %s", run_id, exc)
                raise HTTPException(status_code=500, detail=f"Chart data error: {exc}")
        else:
            base = {"candles": [], "indicators": {}, "has_more": False, "bar_count": 0}

        # Derive the OHLC time window from the candles themselves
        candle_times: list[int] = [c["time"] for c in base["candles"]]
        ohlc_from: int | None = min(candle_times) if candle_times else None
        ohlc_to: int | None = max(candle_times) if candle_times else None

        # Trade markers from DB
        trades, _ = await strategy_repo.get_trades(db, run_id, offset=0, limit=10_000)
        markers: list[dict] = []
        for t in trades:
            dir_label = t.direction.upper()
            dir_color = "#22c55e" if t.direction == "buy" else "#ef4444"
            if t.opened_at:
                entry_html = (
                    f"<b style='color:{dir_color}'>{dir_label}</b> "
                    f"@ {t.entry_price:.4f}<br/>"
                    f"<span style='color:#9ca3af;font-size:10px'>"
                    f"SL {t.sl_price:.4f} &nbsp; TP {t.tp_price:.4f}"
                    f"</span>"
                ) if t.sl_price and t.tp_price else (
                    f"<b style='color:{dir_color}'>{dir_label}</b> @ {t.entry_price:.4f}"
                )
                markers.append({
                    "time": int(t.opened_at.timestamp()),
                    "position": "belowBar" if t.direction == "buy" else "aboveBar",
                    "color": dir_color,
                    "shape": "arrowUp" if t.direction == "buy" else "arrowDown",
                    "text": entry_html,
                })
            if t.closed_at and t.exit_price is not None:
                pnl = t.profit or 0.0
                pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                reason = t.exit_reason or "exit"
                if t.direction == "buy":
                    gross = (t.exit_price - t.entry_price) / t.entry_price
                else:
                    gross = (t.entry_price - t.exit_price) / t.entry_price
                gross_color = "#22c55e" if gross >= 0 else "#ef4444"
                exit_html = (
                    f"<b style='color:#f59e0b'>EXIT</b> ({reason})<br/>"
                    f"<span style='color:#9ca3af;font-size:10px'>"
                    f"{dir_label} {t.entry_price:.4f} → {t.exit_price:.4f}</span><br/>"
                    f"Gross: <span style='color:{gross_color}'>{gross * 100:+.2f}%</span>"
                    f" &nbsp; Net: <b style='color:{pnl_color}'>{pnl * 100:+.2f}%</b>"
                )
                markers.append({
                    "time": int(t.closed_at.timestamp()),
                    "position": "aboveBar" if t.direction == "buy" else "belowBar",
                    "color": "#f59e0b",
                    "shape": "arrowDown" if t.direction == "buy" else "arrowUp",
                    "text": exit_html,
                })
        markers.sort(key=lambda m: m["time"])

        # Snap marker timestamps to nearest preceding candle bar so they render correctly
        # when candles are resampled to a coarser timeframe than the trade timestamps.
        if candle_times:
            import bisect
            candle_times_sorted = sorted(candle_times)

            def snap_to_candle(t: int) -> int | None:
                idx = bisect.bisect_right(candle_times_sorted, t) - 1
                if idx < 0:
                    return None  # before first candle — drop the marker
                return candle_times_sorted[idx]

            snapped: list[dict] = []
            for m in markers:
                st = snap_to_candle(m["time"])
                if st is not None:
                    m["time"] = st
                    snapped.append(m)
            markers = snapped

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

    async def validate_strategy(
        self,
        db: AsyncSession,
        strategy_id: int,
        dataset_id: int,
        definition_override: dict | None = None,
    ) -> dict:
        """Run a synchronous (no-Celery) backtest and return chart data for inline validation.

        Returns the same shape as get_chart_data: {candles, indicators, markers}.
        No database writes — purely ephemeral.
        """
        import asyncio
        import functools

        strategy = await self.get_strategy(db, strategy_id)
        definition = definition_override if definition_override is not None else (strategy.definition or {})

        from data.models import Dataset
        from sqlalchemy import select as sa_select
        result = await db.execute(sa_select(Dataset).where(Dataset.id == dataset_id))
        ds_rec = result.scalar_one_or_none()
        if ds_rec is None or not ds_rec.artifact_path:
            return {"candles": [], "indicators": {}, "markers": []}

        artifact_path = ds_rec.artifact_path
        indicator_specs = definition.get("indicators", [])

        loop = asyncio.get_running_loop()

        # Build candles + indicator series
        base = await loop.run_in_executor(
            None,
            functools.partial(_build_chart_base, artifact_path, indicator_specs),
        )

        # Run backtest synchronously to get trades
        from strategy.engine.backtest import run_backtest
        _win_size = int(definition.get("window_size", 0))
        _sc = float(definition.get("starting_capital", 1.0))
        trades_raw, _, _ = await loop.run_in_executor(
            None,
            functools.partial(run_backtest, definition, artifact_path, window_size=_win_size, starting_capital=_sc),
        )

        candle_times: list[int] = [c["time"] for c in base["candles"]]

        # Build markers from raw trade dicts (same shape as get_chart_data)
        markers: list[dict] = []
        for t in trades_raw:
            dir_label = t["direction"].upper()
            dir_color = "#22c55e" if t["direction"] == "buy" else "#ef4444"
            if t.get("opened_at"):
                opened_ts = int(t["opened_at"].timestamp()) if hasattr(t["opened_at"], "timestamp") else int(t["opened_at"])
                entry_html = (
                    f"<b style='color:{dir_color}'>{dir_label}</b> @ {t['entry_price']:.4f}"
                )
                if t.get("sl_price") and t.get("tp_price"):
                    entry_html += (
                        f"<br/><span style='color:#9ca3af;font-size:10px'>"
                        f"SL {t['sl_price']:.4f} &nbsp; TP {t['tp_price']:.4f}</span>"
                    )
                markers.append({
                    "time": opened_ts,
                    "position": "belowBar" if t["direction"] == "buy" else "aboveBar",
                    "color": dir_color,
                    "shape": "arrowUp" if t["direction"] == "buy" else "arrowDown",
                    "text": entry_html,
                })
            if t.get("closed_at") and t.get("exit_price") is not None:
                closed_ts = int(t["closed_at"].timestamp()) if hasattr(t["closed_at"], "timestamp") else int(t["closed_at"])
                pnl = t.get("profit") or 0.0
                pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                reason = t.get("exit_reason") or "exit"
                gross = ((t["exit_price"] - t["entry_price"]) / t["entry_price"]
                         if t["direction"] == "buy"
                         else (t["entry_price"] - t["exit_price"]) / t["entry_price"])
                gross_color = "#22c55e" if gross >= 0 else "#ef4444"
                exit_html = (
                    f"<b style='color:#f59e0b'>EXIT</b> ({reason})<br/>"
                    f"<span style='color:#9ca3af;font-size:10px'>"
                    f"{dir_label} {t['entry_price']:.4f} → {t['exit_price']:.4f}</span><br/>"
                    f"Gross: <span style='color:{gross_color}'>{gross * 100:+.2f}%</span>"
                    f" &nbsp; Net: <b style='color:{pnl_color}'>{pnl * 100:+.2f}%</b>"
                )
                markers.append({
                    "time": closed_ts,
                    "position": "aboveBar" if t["direction"] == "buy" else "belowBar",
                    "color": "#f59e0b",
                    "shape": "arrowDown" if t["direction"] == "buy" else "arrowUp",
                    "text": exit_html,
                })

        markers.sort(key=lambda m: m["time"])

        # Snap markers to nearest preceding candle bar
        if candle_times:
            import bisect
            candle_times_sorted = sorted(candle_times)

            def snap(t: int) -> int | None:
                idx = bisect.bisect_right(candle_times_sorted, t) - 1
                return candle_times_sorted[idx] if idx >= 0 else None

            snapped = []
            for m in markers:
                st = snap(m["time"])
                if st is not None:
                    m["time"] = st
                    snapped.append(m)
            markers = snapped

        base["markers"] = markers
        base["trade_count"] = len(trades_raw)
        return base

    async def validate_strategy_stream(
        self,
        db: AsyncSession,
        strategy_id: int,
        dataset_id: int,
        definition_override: dict | None = None,
        limit_bars: int | None = None,
    ):
        """Async generator yielding SSE lines for streaming condition validation.

        Uses vectorised condition evaluation (no full backtest) for speed.
        Events:
          init    — {type, candles, indicators, total_bars}
          signals — {type, long_entry, long_exit, short_entry, short_exit}
                    each value is a list of unix timestamps where the block fires
          done    — {type}
          error   — {type, message}
        """
        import asyncio
        import functools
        import json

        strategy = await self.get_strategy(db, strategy_id)
        definition = definition_override if definition_override is not None else (strategy.definition or {})

        from data.models import Dataset
        from sqlalchemy import select as sa_select
        result = await db.execute(sa_select(Dataset).where(Dataset.id == dataset_id))
        ds_rec = result.scalar_one_or_none()
        if ds_rec is None or not ds_rec.artifact_path:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset not found'})}\n\n"
            return

        artifact_path = ds_rec.artifact_path
        indicator_specs = definition.get("indicators", [])
        groups = definition.get("groups", {})

        loop = asyncio.get_running_loop()

        # Build candles + indicators (limited to last limit_bars rows)
        base = await loop.run_in_executor(
            None,
            functools.partial(_build_chart_base, artifact_path, indicator_specs, limit_bars=limit_bars),
        )

        total_bars = len(base["candles"])
        yield f"data: {json.dumps({'type': 'init', 'candles': base['candles'], 'indicators': base['indicators'], 'total_bars': total_bars})}\n\n"

        # Vectorised condition evaluation — runs in thread executor to stay non-blocking
        def compute_signals() -> dict:
            import os
            import bisect
            from pathlib import Path
            from strategy.engine.backtest import _load_df, _normalise
            from strategy.engine.indicators import apply_indicators
            from strategy.engine.conditions import eval_block_series

            store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
            df = _load_df(store / artifact_path)
            if limit_bars and limit_bars < len(df):
                df = df.iloc[-limit_bars:]

            defn = _normalise(definition)
            if indicator_specs:
                df = apply_indicators(df, indicator_specs)

            candle_times_sorted = sorted(c["time"] for c in base["candles"])

            def snap(t: int) -> int | None:
                idx = bisect.bisect_right(candle_times_sorted, t) - 1
                return candle_times_sorted[idx] if idx >= 0 else None

            signals: dict[str, list[int]] = {
                "long_entry": [], "long_exit": [],
                "short_entry": [], "short_exit": [],
            }

            blocks = {
                "long_entry":  defn.get("long", {}).get("entry"),
                "long_exit":   defn.get("long", {}).get("exit"),
                "short_entry": defn.get("short", {}).get("entry"),
                "short_exit":  defn.get("short", {}).get("exit"),
            }

            for key, block in blocks.items():
                if not block or not block.get("conditions"):
                    continue
                try:
                    mask = eval_block_series(df, block, groups)
                    for ts, fired in zip(df.index, mask):
                        if fired:
                            raw_ts = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(ts)
                            snapped = snap(raw_ts)
                            if snapped is not None:
                                signals[key].append(snapped)
                except Exception:
                    pass

            return signals

        try:
            signals = await loop.run_in_executor(None, compute_signals)
            yield f"data: {json.dumps({'type': 'signals', **signals})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

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
# Marker helpers
# ---------------------------------------------------------------------------

def _trade_to_stream_markers(trade: dict, kind: str) -> list[dict]:
    """Convert a partial (open) or complete (close) trade dict to chart marker(s)."""
    markers = []
    direction = trade.get("direction", "buy")
    dir_label = direction.upper()
    dir_color = "#22c55e" if direction == "buy" else "#ef4444"
    entry_price = trade.get("entry_price", 0.0)

    if kind == "open":
        opened_at = trade.get("opened_at")
        if opened_at is None:
            return markers
        ts = int(opened_at.timestamp()) if hasattr(opened_at, "timestamp") else int(opened_at)
        text = f"<b style='color:{dir_color}'>{dir_label}</b> @ {entry_price:.4f}"
        sl = trade.get("sl_price")
        tp = trade.get("tp_price")
        if sl and tp:
            text += f"<br/><span style='color:#9ca3af;font-size:10px'>SL {sl:.4f} &nbsp; TP {tp:.4f}</span>"
        markers.append({
            "time": ts,
            "position": "belowBar" if direction == "buy" else "aboveBar",
            "color": dir_color,
            "shape": "arrowUp" if direction == "buy" else "arrowDown",
            "text": text,
        })

    elif kind == "close":
        closed_at = trade.get("closed_at")
        exit_price = trade.get("exit_price")
        if closed_at is None or exit_price is None:
            return markers
        ts = int(closed_at.timestamp()) if hasattr(closed_at, "timestamp") else int(closed_at)
        pnl = trade.get("profit") or 0.0
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        reason = trade.get("exit_reason") or "exit"
        gross = (
            (exit_price - entry_price) / entry_price if direction == "buy"
            else (entry_price - exit_price) / entry_price
        )
        gross_color = "#22c55e" if gross >= 0 else "#ef4444"
        text = (
            f"<b style='color:#f59e0b'>EXIT</b> ({reason})<br/>"
            f"<span style='color:#9ca3af;font-size:10px'>{dir_label} {entry_price:.4f} → {exit_price:.4f}</span><br/>"
            f"Gross: <span style='color:{gross_color}'>{gross * 100:+.2f}%</span>"
            f" &nbsp; Net: <b style='color:{pnl_color}'>{pnl * 100:+.2f}%</b>"
        )
        markers.append({
            "time": ts,
            "position": "aboveBar" if direction == "buy" else "belowBar",
            "color": "#f59e0b",
            "shape": "arrowDown" if direction == "buy" else "arrowUp",
            "text": text,
        })

    return markers


# ---------------------------------------------------------------------------
# Sync helpers (run in thread executor)
# ---------------------------------------------------------------------------

# Indicator types drawn on the price axis (same scale as OHLC candles)
_OVERLAY_TYPES = {"ema", "sma", "bb", "donchian", "sar"}

# Per-column color overrides; keys may be exact column names or type prefixes
_INDICATOR_COLORS: dict[str, str] = {
    "ema":           "#f59e0b",
    "sma":           "#a78bfa",
    "bb_upper":      "#64748b",
    "bb_middle":     "#94a3b8",
    "bb_lower":      "#64748b",
    "macd_line":     "#0ea5e9",
    "macd_signal":   "#f97316",
    "macd_hist":     "#6b7280",
    "rsi":           "#a855f7",
    "atr":           "#84cc16",
    "slope":         "#22d3ee",
    "adx":           "#fb923c",
    "plus_di":       "#4ade80",
    "minus_di":      "#f87171",
    "stochastic_k":  "#818cf8",
    "stochastic_d":  "#c084fc",
    "donchian":      "#475569",
    "sar":           "#facc15",
    "cci":           "#2dd4bf",
    "roc":           "#38bdf8",
    "renko":         "#d946ef",
    "rangetrend":    "#34d399",
    "candle":        "#e879f9",
    "streak":        "#fbbf24",
}

# Multi-column indicator suffix specs: type → [(suffix, series_type), ...]
# Single-output indicators (ema, rsi, etc.) are handled by the fallback.
_MULTI_COL_SPECS: dict[str, list[tuple[str, str]]] = {
    "macd":       [("_line", "line"), ("_signal", "line"), ("_hist", "histogram")],
    "bb":         [("_upper", "line"), ("_middle", "line"), ("_lower", "line")],
    "adx":        [("", "line"), ("_plus_di", "line"), ("_minus_di", "line")],
    "stochastic": [("_k", "line"), ("_d", "line")],
    "donchian":   [("_upper", "line"), ("_lower", "line"), ("_mid", "line")],
    "renko":      [("_direction", "line"), ("_flip", "line"), ("_momentum", "line"), ("_pos", "line")],
    "candle":     [
        ("_bull_engulf", "line"), ("_bear_engulf", "line"),
        ("_bull_pin", "line"), ("_bear_pin", "line"),
        ("_bull_outside", "line"), ("_bear_outside", "line"),
    ],
}

def _resample_ohlcv(df, step: int):
    """Aggregate rows in chunks of `step`, preserving proper OHLC semantics."""
    import pandas as pd

    chunks = [df.iloc[i:i + step] for i in range(0, len(df), step)]
    records = []
    for chunk in chunks:
        if chunk.empty:
            continue
        rec: dict = {}
        rec["open"] = float(chunk["open"].iloc[0]) if "open" in chunk.columns else float(chunk["close"].iloc[0])
        rec["high"] = float(chunk["high"].max()) if "high" in chunk.columns else float(chunk["close"].max())
        rec["low"] = float(chunk["low"].min()) if "low" in chunk.columns else float(chunk["close"].min())
        rec["close"] = float(chunk["close"].iloc[-1])
        records.append((chunk.index[-1], rec))
    if not records:
        return df.iloc[0:0]
    idx, rows = zip(*records)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _inject_condition_indicators(
    df,
    indicator_specs: list,
    definition: dict,
    indicators: dict,
    candle_times: list,
) -> None:
    """Augment `indicators` in-place with condition-derived series:

    1. For every numeric comparison (left op NUMBER): add a constant dashed threshold
       line in the indicator's existing pane, and make sure the left column is shown.
    2. For every evaluable condition: add a 0/1 boolean series in a new
       "cond:{group}" pane so the user can see exactly when each condition fires.
    """
    import math as _math
    import pandas as pd
    from strategy.engine.conditions import eval_condition_series

    # col → (group, pane)
    col_to_meta: dict[str, tuple[str, str]] = {}
    # indicator-id prefix → (group, pane)  — used for sub-columns not in _MULTI_COL_SPECS
    prefix_to_meta: dict[str, tuple[str, str]] = {}

    for spec in indicator_specs:
        itype = spec["type"].lower()
        iid = spec.get("id", itype)
        pane = "overlay" if itype in _OVERLAY_TYPES else "separate"
        prefix_to_meta[iid] = (itype, pane)
        if itype in _MULTI_COL_SPECS:
            for suffix, _ in _MULTI_COL_SPECS[itype]:
                col_to_meta[f"{iid}{suffix}"] = (itype, pane)
        elif itype == "rangetrend":
            method = (spec.get("params") or {}).get("method", "atr")
            for suffix in ("_is_range", "_direction", f"_{method}"):
                col_to_meta[f"{iid}{suffix}"] = (itype, pane)
        else:
            col_to_meta[iid] = (itype, pane)

    def _resolve_meta(col: str) -> tuple[str, str] | None:
        m = col_to_meta.get(col)
        if m:
            return m
        for prefix, pm in prefix_to_meta.items():
            if col.startswith(prefix + "_") or col == prefix:
                return pm
        return None

    groups_def = definition.get("groups", {})

    def _collect(block: dict) -> list[dict]:
        out: list[dict] = []
        for cond in block.get("conditions", []):
            ctype = cond.get("type") or "comparison"
            if ctype == "group_ref":
                gid = cond.get("group_id", "")
                if gid in groups_def:
                    out.extend(_collect(groups_def[gid]))
            elif ctype not in ("ml_signal", "llm_signal"):
                out.append(cond)
        return out

    # (cond, side_tag) pairs — side+phase aware for separate panes and colors
    _SIDE_COLORS = {
        "long_entry":  "#22c55e",  # green
        "long_exit":   "#f59e0b",  # amber
        "short_entry": "#ef4444",  # red
        "short_exit":  "#a78bfa",  # purple
        "group":       "#60a5fa",  # blue
    }
    _SIDE_LABELS = {
        "long_entry": "LE", "long_exit": "LX",
        "short_entry": "SE", "short_exit": "SX",
        "group": "G",
    }
    all_conds: list[tuple[dict, str]] = []
    for side in ("long", "short"):
        for phase in ("entry", "exit"):
            for c in _collect(definition.get(side, {}).get(phase, {})):
                all_conds.append((c, f"{side}_{phase}"))
    for gdef in groups_def.values():
        for c in _collect(gdef):
            all_conds.append((c, "group"))

    seen: set[tuple] = set()
    _OP_SYM = {">=": "≥", "<=": "≤", "==": "=", "!=": "≠", ">": ">", "<": "<"}

    for cond, side_tag in all_conds:
        ctype = cond.get("type") or "comparison"

        # ── regime ──────────────────────────────────────────────────────────
        if ctype == "regime":
            indicator = cond.get("indicator", "rt")
            mode = cond.get("mode", "range")
            threshold = float(cond.get("threshold", 0.5 if mode in ("range", "trend") else 0.3))
            cond_key = ("regime", indicator, mode, threshold, side_tag)
            if cond_key in seen:
                continue
            seen.add(cond_key)
            group = "rangetrend"
            color = _SIDE_COLORS[side_tag]
            # threshold line in existing rangetrend pane (score-based indicators only)
            range_col = f"{indicator}_range"
            if range_col in df.columns and range_col not in indicators:
                s = _series_from_col(df, range_col, "line", "separate")
                s["group"] = group
                s["color"] = _INDICATOR_COLORS.get("rangetrend", "#34d399")
                indicators[range_col] = s
            if range_col in df.columns and candle_times:
                tkey = f"regime_threshold_{threshold}"
                if tkey not in indicators:
                    indicators[tkey] = {
                        "type": "line", "pane": "separate",
                        "color": _INDICATOR_COLORS.get("rangetrend", "#34d399"),
                        "group": group,
                        "data": [
                            {"time": candle_times[0], "value": threshold},
                            {"time": candle_times[-1], "value": threshold},
                        ],
                        "line_style": "dashed",
                    }
            # 0/1 condition pane (transition-point encoded)
            try:
                met = eval_condition_series(df, cond)
                if met is not None:
                    met_set = {int(ts.timestamp()) for ts, v in zip(df.index, met)
                               if v and isinstance(ts, pd.Timestamp)}
                    label = f"[{_SIDE_LABELS[side_tag]}] regime:{mode}"
                    if label not in indicators:
                        rle: list[dict] = []
                        prev_v: int | None = None
                        for t in candle_times:
                            val = 1 if t in met_set else 0
                            if val != prev_v:
                                rle.append({"time": t, "value": val})
                                prev_v = val
                        if candle_times:
                            lv = 1 if candle_times[-1] in met_set else 0
                            if not rle or rle[-1]["time"] != candle_times[-1]:
                                rle.append({"time": candle_times[-1], "value": lv})
                        indicators[label] = {
                            "type": "line", "pane": "separate", "color": color,
                            "group": f"cond:{group}:{side_tag}",
                            "data": rle,
                            "line_style": "step",
                        }
            except Exception:
                pass
            continue

        # ── standard comparison / streak ────────────────────────────────────
        left = cond.get("left")
        right = cond.get("right")
        op = cond.get("op", ">")
        if not left:
            continue

        meta = _resolve_meta(left)
        if meta is None:
            continue
        group, pane = meta
        cond_color = _SIDE_COLORS[side_tag]

        cond_key = (left, op, str(right), side_tag)
        if cond_key in seen:
            continue
        seen.add(cond_key)

        # Ensure the left column is visible in the chart (uses indicator's own color, not side color)
        if left in df.columns and left not in indicators:
            s = _series_from_col(df, left, "line", pane)
            s["group"] = group
            base_key = next((k for k in _INDICATOR_COLORS if left.startswith(k)), group)
            s["color"] = _INDICATOR_COLORS.get(base_key, _INDICATOR_COLORS.get(group, "#6b7280"))
            indicators[left] = s

        # Constant threshold line — only endpoints needed; frontend fills in between
        if isinstance(right, (int, float)) and not _math.isnan(float(right)):
            thresh_val = float(right)
            tkey = f"{left}_{op}_{thresh_val}_line"
            if tkey not in indicators and candle_times:
                ind_color = _INDICATOR_COLORS.get(group, "#6b7280")
                indicators[tkey] = {
                    "type": "line", "pane": pane, "color": ind_color, "group": group,
                    "data": [
                        {"time": candle_times[0], "value": thresh_val},
                        {"time": candle_times[-1], "value": thresh_val},
                    ],
                    "line_style": "dashed",
                }

        # 0/1 condition-state line — prefixed with side, colored by side
        op_sym = _OP_SYM.get(op, op)
        right_label = right if isinstance(right, str) else str(right)
        side_prefix = _SIDE_LABELS[side_tag]
        cond_label = f"[{side_prefix}] {left} {op_sym} {right_label}"
        if cond_label not in indicators:
            try:
                met = eval_condition_series(df, cond)
                if met is None:
                    continue
                met_set = {int(ts.timestamp()) for ts, v in zip(df.index, met)
                           if v and isinstance(ts, pd.Timestamp)}
                data: list[dict] = []
                prev: int | None = None
                for t in candle_times:
                    val = 1 if t in met_set else 0
                    if val != prev:
                        data.append({"time": t, "value": val})
                        prev = val
                if candle_times:
                    last_val = 1 if candle_times[-1] in met_set else 0
                    if not data or data[-1]["time"] != candle_times[-1]:
                        data.append({"time": candle_times[-1], "value": last_val})
                indicators[cond_label] = {
                    "type": "line", "pane": "separate", "color": cond_color,
                    "group": f"cond:{group}:{side_tag}",
                    "data": data,
                    "line_style": "step",
                }
            except Exception:
                pass


def _build_chart_base(
    artifact_path: str,
    indicator_specs: list,
    from_ts: int | None = None,
    to_ts: int | None = None,
    limit_bars: int | None = None,
    limit: int = _INITIAL_CHART_BARS,
    definition: dict | None = None,
) -> dict:
    """Load parquet, apply indicators, return candles + series.

    limit_bars: hard cap used by the validate panel (overrides limit).
    limit: windowed cap for chart-data requests (default _INITIAL_CHART_BARS).

    Warmup bars are loaded before the window and trimmed after indicator
    computation so every returned bar has valid indicator values.

    Returns dict with keys: candles, indicators, has_more, bar_count.
    """
    import os
    from pathlib import Path

    import pandas as pd

    from strategy.engine.backtest import _load_df
    from strategy.engine.indicators import apply_indicators, estimate_warmup_bars

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = _load_df(store / artifact_path)

    # Filter by upper time bound first so we work on the relevant subset.
    if from_ts is not None or to_ts is not None:
        try:
            unix = df.index.asi8 // 1_000_000_000
        except Exception:
            unix = df.index.astype("datetime64[s]").astype("int64")
        mask = pd.Series(True, index=df.index)
        if from_ts is not None:
            mask = mask & (unix >= from_ts)
        if to_ts is not None:
            mask = mask & (unix <= to_ts)
        df = df.loc[mask]

    if df.empty:
        return {"candles": [], "indicators": {}, "has_more": False, "bar_count": 0}

    # Caller-supplied limit_bars (validate panel) takes precedence over windowed limit.
    cap = limit_bars if limit_bars else limit

    # Load extra warmup bars so indicators are valid at the first returned bar.
    warmup = estimate_warmup_bars(indicator_specs) if indicator_specs else 0
    total_needed = cap + warmup
    has_more = len(df) > total_needed
    if len(df) > total_needed:
        df = df.iloc[-total_needed:]

    if indicator_specs:
        df = apply_indicators(df, indicator_specs)

    # Trim warmup prefix — the remaining rows all have valid indicator values.
    if len(df) > cap:
        df = df.iloc[-cap:]

    # Build candle list — skip rows with invalid prices
    import math as _math
    candles: list[dict] = []
    for ts, row in df.iterrows():
        if not isinstance(ts, pd.Timestamp):
            continue
        o = float(row.get("open", row["close"]))
        h = float(row.get("high", row["close"]))
        lo = float(row.get("low", row["close"]))
        c = float(row["close"])
        if any(_math.isnan(v) or _math.isinf(v) or v <= 0 for v in (o, h, lo, c)):
            continue
        if h < lo:
            h, lo = lo, h
        # Enforce proper OHLC relationship: high must be ≥ max(open,close) and low ≤ min(open,close)
        if h < max(o, c) or lo > min(o, c):
            continue
        candles.append({"time": int(ts.timestamp()), "open": o, "high": h, "low": lo, "close": c})

    # Build indicator series — same resolution as candles
    indicators: dict[str, dict] = {}
    for spec in indicator_specs:
        itype = spec["type"].lower()
        iid = spec.get("id", itype)
        pane = "overlay" if itype in _OVERLAY_TYPES else "separate"

        def _add(col: str, stype: str = "line", _itype: str = itype, _pane: str = pane) -> None:
            if col not in df.columns:
                return
            s = _series_from_col(df, col, stype, _pane)
            s["group"] = _itype
            base_key = next((k for k in _INDICATOR_COLORS if col.startswith(k)), col)
            s["color"] = _INDICATOR_COLORS.get(base_key, "#6b7280")
            indicators[col] = s

        if itype in _MULTI_COL_SPECS:
            for suffix, stype in _MULTI_COL_SPECS[itype]:
                _add(f"{iid}{suffix}", stype)
        elif itype == "rangetrend":
            method = (spec.get("params") or {}).get("method", "atr")
            for suffix, stype in [("_is_range", "line"), ("_direction", "line"), (f"_{method}", "line")]:
                _add(f"{iid}{suffix}", stype)
        else:
            _add(iid)

    if definition:
        try:
            _inject_condition_indicators(
                df, indicator_specs, definition, indicators,
                [c["time"] for c in candles],
            )
        except Exception:
            pass

    return {
        "candles": candles,
        "indicators": indicators,
        "has_more": has_more,
        "bar_count": len(candles),
    }


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
