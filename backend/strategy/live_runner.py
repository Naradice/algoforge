"""
Paper/live strategy execution loop.

Fetches live OHLC data from yfinance, applies indicators, evaluates strategy
conditions bar-by-bar, and logs simulated trades to the DB.

Two modes:
  paper  — simulates execution with real market data (no real orders)
  live   — placeholder; real broker integration is a future extension

The loop can be stopped from the API by setting strategy_runs.status = "stopped".
Check interval configurable via the PAPER_CHECK_INTERVAL_S env var (default 60s).

Call entry point:
    await run_paper(run_id, strategy_rec, model_cache)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# How often to poll for new candles (seconds). Set low for testing.
_CHECK_INTERVAL = int(os.getenv("PAPER_CHECK_INTERVAL_S", "60"))


async def run_paper(
    run_id: int,
    strategy_definition: dict,
    model_cache: dict,
) -> dict:
    """
    Main paper trading loop. Runs until stopped, errored, or data source fails.
    Returns a summary dict.
    """
    from database import async_session_factory
    from strategy.models import StrategyRun, Trade, RunMetric
    from strategy.engine.indicators import apply_indicators
    from strategy.engine.conditions import evaluate_conditions
    from strategy.engine.backtest import _OpenPosition, _check_sl_tp, _close_position, _compute_metrics
    from sqlalchemy import select, update

    symbol = strategy_definition.get("symbol", "AAPL")
    indicator_specs = strategy_definition.get("indicators", [])
    entry_block = strategy_definition.get("entry", {})
    exit_block = strategy_definition.get("exit", {})
    risk = strategy_definition.get("risk", {})

    direction = entry_block.get("direction", "buy")
    sl_pct = float(risk.get("sl_pct", 0.0))
    tp_pct = float(risk.get("tp_pct", 0.0))
    position_size = float(risk.get("position_size", 1.0))

    position: _OpenPosition | None = None
    last_bar_ts = None
    trade_records: list[dict] = []
    bar_index = 0

    logger.info(f"Paper run {run_id} started for {symbol}")

    while True:
        # ── Check if stopped ───────────────────────────────────────────────────
        async with async_session_factory() as db:
            result = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))
            run = result.scalar_one_or_none()
        if run is None or run.status in ("stopped", "error"):
            break

        # ── Fetch latest OHLC from yfinance ────────────────────────────────────
        try:
            df = await asyncio.to_thread(_fetch_ohlc, symbol)
        except Exception as e:
            logger.warning(f"Paper run {run_id}: yfinance fetch failed: {e}")
            await asyncio.sleep(_CHECK_INTERVAL)
            continue

        if df.empty:
            await asyncio.sleep(_CHECK_INTERVAL)
            continue

        # Check for new bar
        latest_ts = df.index[-1]
        if last_bar_ts is not None and latest_ts <= last_bar_ts:
            await asyncio.sleep(_CHECK_INTERVAL)
            continue

        last_bar_ts = latest_ts
        bar_index += 1

        # Apply indicators
        if indicator_specs:
            df = apply_indicators(df, indicator_specs)

        row = df.iloc[-1]
        bar_dt = latest_ts.to_pydatetime().replace(tzinfo=timezone.utc)
        close = float(row.get("close", row.iloc[-1]))
        high = float(row.get("high", close))
        low = float(row.get("low", close))

        ctx = {"df_upto": df, "bar_index": bar_index, "model_cache": model_cache}

        # ── SL/TP check ────────────────────────────────────────────────────────
        if position is not None:
            closed = _check_sl_tp(position, high, low, close, bar_dt, trade_records, symbol)
            if closed:
                await _persist_last_trade(run_id, trade_records, async_session_factory, Trade)
                position = None

        # ── Entry / exit ───────────────────────────────────────────────────────
        if position is None:
            try:
                should_enter = evaluate_conditions(row, entry_block, context=ctx)
            except Exception:
                should_enter = False

            if should_enter:
                sl_price = close * (1 - sl_pct) if sl_pct > 0 else None
                tp_price = close * (1 + tp_pct) if tp_pct > 0 else None
                if direction == "sell":
                    sl_price = close * (1 + sl_pct) if sl_pct > 0 else None
                    tp_price = close * (1 - tp_pct) if tp_pct > 0 else None
                position = _OpenPosition(
                    direction=direction,
                    entry_price=close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    volume=position_size,
                    opened_at=bar_dt,
                )
                logger.info(f"Paper run {run_id}: entered {direction} @ {close:.4f}")
        else:
            try:
                should_exit = evaluate_conditions(row, exit_block, context=ctx)
            except Exception:
                should_exit = False

            if should_exit:
                _close_position(position, close, bar_dt, "signal", trade_records, symbol)
                await _persist_last_trade(run_id, trade_records, async_session_factory, Trade)
                position = None

        # ── Update progress message ────────────────────────────────────────────
        async with async_session_factory() as db:
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    message=f"Bar {bar_index} | {symbol} close={close:.4f} | trades={len(trade_records)}"
                )
            )
            await db.commit()

        await asyncio.sleep(_CHECK_INTERVAL)

    # ── Finalise ──────────────────────────────────────────────────────────────
    if position is not None:
        last_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
        _close_position(position, last_close, datetime.now(timezone.utc), "end_of_run", trade_records, symbol)
        await _persist_last_trade(run_id, trade_records, async_session_factory, Trade)

    metrics = _compute_metrics(trade_records, df if not df.empty else None)

    async with async_session_factory() as db:
        for key, value in metrics.items():
            db.add(RunMetric(run_id=run_id, key=key, value=float(value)))
        await db.execute(
            update(StrategyRun).where(StrategyRun.id == run_id).values(
                status="completed",
                progress_pct=100.0,
                ended_at=datetime.now(timezone.utc),
                message=f"{metrics.get('total_trades', 0)} trades, PnL {metrics.get('total_pnl', 0):.4f}",
            )
        )
        await db.commit()

    logger.info(f"Paper run {run_id} completed: {metrics}")
    return metrics


def _fetch_ohlc(symbol: str):
    import yfinance as yf  # type: ignore
    import pandas as pd

    ticker = yf.Ticker(symbol)
    df = ticker.history(period="3mo", interval="1d")
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


async def _persist_last_trade(run_id: int, trade_records: list[dict], session_factory, Trade) -> None:
    if not trade_records:
        return
    t = trade_records[-1]
    async with session_factory() as db:
        db.add(Trade(
            run_id=run_id,
            symbol=t["symbol"],
            direction=t["direction"],
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            volume=t["volume"],
            sl_price=t["sl_price"],
            tp_price=t["tp_price"],
            profit=t["profit"],
            opened_at=t["opened_at"],
            closed_at=t["closed_at"],
        ))
        await db.commit()
