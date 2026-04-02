"""
Bar-by-bar backtest runner.

Strategy definition format (stored in strategies.definition JSONB):
{
    "symbol": "AAPL",
    "indicators": [
        {"id": "macd", "type": "macd", "params": {"fast": 12, "slow": 26, "signal_period": 9}},
        {"id": "rsi",  "type": "rsi",  "params": {"period": 14}},
        {"id": "atr",  "type": "atr",  "params": {"period": 14}}
    ],
    "entry": {
        "direction": "buy",
        "conditions": [
            {"left": "macd_line",   "op": ">", "right": "macd_signal"},
            {"left": "rsi",         "op": "<", "right": 70}
        ],
        "logic": "and"
    },
    "exit": {
        "conditions": [
            {"left": "macd_line", "op": "<", "right": "macd_signal"}
        ],
        "logic": "or"
    },
    "risk": {
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "position_size": 1.0
    }
}

Returns (trades, metrics, equity_curve).

Each trade dict keys: symbol, direction, entry_price, exit_price, volume,
                      sl_price, tp_price, profit, opened_at, closed_at.
equity_curve: list[dict] with keys timestamp, equity, drawdown.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from strategy.engine.indicators import apply_indicators
from strategy.engine.conditions import evaluate_conditions


@dataclass
class _OpenPosition:
    direction: str       # "buy" | "sell"
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    volume: float
    opened_at: datetime


def run_backtest(
    definition: dict,
    artifact_path: str,
    on_progress: Callable[[float], None] | None = None,
    model_cache: dict | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Load the dataset parquet and run a bar-by-bar simulation.

    on_progress(pct: float) is called periodically (0.0–100.0).
    model_cache: pre-loaded model metadata for ml_signal conditions.
    Returns (trades, summary_metrics).
    """
    from strategy.engine.llm_condition import clear_cache as _clear_llm_cache
    _clear_llm_cache()

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = _load_df(store / artifact_path)

    # Apply indicators
    indicator_specs = definition.get("indicators", [])
    if indicator_specs:
        df = apply_indicators(df, indicator_specs)

    entry_block = definition.get("entry", {})
    exit_block = definition.get("exit", {})
    risk = definition.get("risk", {})

    direction = entry_block.get("direction", "buy")
    sl_pct = float(risk.get("sl_pct", 0.0))
    tp_pct = float(risk.get("tp_pct", 0.0))
    position_size = float(risk.get("position_size", 1.0))

    symbol = definition.get("symbol", "unknown")
    _model_cache = model_cache or {}

    trades: list[dict] = []
    position: _OpenPosition | None = None
    n = len(df)

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        high = float(row["high"]) if "high" in row.index else close
        low = float(row["low"]) if "low" in row.index else close

        if isinstance(ts, pd.Timestamp):
            bar_dt = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            bar_dt = datetime.now(timezone.utc)

        # Context passed to condition evaluator for ML/LLM conditions
        ctx = {
            "df_upto": df.iloc[: i + 1],
            "bar_index": i,
            "model_cache": _model_cache,
        }

        if position is not None:
            # Check SL/TP within bar range
            closed = _check_sl_tp(position, high, low, close, bar_dt, trades, symbol)
            if closed:
                position = None

        if position is None:
            # Check entry conditions
            try:
                should_enter = evaluate_conditions(row, entry_block, context=ctx)
            except (KeyError, ValueError):
                should_enter = False

            if should_enter:
                entry_price = close
                sl_price = entry_price * (1 - sl_pct) if sl_pct > 0 else None
                tp_price = entry_price * (1 + tp_pct) if tp_pct > 0 else None
                if direction == "sell":
                    sl_price = entry_price * (1 + sl_pct) if sl_pct > 0 else None
                    tp_price = entry_price * (1 - tp_pct) if tp_pct > 0 else None
                position = _OpenPosition(
                    direction=direction,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    volume=position_size,
                    opened_at=bar_dt,
                )
        else:
            # Check exit conditions (close at this bar's close)
            try:
                should_exit = evaluate_conditions(row, exit_block, context=ctx)
            except (KeyError, ValueError):
                should_exit = False

            if should_exit:
                _close_position(position, close, bar_dt, "signal", trades, symbol)
                position = None

        if on_progress and (i % max(1, n // 100) == 0):
            on_progress(100.0 * i / n)

    # Close any open position at end of data
    if position is not None and n > 0:
        last_close = float(df["close"].iloc[-1])
        last_dt = _row_dt(df.index[-1])
        _close_position(position, last_close, last_dt, "end_of_data", trades, symbol)

    metrics, equity_curve = _compute_metrics(trades, df)
    return trades, metrics, equity_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MAX_TICK_FILES = 100   # 100 × 10 000 ticks → resamples to ~3 333 M1 candles

def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")

    if path.is_dir():
        # DDM tick directory: load a capped sample and resample to M1 OHLC
        from data.parquet_reader import load_ddm_ticks
        tick_df = load_ddm_ticks(path, max_files=_MAX_TICK_FILES)
        ohlc = tick_df["price"].resample("1min").ohlc()
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = tick_df["price"].resample("1min").count()
        df = ohlc.dropna()
    else:
        df = pd.read_parquet(path)

    df.columns = [c.lower() for c in df.columns]
    for required in ("close",):
        if required not in df.columns:
            raise ValueError(f"Dataset missing required column: {required!r}")
    return df.sort_index()


def _row_dt(ts) -> datetime:
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _check_sl_tp(
    pos: _OpenPosition,
    high: float,
    low: float,
    close: float,
    bar_dt: datetime,
    trades: list[dict],
    symbol: str,
) -> bool:
    """Return True if position was closed by SL or TP."""
    if pos.direction == "buy":
        if pos.tp_price and high >= pos.tp_price:
            _close_position(pos, pos.tp_price, bar_dt, "tp", trades, symbol)
            return True
        if pos.sl_price and low <= pos.sl_price:
            _close_position(pos, pos.sl_price, bar_dt, "sl", trades, symbol)
            return True
    else:  # sell
        if pos.tp_price and low <= pos.tp_price:
            _close_position(pos, pos.tp_price, bar_dt, "tp", trades, symbol)
            return True
        if pos.sl_price and high >= pos.sl_price:
            _close_position(pos, pos.sl_price, bar_dt, "sl", trades, symbol)
            return True
    return False


def _close_position(
    pos: _OpenPosition,
    exit_price: float,
    closed_at: datetime,
    reason: str,
    trades: list[dict],
    symbol: str,
) -> None:
    if pos.direction == "buy":
        profit = (exit_price - pos.entry_price) / pos.entry_price * pos.volume
    else:
        profit = (pos.entry_price - exit_price) / pos.entry_price * pos.volume

    trades.append({
        "symbol": symbol,
        "direction": pos.direction,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "volume": pos.volume,
        "sl_price": pos.sl_price,
        "tp_price": pos.tp_price,
        "profit": round(profit, 6),
        "opened_at": pos.opened_at,
        "closed_at": closed_at,
        "exit_reason": reason,
    })


def _compute_metrics(trades: list[dict], df: pd.DataFrame) -> tuple[dict, list[dict]]:
    empty_curve: list[dict] = []
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }, empty_curve

    profits = [t["profit"] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    total_pnl = sum(profits)
    win_rate = len(wins) / len(profits) if profits else 0.0
    avg_trade_pnl = total_pnl / len(profits)
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

    # Equity curve (per-trade points)
    equity = np.cumsum([0.0] + profits)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    max_drawdown = float(abs(drawdowns.min()))

    equity_curve = []
    for i, t in enumerate(trades):
        ts = t.get("closed_at")
        equity_curve.append({
            "timestamp": ts.isoformat() if ts else None,
            "equity": round(float(equity[i + 1]), 6),
            "drawdown": round(float(drawdowns[i + 1]), 6),
        })

    # Approximate annualised Sharpe from trade returns
    if len(profits) > 1:
        p_arr = np.array(profits)
        sharpe_ratio = float(p_arr.mean() / (p_arr.std() + 1e-12) * math.sqrt(252 / max(1, len(df) / len(profits))))
    else:
        sharpe_ratio = 0.0

    metrics = {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 6),
        "profit_factor": round(min(profit_factor, 999.0), 4),
        "avg_trade_pnl": round(avg_trade_pnl, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
    }
    return metrics, equity_curve
