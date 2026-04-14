"""
Bar-by-bar backtest runner — realistic execution model.

Key execution rules
-------------------
* Signal fires on bar N → fill at OPEN of bar N+1  (no look-ahead bias)
* Slippage applied at fill: buy fills high, sell fills low
* Commission deducted from each trade profit
* SL/TP checked against bar high/low; TP checked before SL (pessimistic)
* Both long and short positions can be open simultaneously
* Walk-forward split: data tagged IS (in-sample) / OOS (out-of-sample)

Definition format (new)
-----------------------
{
  "symbol": "AAPL",
  "indicators": [...],
  "long":  { "entry": <block>, "exit": <block> },
  "short": { "entry": <block>, "exit": <block> },
  "risk": {
    "risk_type":           "fixed" | "percent_equity" | "atr",
    "position_size":       1.0,          # fixed: fraction of equity (default 1.0)
    "risk_pct":            0.01,         # percent_equity / atr: risk per trade
    "atr_multiplier":      2.0,          # atr: stop distance in ATR multiples
    "sl_pct":              0.02,
    "tp_pct":              0.04,
    "slippage_pct":        0.0005,       # 0.05% half-spread per fill
    "commission_pct":      0.001,        # 0.1% round-trip (deducted per trade)
    "max_positions":       1,            # max simultaneous open trades
    "daily_loss_limit_pct":0.0,          # 0 = disabled; circuit breaker
    "cooldown_bars":       0,            # bars to skip after a loss
  }
}

Backward-compatible: old { "entry": {...}, "exit": {...} } is auto-normalised.

Returns (trades, metrics, equity_curve).

Trade dict keys: symbol, direction, entry_price, exit_price, volume,
                 sl_price, tp_price, profit, opened_at, closed_at,
                 exit_reason, phase ("is"|"oos"), mae, mfe.
equity_curve: list[dict] with keys timestamp, equity, drawdown, phase.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from strategy.engine.indicators import apply_indicators
from strategy.engine.conditions import evaluate_conditions


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _OpenPosition:
    direction: str       # "buy" | "sell"
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    volume: float
    opened_at: datetime
    bar_index: int       # bar that triggered entry
    # MAE / MFE tracking
    mae: float = 0.0     # maximum adverse excursion (always positive)
    mfe: float = 0.0     # maximum favorable excursion (always positive)


@dataclass
class _RiskParams:
    risk_type: str           = "fixed"
    position_size: float     = 1.0
    risk_pct: float          = 0.01
    atr_multiplier: float    = 2.0
    sl_pct: float            = 0.02
    tp_pct: float            = 0.04
    slippage_pct: float      = 0.0005
    commission_pct: float    = 0.001
    max_positions: int       = 1
    daily_loss_limit_pct: float = 0.0
    cooldown_bars: int       = 0

    @classmethod
    def from_dict(cls, d: dict) -> "_RiskParams":
        return cls(
            risk_type            = str(d.get("risk_type", "fixed")),
            position_size        = float(d.get("position_size", 1.0)),
            risk_pct             = float(d.get("risk_pct", 0.01)),
            atr_multiplier       = float(d.get("atr_multiplier", 2.0)),
            sl_pct               = float(d.get("sl_pct", 0.02)),
            tp_pct               = float(d.get("tp_pct", 0.04)),
            slippage_pct         = float(d.get("slippage_pct", 0.0005)),
            commission_pct       = float(d.get("commission_pct", 0.001)),
            max_positions        = int(d.get("max_positions", 1)),
            daily_loss_limit_pct = float(d.get("daily_loss_limit_pct", 0.0)),
            cooldown_bars        = int(d.get("cooldown_bars", 0)),
        )


# ---------------------------------------------------------------------------
# Definition normalisation
# ---------------------------------------------------------------------------

def _normalise(definition: dict) -> dict:
    """Convert old { entry, exit } format to { long/short: { entry, exit } }."""
    if "long" in definition or "short" in definition:
        return definition

    entry = definition.get("entry", {})
    exit_ = definition.get("exit", {})
    direction = entry.get("direction", "buy")

    result = {k: v for k, v in definition.items() if k not in ("entry", "exit")}
    if direction == "buy":
        result["long"] = {"entry": entry, "exit": exit_}
    else:
        result["short"] = {"entry": entry, "exit": exit_}
    return result


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _calc_volume(rp: _RiskParams, equity: float, entry_price: float, atr: float | None) -> float:
    if rp.risk_type == "percent_equity" and rp.sl_pct > 0:
        # Risk X% of equity; stop is sl_pct of price
        return (equity * rp.risk_pct) / (entry_price * rp.sl_pct)
    if rp.risk_type == "atr" and atr is not None and atr > 0:
        stop_dist = atr * rp.atr_multiplier
        return (equity * rp.risk_pct) / stop_dist
    return rp.position_size


def _fill_price(mid: float, direction: str, slippage_pct: float) -> float:
    """Apply slippage: buys fill above mid, sells fill below mid."""
    if direction == "buy":
        return mid * (1 + slippage_pct)
    return mid * (1 - slippage_pct)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    definition: dict,
    artifact_path: str,
    on_progress: Callable[[float], None] | None = None,
    model_cache: dict | None = None,
    walk_forward_ratio: float = 0.0,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Load the dataset parquet and run a bar-by-bar simulation.

    walk_forward_ratio: fraction of bars used as in-sample (0 = disabled).
    Returns (trades, summary_metrics, equity_curve).
    """
    from strategy.engine.llm_condition import clear_cache as _clear_llm_cache
    _clear_llm_cache()

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = _load_df(store / artifact_path)

    definition = _normalise(definition)

    indicator_specs = definition.get("indicators", [])
    if indicator_specs:
        df = apply_indicators(df, indicator_specs)

    rp = _RiskParams.from_dict(definition.get("risk", {}))
    symbol = definition.get("symbol", "unknown")
    _model_cache = model_cache or {}

    long_def  = definition.get("long",  {})
    short_def = definition.get("short", {})

    # ATR column for dynamic sizing
    atr_col = next((s["id"] for s in indicator_specs if s.get("type") == "atr"), None)

    n = len(df)
    split_idx = int(n * walk_forward_ratio) if 0 < walk_forward_ratio < 1 else n

    trades: list[dict] = []
    equity_curve: list[dict] = []

    # Simulation state
    pos_long:  _OpenPosition | None = None
    pos_short: _OpenPosition | None = None
    equity = 1.0      # normalised starting equity
    pending_long_entry  = False
    pending_short_entry = False
    last_loss_bar = -9999
    trading_halted_date: date | None = None
    day_loss: dict[date, float] = {}

    rows = list(df.iterrows())

    for i, (ts, row) in enumerate(rows):
        phase = "is" if i < split_idx else "oos"
        close = float(row["close"])
        high  = float(row["high"])  if "high"  in row.index else close
        low   = float(row["low"])   if "low"   in row.index else close
        open_ = float(row["open"])  if "open"  in row.index else close
        atr   = float(row[atr_col]) if atr_col and atr_col in row.index and not math.isnan(float(row[atr_col])) else None

        bar_dt = _row_dt(ts)
        bar_date = bar_dt.date()

        # ── Fill pending entries at this bar's open ───────────────────────
        num_open = (1 if pos_long else 0) + (1 if pos_short else 0)

        if pending_long_entry and pos_long is None and num_open < rp.max_positions:
            fill = _fill_price(open_, "buy", rp.slippage_pct)
            sl = fill * (1 - rp.sl_pct) if rp.sl_pct > 0 else None
            tp = fill * (1 + rp.tp_pct) if rp.tp_pct > 0 else None
            vol = _calc_volume(rp, equity, fill, atr)
            pos_long = _OpenPosition("buy", fill, sl, tp, vol, bar_dt, i)

        if pending_short_entry and pos_short is None and num_open < rp.max_positions:
            fill = _fill_price(open_, "sell", rp.slippage_pct)
            sl = fill * (1 + rp.sl_pct) if rp.sl_pct > 0 else None
            tp = fill * (1 - rp.tp_pct) if rp.tp_pct > 0 else None
            vol = _calc_volume(rp, equity, fill, atr)
            pos_short = _OpenPosition("sell", fill, sl, tp, vol, bar_dt, i)

        pending_long_entry = False
        pending_short_entry = False

        # ── Update MAE / MFE on open positions ───────────────────────────
        if pos_long:
            fav  = high - pos_long.entry_price
            adv  = pos_long.entry_price - low
            pos_long.mfe = max(pos_long.mfe, fav)
            pos_long.mae = max(pos_long.mae, adv)
        if pos_short:
            fav  = pos_short.entry_price - low
            adv  = high - pos_short.entry_price
            pos_short.mfe = max(pos_short.mfe, fav)
            pos_short.mae = max(pos_short.mae, adv)

        # ── SL / TP checks ───────────────────────────────────────────────
        if pos_long:
            closed, pos_long = _check_sl_tp(pos_long, high, low, close, bar_dt, trades, symbol, rp, equity, phase)
            if closed:
                pnl = trades[-1]["profit"]
                equity += pnl
                _record_day_loss(day_loss, bar_date, pnl)
                if pnl < 0:
                    last_loss_bar = i

        if pos_short:
            closed, pos_short = _check_sl_tp(pos_short, high, low, close, bar_dt, trades, symbol, rp, equity, phase)
            if closed:
                pnl = trades[-1]["profit"]
                equity += pnl
                _record_day_loss(day_loss, bar_date, pnl)
                if pnl < 0:
                    last_loss_bar = i

        # ── Daily loss circuit breaker ────────────────────────────────────
        halted = (
            rp.daily_loss_limit_pct > 0
            and bar_date in day_loss
            and day_loss[bar_date] >= equity * rp.daily_loss_limit_pct
        )

        # ── Cooldown check ────────────────────────────────────────────────
        in_cooldown = rp.cooldown_bars > 0 and (i - last_loss_bar) < rp.cooldown_bars

        can_enter = not halted and not in_cooldown

        # ── Context for ML / LLM conditions ──────────────────────────────
        ctx = {"df_upto": df.iloc[: i + 1], "bar_index": i, "model_cache": _model_cache}

        # ── Long exit conditions ──────────────────────────────────────────
        if pos_long and long_def.get("exit"):
            try:
                if evaluate_conditions(row, long_def["exit"], context=ctx):
                    fill = _fill_price(close, "sell", rp.slippage_pct)
                    pnl = _close(pos_long, fill, bar_dt, "signal", trades, symbol, rp, phase)
                    equity += pnl
                    _record_day_loss(day_loss, bar_date, pnl)
                    if pnl < 0:
                        last_loss_bar = i
                    pos_long = None
            except (KeyError, ValueError):
                pass

        # ── Short exit conditions ─────────────────────────────────────────
        if pos_short and short_def.get("exit"):
            try:
                if evaluate_conditions(row, short_def["exit"], context=ctx):
                    fill = _fill_price(close, "buy", rp.slippage_pct)
                    pnl = _close(pos_short, fill, bar_dt, "signal", trades, symbol, rp, phase)
                    equity += pnl
                    _record_day_loss(day_loss, bar_date, pnl)
                    if pnl < 0:
                        last_loss_bar = i
                    pos_short = None
            except (KeyError, ValueError):
                pass

        # ── Entry signals (fill next bar) ─────────────────────────────────
        if can_enter:
            num_open = (1 if pos_long else 0) + (1 if pos_short else 0)

            if pos_long is None and long_def.get("entry") and num_open < rp.max_positions:
                try:
                    if evaluate_conditions(row, long_def["entry"], context=ctx):
                        pending_long_entry = True
                except (KeyError, ValueError):
                    pass

            if pos_short is None and short_def.get("entry") and num_open < rp.max_positions:
                try:
                    if evaluate_conditions(row, short_def["entry"], context=ctx):
                        pending_short_entry = True
                except (KeyError, ValueError):
                    pass

        # ── Equity curve snapshot (every bar) ────────────────────────────
        unrealised = 0.0
        if pos_long:
            unrealised += (close - pos_long.entry_price) / pos_long.entry_price * pos_long.volume
        if pos_short:
            unrealised += (pos_short.entry_price - close) / pos_short.entry_price * pos_short.volume

        equity_curve.append({
            "timestamp": bar_dt.isoformat(),
            "equity": round(equity + unrealised, 6),
            "drawdown": 0.0,   # filled in below
            "phase": phase,
        })

        if on_progress and (i % max(1, n // 100) == 0):
            on_progress(100.0 * i / n)

    # ── Close any still-open positions at end of data ────────────────────
    if len(rows) > 0:
        last_ts, last_row = rows[-1]
        last_close = float(last_row["close"])
        last_dt = _row_dt(last_ts)
        last_phase = "is" if (n - 1) < split_idx else "oos"
        for pos in [pos_long, pos_short]:
            if pos:
                fill = _fill_price(last_close, "sell" if pos.direction == "buy" else "buy", rp.slippage_pct)
                pnl = _close(pos, fill, last_dt, "end_of_data", trades, symbol, rp, last_phase)
                equity += pnl

    # ── Compute drawdown on equity curve ─────────────────────────────────
    eq_vals = np.array([p["equity"] for p in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    drawdowns = eq_vals - peak
    for j, p in enumerate(equity_curve):
        p["drawdown"] = round(float(drawdowns[j]), 6)

    metrics = _compute_metrics(trades, equity_curve, df, split_idx)
    return trades, metrics, equity_curve


# ---------------------------------------------------------------------------
# SL / TP check
# ---------------------------------------------------------------------------

def _check_sl_tp(
    pos: _OpenPosition,
    high: float, low: float, close: float,
    bar_dt: datetime,
    trades: list[dict],
    symbol: str,
    rp: _RiskParams,
    equity: float,
    phase: str,
) -> tuple[bool, _OpenPosition | None]:
    """Check SL/TP against bar range. TP before SL (pessimistic for P&L)."""
    if pos.direction == "buy":
        if pos.tp_price and high >= pos.tp_price:
            fill = _fill_price(pos.tp_price, "sell", rp.slippage_pct)
            _close(pos, fill, bar_dt, "tp", trades, symbol, rp, phase)
            return True, None
        if pos.sl_price and low <= pos.sl_price:
            fill = _fill_price(pos.sl_price, "sell", rp.slippage_pct)
            _close(pos, fill, bar_dt, "sl", trades, symbol, rp, phase)
            return True, None
    else:  # sell
        if pos.tp_price and low <= pos.tp_price:
            fill = _fill_price(pos.tp_price, "buy", rp.slippage_pct)
            _close(pos, fill, bar_dt, "tp", trades, symbol, rp, phase)
            return True, None
        if pos.sl_price and high >= pos.sl_price:
            fill = _fill_price(pos.sl_price, "buy", rp.slippage_pct)
            _close(pos, fill, bar_dt, "sl", trades, symbol, rp, phase)
            return True, None
    return False, pos


# ---------------------------------------------------------------------------
# Close position
# ---------------------------------------------------------------------------

def _close(
    pos: _OpenPosition,
    exit_price: float,
    closed_at: datetime,
    reason: str,
    trades: list[dict],
    symbol: str,
    rp: _RiskParams,
    phase: str,
) -> float:
    if pos.direction == "buy":
        gross = (exit_price - pos.entry_price) / pos.entry_price * pos.volume
    else:
        gross = (pos.entry_price - exit_price) / pos.entry_price * pos.volume

    commission = rp.commission_pct * pos.volume
    profit = round(gross - commission, 6)

    mae_pct = pos.mae / pos.entry_price if pos.entry_price > 0 else 0.0
    mfe_pct = pos.mfe / pos.entry_price if pos.entry_price > 0 else 0.0

    trades.append({
        "symbol":      symbol,
        "direction":   pos.direction,
        "entry_price": pos.entry_price,
        "exit_price":  exit_price,
        "volume":      pos.volume,
        "sl_price":    pos.sl_price,
        "tp_price":    pos.tp_price,
        "profit":      profit,
        "opened_at":   pos.opened_at,
        "closed_at":   closed_at,
        "exit_reason": reason,
        "phase":       phase,
        "mae":         round(mae_pct, 6),
        "mfe":         round(mfe_pct, 6),
    })
    return profit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_day_loss(day_loss: dict[date, float], d: date, pnl: float) -> None:
    if pnl < 0:
        day_loss[d] = day_loss.get(d, 0.0) + abs(pnl)


_MAX_TICK_FILES = 100

def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")
    if path.is_dir():
        from data.parquet_reader import load_ddm_ticks
        tick_df = load_ddm_ticks(path, max_files=_MAX_TICK_FILES)
        ohlc = tick_df["price"].resample("1min").ohlc()
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = tick_df["price"].resample("1min").count()
        df = ohlc.dropna()
    else:
        df = pd.read_parquet(path)

    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        raise ValueError("Dataset missing required column: 'close'")
    return df.sort_index()


def _row_dt(ts) -> datetime:
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics_for(trades: list[dict]) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "profit_factor": 0.0, "avg_trade_pnl": 0.0,
            "max_drawdown": 0.0, "sharpe_ratio": 0.0,
            "avg_mae": 0.0, "avg_mfe": 0.0, "max_consecutive_losses": 0,
        }
    profits = [t["profit"] for t in trades]
    wins    = [p for p in profits if p > 0]
    losses  = [p for p in profits if p <= 0]

    total_pnl     = sum(profits)
    win_rate      = len(wins) / len(profits)
    avg_trade_pnl = total_pnl / len(profits)
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

    # Drawdown from equity curve of these trades
    equity = np.cumsum([0.0] + profits)
    peak   = np.maximum.accumulate(equity)
    max_dd = float(abs((equity - peak).min()))

    # Sharpe from trade returns (normalised)
    p_arr = np.array(profits)
    sharpe = float(p_arr.mean() / (p_arr.std() + 1e-12) * math.sqrt(252)) if len(profits) > 1 else 0.0

    # MAE / MFE averages
    avg_mae = float(np.mean([t.get("mae", 0.0) for t in trades]))
    avg_mfe = float(np.mean([t.get("mfe", 0.0) for t in trades]))

    # Max consecutive losses
    max_consec = cur = 0
    for p in profits:
        if p < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    return {
        "total_trades":          len(trades),
        "win_rate":              round(win_rate, 4),
        "total_pnl":             round(total_pnl, 6),
        "profit_factor":         round(min(profit_factor, 999.0), 4),
        "avg_trade_pnl":         round(avg_trade_pnl, 6),
        "max_drawdown":          round(max_dd, 6),
        "sharpe_ratio":          round(sharpe, 4),
        "avg_mae":               round(avg_mae, 6),
        "avg_mfe":               round(avg_mfe, 6),
        "max_consecutive_losses":max_consec,
    }


def _compute_metrics(
    trades: list[dict],
    equity_curve: list[dict],
    df: pd.DataFrame,
    split_idx: int,
) -> dict:
    walk_forward = 0 < split_idx < len(df)

    if not walk_forward:
        return _metrics_for(trades)

    is_trades  = [t for t in trades if t.get("phase") == "is"]
    oos_trades = [t for t in trades if t.get("phase") == "oos"]

    is_m  = {f"is_{k}":  v for k, v in _metrics_for(is_trades).items()}
    oos_m = {f"oos_{k}": v for k, v in _metrics_for(oos_trades).items()}

    combined = _metrics_for(trades)
    return {**combined, **is_m, **oos_m}
