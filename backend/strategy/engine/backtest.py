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
    "daily_loss_limit_pct":    0.0,      # 0 = disabled; circuit breaker
    "cooldown_bars":           0,        # bars to skip after a loss
    "trailing_stop":           false,    # ATR-based trailing stop (requires atr indicator)
    "trailing_atr_multiplier": 3.0,      # distance from close in ATR multiples
    "trailing_clip_with_price": false,   # floor stop at entry price (breakeven guarantee)
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
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from strategy.engine.indicators import apply_indicators, estimate_warmup_bars
from strategy.engine.conditions import evaluate_conditions
from strategy.engine.execution import RiskParams, OpenPosition, calc_volume, fill_price, check_sl_tp, close_trade


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
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    definition: dict,
    artifact_path: str,
    on_progress: Callable[[float], None] | None = None,
    on_trade: Callable[[dict, str], None] | None = None,
    stop_event=None,  # threading.Event — set to abort early
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
    raw_df = _load_df(store / artifact_path)

    definition = _normalise(definition)

    indicator_specs = definition.get("indicators", [])
    warmup_bars = estimate_warmup_bars(indicator_specs)

    rp = RiskParams.from_dict(definition.get("risk", {}))
    symbol = definition.get("symbol", "unknown")
    _model_cache = model_cache or {}

    long_def  = definition.get("long",  {})
    short_def = definition.get("short", {})

    # ATR column for dynamic sizing
    atr_col = next((s["id"] for s in indicator_specs if s.get("type") == "atr"), None)

    n = len(raw_df)
    split_idx = int(n * walk_forward_ratio) if 0 < walk_forward_ratio < 1 else n

    trades: list[dict] = []
    equity_curve: list[dict] = []

    # Simulation state
    pos_long: OpenPosition | None = None
    pos_short: OpenPosition | None = None
    equity = 1.0      # normalised starting equity
    pending_long_entry  = False
    pending_short_entry = False
    last_loss_bar = -9999
    trading_halted_date: date | None = None
    day_loss: dict[date, float] = {}

    # Pre-compute indicators on the full dataset once (O(n) instead of O(n²)).
    # Standard backward-looking indicators give the same value at bar i regardless
    # of whether computed on the full series or only on data up to bar i.
    # Skip pre-computation when n <= warmup_bars: the whole dataset is warmup,
    # so conditions are never evaluated and indicator columns are never needed.
    if indicator_specs and n > warmup_bars:
        df_with_indicators = apply_indicators(raw_df, indicator_specs)
    else:
        df_with_indicators = raw_df

    rows = list(raw_df.iterrows())

    for i, (ts, _) in enumerate(rows):
        if stop_event is not None and stop_event.is_set():
            break
        phase = "is" if i < split_idx else "oos"
        bar_dt = _row_dt(ts)
        bar_date = bar_dt.date()

        if (i + 1) < warmup_bars:
            equity_curve.append({
                "timestamp": bar_dt.isoformat(),
                "equity": round(equity, 6),
                "drawdown": 0.0,
                "phase": phase,
            })
            if on_progress and (i % max(1, n // 200) == 0):
                on_progress(100.0 * i / n)
            continue

        row = df_with_indicators.iloc[i]
        df_upto = df_with_indicators.iloc[: i + 1]
        close = float(row["close"])
        high  = float(row["high"])  if "high"  in row.index else close
        low   = float(row["low"])   if "low"   in row.index else close
        open_ = float(row["open"])  if "open"  in row.index else close
        atr   = float(row[atr_col]) if atr_col and atr_col in row.index and not math.isnan(float(row[atr_col])) else None

        # ── Fill pending entries at this bar's open ───────────────────────
        # num_open is updated after each fill so max_positions is respected
        # even when both long and short are pending simultaneously.
        num_open = (1 if pos_long else 0) + (1 if pos_short else 0)

        if pending_long_entry and pos_long is None and num_open < rp.max_positions:
            fill = fill_price(open_, "buy", rp.slippage_pct)
            sl = fill * (1 - rp.sl_pct) if rp.sl_pct > 0 else None
            tp = fill * (1 + rp.tp_pct) if rp.tp_pct > 0 else None
            vol = calc_volume(rp, equity, fill, atr)
            pos_long = OpenPosition("buy", fill, sl, tp, vol, bar_dt, i)
            num_open += 1
            if on_trade:
                on_trade({"direction": "buy", "entry_price": fill, "sl_price": sl, "tp_price": tp,
                          "opened_at": bar_dt, "symbol": symbol}, "open")

        if pending_short_entry and pos_short is None and num_open < rp.max_positions:
            fill = fill_price(open_, "sell", rp.slippage_pct)
            sl = fill * (1 + rp.sl_pct) if rp.sl_pct > 0 else None
            tp = fill * (1 - rp.tp_pct) if rp.tp_pct > 0 else None
            vol = calc_volume(rp, equity, fill, atr)
            pos_short = OpenPosition("sell", fill, sl, tp, vol, bar_dt, i)
            num_open += 1
            if on_trade:
                on_trade({"direction": "sell", "entry_price": fill, "sl_price": sl, "tp_price": tp,
                          "opened_at": bar_dt, "symbol": symbol}, "open")

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
        # Must run BEFORE trailing stop update so the new stop (based on this
        # bar's close) cannot immediately be triggered by this bar's low/high.
        # trade_strategy checks SL/TP when fetching each new bar, before updating stops.
        if pos_long:
            closed, pos_long = check_sl_tp(pos_long, high, low, bar_dt, trades, symbol, rp, phase)
            if closed:
                pnl = trades[-1]["profit"]
                equity += pnl
                _record_day_loss(day_loss, bar_date, pnl)
                if pnl < 0:
                    last_loss_bar = i
                if on_trade:
                    on_trade(trades[-1], "close")

        if pos_short:
            closed, pos_short = check_sl_tp(pos_short, high, low, bar_dt, trades, symbol, rp, phase)
            if closed:
                pnl = trades[-1]["profit"]
                equity += pnl
                _record_day_loss(day_loss, bar_date, pnl)
                if pnl < 0:
                    last_loss_bar = i
                if on_trade:
                    on_trade(trades[-1], "close")

        # ── Trailing stop update ─────────────────────────────────────────
        # Runs AFTER SL/TP check: the new stop level takes effect from the
        # next bar onward, matching trade_strategy's TrailingStopByATR behaviour.
        if rp.trailing_stop and atr is not None:
            trail_dist = rp.trailing_atr_multiplier * atr
            if pos_long and close > pos_long.entry_price:
                new_sl = close - trail_dist
                if rp.trailing_clip_with_price and new_sl < pos_long.entry_price:
                    new_sl = pos_long.entry_price
                if pos_long.sl_price is None or new_sl > pos_long.sl_price:
                    pos_long.sl_price = new_sl
            if pos_short and close < pos_short.entry_price:
                new_sl = close + trail_dist
                if rp.trailing_clip_with_price and new_sl > pos_short.entry_price:
                    new_sl = pos_short.entry_price
                if pos_short.sl_price is None or new_sl < pos_short.sl_price:
                    pos_short.sl_price = new_sl

        # ── Daily loss circuit breaker ────────────────────────────────────
        halted = (
            rp.daily_loss_limit_pct > 0
            and bar_date in day_loss
            and day_loss[bar_date] >= equity * rp.daily_loss_limit_pct
        )

        # ── Cooldown check ────────────────────────────────────────────────
        in_cooldown = rp.cooldown_bars > 0 and (i - last_loss_bar) < rp.cooldown_bars

        can_enter = not halted and not in_cooldown

        # ── Context for ML / LLM / group_ref conditions ──────────────────
        ctx = {"df_upto": df_upto, "bar_index": i, "model_cache": _model_cache,
               "groups": definition.get("groups", {})}

        # ── Long exit conditions ──────────────────────────────────────────
        if pos_long and long_def.get("exit"):
            try:
                if evaluate_conditions(row, long_def["exit"], context=ctx):
                    fill = fill_price(close, "sell", rp.slippage_pct)
                    pnl = close_trade(pos_long, fill, bar_dt, "signal", trades, symbol, rp, phase)
                    equity += pnl
                    _record_day_loss(day_loss, bar_date, pnl)
                    if pnl < 0:
                        last_loss_bar = i
                    if on_trade:
                        on_trade(trades[-1], "close")
                    pos_long = None
            except (KeyError, ValueError):
                pass

        # ── Short exit conditions ─────────────────────────────────────────
        if pos_short and short_def.get("exit"):
            try:
                if evaluate_conditions(row, short_def["exit"], context=ctx):
                    fill = fill_price(close, "buy", rp.slippage_pct)
                    pnl = close_trade(pos_short, fill, bar_dt, "signal", trades, symbol, rp, phase)
                    equity += pnl
                    _record_day_loss(day_loss, bar_date, pnl)
                    if pnl < 0:
                        last_loss_bar = i
                    if on_trade:
                        on_trade(trades[-1], "close")
                    pos_short = None
            except (KeyError, ValueError):
                pass

        # ── Entry signals (fill next bar) ─────────────────────────────────
        # Re-count after exits so a same-bar exit+entry pair is allowed.
        # num_open is incremented after each queued entry so both directions
        # cannot be queued simultaneously when max_positions=1.
        if can_enter:
            num_open = (1 if pos_long else 0) + (1 if pos_short else 0)

            if pos_long is None and long_def.get("entry") and num_open < rp.max_positions:
                try:
                    if evaluate_conditions(row, long_def["entry"], context=ctx):
                        pending_long_entry = True
                        num_open += 1
                except (KeyError, ValueError):
                    pass

            if pos_short is None and short_def.get("entry") and num_open < rp.max_positions:
                try:
                    if evaluate_conditions(row, short_def["entry"], context=ctx):
                        pending_short_entry = True
                        num_open += 1
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

        if on_progress and (i % max(1, n // 200) == 0):
            on_progress(100.0 * i / n)

    # ── Close any still-open positions at end of data ────────────────────
    if len(rows) > 0:
        last_ts, last_row = rows[-1]
        last_close = float(last_row["close"])
        last_dt = _row_dt(last_ts)
        last_phase = "is" if (n - 1) < split_idx else "oos"
        for pos in [pos_long, pos_short]:
            if pos:
                fill = fill_price(last_close, "sell" if pos.direction == "buy" else "buy", rp.slippage_pct)
                pnl = close_trade(pos, fill, last_dt, "end_of_data", trades, symbol, rp, last_phase)
                equity += pnl

    # ── Compute drawdown on equity curve ─────────────────────────────────
    eq_vals = np.array([p["equity"] for p in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    drawdowns = eq_vals - peak
    for j, p in enumerate(equity_curve):
        p["drawdown"] = round(float(drawdowns[j]), 6)

    metrics = _compute_metrics(trades, equity_curve, raw_df, split_idx)
    return trades, metrics, equity_curve
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
