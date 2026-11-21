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
    "trailing_stop":            false,    # ATR-based trailing stop (requires atr indicator)
    "trailing_atr_multiplier":  3.0,     # distance from close in ATR multiples
    "trailing_clip_with_price": false,   # clip stop to entry price (breakeven floor)
    "trailing_only_in_profit":  true,    # only move stop when position is profitable (matches trade_strategy default)
    "swap_col":                 "carry", # optional: indicator column with annual long-side carry; accrued per day held
    "swap_col_short":           "",      # optional: separate (already signed) accrual column for sells, e.g. "carry_short"
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
from datetime import datetime, timedelta, timezone, date
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
    """
    Normalises strategy definitions:
    - Transpiles analyst_* forecast strategies into standard indicators + long/short blocks.
    - Converts old { entry, exit } format to { long/short: { entry, exit } }.
    """
    strat_type = definition.get("strategy_type", "")
    if strat_type.startswith("analyst_"):
        if strat_type not in {"analyst_range_reversion", "analyst_breakout_momentum"}:
            raise ValueError(f"Unsupported analyst strategy type: {strat_type!r}")
        params = definition.get("parameters", {})
        sym = definition.get("symbol", "USDJPY")
        source = params.get("source", "FX168")
        tol_pips = float(params.get("tolerance_pips", 10.0) or 0.0)
        sl_pips = params.get("sl_pips")
        tp_pips = params.get("tp_pips")
        ml_filter = params.get("ml_filter")
        ml_enabled = isinstance(ml_filter, dict) and ml_filter.get("enabled", False)

        indicators = list(definition.get("indicators", []))
        if not any(i.get("type") == "analyst_forecast" for i in indicators):
            indicators.append({
                "id": "forecast",
                "type": "analyst_forecast",
                "params": {"symbol": sym, "source": source, "tolerance_pips": tol_pips}
            })

        if ml_enabled and strat_type == "analyst_range_reversion":
            indicators.append({"id": "rebound_features", "type": "rebound_features"})

        long_conds = [{"left": "has_forecast", "op": "==", "right": 1}]
        short_conds = [{"left": "has_forecast", "op": "==", "right": 1}]

        if strat_type == "analyst_range_reversion":
            long_conds.append({"left": "low", "op": "<=", "right": "buy_boundary"})
            short_conds.append({"left": "high", "op": ">=", "right": "sell_boundary"})
            if ml_enabled:
                long_conds.append({
                    "type": "ml_rebound_filter",
                    "threshold": ml_filter.get("threshold", 0.50),
                    "model_path": ml_filter.get("model_path"),
                    "direction": "buy"
                })
                short_conds.append({
                    "type": "ml_rebound_filter",
                    "threshold": ml_filter.get("threshold", 0.50),
                    "model_path": ml_filter.get("model_path"),
                    "direction": "sell"
                })
        elif strat_type == "analyst_breakout_momentum":
            long_conds.append({"left": "high", "op": ">", "right": "expected_high"})
            short_conds.append({"left": "low", "op": "<", "right": "expected_low"})

        risk = dict(definition.get("risk", {}))
        pip_unit = 0.01 if "JPY" in sym.upper() else 0.0001
        risk["pip_size"] = pip_unit
        risk.setdefault("sl_pct", 0.0)
        risk.setdefault("tp_pct", 0.0)
        if sl_pips is not None and str(sl_pips).lower() != "none" and "sl_pips" not in risk:
            risk["sl_pips"] = float(sl_pips)
        if tp_pips is not None and str(tp_pips).lower() not in ("none", "opposite") and "tp_pips" not in risk:
            risk["tp_pips"] = float(tp_pips)
        if "max_positions" not in risk:
            risk["max_positions"] = 1

        return {
            **definition,
            "indicators": indicators,
            "long": {"entry": {"conditions": long_conds, "logic": "and"}, "exit": {}},
            "short": {"entry": {"conditions": short_conds, "logic": "and"}, "exit": {}},
            "risk": risk,
        }

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
    window_size: int = 0,
    starting_capital: float = 1.0,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Load the dataset parquet and run a bar-by-bar simulation.

    walk_forward_ratio: fraction of bars used as in-sample (0 = disabled).
    window_size: rolling indicator window in bars (0 = full history). Use N > 0
        for path-dependent indicators like Renko to match live trading behaviour;
        indicators are recomputed on the last N bars at each step (slower).
    Returns (trades, summary_metrics, equity_curve).
    """
    from strategy.engine.llm_condition import clear_cache as _clear_llm_cache
    _clear_llm_cache()

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    raw_df = _load_df(store / artifact_path)
    if from_ts is not None or to_ts is not None:
        index = pd.to_datetime(raw_df.index, utc=True)
        mask = np.ones(len(index), dtype=bool)
        if from_ts is not None:
            mask &= index >= pd.to_datetime(from_ts, utc=True)
        if to_ts is not None:
            mask &= index <= pd.to_datetime(to_ts, utc=True)
        raw_df = raw_df.loc[mask]
        if raw_df.empty:
            raise ValueError("No market data in the requested backtest period")

    # Note: analyst_* strategies are transpiled by _normalise() into standard indicators/conditions.

    definition = _normalise(definition)
    analyst = definition.get("strategy_type", "").startswith("analyst_")
    analyst_params = definition.get("parameters", {}) if analyst else {}
    time_stop = analyst and analyst_params.get("exit_at_close", True)
    opposite_tp = analyst and str(analyst_params.get("tp_pips")).lower() == "opposite"
    if analyst:
        from strategy.engine.forecast_levels import ANALYST_CLOSE_UTC

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
    entered_on = {}
    analyst_signal_at = None
    long_target = short_target = None
    pending_long_entry  = False
    pending_short_entry = False
    last_loss_bar = -9999
    trading_halted_date: date | None = None
    day_loss: dict[date, float] = {}

    # Rolling-window mode: recompute indicators on the last window_size bars at each
    # step. Required for path-dependent indicators (e.g. Renko) to match live
    # behaviour. Full-history mode (window_size=0): precompute once — O(n) vs O(n×w).
    effective_warmup = max(warmup_bars, window_size) if window_size > 0 else warmup_bars
    if window_size > 0:
        df_with_indicators = None  # computed per-bar below
    elif indicator_specs and n > effective_warmup:
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

        if (i + 1) < effective_warmup:
            equity_curve.append({
                "timestamp": bar_dt.isoformat(),
                "equity": round(equity, 6),
                "drawdown": 0.0,
                "phase": phase,
            })
            if on_progress and (i % max(1, n // 200) == 0):
                on_progress(100.0 * i / n)
            continue

        if window_size > 0:
            start = max(0, i - window_size + 1)
            slice_df = raw_df.iloc[start : i + 1]
            if indicator_specs:
                slice_df = apply_indicators(slice_df, indicator_specs)
            row = slice_df.iloc[-1]
            df_upto = slice_df
        else:
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

        entry_window = not analyst or (
            analyst_signal_at is not None
            and bar_dt == analyst_signal_at + timedelta(minutes=30)
            and bar_dt.date() == analyst_signal_at.date()
            and bar_dt.time() < ANALYST_CLOSE_UTC
        )

        if time_stop:
            for position in (pos_long, pos_short):
                if position and (bar_dt.time() >= ANALYST_CLOSE_UTC or bar_date != position.opened_at.date()):
                    side = "sell" if position.direction == "buy" else "buy"
                    pnl = close_trade(position, fill_price(open_, side, rp.slippage_pct),
                                      bar_dt, "day_close", trades, symbol, rp, phase)
                    equity += pnl
                    _record_day_loss(day_loss, bar_date, pnl)
                    if pnl < 0:
                        last_loss_bar = i
                    if on_trade:
                        on_trade(trades[-1], "close")
                    if position.direction == "buy":
                        pos_long = None
                    else:
                        pos_short = None
            num_open = (1 if pos_long else 0) + (1 if pos_short else 0)

        if entry_window and pending_long_entry and pos_long is None and num_open < rp.max_positions:
            fill = fill_price(open_, "buy", rp.slippage_pct)
            if rp.sl_pips > 0:
                sl = fill - (rp.sl_pips * rp.pip_size)
            elif rp.sl_pct > 0:
                sl = fill * (1 - rp.sl_pct)
            else:
                sl = None

            if rp.tp_pips > 0:
                tp = fill + (rp.tp_pips * rp.pip_size)
            elif rp.tp_pct > 0:
                tp = fill * (1 + rp.tp_pct)
            else:
                tp = None

            if opposite_tp and long_target > fill:
                tp = long_target
            if analyst:
                entered_on["buy"] = bar_date
            vol = calc_volume(rp, equity, fill, atr)
            pos_long = OpenPosition("buy", fill, sl, tp, vol, bar_dt, i)
            num_open += 1
            if on_trade:
                on_trade({"direction": "buy", "entry_price": fill, "sl_price": sl, "tp_price": tp,
                          "opened_at": bar_dt, "symbol": symbol}, "open")

        if entry_window and pending_short_entry and pos_short is None and num_open < rp.max_positions:
            fill = fill_price(open_, "sell", rp.slippage_pct)
            if rp.sl_pips > 0:
                sl = fill + (rp.sl_pips * rp.pip_size)
            elif rp.sl_pct > 0:
                sl = fill * (1 + rp.sl_pct)
            else:
                sl = None

            if rp.tp_pips > 0:
                tp = fill - (rp.tp_pips * rp.pip_size)
            elif rp.tp_pct > 0:
                tp = fill * (1 - rp.tp_pct)
            else:
                tp = None

            if opposite_tp and short_target < fill:
                tp = short_target
            if analyst:
                entered_on["sell"] = bar_date
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
        # trailing_only_in_profit=True (default): stop is only moved when the
        # position is currently profitable, matching TrailingStopByATR's
        # "close > entry_price" guard in trade_strategy.
        if rp.trailing_stop and atr is not None:
            trail_dist = rp.trailing_atr_multiplier * atr
            long_in_profit = (not rp.trailing_only_in_profit) or (close > pos_long.entry_price) if pos_long else False
            short_in_profit = (not rp.trailing_only_in_profit) or (close < pos_short.entry_price) if pos_short else False
            if pos_long and long_in_profit:
                new_sl = close - trail_dist
                if rp.trailing_clip_with_price and new_sl < pos_long.entry_price:
                    new_sl = pos_long.entry_price
                if pos_long.sl_price is None or new_sl > pos_long.sl_price:
                    pos_long.sl_price = new_sl
            if pos_short and short_in_profit:
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

            if (pos_long is None and long_def.get("entry") and num_open < rp.max_positions
                    and (not analyst or entered_on.get("buy") != bar_date)):
                try:
                    if evaluate_conditions(row, long_def["entry"], context=ctx):
                        pending_long_entry = True
                        analyst_signal_at = bar_dt
                        if opposite_tp:
                            long_target = float(row["expected_high"])
                        num_open += 1
                except (KeyError, ValueError):
                    pass

            if (pos_short is None and short_def.get("entry") and num_open < rp.max_positions
                    and (not analyst or entered_on.get("sell") != bar_date)):
                try:
                    if evaluate_conditions(row, short_def["entry"], context=ctx):
                        pending_short_entry = True
                        analyst_signal_at = bar_dt
                        if opposite_tp:
                            short_target = float(row["expected_low"])
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

    # ── Swap / carry accrual (risk.swap_col): annual rate differential accrued per calendar day held ──
    if rp.swap_col and trades and df_with_indicators is not None and rp.swap_col in df_with_indicators.columns:
        short_rate = (df_with_indicators[rp.swap_col_short]
                      if rp.swap_col_short and rp.swap_col_short in df_with_indicators.columns else None)
        _apply_swap(trades, equity_curve, df_with_indicators[rp.swap_col], short_rate)

    # ── Compute drawdown on equity curve ─────────────────────────────────
    eq_vals = np.array([p["equity"] for p in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    drawdowns = eq_vals - peak
    for j, p in enumerate(equity_curve):
        p["drawdown"] = round(float(drawdowns[j]), 6)

    metrics = _compute_metrics(trades, equity_curve, raw_df, split_idx)

    if starting_capital != 1.0:
        sc = starting_capital
        for t in trades:
            t["profit"] = round(t["profit"] * sc, 6)
        for p in equity_curve:
            p["equity"] = round(p["equity"] * sc, 4)
            p["drawdown"] = round(p["drawdown"] * sc, 4)
        metrics = _compute_metrics(trades, equity_curve, raw_df, split_idx)

    return trades, metrics, equity_curve
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_swap(trades: list[dict], equity_curve: list[dict], rate: pd.Series,
                short_rate: pd.Series | None = None) -> None:
    """Add carry accrual to every trade's profit and shift the equity curve accordingly.

    accrual = side * volume * mean(rate over holding period) * calendar_days / 365, where `rate` is the annual
    long-side differential (base - quote) and side is +1 for buy, -1 for sell. Profit is in the engine's normalised
    equity units (volume is a fraction of equity), so this adds directly.
    """
    idx = pd.DatetimeIndex(rate.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    r = pd.Series(rate.values, index=idx)
    rs = pd.Series(short_rate.values, index=idx) if short_rate is not None else None
    steps = []
    for t in trades:
        a, b = pd.Timestamp(t["opened_at"]), pd.Timestamp(t["closed_at"])
        a = a.tz_convert(None) if a.tzinfo else a
        b = b.tz_convert(None) if b.tzinfo else b
        src = rs if (rs is not None and t["direction"] != "buy") else r
        seg = src[(src.index >= a) & (src.index <= b)]
        if seg.empty or b <= a:
            continue
        side = 1.0 if (t["direction"] == "buy" or src is rs) else -1.0
        acc = side * float(t["volume"]) * float(seg.mean()) * (b - a).days / 365.0
        t["profit"] = round(t["profit"] + acc, 6)
        steps.append((b, acc))
    if not steps:
        return
    steps.sort()
    ts = [pd.Timestamp(p["timestamp"]) for p in equity_curve]
    ts = [x.tz_convert(None) if x.tzinfo else x for x in ts]
    j, cum = 0, 0.0
    for k, p in enumerate(equity_curve):
        while j < len(steps) and steps[j][0] <= ts[k]:
            cum += steps[j][1]; j += 1
        p["equity"] = round(p["equity"] + cum, 6)


def _record_day_loss(day_loss: dict[date, float], d: date, pnl: float) -> None:
    if pnl < 0:
        day_loss[d] = day_loss.get(d, 0.0) + abs(pnl)


_MAX_TICK_FILES = 100


def _load_df_windowed(
    path: Path,
    n_bars: int,
    to_ts: float | None = None,
    warmup_bars: int = 0,
) -> tuple[pd.DataFrame, bool]:
    """Load the last (n_bars + warmup_bars) OHLC bars, optionally capped at to_ts.

    Returns (df, has_more). df includes warmup rows at the start so callers can
    apply indicators and then trim the warmup prefix before returning to clients.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")

    total_needed = n_bars + warmup_bars

    if path.is_dir():
        from data.parquet_reader import load_ddm_ticks_windowed
        n_files = max(20, total_needed // 300 + 10)
        tick_df, file_has_more = load_ddm_ticks_windowed(path, n_files=n_files, to_ts=to_ts)
        if tick_df.empty:
            empty = pd.DataFrame(columns=["open", "high", "low", "close"])
            return empty, False
        ohlc = tick_df["price"].resample("1min").ohlc()
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = tick_df["price"].resample("1min").count()
        df = ohlc.dropna().sort_index()
        df.columns = [c.lower() for c in df.columns]
        has_more = file_has_more or len(df) > total_needed
    else:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            raise ValueError("Dataset missing required column: 'close'")
        df = df.sort_index()

        if to_ts is not None:
            try:
                if df.index.tz is not None:
                    unix = df.index.tz_convert("UTC").tz_localize(None).astype("datetime64[s]").astype("int64")
                else:
                    unix = df.index.astype("datetime64[s]").astype("int64")
            except Exception:
                unix = pd.Series([int(t.timestamp()) if isinstance(t, pd.Timestamp) else 0 for t in df.index], index=df.index)
            df = df[unix <= int(to_ts)]

        has_more = len(df) > total_needed

    if len(df) > total_needed:
        df = df.iloc[-total_needed:]

    return df, has_more


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Fallback candidates across root artifacts and backend artifacts
        candidates = [
            Path("../artifacts") / path.name,
            Path("../artifacts") / "datasets" / path.name,
            Path("artifacts") / "datasets" / path.name,
            Path(__file__).resolve().parents[2] / "artifacts" / "datasets" / path.name,
            Path(__file__).resolve().parents[3] / "artifacts" / "datasets" / path.name,
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
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
        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()
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

    # Overfitting score: OOS PnL / IS PnL. 1.0 = perfect generalisation, < 0 = OOS losing.
    # Capped at [-2, 2] to keep it displayable.
    is_pnl  = is_m.get("is_total_pnl",  0.0)
    oos_pnl = oos_m.get("oos_total_pnl", 0.0)
    if abs(is_pnl) > 1e-9:
        oos_consistency = round(max(-2.0, min(2.0, oos_pnl / is_pnl)), 4)
    elif oos_pnl >= 0:
        oos_consistency = 1.0
    else:
        oos_consistency = -1.0
    combined["oos_consistency"] = oos_consistency

    return {**combined, **is_m, **oos_m}
