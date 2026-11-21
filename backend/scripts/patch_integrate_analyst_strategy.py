"""
Patch script to integrate analyst strategies into the standard backtest engine
and deprecate analyst_strategy.py.
"""
from pathlib import Path

backtest_path = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\engine\backtest.py")
content = backtest_path.read_text(encoding="utf-8")

# 1. Update _normalise function to transpile analyst_range_reversion and analyst_breakout_momentum
old_normalise = '''def _normalise(definition: dict) -> dict:
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
    return result'''

new_normalise = '''def _normalise(definition: dict) -> dict:
    """
    Normalises strategy definitions:
    - Transpiles analyst_* forecast strategies into standard indicators + long/short blocks.
    - Converts old { entry, exit } format to { long/short: { entry, exit } }.
    """
    strat_type = definition.get("strategy_type", "")
    if strat_type.startswith("analyst_"):
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
    return result'''

assert old_normalise in content, "old_normalise not found in backtest.py"
content = content.replace(old_normalise, new_normalise)

# 2. Remove the analyst_ bypass in run_backtest
old_bypass = '''    # Support specialized analyst forecast strategies
    strat_type = definition.get("strategy_type", "")
    if strat_type.startswith("analyst_"):
        from strategy.engine.analyst_strategy import run_analyst_backtest
        return run_analyst_backtest(
            definition,
            raw_df,
            on_progress=on_progress,
            starting_capital=starting_capital if starting_capital > 1.0 else 1_000_000.0,
        )'''

assert old_bypass in content, "old_bypass not found in backtest.py"
content = content.replace(old_bypass, "    # Note: analyst_* strategies are transpiled by _normalise() into standard indicators/conditions.")

# 3. Update pending order SL/TP calculation for sl_pips and tp_pips
old_fills = '''        if pending_long_entry and pos_long is None and num_open < rp.max_positions:
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
                          "opened_at": bar_dt, "symbol": symbol}, "open")'''

new_fills = '''        if pending_long_entry and pos_long is None and num_open < rp.max_positions:
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

            vol = calc_volume(rp, equity, fill, atr)
            pos_long = OpenPosition("buy", fill, sl, tp, vol, bar_dt, i)
            num_open += 1
            if on_trade:
                on_trade({"direction": "buy", "entry_price": fill, "sl_price": sl, "tp_price": tp,
                          "opened_at": bar_dt, "symbol": symbol}, "open")

        if pending_short_entry and pos_short is None and num_open < rp.max_positions:
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

            vol = calc_volume(rp, equity, fill, atr)
            pos_short = OpenPosition("sell", fill, sl, tp, vol, bar_dt, i)
            num_open += 1
            if on_trade:
                on_trade({"direction": "sell", "entry_price": fill, "sl_price": sl, "tp_price": tp,
                          "opened_at": bar_dt, "symbol": symbol}, "open")'''

assert old_fills in content, "old_fills not found in backtest.py"
content = content.replace(old_fills, new_fills)

backtest_path.write_text(content, encoding="utf-8")
print("Successfully patched backtest.py!")
