"""
Evaluate entry/exit conditions from a strategy definition.

Standard comparison condition:
    {
        "left": "macd_line", "op": ">", "right": "macd_signal"
    }
    {
        "left": "rsi", "op": "<", "right": 70
    }

Streak condition — counts consecutive bars where a comparison holds true:
    {
        "type": "streak",
        "left": "macd_line",
        "op": ">",
        "right": "macd_signal",
        "min_streak": 3
    }
    Returns True when the sub-condition (left op right) has been continuously True
    for at least `min_streak` bars up to and including the current bar.
    Requires df_upto in context (always present during backtest/paper runs).

Regime condition — requires a `rangetrend` indicator in the same strategy:
    {
        "type": "regime",
        "mode": "range",       # "range" | "trend" | "bull" | "bear"
        "indicator": "rt",     # id of the rangetrend indicator (default "rt")
        "threshold": 0.6       # range/trend: {id}_range threshold (default 0.5)
                               # bull/bear:   abs({id}_trend) threshold (default 0.3)
    }

    mode semantics:
        range  →  {id}_range  >  threshold   (low volatility, mean-reverting)
        trend  →  {id}_range  <= threshold   (directional, breakout-friendly)
        bull   →  {id}_trend  >  threshold   (uptrend)
        bear   →  {id}_trend  < -threshold   (downtrend)

ML signal condition:
    {
        "type": "ml_signal",
        "model_id": 1,
        "direction": "buy",
        "step": 1,
        "min_confidence": 0.0
    }

LLM signal condition:
    {
        "type": "llm_signal",
        "direction": "buy",
        "lookback": 10,
        "model": "gemini-2.0-flash"
    }

A condition block groups conditions with and/or logic:
    {
        "conditions": [...],
        "logic": "and"   # "and" | "or"
    }

Context dict (optional, passed through from backtest runner):
    {
        "df_upto": pd.DataFrame,      # full history up to current bar (for ML/LLM)
        "bar_index": int,             # integer bar index (for LLM cache key)
        "model_cache": dict[int, Any] # pre-loaded model metadata for ml_signal
    }
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def evaluate_conditions(
    row: pd.Series,
    condition_block: dict,
    context: dict[str, Any] | None = None,
) -> bool:
    """Return True if all/any (and/or) conditions are met for a single bar."""
    conditions = condition_block.get("conditions", [])
    logic = condition_block.get("logic", "and").lower()

    if not conditions:
        return False

    results = [_eval_one(row, c, context) for c in conditions]

    if logic == "or":
        return any(results)
    return all(results)


def _eval_one(
    row: pd.Series,
    cond: dict,
    context: dict[str, Any] | None,
) -> bool:
    ctype = cond.get("type")

    if ctype == "streak":
        df_upto = context.get("df_upto") if context else None
        if df_upto is None or len(df_upto) == 0:
            return False
        return _eval_streak(df_upto, cond)

    if ctype == "group_ref":
        groups = context.get("groups", {}) if context else {}
        group_id = cond.get("group_id", "")
        group_def = groups.get(group_id)
        if group_def is None:
            raise KeyError(f"Condition group '{group_id}' not found in definition.")
        return evaluate_conditions(row, group_def, context=context)

    if ctype == "regime":
        return _eval_regime(row, cond)

    if ctype == "ml_signal":
        from strategy.engine.ml_condition import evaluate_ml_condition
        df_upto = context.get("df_upto") if context else None
        model_cache = context.get("model_cache", {}) if context else {}
        if df_upto is None:
            return False
        return evaluate_ml_condition(df_upto, cond, model_cache)

    if ctype == "llm_signal":
        from strategy.engine.llm_condition import evaluate_llm_condition
        df_upto = context.get("df_upto") if context else None
        bar_index = context.get("bar_index", 0) if context else 0
        if df_upto is None:
            return False
        return evaluate_llm_condition(df_upto, bar_index, cond)

    # Default: standard comparison condition
    return _eval_comparison(row, cond)


def _eval_regime(row: pd.Series, cond: dict) -> bool:
    indicator = cond.get("indicator", "rt")
    mode = cond.get("mode", "range").lower()
    threshold = float(cond.get("threshold", 0.5 if mode in ("range", "trend") else 0.3))

    has_score = f"{indicator}_range" in row.index
    has_bool = f"{indicator}_is_range" in row.index

    if mode == "range":
        if has_bool:
            return int(row[f"{indicator}_is_range"]) == 1
        if has_score:
            return float(row[f"{indicator}_range"]) > threshold
        raise KeyError(f"No range columns found for indicator '{indicator}' — add a rangetrend indicator with that id.")

    if mode == "trend":
        if has_bool:
            return int(row[f"{indicator}_is_range"]) == 0
        if has_score:
            return float(row[f"{indicator}_range"]) <= threshold
        raise KeyError(f"No range columns found for indicator '{indicator}' — add a rangetrend indicator with that id.")

    if mode in ("bull", "bear"):
        col = f"{indicator}_trend"
        if col not in row.index:
            raise KeyError(f"Column '{col}' not found — bull/bear mode requires method='bband'.")
        val = float(row[col])
        if math.isnan(val):
            return False
        return val > threshold if mode == "bull" else val < -threshold

    raise ValueError(f"Unknown regime mode: {mode!r}. Use 'range', 'trend', 'bull', or 'bear'.")


def _eval_comparison(row: pd.Series, cond: dict) -> bool:
    left_val = _resolve(row, cond["left"])
    right_val = _resolve(row, cond["right"])

    if math.isnan(left_val) or math.isnan(right_val):
        return False

    op = cond["op"]
    if op == ">":
        return left_val > right_val
    elif op == "<":
        return left_val < right_val
    elif op == ">=":
        return left_val >= right_val
    elif op == "<=":
        return left_val <= right_val
    elif op == "==":
        return left_val == right_val
    elif op == "!=":
        return left_val != right_val
    else:
        raise ValueError(f"Unknown operator: {op!r}")


def _eval_streak(df_upto: pd.DataFrame, cond: dict) -> bool:
    """Count consecutive trailing bars where the sub-condition holds; return True if >= min_streak."""
    min_streak = int(cond.get("min_streak", 1))
    sub = {"left": cond["left"], "op": cond["op"], "right": cond["right"]}
    count = 0
    for i in range(len(df_upto) - 1, -1, -1):
        if _eval_comparison(df_upto.iloc[i], sub):
            count += 1
            if count >= min_streak:
                return True
        else:
            break
    return False


def eval_condition_series(df: pd.DataFrame, cond: dict) -> "pd.Series | None":
    """Vectorised evaluation of a single condition over a full DataFrame.

    Returns a boolean Series aligned to df.index, or None for conditions that
    require per-bar context (ml_signal, llm_signal) and cannot be vectorised.
    """
    ctype = cond.get("type") or "comparison"

    if ctype in ("ml_signal", "llm_signal"):
        return None

    if ctype == "streak":
        min_streak = int(cond.get("min_streak", 1))
        sub = {"left": cond["left"], "op": cond["op"], "right": cond["right"]}
        bool_s = eval_condition_series(df, sub)
        if bool_s is None:
            return None
        # Cumulative consecutive-True count, reset to 0 on False
        groups = (bool_s != bool_s.shift()).cumsum()
        streak_s = bool_s.groupby(groups).cumcount() + 1
        return streak_s.where(bool_s, 0) >= min_streak

    if ctype == "regime":
        indicator = cond.get("indicator", "rt")
        mode = cond.get("mode", "range").lower()
        threshold = float(cond.get("threshold", 0.5 if mode in ("range", "trend") else 0.3))
        has_bool = f"{indicator}_is_range" in df.columns
        has_score = f"{indicator}_range" in df.columns
        if mode == "range":
            if has_bool:
                return df[f"{indicator}_is_range"] == 1
            if has_score:
                return df[f"{indicator}_range"] > threshold
        elif mode == "trend":
            if has_bool:
                return df[f"{indicator}_is_range"] == 0
            if has_score:
                return df[f"{indicator}_range"] <= threshold
        elif mode == "bull":
            col = f"{indicator}_trend"
            if col in df.columns:
                return df[col] > threshold
        elif mode == "bear":
            col = f"{indicator}_trend"
            if col in df.columns:
                return df[col] < -threshold
        return pd.Series(False, index=df.index)

    # comparison
    left_key = cond.get("left", "close")
    right_key = cond.get("right", 0)
    op = cond.get("op", ">")
    left_s = df[left_key] if left_key in df.columns else pd.Series(float(left_key), index=df.index)
    right_s = df[right_key] if isinstance(right_key, str) and right_key in df.columns else pd.Series(float(right_key), index=df.index)
    ops = {">": left_s > right_s, "<": left_s < right_s, ">=": left_s >= right_s,
           "<=": left_s <= right_s, "==": left_s == right_s, "!=": left_s != right_s}
    return ops.get(op, pd.Series(False, index=df.index)).fillna(False)


def eval_block_series(df: "pd.DataFrame", block: dict, groups: dict | None = None) -> "pd.Series":
    """Vectorised evaluation of a full condition block (AND/OR of conditions).

    ML/LLM conditions that cannot be vectorised are skipped (treated as True so they
    don't suppress signals from other conditions).
    Returns a boolean Series aligned to df.index.
    """
    import pandas as pd

    conditions = block.get("conditions", [])
    logic = block.get("logic", "and").lower()
    if not conditions:
        return pd.Series(False, index=df.index)

    series_list: list[pd.Series] = []
    for cond in conditions:
        ctype = cond.get("type") or "comparison"
        if ctype == "group_ref":
            group_id = cond.get("group_id", "")
            group_def = (groups or {}).get(group_id)
            s = eval_block_series(df, group_def, groups) if group_def else pd.Series(False, index=df.index)
        elif ctype in ("ml_signal", "llm_signal"):
            s = pd.Series(True, index=df.index)  # skip non-vectorizable
        else:
            s = eval_condition_series(df, cond)
            if s is None:
                s = pd.Series(True, index=df.index)
        series_list.append(s.fillna(False))

    result = series_list[0]
    for s in series_list[1:]:
        result = result | s if logic == "or" else result & s
    return result.fillna(False)


def _resolve(row: pd.Series, operand) -> float:
    if isinstance(operand, str):
        val = row.get(operand)
        if val is None:
            raise KeyError(f"Column '{operand}' not found. Available: {list(row.index)}")
        return float(val)
    return float(operand)
