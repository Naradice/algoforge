"""
Evaluate entry/exit conditions from a strategy definition.

Standard comparison condition:
    {
        "left": "macd_line", "op": ">", "right": "macd_signal"
    }
    {
        "left": "rsi", "op": "<", "right": 70
    }

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


def _resolve(row: pd.Series, operand) -> float:
    if isinstance(operand, str):
        val = row.get(operand)
        if val is None:
            raise KeyError(f"Column '{operand}' not found. Available: {list(row.index)}")
        return float(val)
    return float(operand)
