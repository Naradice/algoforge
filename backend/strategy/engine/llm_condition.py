"""
LLM market signal condition — synchronous, for use inside the backtest executor thread.

Calls Google Gemini (via google-genai) with a structured prompt containing recent
OHLC + indicator data, then parses the response for a directional signal.

Results are cached by (bar_index, condition_key) to avoid repeated API calls during
backtest replay. In live/paper mode caching is disabled.

Condition spec:
    {
        "type": "llm_signal",
        "direction": "buy",         # required signal direction
        "lookback": 10,             # bars of history to include in prompt (default 10)
        "model": "gemini-2.0-flash",# Gemini model (default from LLM_MODEL env var)
        "columns": ["close", "volume"],  # columns to include (default: all numeric)
        "cache": true               # cache results per bar (default: true)
    }

The LLM must respond with EXACTLY one of: BUY / SELL / HOLD
Any other response is treated as HOLD.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Module-level cache: (cache_key) → signal string
_signal_cache: dict[str, str] = {}

_SYSTEM_PROMPT = """You are a quantitative trading signal generator.
Given recent OHLC and indicator data, analyse the market and respond with EXACTLY one word:
BUY — if conditions favour a long entry
SELL — if conditions favour a short entry
HOLD — if no clear edge is present

Respond with only one word. No explanation."""


def evaluate_llm_condition(
    df_upto: pd.DataFrame,
    bar_index: int,
    cond: dict,
) -> bool:
    """
    df_upto: DataFrame slice up to and including the current bar.
    bar_index: integer index of the current bar (for cache key).
    cond: llm_signal condition spec.
    """
    required_direction: str = cond.get("direction", "buy").upper()
    lookback: int = int(cond.get("lookback", 10))
    llm_model: str = cond.get("model") or os.getenv("LLM_MODEL", "gemini-2.0-flash")
    columns: list[str] | None = cond.get("columns")
    use_cache: bool = cond.get("cache", True)

    # Build the data slice
    slice_df = df_upto.tail(lookback)
    if columns:
        cols = [c for c in columns if c in slice_df.columns]
        slice_df = slice_df[cols] if cols else slice_df

    data_str = slice_df.select_dtypes(include="number").round(5).to_csv()

    # Cache key
    if use_cache:
        key = hashlib.md5(f"{bar_index}:{required_direction}:{data_str}".encode()).hexdigest()
        if key in _signal_cache:
            signal = _signal_cache[key]
            return _signal_matches(signal, required_direction)

    signal = _call_gemini(llm_model, data_str)

    if use_cache:
        _signal_cache[key] = signal

    return _signal_matches(signal, required_direction)


def _call_gemini(model_name: str, data_csv: str) -> str:
    try:
        from google import genai  # type: ignore
    except ImportError:
        logger.warning("google-genai not installed — llm_signal always returns HOLD")
        return "HOLD"

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GOOGLE_API_KEY set — llm_signal always returns HOLD")
        return "HOLD"

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"{_SYSTEM_PROMPT}\n\nMarket data (CSV):\n{data_csv}"
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = (response.text or "").strip().upper()
        for word in ("BUY", "SELL", "HOLD"):
            if word in text:
                return word
        return "HOLD"
    except Exception as e:
        logger.warning(f"llm_signal: Gemini call failed: {e}")
        return "HOLD"


def _signal_matches(signal: str, required: str) -> bool:
    return signal == required.upper()


def clear_cache() -> None:
    """Clear the LLM signal cache (useful between backtest runs)."""
    _signal_cache.clear()
