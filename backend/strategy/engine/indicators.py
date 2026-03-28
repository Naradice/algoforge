"""
Apply technical indicators to an OHLC DataFrame from a strategy definition.

Each indicator spec in the definition has:
    {"id": "<column_name>", "type": "<indicator_type>", "params": {...}}

Supported types and output columns:
    macd   → <id>_line, <id>_signal, <id>_hist   (default id: macd)
    rsi    → <id>                                  (default id: rsi)
    atr    → <id>                                  (default id: atr)
    ema    → <id>
    sma    → <id>
    bb     → <id>_upper, <id>_middle, <id>_lower  (default id: bb)
    slope  → <id>                                  (default id: slope)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_indicators(df: pd.DataFrame, indicator_specs: list[dict]) -> pd.DataFrame:
    """Return a copy of df with all indicator columns added."""
    df = df.copy()
    for spec in indicator_specs:
        itype = spec["type"].lower()
        iid = spec.get("id", itype)
        params = spec.get("params", {})
        _apply_one(df, itype, iid, params)
    return df


def _apply_one(df: pd.DataFrame, itype: str, iid: str, params: dict) -> None:
    col = params.get("column", "close")
    src = df[col].astype(float) if col in df.columns else df["close"].astype(float)

    if itype == "ema":
        period = int(params.get("period", 20))
        df[iid] = src.ewm(span=period, adjust=False).mean()

    elif itype == "sma":
        period = int(params.get("period", 20))
        df[iid] = src.rolling(period).mean()

    elif itype == "macd":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_period = int(params.get("signal_period", 9))
        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        df[f"{iid}_line"] = macd_line
        df[f"{iid}_signal"] = signal_line
        df[f"{iid}_hist"] = macd_line - signal_line

    elif itype == "rsi":
        period = int(params.get("period", 14))
        delta = src.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df[iid] = 100 - (100 / (1 + rs))

    elif itype == "atr":
        period = int(params.get("period", 14))
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df[iid] = tr.ewm(com=period - 1, adjust=False).mean()

    elif itype == "bb":
        period = int(params.get("period", 20))
        std_mult = float(params.get("std", 2.0))
        middle = src.rolling(period).mean()
        std = src.rolling(period).std()
        df[f"{iid}_upper"] = middle + std_mult * std
        df[f"{iid}_middle"] = middle
        df[f"{iid}_lower"] = middle - std_mult * std

    elif itype == "slope":
        period = int(params.get("period", 5))
        x = np.arange(period, dtype=float)
        x -= x.mean()

        def _slope(window: np.ndarray) -> float:
            if np.isnan(window).any():
                return np.nan
            return float(np.polyfit(x, window, 1)[0])

        df[iid] = src.rolling(period).apply(_slope, raw=True)

    else:
        raise ValueError(f"Unknown indicator type: {itype!r}")
