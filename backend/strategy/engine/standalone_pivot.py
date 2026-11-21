"""
Standalone Mathematical Pivot Point Engine for AlgoForge.
Zero external report dependency -- calculates Classic, Camarilla, and Fibonacci pivots
directly from historical OHLCV bar data.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_daily_pivots(
    df_bars: pd.DataFrame,
    pivot_type: str = "classic",
) -> pd.DataFrame:
    """
    Computes daily pivot boundaries from intraday or daily bars.
    Returns DataFrame indexed by date with columns: ['P', 'S1', 'R1', 'S2', 'R2', 'expected_low', 'expected_high'].
    """
    df = df_bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = [c for c in ["time", "datetime", "Date", "Time"] if c in df.columns]
        if time_col:
            df[time_col[0]] = pd.to_datetime(df[time_col[0]], utc=True)
            df.set_index(time_col[0], inplace=True)
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    df["date_key"] = df.index.date
    daily = df.groupby("date_key").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).sort_index()

    H = daily["high"]
    L = daily["low"]
    C = daily["close"]

    if pivot_type.lower() == "classic":
        # Standard Floor Classic Pivot
        P = (H + L + C) / 3.0
        R1 = 2 * P - L
        S1 = 2 * P - H
        R2 = P + (H - L)
        S2 = P - (H - L)
    elif pivot_type.lower() == "fibonacci":
        P = (H + L + C) / 3.0
        rng = H - L
        R1 = P + 0.382 * rng
        S1 = P - 0.382 * rng
        R2 = P + 0.618 * rng
        S2 = P - 0.618 * rng
    elif pivot_type.lower() == "camarilla":
        rng = H - L
        P = (H + L + C) / 3.0
        R1 = C + rng * 1.1 / 12.0
        S1 = C - rng * 1.1 / 12.0
        R2 = C + rng * 1.1 / 6.0
        S2 = C - rng * 1.1 / 6.0
    else:
        raise ValueError(f"Unsupported pivot_type: {pivot_type}")

    df_pivots = pd.DataFrame({
        "date": daily.index,
        "P": P,
        "S1": S1,
        "R1": R1,
        "S2": S2,
        "R2": R2,
        "expected_low": S1,
        "expected_high": R1,
        "source": f"Standalone_{pivot_type.capitalize()}",
        "horizon": "Daily",
    })

    # Shift by 1 day: today's pivot relies on yesterday's completed day
    df_pivots[["P", "S1", "R1", "S2", "R2", "expected_low", "expected_high"]] = (
        df_pivots[["P", "S1", "R1", "S2", "R2", "expected_low", "expected_high"]].shift(1)
    )
    df_pivots = df_pivots.dropna().reset_index(drop=True)
    return df_pivots
