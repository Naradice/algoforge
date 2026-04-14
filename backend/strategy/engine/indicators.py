"""
Apply technical indicators to an OHLC DataFrame using finance_client's fprocess library
as the single source of truth for all indicator math.

A thin rename layer maps finance_client's output column names to the convention used
in AlgoForge strategy definitions so existing saved strategies keep working:

    macd   → {id}_line, {id}_signal, {id}_hist   (default id: macd)
    rsi    → {id}                                  (default id: rsi)
    atr    → {id}                                  (default id: atr)
    ema    → {id}
    sma    → {id}
    bb     → {id}_upper, {id}_middle, {id}_lower  (default id: bb)
    slope  → {id}                                  (default id: slope)
"""

from __future__ import annotations

import pandas as pd
from finance_client.fprocess.fprocess.idcprocess import (
    ATRProcess,
    BBANDProcess,
    EMAProcess,
    MAProcess,
    MACDProcess,
    RSIProcess,
    SlopeProcess,
)


def apply_indicators(df: pd.DataFrame, indicator_specs: list[dict]) -> pd.DataFrame:
    """Return a copy of df with all indicator columns added."""
    df = df.copy()
    for spec in indicator_specs:
        itype = spec["type"].lower()
        iid = spec.get("id", itype)
        params = spec.get("params", {})
        df = _apply_one(df, itype, iid, params)
    return df


def _apply_one(df: pd.DataFrame, itype: str, iid: str, params: dict) -> pd.DataFrame:
    """Dispatch to the appropriate finance_client Process and normalise column names."""
    col = params.get("column", "close")

    if itype == "ema":
        period = int(params.get("period", 20))
        proc = EMAProcess(key=iid, window=period, column=col)
        df = proc.run(df)
        # EMAProcess names the output column after `key` → matches {iid} convention ✓

    elif itype == "sma":
        period = int(params.get("period", 20))
        proc = MAProcess(key=iid, window=period, column=col)
        df = proc.run(df)
        # MAProcess names the output column "{iid}_MA" → rename to {iid}
        df = df.rename(columns={f"{iid}_MA": iid})

    elif itype == "macd":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_period = int(params.get("signal_period", 9))
        proc = MACDProcess(
            key=iid,
            target_column=col,
            short_window=fast,
            long_window=slow,
            signal_window=signal_period,
        )
        df = proc.run(df)
        # MACDProcess produces: "MACD" (hardcoded), "{iid}_Signal", "{iid}_S_EMA", "{iid}_L_EMA"
        # Rename to AlgoForge convention; compute histogram column
        df = df.rename(columns={
            "MACD": f"{iid}_line",
            f"{iid}_Signal": f"{iid}_signal",
        })
        df[f"{iid}_hist"] = df[f"{iid}_line"] - df[f"{iid}_signal"]

    elif itype == "rsi":
        period = int(params.get("period", 14))
        # RSIProcess uses ohlc_column[0] as the price source — pass (col,) for single column
        proc = RSIProcess(key=iid, window=period, ohlc_column_name=(col,))
        df = proc.run(df)
        # RSIProcess names the RSI column after `key` → matches {iid} convention ✓
        # Extra columns ({iid}_Gain, {iid}_Loss) are left in df but never referenced by conditions

    elif itype == "atr":
        period = int(params.get("period", 14))
        # ATR needs OHLC column names; our DataFrames are always lowercased by _load_df
        proc = ATRProcess(
            key=iid,
            window=period,
            ohlc_column_name=("open", "high", "low", "close"),
        )
        df = proc.run(df)
        # ATRProcess names the output column after `key` → matches {iid} convention ✓

    elif itype == "bb":
        period = int(params.get("period", 20))
        std_mult = float(params.get("std", 2.0))
        proc = BBANDProcess(key=iid, window=period, alpha=std_mult, target_column=col)
        df = proc.run(df)
        # BBANDProcess produces "{iid}_UV", "{iid}_MV", "{iid}_LV"
        # Rename to AlgoForge convention
        df = df.rename(columns={
            f"{iid}_UV": f"{iid}_upper",
            f"{iid}_MV": f"{iid}_middle",
            f"{iid}_LV": f"{iid}_lower",
        })

    elif itype == "slope":
        period = int(params.get("period", 5))
        proc = SlopeProcess(key=iid, target_column=col, window=period)
        df = proc.run(df)
        # SlopeProcess produces "{iid}_slope" → rename to {iid}
        df = df.rename(columns={f"{iid}_slope": iid})

    else:
        raise ValueError(f"Unknown indicator type: {itype!r}")

    return df
