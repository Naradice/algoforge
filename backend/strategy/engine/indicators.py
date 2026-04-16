"""
Apply technical indicators to an OHLC DataFrame using finance_client's fprocess library
as the single source of truth for all indicator math.

A thin rename layer maps finance_client's output column names to the convention used
in AlgoForge strategy definitions so existing saved strategies keep working:

    macd        → {id}_line, {id}_signal, {id}_hist   (default id: macd)
    rsi         → {id}                                  (default id: rsi)
    atr         → {id}                                  (default id: atr)
    ema         → {id}
    sma         → {id}
    bb          → {id}_upper, {id}_middle, {id}_lower  (default id: bb)
    slope       → {id}                                  (default id: slope)
    cci         → {id}                                  (default id: cci)
    rangetrend  → {id}_trend, {id}_range               (default id: rt)
    lrmomentum  → {id}_momentum                        (default id: lrm)
    adx         → {id}, {id}_plus_di, {id}_minus_di   (default id: adx)
    stochastic  → {id}_k, {id}_d                       (default id: stoch)
    sar         → {id}                                  (default id: sar)
    donchian    → {id}_upper, {id}_lower, {id}_mid     (default id: dc)
    roc         → {id}                                  (default id: roc)
    candle      → {id}_bull_engulf, {id}_bear_engulf,  (default id: candle)
                  {id}_bull_pin, {id}_bear_pin,
                  {id}_bull_outside, {id}_bear_outside
"""

from __future__ import annotations

import pandas as pd
from finance_client.fprocess.fprocess.idcprocess import (
    ATRProcess,
    BBANDProcess,
    CCIProcess,
    EMAProcess,
    LinearRegressionMomentumProcess,
    MAProcess,
    MACDProcess,
    RSIProcess,
    SlopeProcess,
)
from finance_client.fprocess.fprocess.indicaters import technical


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

    elif itype == "cci":
        period = int(params.get("period", 14))
        proc = CCIProcess(key=iid, window=period, ohlc_column=("open", "high", "low", "close"))
        df = proc.run(df)
        # CCIProcess names the output column after `key` → matches {iid} convention ✓

    elif itype == "rangetrend":
        # Reimplemented directly — the upstream RangeTrendProcess has unfixable MultiIndex bugs.
        # Logic mirrors the original: trend from BB-mean slope, range from BB-width pct-change.
        slope_window = int(params.get("slope_window", 4))
        bb_period = int(params.get("bb_period", 20))
        bb_std = float(params.get("bb_std", 2.0))
        _bb_key = f"__{iid}_bb"
        _proc = BBANDProcess(key=_bb_key, window=bb_period, alpha=bb_std, target_column=col)
        _tmp = _proc.run(df)
        _mid = _tmp[f"{_bb_key}_MV"]
        _width = _tmp[f"{_bb_key}_Width"]
        # trend: normalised slope of the BB middle band, clamped to [-1, 1]
        _slope = (_mid - _mid.shift(slope_window)) / slope_window
        _slope_std = _slope.std()
        df[f"{iid}_trend"] = (_slope / _slope_std).clip(-1, 1) if _slope_std > 0 else 0.0
        # range: how "stable" the band width is — 1 = tight/stable, 0 = expanding fast
        _width_diff = _width.diff().replace(0, float("nan"))
        _pct = _width_diff.pct_change()
        _pct_std = _pct.std()
        df[f"{iid}_range"] = (1 / (1 + (_pct / _pct_std).abs())) if _pct_std > 0 else 0.5

    elif itype == "lrmomentum":
        period = int(params.get("period", 90))
        proc = LinearRegressionMomentumProcess(key=iid, window=period, column=col)
        df = proc.run(df)
        # Produces "{iid}_momentum" ✓

    elif itype == "adx":
        period = int(params.get("period", 14))
        result = technical.ADXFromOHLC(
            df,
            window=period,
            ohlc_columns=("open", "high", "low", "close"),
            plus_di_name=f"{iid}_plus_di",
            minus_di_name=f"{iid}_minus_di",
            adx_name=iid,
        )
        result.index = df.index
        df = pd.concat([df, result], axis=1)

    elif itype == "stochastic":
        k_period = int(params.get("k_period", 14))
        d_period = int(params.get("d_period", 3))
        result = technical.StochasticOscillatorFromOHLC(
            df,
            k_window=k_period,
            d_window=d_period,
            ohlc_columns=("open", "high", "low", "close"),
            k_name=f"{iid}_k",
            d_name=f"{iid}_d",
        )
        result.index = df.index
        df = pd.concat([df, result], axis=1)

    elif itype == "sar":
        af_start = float(params.get("af_start", 0.02))
        af_increment = float(params.get("af_increment", 0.02))
        af_max = float(params.get("af_max", 0.2))
        result = technical.ParabolicSARFromOHLC(
            df,
            ohlc_columns=("open", "high", "low", "close"),
            af_start=af_start,
            af_increment=af_increment,
            af_max=af_max,
            sar_name=iid,
        )
        result.index = df.index
        df = pd.concat([df, result], axis=1)

    elif itype == "donchian":
        period = int(params.get("period", 20))
        df[f"{iid}_upper"] = df["high"].rolling(period).max()
        df[f"{iid}_lower"] = df["low"].rolling(period).min()
        df[f"{iid}_mid"] = (df[f"{iid}_upper"] + df[f"{iid}_lower"]) / 2

    elif itype == "roc":
        period = int(params.get("period", 10))
        df[iid] = df[col].pct_change(periods=period) * 100

    elif itype == "candle":
        ratio = float(params.get("ratio", 2.0))
        ohlc = ("open", "high", "low", "close")
        df[f"{iid}_bull_engulf"] = technical.bullish_engulfing(df, "open", "close").astype(int)
        df[f"{iid}_bear_engulf"] = technical.bearish_engulfing(df, "open", "close").astype(int)
        df[f"{iid}_bull_pin"] = technical.bullish_pinbar(df, ohlc, ratio=ratio).astype(int)
        df[f"{iid}_bear_pin"] = technical.bearish_pinbar(df, ohlc, ratio=ratio).astype(int)
        df[f"{iid}_bull_outside"] = technical.bullish_outside(df, ohlc).astype(int)
        df[f"{iid}_bear_outside"] = technical.bearish_outside(df, ohlc).astype(int)

    else:
        raise ValueError(f"Unknown indicator type: {itype!r}")

    return df
