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
    renko       → {id}_direction (+1/-1 last brick direction, carried fwd), {id}_flip (+1 bullish flip, -1 bearish flip, 0 none),
                  {id}_momentum (raw ffilled diff — matches trade_strategy renko_cons_num; compare with >= threshold),
                  {id}_pos (continuous brick position: (price - last_ref) / brick_size, same as finance_client renko_BrickSize)
    streak      → {id}  integer count of consecutive bars where left op right is true (default id: streak)
    roc         → {id}                                  (default id: roc)
    rolling     → {id}  rolling aggregate (sum/mean/std/min/max/diff/pct_change) of an existing column
    derived     → {id}  result of an arbitrary pandas expression evaluated against the DataFrame
    candle      → {id}_bull_engulf, {id}_bear_engulf,  (default id: candle)
                  {id}_bull_pin, {id}_bear_pin,
                  {id}_bull_outside, {id}_bear_outside
"""

from __future__ import annotations

import pandas as pd
from finance_client.fprocess.fprocess import regime as fprocess_regime
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


def estimate_warmup_bars(indicator_specs: list[dict]) -> int:
    """Return the minimum history length required before indicators are safe to evaluate."""
    if not indicator_specs:
        return 1
    return max(_estimate_one_warmup(spec) for spec in indicator_specs)


def _estimate_one_warmup(spec: dict) -> int:
    itype = str(spec.get("type", "")).lower()
    params = spec.get("params", {})

    if itype == "macd":
        return int(params.get("slow", 26)) + int(params.get("signal_period", 9)) - 2
    if itype in ("rsi", "atr", "ema", "sma", "bb", "slope", "cci", "adx", "donchian"):
        return int(params.get("period", 14))
    if itype == "stochastic":
        return max(int(params.get("k_period", 14)), int(params.get("d_period", 3)))
    if itype == "renko":
        brick_size = params.get("brick_size", None)
        if brick_size not in (None, "", 0, "0"):
            return 2
        return int(params.get("atr_window", 14)) + 30
    if itype == "rangetrend":
        method = str(params.get("method", "bband"))
        if method == "bband":
            return int(params.get("bb_period", 20)) + int(params.get("slope_window", 4))
        if method == "atr":
            return max(int(params.get("mean_window", 100)), int(params.get("atr_window", 14)))
        if method == "bollinger":
            return max(int(params.get("std_window", 200)), int(params.get("window", 20)))
        if method == "swing":
            return int(params.get("window", 50))
        if method == "adx":
            return int(params.get("adx_window", 14))
        if method == "ma_deviation":
            return max(int(params.get("short_window", 10)), int(params.get("long_window", 50)))
        return 1
    if itype == "lrmomentum":
        return int(params.get("period", 90))
    if itype in ("roc",):
        return int(params.get("period", 10))
    if itype in ("sar", "candle"):
        return 2
    if itype == "rolling":
        return int(params.get("window", 2))
    if itype == "derived":
        return 1
    return 1


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
        method = str(params.get("method", "bband"))
        ohlc = ("open", "high", "low", "close")
        if method == "bband":
            slope_window = int(params.get("slope_window", 4))
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            _bb_key = f"__{iid}_bb"
            _proc = BBANDProcess(key=_bb_key, window=bb_period, alpha=bb_std, target_column=col)
            _tmp = _proc.run(df)
            _mid = _tmp[f"{_bb_key}_MV"]
            _width = _tmp[f"{_bb_key}_Width"]
            _slope = (_mid - _mid.shift(slope_window)) / slope_window
            _slope_std = _slope.std()
            df[f"{iid}_trend"] = (_slope / _slope_std).clip(-1, 1) if _slope_std > 0 else 0.0
            _width_diff = _width.diff().replace(0, float("nan"))
            _pct = _width_diff.pct_change()
            _pct_std = _pct.std()
            df[f"{iid}_range"] = (1 / (1 + (_pct / _pct_std).abs())) if _pct_std > 0 else 0.5
        elif method == "atr":
            is_range = fprocess_regime.range_detection_by_atr(
                df,
                mean_window=int(params.get("mean_window", 100)),
                atr_window=int(params.get("atr_window", 14)),
                range_threshold=float(params.get("range_threshold", 0.7)),
                ohlc_columns=ohlc,
            )
            df[f"{iid}_is_range"] = is_range.astype(int)
        elif method == "bollinger":
            is_range = fprocess_regime.range_detection_by_bollinger(
                df,
                std_window=int(params.get("std_window", 200)),
                window=int(params.get("window", 20)),
                std_threshold=float(params.get("std_threshold", 0.6)),
                ohlc_columns=ohlc,
            )
            df[f"{iid}_is_range"] = is_range.astype(int)
        elif method == "swing":
            is_range = fprocess_regime.range_detection_by_swing_width(
                df,
                window=int(params.get("window", 50)),
                width_threshold=float(params.get("width_threshold", 0.015)),
                ohlc_columns=ohlc,
            )
            df[f"{iid}_is_range"] = is_range.astype(int)
        elif method == "adx":
            is_range = fprocess_regime.range_detection_by_adx(
                df,
                adx_window=int(params.get("adx_window", 14)),
                adx_threshold=float(params.get("adx_threshold", 25)),
                ohlc_columns=ohlc,
            )
            df[f"{iid}_is_range"] = is_range.astype(int)
        elif method == "ma_deviation":
            is_range = fprocess_regime.range_detection_by_ma_deviation(
                df,
                short_window=int(params.get("short_window", 10)),
                long_window=int(params.get("long_window", 50)),
                deviation_threshold=float(params.get("deviation_threshold", 0.005)),
                ohlc_columns=ohlc,
            )
            df[f"{iid}_is_range"] = is_range.astype(int)
        else:
            raise ValueError(f"Unknown rangetrend method: {method!r}. Use bband, atr, bollinger, swing, adx, or ma_deviation.")

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

    elif itype == "renko":
        atr_window = int(params.get("atr_window", 14))
        brick_size = params.get("brick_size", None)
        fixed = float(brick_size) if brick_size not in (None, "", 0, "0") else None
        renko_df = technical.RenkoFromOHLC(
            df,
            ohlc_columns=("open", "high", "low", "close"),
            brick_size=fixed,
            atr_window=atr_window,
            total_brick_name=f"{iid}_count",
            brick_size_name=f"__{iid}_raw",
        )
        renko_df.index = df.index
        count = renko_df[f"{iid}_count"]
        raw_diff = count.diff()
        # Per-bar brick direction: +1 if up brick(s) formed, -1 if down, 0 if no new brick
        brick_dir = raw_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        # Carry forward the last known direction
        direction = brick_dir.where(brick_dir != 0).ffill().fillna(0).astype(int)
        # Flip fires only on the bar a new brick changes direction
        prev_dir = direction.shift(1).fillna(0).astype(int)
        flip = brick_dir.copy()
        flip[(brick_dir == 0) | (brick_dir == prev_dir)] = 0
        # momentum: forward-filled raw diff. This is useful for debugging, but it is
        # not the same as trade_strategy's renko_cons_num, which sums the trailing
        # N bars of this forward-filled series. For legacy MACDRenko threshold=2,
        # prefer a streak condition on {id}_direction instead.
        momentum = raw_diff.replace(0, float("nan")).ffill().fillna(0)
        df[f"{iid}_direction"] = direction
        df[f"{iid}_flip"] = flip.astype(int)
        df[f"{iid}_momentum"] = momentum
        # Continuous brick value: (price - last_ref) / brick_size, same as finance_client
        # renko_BrickSize. +2 means 2 up-bricks from last reference; -2 means 2 down-bricks.
        df[f"{iid}_pos"] = renko_df[f"__{iid}_raw"]

    elif itype == "streak":
        left_key = str(params.get("left", "close"))
        op = str(params.get("op", ">"))
        right_key = params.get("right", 0)
        left_s = df[left_key] if left_key in df.columns else pd.Series(float(left_key), index=df.index)
        right_s = df[str(right_key)] if isinstance(right_key, str) and right_key in df.columns else pd.Series(float(right_key), index=df.index)
        ops = {">": left_s > right_s, "<": left_s < right_s, ">=": left_s >= right_s,
               "<=": left_s <= right_s, "==": left_s == right_s, "!=": left_s != right_s}
        cond = ops.get(op, pd.Series(False, index=df.index)).fillna(False)
        # Vectorised consecutive-True counter: group by run boundaries, cumsum within each run
        groups = (cond != cond.shift()).cumsum()
        df[iid] = cond.astype(int).groupby(groups).cumsum()

    elif itype == "rolling":
        src = str(params.get("column", "close"))
        fn = str(params.get("function", "sum"))
        window = int(params.get("window", 2))
        if src not in df.columns:
            raise ValueError(f"Rolling indicator '{iid}': source column '{src}' not found. "
                             f"Make sure the source indicator is listed before this one.")
        s = df[src]
        if fn == "sum":          df[iid] = s.rolling(window).sum()
        elif fn == "mean":       df[iid] = s.rolling(window).mean()
        elif fn == "std":        df[iid] = s.rolling(window).std()
        elif fn == "min":        df[iid] = s.rolling(window).min()
        elif fn == "max":        df[iid] = s.rolling(window).max()
        elif fn == "diff":       df[iid] = s.diff(window)
        elif fn == "pct_change": df[iid] = s.pct_change(window)
        else:
            raise ValueError(f"Rolling indicator '{iid}': unknown function {fn!r}. "
                             f"Use sum, mean, std, min, max, diff, or pct_change.")

    elif itype == "derived":
        expr = str(params.get("expr", "close"))
        namespace = {c: df[c] for c in df.columns}
        try:
            result = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
            df[iid] = result
        except Exception as exc:
            raise ValueError(f"Derived indicator '{iid}': expression error — {exc}") from exc

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
