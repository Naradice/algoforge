"""
Build Rebound/Breakout Classification Dataset from M30 bars and Analyst Forecasts.
Extracts event-level features at the exact moment price touches the boundary.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from strategy.engine.analyst_strategy import get_forecast_records


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators on M30 bars."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Returns (momentum)
    for p in [1, 2, 3, 6, 12, 24]:
        df[f"ret_{p}"] = close.pct_change(p)

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = (high - low) / (df["atr_14"] + 1e-6)

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-6)
    df["bb_pct"] = (close - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-6)

    # Candle body & wick characteristics
    candle_range = (high - low).replace(0, 1e-6)
    df["body_ratio"] = (close - df["open"]).abs() / candle_range
    df["upper_wick_ratio"] = (high - np.maximum(close, df["open"])) / candle_range
    df["lower_wick_ratio"] = (np.minimum(close, df["open"]) - low) / candle_range

    return df


def extract_touch_events(
    symbol: str = "USDJPY",
    source: str = "all",
    sl_pips: float = 15.0,
    tp_pips: float = 40.0,
    tolerance_pips: float = 10.0,
) -> pd.DataFrame:
    m30_store_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"
    if not m30_store_path.exists():
        raise FileNotFoundError(f"M30 dataset not found at {m30_store_path}")

    raw_df = pd.read_parquet(m30_store_path)
    if not isinstance(raw_df.index, pd.DatetimeIndex) and "time" in raw_df.columns:
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        raw_df.set_index("time", inplace=True)
    raw_df.rename(columns={c: str(c).lower().strip() for c in raw_df.columns}, inplace=True)

    df_ohlc = calculate_indicators(raw_df)
    df_ohlc["date_key"] = df_ohlc.index.date

    daily_ohlc = df_ohlc.groupby("date_key").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    })

    df_fc = get_forecast_records(symbol, source=source)
    if df_fc.empty:
        raise ValueError(f"No forecast records found for {symbol} (source={source})")

    sl_yen = sl_pips * 0.01
    tp_yen = tp_pips * 0.01
    tol_yen = tolerance_pips * 0.01
    lot_multiplier = 100_000.0

    events = []

    def get_pub_time(src: str):
        s_low = src.lower()
        if "minna" in s_low or "みんな" in s_low:
            return datetime.strptime("00:00", "%H:%M").time()
        return datetime.strptime("00:30", "%H:%M").time()

    for idx, fc_row in df_fc.iterrows():
        fc_date = fc_row["date"]
        exp_low = float(fc_row["expected_low"])
        exp_high = float(fc_row["expected_high"])
        fc_source = str(fc_row.get("source") or "Report")
        pub_time = get_pub_time(fc_source)

        if fc_date not in daily_ohlc.index:
            candidates = [d for d in daily_ohlc.index if fc_date <= d <= fc_date + timedelta(days=4)]
            if not candidates:
                continue
            act_date = candidates[0]
        else:
            act_date = fc_date

        day_bars = df_ohlc[df_ohlc["date_key"] == act_date]
        if day_bars.empty:
            continue

        last_bar_time = day_bars.index[-1]
        last_close_p = float(day_bars.iloc[-1]["close"])
        post_pub_bars = day_bars[day_bars.index.time >= pub_time]
        if post_pub_bars.empty:
            continue

        buy_entered = False
        sell_entered = False

        for b_idx, (b_time, b) in enumerate(post_pub_bars.iterrows()):
            b_high = float(b["high"])
            b_low = float(b["low"])
            b_open = float(b["open"])
            b_close = float(b["close"])

            # --- Buy Fade: Low dipped to exp_low + tol_yen ---
            if not buy_entered and b_low <= (exp_low + tol_yen):
                buy_entered = True
                entry_p = exp_low
                exit_p = min(exp_high, last_close_p)
                exit_reason = "close"
                exit_time = last_bar_time

                remaining = post_pub_bars.iloc[b_idx:]
                for _, rb in remaining.iterrows():
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    if sl_yen is not None and rb_low <= (entry_p - sl_yen):
                        exit_p = entry_p - sl_yen
                        exit_reason = "sl"
                        exit_time = rb.name
                        break
                    elif tp_yen is not None and rb_high >= (entry_p + tp_yen):
                        exit_p = entry_p + tp_yen
                        exit_reason = "tp"
                        exit_time = rb.name
                        break
                    elif rb_high >= exp_high:
                        exit_p = exp_high
                        exit_reason = "tp_boundary"
                        exit_time = rb.name
                        break

                pnl = (exit_p - entry_p) * lot_multiplier
                target = 1 if pnl > 0 else 0

                # Features at bar b
                events.append({
                    "time": b_time,
                    "date": act_date,
                    "source": fc_source,
                    "direction": "buy",
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "target_rebound": target,
                    # Boundary context
                    "expected_low": exp_low,
                    "expected_high": exp_high,
                    "expected_range_pips": (exp_high - exp_low) / 0.01,
                    "dist_to_boundary_pips": (b_low - exp_low) / 0.01,
                    # Momentum features
                    "ret_1": b.get("ret_1", 0.0),
                    "ret_2": b.get("ret_2", 0.0),
                    "ret_3": b.get("ret_3", 0.0),
                    "ret_6": b.get("ret_6", 0.0),
                    "ret_12": b.get("ret_12", 0.0),
                    "ret_24": b.get("ret_24", 0.0),
                    # Technicals
                    "atr_14": b.get("atr_14", 0.0),
                    "atr_ratio": b.get("atr_ratio", 1.0),
                    "rsi_14": b.get("rsi_14", 50.0),
                    "bb_width": b.get("bb_width", 0.0),
                    "bb_pct": b.get("bb_pct", 0.5),
                    # Candle shape
                    "body_ratio": b.get("body_ratio", 0.5),
                    "upper_wick_ratio": b.get("upper_wick_ratio", 0.25),
                    "lower_wick_ratio": b.get("lower_wick_ratio", 0.25),
                    # Time features
                    "hour": b_time.hour,
                    "dayofweek": b_time.dayofweek,
                    "is_asian": 1 if 0 <= b_time.hour < 7 else 0,
                    "is_london": 1 if 7 <= b_time.hour < 13 else 0,
                    "is_ny": 1 if 13 <= b_time.hour < 21 else 0,
                })

            # --- Sell Fade: High touched exp_high - tol_yen ---
            if not sell_entered and b_high >= (exp_high - tol_yen):
                sell_entered = True
                entry_p = exp_high
                exit_p = max(exp_low, last_close_p)
                exit_reason = "close"
                exit_time = last_bar_time

                remaining = post_pub_bars.iloc[b_idx:]
                for _, rb in remaining.iterrows():
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    if sl_yen is not None and rb_high >= (entry_p + sl_yen):
                        exit_p = entry_p + sl_yen
                        exit_reason = "sl"
                        exit_time = rb.name
                        break
                    elif tp_yen is not None and rb_low <= (entry_p - tp_yen):
                        exit_p = entry_p - tp_yen
                        exit_reason = "tp"
                        exit_time = rb.name
                        break
                    elif rb_low <= exp_low:
                        exit_p = exp_low
                        exit_reason = "tp_boundary"
                        exit_time = rb.name
                        break

                pnl = (entry_p - exit_p) * lot_multiplier
                target = 1 if pnl > 0 else 0

                events.append({
                    "time": b_time,
                    "date": act_date,
                    "source": fc_source,
                    "direction": "sell",
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "target_rebound": target,
                    # Boundary context
                    "expected_low": exp_low,
                    "expected_high": exp_high,
                    "expected_range_pips": (exp_high - exp_low) / 0.01,
                    "dist_to_boundary_pips": (exp_high - b_high) / 0.01,
                    # Momentum features
                    "ret_1": b.get("ret_1", 0.0),
                    "ret_2": b.get("ret_2", 0.0),
                    "ret_3": b.get("ret_3", 0.0),
                    "ret_6": b.get("ret_6", 0.0),
                    "ret_12": b.get("ret_12", 0.0),
                    "ret_24": b.get("ret_24", 0.0),
                    # Technicals
                    "atr_14": b.get("atr_14", 0.0),
                    "atr_ratio": b.get("atr_ratio", 1.0),
                    "rsi_14": b.get("rsi_14", 50.0),
                    "bb_width": b.get("bb_width", 0.0),
                    "bb_pct": b.get("bb_pct", 0.5),
                    # Candle shape
                    "body_ratio": b.get("body_ratio", 0.5),
                    "upper_wick_ratio": b.get("upper_wick_ratio", 0.25),
                    "lower_wick_ratio": b.get("lower_wick_ratio", 0.25),
                    # Time features
                    "hour": b_time.hour,
                    "dayofweek": b_time.dayofweek,
                    "is_asian": 1 if 0 <= b_time.hour < 7 else 0,
                    "is_london": 1 if 7 <= b_time.hour < 13 else 0,
                    "is_ny": 1 if 13 <= b_time.hour < 21 else 0,
                })

    df_events = pd.DataFrame(events)
    return df_events


if __name__ == "__main__":
    print("Building touch events dataset...")
    df_events = extract_touch_events(symbol="USDJPY", source="all")
    print(f"Total touch events extracted: {len(df_events)}")
    print(f"Sources: {df_events['source'].value_counts().to_dict()}")
    print(f"Rebound target distribution:\n{df_events['target_rebound'].value_counts(normalize=True)}")
    print(f"Exit reasons:\n{df_events['exit_reason'].value_counts().to_dict()}")

    out_path = backend_dir / "artifacts" / "datasets" / "rebound_events.parquet"
    df_events.to_parquet(out_path, index=False)
    print(f"Saved dataset to {out_path}")
