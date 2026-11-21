"""
Feature extraction and model inference utilities for Rebound Classifier.
Shared across backtesting (analyst_strategy.py) and live execution (live_runner.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURE_COLS = [
    "is_buy",
    "expected_range_pips",
    "dist_to_boundary_pips",
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24",
    "atr_14", "atr_ratio", "rsi_14", "bb_width", "bb_pct",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "hour", "dayofweek", "is_asian", "is_london", "is_ny"
]

_CACHED_BOOSTER: lgb.Booster | None = None
_CACHED_MODEL_PATH: str | None = None


def get_rebound_booster(model_path: str | Path | None = None) -> lgb.Booster:
    global _CACHED_BOOSTER, _CACHED_MODEL_PATH
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "models" / "rebound_classifier" / "lgbm_rebound_model.txt"
    model_str = str(model_path)
    if _CACHED_BOOSTER is None or _CACHED_MODEL_PATH != model_str:
        if not Path(model_str).exists():
            raise FileNotFoundError(f"Model file not found at: {model_str}")
        _CACHED_BOOSTER = lgb.Booster(model_file=model_str)
        _CACHED_MODEL_PATH = model_str
    return _CACHED_BOOSTER


def compute_m30_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators on M30 bars DataFrame."""
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


def predict_rebound_prob(
    bar_series: pd.Series,
    direction: str,
    exp_low: float,
    exp_high: float,
    touch_price: float,
    bar_time: pd.Timestamp,
    model_path: str | Path | None = None,
) -> float:
    """Predicts rebound probability (0.0 to 1.0) given current bar and boundary context."""
    booster = get_rebound_booster(model_path)

    is_buy = 1 if direction.lower() == "buy" else 0
    exp_range_pips = (exp_high - exp_low) / 0.01

    if is_buy:
        dist_to_boundary_pips = (float(bar_series.get("low", touch_price)) - exp_low) / 0.01
    else:
        dist_to_boundary_pips = (exp_high - float(bar_series.get("high", touch_price))) / 0.01

    hour = bar_time.hour
    dayofweek = getattr(bar_time, "dayofweek", bar_time.weekday() if hasattr(bar_time, "weekday") else 0)

    feat_dict = {
        "is_buy": is_buy,
        "expected_range_pips": exp_range_pips,
        "dist_to_boundary_pips": dist_to_boundary_pips,
        "ret_1": float(bar_series.get("ret_1", 0.0)),
        "ret_2": float(bar_series.get("ret_2", 0.0)),
        "ret_3": float(bar_series.get("ret_3", 0.0)),
        "ret_6": float(bar_series.get("ret_6", 0.0)),
        "ret_12": float(bar_series.get("ret_12", 0.0)),
        "ret_24": float(bar_series.get("ret_24", 0.0)),
        "atr_14": float(bar_series.get("atr_14", 0.0)),
        "atr_ratio": float(bar_series.get("atr_ratio", 1.0)),
        "rsi_14": float(bar_series.get("rsi_14", 50.0)),
        "bb_width": float(bar_series.get("bb_width", 0.0)),
        "bb_pct": float(bar_series.get("bb_pct", 0.5)),
        "body_ratio": float(bar_series.get("body_ratio", 0.5)),
        "upper_wick_ratio": float(bar_series.get("upper_wick_ratio", 0.25)),
        "lower_wick_ratio": float(bar_series.get("lower_wick_ratio", 0.25)),
        "hour": hour,
        "dayofweek": dayofweek,
        "is_asian": 1 if 0 <= hour < 7 else 0,
        "is_london": 1 if 7 <= hour < 13 else 0,
        "is_ny": 1 if 13 <= hour < 21 else 0,
    }

    df_row = pd.DataFrame([feat_dict])[FEATURE_COLS]
    prob = float(booster.predict(df_row)[0])
    return prob
