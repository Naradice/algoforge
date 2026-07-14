"""
Synthetic function collector — generates simple, deterministic time series from closed-form
formulas, for quickly sanity-checking models against a known ground truth (does AR/ARMA or a
neural net actually recover a known periodicity, or not?).

Datasource config shape (stored in datasources.config):
    {
        "function": "sine" | "sine_sum",
        "period": 50,        # T -- base period, in bars
        "amplitude": 1.0,    # A -- wave amplitude ("sine": the only wave; "sine_sum": the 2nd wave)
        "freq_ratio": 5,     # sine_sum only -- 2nd wave oscillates this many times faster than the base
        "base_price": 100.0, # vertical offset so the series looks like a price series
        "noise": 0.0,        # gaussian noise std dev added on top; 0 = pure deterministic
        "length": 2000,      # number of bars to generate
        "timeframe": "M5",   # bar spacing
        "seed": 42,          # noise RNG seed (only used when noise > 0)
        "start_ts": "2024-01-01",  # first bar timestamp
    }

Formulas (t = bar index, 0..length-1):
    "sine":     x_t = base_price + amplitude * sin(2*pi * t / period)
    "sine_sum": x_t = base_price + sin(2*pi * t / period) + amplitude * sin(2*pi * freq_ratio * t / period)

Both are a practical reading of "x_t periodic with period T" and "sin(t) + A*sin(T*t)",
reparameterized around a bar-count period so the result is a usable series at any timeframe --
raw sin(t) with integer t oscillates every ~6.3 bars, too fast to be a useful comparison signal.

Returns: CollectResult(artifact_path, row_count, from_ts, to_ts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

_PANDAS_FREQ = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W", "MN": "1MS",
}


@dataclass
class CollectResult:
    artifact_path: str   # relative to ARTIFACT_STORE
    row_count: int
    from_ts: datetime
    to_ts: datetime


def _generate_series(function: str, length: int, period: float, amplitude: float, freq_ratio: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    if function == "sine":
        return amplitude * np.sin(2 * np.pi * t / period)
    if function == "sine_sum":
        return np.sin(2 * np.pi * t / period) + amplitude * np.sin(2 * np.pi * freq_ratio * t / period)
    raise ValueError(f"Unknown synthetic function: {function!r} (expected 'sine' or 'sine_sum')")


def collect(datasource_id: int, config: dict) -> CollectResult:
    function = config.get("function", "sine")
    length = int(config.get("length", 2000))
    period = float(config.get("period", 50))
    amplitude = float(config.get("amplitude", 1.0))
    freq_ratio = float(config.get("freq_ratio", 5))
    base_price = float(config.get("base_price", 100.0))
    noise = float(config.get("noise", 0.0))
    seed = int(config.get("seed", 42))
    timeframe = config.get("timeframe", "M5")
    start_ts = config.get("start_ts") or "2024-01-01"

    if length < 2:
        raise ValueError("length must be at least 2")
    if period <= 0:
        raise ValueError("period must be positive")

    values = base_price + _generate_series(function, length, period, amplitude, freq_ratio)
    if noise > 0:
        rng = np.random.default_rng(seed)
        values = values + rng.normal(0, noise, length)

    freq = _PANDAS_FREQ.get(timeframe, "5min")
    index = pd.date_range(start=pd.Timestamp(start_ts, tz="UTC"), periods=length, freq=freq)

    df = pd.DataFrame({
        "open": values, "high": values, "low": values, "close": values,
        "volume": np.ones(length),
    }, index=index)
    df.index.name = "datetime"

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/{function}_{timeframe}.parquet"
    df.to_parquet(ARTIFACT_STORE / artifact_rel)

    from data.artifact_store import upload as _upload
    _upload(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(df),
        from_ts=df.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
        to_ts=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
    )
