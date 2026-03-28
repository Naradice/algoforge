"""
OHLC collector — downloads historical candle data via yfinance or Alpha Vantage.

Datasource config shape (stored in datasources.config):
    {
        "client": "yfinance" | "vantage",
        "symbol": "EURUSD=X",       # yfinance ticker format
        "timeframe": "H1",           # M1 M5 M15 M30 H1 H4 D1 W1
        "from_ts": "2020-01-01",     # ISO date string
        "to_ts": "2024-12-31",       # ISO date string; null = today
        "adjust_close": false,       # yfinance only
        "api_key": "...",            # vantage only
    }

Returns:
    CollectResult(artifact_path, row_count, from_ts, to_ts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Timeframe → yfinance interval string
_YF_INTERVAL: dict[str, str] = {
    "M1": "1m",
    "M2": "2m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "1h",   # we resample 1h → 4h after download
    "D1": "1d",
    "W1": "1wk",
    "MN": "1mo",
}

# yfinance maximum lookback in days per interval (None = unlimited)
_YF_MAX_LOOKBACK_DAYS: dict[str, int | None] = {
    "1m": 7,
    "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "1h": 730,
    "1d": None, "1wk": None, "1mo": None,
}

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))


@dataclass
class CollectResult:
    artifact_path: str   # relative to ARTIFACT_STORE
    row_count: int
    from_ts: datetime
    to_ts: datetime


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(how="all")


def collect(datasource_id: int, config: dict) -> CollectResult:
    """
    Synchronous download — called inside an arq worker (runs in executor or directly).
    """
    client = config.get("client", "yfinance")
    symbol: str = config["symbol"]
    timeframe: str = config.get("timeframe", "H1")
    from_str: str | None = config.get("from_ts")
    to_str: str | None = config.get("to_ts")

    from_dt = pd.Timestamp(from_str) if from_str else pd.Timestamp("2000-01-01")
    to_dt = pd.Timestamp(to_str) if to_str else pd.Timestamp.now()

    if client == "yfinance":
        df = _download_yfinance(symbol, timeframe, from_dt, to_dt, config)
    elif client == "vantage":
        df = _download_vantage(symbol, timeframe, from_dt, to_dt, config)
    else:
        raise ValueError(f"Unknown OHLC client: {client!r}")

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {timeframe} ({from_dt} to {to_dt})")

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"

    # Save to parquet
    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/{symbol.replace('/', '_')}_{timeframe}.parquet"
    df.to_parquet(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(df),
        from_ts=df.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
        to_ts=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
    )


def _download_yfinance(
    symbol: str, timeframe: str, from_dt: pd.Timestamp, to_dt: pd.Timestamp, config: dict
) -> pd.DataFrame:
    import yfinance as yf  # type: ignore

    interval = _YF_INTERVAL.get(timeframe, "1d")

    # Clamp from_dt to yfinance's lookback limit for this interval
    max_days = _YF_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is not None:
        earliest_allowed = to_dt - pd.Timedelta(days=max_days)
        if from_dt < earliest_allowed:
            from_dt = earliest_allowed

    df: pd.DataFrame = yf.download(
        symbol,
        start=from_dt.strftime("%Y-%m-%d"),
        end=to_dt.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=config.get("adjust_close", False),
        progress=False,
    )

    if df.empty:
        return df

    # yfinance returns MultiIndex columns when downloading a single ticker in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if timeframe == "H4":
        df = _resample_4h(df)

    df.index = pd.to_datetime(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def _download_vantage(
    symbol: str, timeframe: str, from_dt: pd.Timestamp, to_dt: pd.Timestamp, config: dict
) -> pd.DataFrame:
    """Alpha Vantage download via alpha_vantage library."""
    from alpha_vantage.foreignexchange import ForeignExchange  # type: ignore
    from alpha_vantage.timeseries import TimeSeries  # type: ignore

    api_key = config.get("api_key") or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        raise ValueError("Alpha Vantage api_key required in datasource config")

    _AV_INTERVAL = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "60min"}

    if "/" in symbol:
        from_sym, to_sym = symbol.split("/")
        client = ForeignExchange(key=api_key, output_format="pandas")
        if timeframe in _AV_INTERVAL:
            data, _ = client.get_currency_exchange_intraday(from_sym, to_sym, interval=_AV_INTERVAL[timeframe], outputsize="full")
        else:
            data, _ = client.get_currency_exchange_daily(from_sym, to_sym, outputsize="full")
    else:
        ts = TimeSeries(key=api_key, output_format="pandas")
        if timeframe in _AV_INTERVAL:
            data, _ = ts.get_intraday(symbol, interval=_AV_INTERVAL[timeframe], outputsize="full")
        else:
            data, _ = ts.get_daily(symbol, outputsize="full")

    # Alpha Vantage returns columns like "1. open", "2. high", etc.
    rename = {}
    for col in data.columns:
        for key in ["open", "high", "low", "close", "volume"]:
            if key in col.lower():
                rename[col] = key.capitalize()
                break
    data = data.rename(columns=rename)

    for col in ["Open", "High", "Low", "Close"]:
        if col not in data.columns:
            data[col] = float("nan")
    if "Volume" not in data.columns:
        data["Volume"] = 0.0

    data.index = pd.to_datetime(data.index).tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC")
    data = data.sort_index()
    data = data[(data.index >= from_dt.tz_localize("UTC")) & (data.index <= to_dt.tz_localize("UTC"))]
    return data[["Open", "High", "Low", "Close", "Volume"]]
