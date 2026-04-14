"""
OHLC collector — downloads historical candle data via finance_client providers.

Datasource config shape (stored in datasources.config):
    {
        "client": "yfinance" | "vantage",
        "symbol": "EURUSD=X",       # provider-specific ticker format
        "timeframe": "H1",           # M1 M5 M15 M30 H1 H4 D1 W1 MN
        "from_ts": "2020-01-01",     # ISO date string
        "to_ts": "2024-12-31",       # ISO date string; null = today
        "adjust_close": false,       # yfinance only
        "api_key": "...",            # vantage only
        "finance_target": "fx",      # vantage only: "fx" | "stock" | "crypto"
    }

Returns:
    CollectResult(artifact_path, row_count, from_ts, to_ts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import pandas as pd
from finance_client import frames as Frame
from finance_client.yfinance import download_ohlc as yf_download_ohlc
from finance_client.vantage import download_ohlc as vantage_download_ohlc

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

# AlgoForge timeframe string → finance_client Frame constant
_FRAME_MAP: dict[str, int] = {
    "M1":  Frame.MIN1,
    "M5":  Frame.MIN5,
    "M15": Frame.MIN15,
    "M30": Frame.MIN30,
    "H1":  Frame.H1,
    "H4":  Frame.H4,   # downloaded as H1 then resampled (yfinance has no native H4)
    "D1":  Frame.D1,
    "W1":  Frame.W1,
    "MN":  Frame.MO1,
}

# yfinance does not support H4 natively — download H1 and resample
_YF_RESAMPLE_FRAMES = {"H4"}


@dataclass
class CollectResult:
    artifact_path: str   # relative to ARTIFACT_STORE
    row_count: int
    from_ts: "datetime"
    to_ts: "datetime"


def collect(datasource_id: int, config: dict, incremental: dict | None = None) -> CollectResult:
    """Download OHLC data and save to parquet. Called inside a Celery worker.

    Args:
        datasource_id: ID of the datasource being collected.
        config: Datasource config dict (symbol, timeframe, client, etc.).
        incremental: When provided, appends new bars to an existing dataset instead of
            rewriting from scratch. Expected keys:
                - "dataset_id"    (int)      existing Dataset row id
                - "artifact_path" (str)      relative path to existing parquet
                - "to_ts"         (datetime) last timestamp in existing dataset
    """
    from datetime import datetime

    client = config.get("client", "yfinance")
    symbol: str = config["symbol"]
    timeframe: str = config.get("timeframe", "H1")
    to_str: str | None = config.get("to_ts")
    to_dt = pd.Timestamp(to_str, tz="UTC") if to_str else pd.Timestamp.now(tz="UTC")

    if incremental is not None:
        from_dt = pd.Timestamp(incremental["to_ts"], tz="UTC")
    else:
        from_str: str | None = config.get("from_ts")
        from_dt = pd.Timestamp(from_str, tz="UTC") if from_str else pd.Timestamp("2000-01-01", tz="UTC")

    if client == "yfinance":
        df = _collect_yfinance(symbol, timeframe, from_dt, to_dt, config)
    elif client == "vantage":
        df = _collect_vantage(symbol, timeframe, from_dt, to_dt, config)
    else:
        raise ValueError(f"Unknown OHLC client: {client!r}")

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"

    if incremental is not None:
        artifact_rel = incremental["artifact_path"]
        if df.empty:
            # No new bars — report existing dataset stats unchanged
            existing = pd.read_parquet(ARTIFACT_STORE / artifact_rel)
            return CollectResult(
                artifact_path=artifact_rel,
                row_count=len(existing),
                from_ts=existing.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
                to_ts=existing.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
            )
        from data.collectors._utils import merge_into_parquet
        total_rows = merge_into_parquet(ARTIFACT_STORE / artifact_rel, df)
        merged = pd.read_parquet(ARTIFACT_STORE / artifact_rel)
        from data.artifact_store import upload as _upload
        _upload(ARTIFACT_STORE / artifact_rel)
        return CollectResult(
            artifact_path=artifact_rel,
            row_count=total_rows,
            from_ts=merged.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
            to_ts=merged.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
        )

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {timeframe} ({from_dt} – {to_dt})")

    # Full collection — write fresh parquet
    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/{symbol.replace('/', '_')}_{timeframe}.parquet"
    df.to_parquet(ARTIFACT_STORE / artifact_rel)
    from data.artifact_store import upload as _upload
    _upload(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(df),
        from_ts=df.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
        to_ts=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
    )


def _collect_yfinance(
    symbol: str, timeframe: str, from_dt: pd.Timestamp, to_dt: pd.Timestamp, config: dict
) -> pd.DataFrame:
    # H4 is not natively supported — download H1 and resample
    download_tf = "H1" if timeframe == "H4" else timeframe
    frame = _FRAME_MAP.get(download_tf, Frame.D1)

    df = yf_download_ohlc(
        symbols=symbol,
        frame=frame,
        adjust_close=config.get("adjust_close", False),
    )
    if df.empty:
        return df

    df = _normalise_index(df)
    df = _filter_date_range(df, from_dt, to_dt)

    if timeframe == "H4":
        df = df.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(how="all")

    return df[["Open", "High", "Low", "Close", "Volume"]]


def _collect_vantage(
    symbol: str, timeframe: str, from_dt: pd.Timestamp, to_dt: pd.Timestamp, config: dict
) -> pd.DataFrame:
    from finance_client.vantage import target as Target

    api_key = config.get("api_key") or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        raise ValueError("Alpha Vantage api_key required in datasource config or ALPHA_VANTAGE_API_KEY env var")

    _target_map = {
        "fx": Target.FX,
        "stock": Target.STOCK,
        "crypto": Target.CRYPTO_CURRENCY,
    }
    finance_target = _target_map.get(config.get("finance_target", "fx"), Target.FX)
    frame = _FRAME_MAP.get(timeframe, Frame.D1)

    df = vantage_download_ohlc(
        api_key=api_key,
        symbols=symbol,
        frame=frame,
        finance_target=finance_target,
    )
    if df.empty:
        return df

    # VantageClient uses lowercase column names — capitalise to match standard
    df.columns = [c.capitalize() for c in df.columns]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = _normalise_index(df)
    df = _filter_date_range(df, from_dt, to_dt)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a UTC-aware DatetimeIndex."""
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def _filter_date_range(df: pd.DataFrame, from_dt: pd.Timestamp, to_dt: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= from_dt) & (df.index <= to_dt)]
