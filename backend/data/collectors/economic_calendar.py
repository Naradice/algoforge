"""
Economic calendar collector — downloads historical economic indicator releases.

Datasource config shape:
    {
        "source": "alpha_vantage",          # alpha_vantage | fred
        "api_key": "...",                    # AV key; for FRED use fred_api_key or FRED_API_KEY env
        "fred_api_key": "...",              # FRED key (fred source only)
        "indicators": ["CPI", "NONFARM_PAYROLL", "UNEMPLOYMENT", "FEDERAL_FUNDS_RATE"],
        "interval": "monthly",              # monthly | quarterly | annual (AV only; FRED ignores)
        "from_ts": "2020-01-01",
        "to_ts": ""
    }

Alpha Vantage indicator names (source="alpha_vantage"):
    REAL_GDP, REAL_GDP_PER_CAPITA, TREASURY_YIELD, FEDERAL_FUNDS_RATE,
    CPI, INFLATION, RETAIL_SALES, DURABLES, UNEMPLOYMENT, NONFARM_PAYROLL

FRED series IDs (source="fred"):
    CPIAUCSL (CPI), PAYEMS (Nonfarm Payroll), UNRATE (Unemployment Rate),
    FEDFUNDS (Federal Funds Rate), GDP, RSXFS (Retail Sales), DGORDER (Durable Goods),
    T10Y2Y (Treasury Yield Spread), VIXCLS (VIX)

Output parquet columns (long format, indexed by datetime UTC):
    indicator, value, unit
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

# Maps human-friendly indicator name → FRED series ID
_FRED_SERIES: dict[str, str] = {
    "CPI":                  "CPIAUCSL",
    "NONFARM_PAYROLL":      "PAYEMS",
    "UNEMPLOYMENT":         "UNRATE",
    "FEDERAL_FUNDS_RATE":   "FEDFUNDS",
    "GDP":                  "GDP",
    "REAL_GDP":             "GDPC1",
    "RETAIL_SALES":         "RSXFS",
    "DURABLES":             "DGORDER",
    "TREASURY_YIELD":       "DGS10",
    "VIX":                  "VIXCLS",
}

# Alpha Vantage indicator functions that support an interval param
_AV_WITH_INTERVAL = {"REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE",
                     "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"}


@dataclass
class CollectResult:
    artifact_path: str  # relative to ARTIFACT_STORE
    row_count: int
    from_ts: datetime
    to_ts: datetime


def collect(datasource_id: int, config: dict) -> CollectResult:
    source = config.get("source", "alpha_vantage")
    indicators: list[str] = config.get("indicators") or ["CPI", "NONFARM_PAYROLL", "UNEMPLOYMENT", "FEDERAL_FUNDS_RATE"]
    from_str: str | None = config.get("from_ts")
    to_str: str | None = config.get("to_ts")

    from_dt = pd.Timestamp(from_str, tz="UTC") if from_str else pd.Timestamp("2000-01-01", tz="UTC")
    to_dt = pd.Timestamp(to_str, tz="UTC") if to_str else pd.Timestamp.now(tz="UTC")

    frames: list[pd.DataFrame] = []
    if source == "alpha_vantage":
        api_key = config.get("api_key") or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            raise ValueError("Alpha Vantage api_key required in datasource config or ALPHA_VANTAGE_API_KEY env")
        interval = config.get("interval", "monthly")
        for indicator in indicators:
            df = _fetch_alpha_vantage(indicator, api_key, interval, from_dt, to_dt)
            if not df.empty:
                frames.append(df)
            time.sleep(0.5)  # respect AV rate limit (5 calls/min free tier)

    elif source == "fred":
        api_key = config.get("fred_api_key") or config.get("api_key") or os.getenv("FRED_API_KEY", "")
        if not api_key:
            raise ValueError("FRED api_key required in datasource config (fred_api_key) or FRED_API_KEY env")
        for indicator in indicators:
            df = _fetch_fred(indicator, api_key, from_dt, to_dt)
            if not df.empty:
                frames.append(df)
            time.sleep(0.2)
    else:
        raise ValueError(f"Unknown economic calendar source: {source!r}")

    if not frames:
        raise RuntimeError(f"No economic data returned for indicators {indicators}")

    combined = pd.concat(frames, ignore_index=False)
    combined = combined.sort_index()

    # Write partitioned parquet (year/month partitions)
    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}" / "economic_calendar"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined["year"] = combined.index.year
    combined["month"] = combined.index.month

    table = pa.Table.from_pandas(combined.reset_index(), preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["year", "month"],
    )

    artifact_rel = f"datasets/src_{datasource_id}/economic_calendar"

    actual_from = combined.index.min().to_pydatetime().replace(tzinfo=timezone.utc)
    actual_to = combined.index.max().to_pydatetime().replace(tzinfo=timezone.utc)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(combined),
        from_ts=actual_from,
        to_ts=actual_to,
    )


def _fetch_alpha_vantage(
    indicator: str, api_key: str, interval: str,
    from_dt: pd.Timestamp, to_dt: pd.Timestamp,
) -> pd.DataFrame:
    import requests  # type: ignore

    if indicator not in _AV_WITH_INTERVAL:
        raise ValueError(f"Unknown Alpha Vantage economic indicator: {indicator!r}. "
                         f"Valid options: {sorted(_AV_WITH_INTERVAL)}")

    params: dict = {"function": indicator, "apikey": api_key, "datatype": "json"}
    if indicator in {"REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE",
                     "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"}:
        params["interval"] = interval

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "Information" in payload:
        raise RuntimeError(f"Alpha Vantage API limit reached: {payload['Information']}")
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage API note: {payload['Note']}")

    raw_data = payload.get("data", [])
    unit = payload.get("unit", "")

    if not raw_data:
        return pd.DataFrame()

    rows = [(d["date"], float(d["value"]) if d["value"] != "." else float("nan")) for d in raw_data]
    df = pd.DataFrame(rows, columns=["datetime", "value"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[(df.index >= from_dt) & (df.index <= to_dt)]
    df["indicator"] = indicator
    df["unit"] = unit
    return df[["indicator", "value", "unit"]]


def _fetch_fred(
    indicator: str, api_key: str,
    from_dt: pd.Timestamp, to_dt: pd.Timestamp,
) -> pd.DataFrame:
    import requests  # type: ignore

    series_id = _FRED_SERIES.get(indicator, indicator)  # allow raw series IDs too

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": from_dt.strftime("%Y-%m-%d"),
        "observation_end": to_dt.strftime("%Y-%m-%d"),
    }

    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame()

    rows = []
    for obs in observations:
        if obs["value"] == ".":
            continue
        rows.append((obs["date"], float(obs["value"])))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["datetime", "value"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df["indicator"] = indicator
    df["unit"] = ""
    return df[["indicator", "value", "unit"]]
