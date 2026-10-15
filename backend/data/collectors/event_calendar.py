"""
Economic *event* calendar collector — scheduled releases with forecast / previous / actual.

Unlike ``economic_calendar`` (value series indexed by reference period), this keeps the **release
timestamp**, so strategies can reason about "minutes to the next high-impact release" and the surprise
of the last one.

Datasource config:
    {
        "sources": ["faireconomy", "alfred", "csv"],   # any subset, merged by (datetime, currency, title)
        # faireconomy: public weekly JSON (schedule + forecast + previous; forward-looking snapshots that
        # accumulate by re-running the collection)
        "faireconomy_url": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        # alfred: first-release value + release date for FRED series (history); fred_api_key optional
        "alfred_series": {"NFP": ["PAYEMS", "USD", "Non-Farm Employment Change", "diff"]},
        "from_ts": "2015-01-01",
        # csv/parquet: user-supplied history (e.g. a scraper's output). Columns: datetime,currency,title,
        # [impact,forecast,previous,actual]
        "csv_path": "L:/data/calendar/history.csv"
    }

Output parquet (one file; the union of the previous store and the new rows):
    datetime (UTC), currency, title, impact (low|medium|high|holiday), forecast, previous, actual (floats),
    forecast_raw / previous_raw / actual_raw (provider strings such as "0.3%" or "175K"), source
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
COLUMNS = ["datetime", "currency", "title", "impact", "forecast", "previous", "actual", "source"]
_IMPACT = {"high": "high", "medium": "medium", "low": "low", "holiday": "holiday",
           "non-economic": "low", "3": "high", "2": "medium", "1": "low"}

# series_key -> [FRED id, currency, title, transform]
DEFAULT_ALFRED_SERIES = {
    "NFP": ["PAYEMS", "USD", "Non-Farm Employment Change", "diff"],
    "CPI": ["CPIAUCSL", "USD", "CPI m/m", "pct"],
    "UNRATE": ["UNRATE", "USD", "Unemployment Rate", "level"],
    "RETAIL": ["RSAFS", "USD", "Retail Sales m/m", "pct"],
    "INDPRO": ["INDPRO", "USD", "Industrial Production m/m", "pct"],
}


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


def parse_number(x) -> float:
    """'175K' -> 175000, '0.3%' -> 0.3, '-1.2B' -> -1.2e9, '' -> nan."""
    if x is None:
        return float("nan")
    s = str(x).strip().replace(",", "")
    if not s or s.lower() in ("nan", "none", "-"):
        return float("nan")
    m = re.match(r"^[<>]?(-?\d+(?:\.\d+)?)\s*([kKmMbBtT%]?)", s)
    if not m:
        return float("nan")
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(m.group(2).lower(), 1.0)
    return float(m.group(1)) * mult


def _norm_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["impact"] = df["impact"].astype(str).str.lower().map(lambda v: _IMPACT.get(v, v))
    df["currency"] = df["currency"].astype(str).str.upper()
    for c in ("forecast", "previous", "actual"):
        raw = df[c].astype("object")
        df[c + "_raw"] = raw.where(raw.notna(), None)
        df[c] = raw.map(parse_number)
    df["source"] = df["source"].fillna(source)
    keep = COLUMNS + ["forecast_raw", "previous_raw", "actual_raw"]
    return df.dropna(subset=["datetime"])[keep]


def fetch_faireconomy(url: str) -> pd.DataFrame:
    import requests  # type: ignore

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"faireconomy: unexpected payload {str(rows)[:120]}")
    df = pd.DataFrame(rows).rename(columns={"country": "currency", "date": "datetime"})
    return _norm_frame(df, "faireconomy")


def _first_releases(obs: pd.DataFrame) -> pd.DataFrame:
    """obs: columns obs (period), rt (real-time start = release date), value -> first release per period."""
    obs = obs.dropna()
    return obs.sort_values("rt").groupby("obs", as_index=False).first().sort_values("obs").reset_index(drop=True)


def alfred_to_events(obs: pd.DataFrame, currency: str, title: str, mode: str, series_id: str) -> pd.DataFrame:
    first = _first_releases(obs)
    lvl = first["value"]
    if mode == "diff":
        act = lvl.diff()
    elif mode == "pct":
        act = lvl.pct_change() * 100
    else:
        act = lvl
    out = pd.DataFrame({"datetime": pd.to_datetime(first["rt"].values, utc=True), "currency": currency,
                        "title": title, "impact": "high", "actual": act.values,
                        "previous": act.shift(1).values, "forecast": None, "source": f"alfred:{series_id}"})
    return _norm_frame(out.dropna(subset=["actual"]), "alfred")


def fetch_alfred_series(series_id: str, currency: str, title: str, mode: str, from_dt: pd.Timestamp,
                        api_key: str = "") -> pd.DataFrame:
    """First-release actuals with their release dates (ALFRED vintages).

    mode: "level" first-release level | "diff" change vs the previous first release | "pct" percent change.
    """
    import requests  # type: ignore

    if api_key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                                 "realtime_start": "1990-01-01", "realtime_end": "9999-12-31",
                                 "observation_start": from_dt.strftime("%Y-%m-%d")}, timeout=120)
        r.raise_for_status()
        obs = pd.DataFrame(r.json().get("observations", []))
        if obs.empty:
            return pd.DataFrame(columns=COLUMNS)
        obs = obs.rename(columns={"date": "obs", "realtime_start": "rt"})[["obs", "rt", "value"]]
    else:
        r = requests.get("https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                         params={"id": series_id, "cosd": from_dt.strftime("%Y-%m-%d"),
                                 "realtime_start": "1990-01-01", "realtime_end": "9999-12-31"}, timeout=120)
        r.raise_for_status()
        wide = pd.read_csv(io.StringIO(r.text))      # observation_date, <ID>_YYYYMMDD (one column per vintage)
        long = wide.melt(id_vars=wide.columns[0], var_name="vintage", value_name="value").dropna(subset=["value"])
        long["rt"] = pd.to_datetime(long["vintage"].str.extract(r"_(\d{8})$")[0], format="%Y%m%d", errors="coerce")
        obs = long.rename(columns={wide.columns[0]: "obs"})[["obs", "rt", "value"]]
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs["obs"] = pd.to_datetime(obs["obs"])
    obs["rt"] = pd.to_datetime(obs["rt"])
    return alfred_to_events(obs, currency, title, mode, series_id)


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    if "datetime" not in df.columns:
        for alt in ("date", "time", "timestamp", "release_time"):
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"})
                break
    if "currency" not in df.columns and "country" in df.columns:
        df = df.rename(columns={"country": "currency"})
    return _norm_frame(df, "csv")


def merge_events(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union; per (datetime, currency, title) keep the row with the most fields filled (later rows win ties)."""
    parts = [x for x in (old, new) if x is not None and len(x)]
    allr = pd.concat(parts, ignore_index=True)
    allr["_fill"] = allr[["forecast", "previous", "actual"]].notna().sum(axis=1)
    allr["_ord"] = range(len(allr))
    allr = allr.sort_values(["_fill", "_ord"]).drop_duplicates(["datetime", "currency", "title"], keep="last")
    return allr.drop(columns=["_fill", "_ord"]).sort_values("datetime").reset_index(drop=True)


def collect(datasource_id: int, config: dict) -> CollectResult:
    sources = config.get("sources") or ["faireconomy"]
    from_dt = pd.Timestamp(config.get("from_ts") or "2010-01-01", tz="UTC")
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for src in sources:
        try:
            if src == "faireconomy":
                frames.append(fetch_faireconomy(config.get(
                    "faireconomy_url", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")))
            elif src == "alfred":
                key = config.get("fred_api_key") or os.getenv("FRED_API_KEY", "")
                for sid, ccy, title, mode in (config.get("alfred_series") or DEFAULT_ALFRED_SERIES).values():
                    frames.append(fetch_alfred_series(sid, ccy, title, mode, from_dt, key))
            elif src == "csv":
                frames.append(_read_csv(config["csv_path"]))
            else:
                raise ValueError(f"unknown event source {src!r}")
        except Exception as exc:  # one failing source must not discard the others
            errors.append(f"{src}: {exc}")
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("event_calendar: no source returned data — " + "; ".join(errors))

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "events.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    merged = merge_events(old, pd.concat(frames, ignore_index=True))
    merged.to_parquet(path)
    if errors:
        logging.getLogger(__name__).warning("event_calendar partial failure: %s", errors)
    return CollectResult(artifact_path=f"datasets/src_{datasource_id}/events.parquet", row_count=len(merged),
                         from_ts=merged["datetime"].min().to_pydatetime(),
                         to_ts=merged["datetime"].max().to_pydatetime())
