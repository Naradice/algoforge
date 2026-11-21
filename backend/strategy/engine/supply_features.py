"""Point-in-time supply & demand event features for AlgoForge backtest engine.

Strictly enforces point-in-time alignment:
- JPX daily margin notices released at 16:00-17:00 JST are only visible starting next trading day's open.
- EDINET large holding reports submitted during market hours are only visible after market close or next morning.
- Prevents look-ahead bias completely.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_supply_events(
    path: str | None = None,
    store: str | None = None,
) -> pd.DataFrame:
    """Load supply/demand event records from Parquet or CSV.

    Columns expected:
    - code: stock ticker / local code (e.g. '7735' or '7735.T')
    - disclosure_time: ISO datetime string or Timestamp of actual public release
    - event_type: 'large_holding' | 'daily_margin' | 'regulation'
    - holder_name: optional
    - ratio_change: float (% point change, e.g. +1.25)
    - holding_ratio: float (% current holding)
    - margin_ratio: float (loan/margin ratio)
    - is_squeeze_alert: bool
    """
    if not path:
        return pd.DataFrame(columns=[
            "code", "disclosure_time", "event_type", "ratio_change",
            "holding_ratio", "margin_ratio", "is_squeeze_alert",
        ])

    p = Path(path)
    if not p.exists():
        candidate = Path(store or os.getenv("ARTIFACT_STORE_PATH", "artifacts")) / path
        if candidate.exists():
            p = candidate
        else:
            # Try removing leading 'artifacts/' if double-prefixed
            parts = p.parts
            if len(parts) > 1 and parts[0] == "artifacts":
                cand2 = Path(store or os.getenv("ARTIFACT_STORE_PATH", "artifacts")).joinpath(*parts[1:])
                if cand2.exists():
                    p = cand2

    if not p.exists():
        return pd.DataFrame(columns=[
            "code", "disclosure_time", "event_type", "ratio_change",
            "holding_ratio", "margin_ratio", "is_squeeze_alert",
        ])

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported event file format: {p.suffix}")

    if "disclosure_time" in df.columns:
        df["disclosure_time"] = pd.to_datetime(df["disclosure_time"], utc=True)
        df = df.sort_values("disclosure_time").reset_index(drop=True)
    return df


def supply_event_features(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    code: str,
    lookback_days: int = 10,
    market_close_time_jst: str = "15:30",
) -> pd.DataFrame:
    """Align supply/demand events to price bars with strict Point-in-Time discipline.

    For a daily bar dated D:
    - If event disclosed on day D after 15:30 JST (or before 09:00 next day):
      it becomes visible on day D+1.
    - If event disclosed on day D during market hours (before 15:30 JST):
      it becomes actionable on day D's close or D+1's open.
    - An event older than `lookback_days` decays to 0.

    Output features aligned with `index`:
    - event_active: 1 if any supply/demand event was disclosed within lookback_days, 0 otherwise
    - institutional_buy: 1 if large holding report indicated >= +1.0% increase within lookback_days
    - squeeze_alert: 1 if JPX margin report triggered squeeze alert within lookback_days
    - latest_ratio_change: float ratio change from the latest report
    - days_since_event: integer days since the last disclosed event (NaN if none)
    """
    n = len(index)
    res = pd.DataFrame(
        {
            "event_active": np.zeros(n, dtype=int),
            "institutional_buy": np.zeros(n, dtype=int),
            "squeeze_alert": np.zeros(n, dtype=int),
            "latest_ratio_change": np.zeros(n, dtype=float),
            "days_since_event": np.full(n, np.nan, dtype=float),
        },
        index=index,
    )

    if events is None or events.empty:
        return res

    # Clean code comparison (strip '.T' if present)
    clean_code = str(code).replace(".T", "").strip()
    ev = events.copy()
    if "code" in ev.columns:
        ev["clean_code"] = ev["code"].astype(str).str.replace(".T", "", regex=False).str.strip()
        ev = ev[ev["clean_code"] == clean_code]

    if ev.empty:
        return res

    # Ensure UTC DatetimeIndex
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx_utc = idx.tz_localize("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    # In Japanese market (JST = UTC+9), market closes at 15:30 JST = 06:30 UTC.
    # Daily bars typically have midnight UTC or 00:00:00 timestamps.
    # We map each event's disclosure timestamp to the first bar index where it becomes effective.
    # An event disclosed after market close on date D is effective on the bar corresponding to date D+1.
    
    # Calculate effective date for each event in JST:
    # If disclosure_time JST hour >= 15 (and minute >= 30) -> effective next calendar trading day
    ev_times = pd.to_datetime(ev["disclosure_time"], utc=True)
    
    # Pre-parse event properties
    r_changes = ev.get("ratio_change", pd.Series(0.0, index=ev.index)).fillna(0.0).values
    is_squeeze = ev.get("is_squeeze_alert", pd.Series(False, index=ev.index)).fillna(False).values
    ev_types = ev.get("event_type", pd.Series("", index=ev.index)).fillna("").values

    # Convert bar times and event times to dates for comparison
    bar_dates = idx_utc.normalize().asi8  # Nanosecond timestamp of date midnight UTC

    for i in range(len(idx_utc)):
        bar_t = idx_utc[i]
        # For daily bar i, the bar opens around 09:00 JST (00:00 UTC) of date D.
        # Any event whose disclosure time is strictly prior to 09:00 JST of date D is visible!
        # In UTC, 09:00 JST is 00:00 UTC of that same day.
        # If the bar represents date D, events disclosed up to bar_t (or midnight UTC of date D) are valid.
        cutoff_utc = bar_t

        past_mask = ev_times <= cutoff_utc
        if not np.any(past_mask):
            continue

        past_ev = ev[past_mask]
        past_times = ev_times[past_mask]
        past_r_changes = r_changes[past_mask]
        past_squeeze = is_squeeze[past_mask]

        # Latest event
        latest_idx = len(past_ev) - 1
        last_time = past_times.iloc[latest_idx]
        delta_days = (bar_t - last_time).total_seconds() / 86400.0

        if delta_days <= lookback_days:
            res.iloc[i, res.columns.get_loc("days_since_event")] = round(delta_days, 1)
            res.iloc[i, res.columns.get_loc("event_active")] = 1
            res.iloc[i, res.columns.get_loc("latest_ratio_change")] = float(past_r_changes[latest_idx])

            # Check if any event in lookback window was institutional buy or squeeze
            recent_mask = (cutoff_utc - past_times).dt.total_seconds() <= (lookback_days * 86400.0)
            recent_r = past_r_changes[recent_mask]
            recent_sq = past_squeeze[recent_mask]

            if np.any(recent_r >= 1.0):
                res.iloc[i, res.columns.get_loc("institutional_buy")] = 1
            if np.any(recent_sq):
                res.iloc[i, res.columns.get_loc("squeeze_alert")] = 1

    return res
