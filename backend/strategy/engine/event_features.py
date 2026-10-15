"""Event-calendar features for the `event` indicator (windows around releases, last surprise)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}


def load_events(path: str | None, store: str | None = None) -> pd.DataFrame:
    if not path:
        raise ValueError("event indicator needs params.calendar (artifact path of the event_calendar parquet)")
    p = Path(path)
    if not p.is_absolute():
        p = Path(store or os.getenv("ARTIFACT_STORE_PATH", "artifacts")) / path
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def event_features(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    currencies: list[str],
    min_impact: str = "high",
    pre_minutes: int = 60,
    post_minutes: int = 60,
    title_contains: str | None = None,
    surprise_window: int = 24,
) -> pd.DataFrame:
    """Point-in-time features aligned to `index` (bar timestamps, interpreted as UTC when naive).

    near            1 if a matching release is within [-pre_minutes, +post_minutes] of the bar timestamp
    mins_to_next    minutes until the next matching scheduled release (NaN if none)
    mins_since_last minutes since the last matching release (NaN if none)
    surprise        (actual - forecast) of the last released event, divided by the std of the previous
                    `surprise_window` surprises (NaN until two events exist; uses only events released <= t)
    """
    ev = events
    if currencies:
        ev = ev[ev["currency"].isin([c.upper() for c in currencies])]
    ev = ev[ev["impact"].map(lambda v: _IMPACT_RANK.get(str(v), 0)) >= _IMPACT_RANK.get(min_impact, 3)]
    if title_contains:
        ev = ev[ev["title"].str.contains(title_contains, case=False, na=False)]
    ev = ev.sort_values("datetime")

    idx = pd.DatetimeIndex(index)
    t = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    # pandas >= 3 may hold s/ms/us resolution: force ns before taking integer views
    tv = t.as_unit("ns").asi8.astype("int64")
    ev_t = (ev["datetime"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
            if len(ev) else np.array([], dtype="int64"))
    minute = 60 * 10**9

    nxt = np.full(len(t), np.nan)
    prv = np.full(len(t), np.nan)
    if len(ev_t):
        pos = np.searchsorted(ev_t, tv, side="left")         # first event >= t
        has_next = pos < len(ev_t)
        nxt[has_next] = (ev_t[pos[has_next]] - tv[has_next]) / minute
        pos_r = np.searchsorted(ev_t, tv, side="right") - 1    # last event <= t
        has_prev = pos_r >= 0
        prv[has_prev] = (tv[has_prev] - ev_t[pos_r[has_prev]]) / minute
    # a scheduled release inside the bar itself counts as "near" (schedule is public ahead of time)
    bar_min = float(np.median(np.diff(tv)) / minute) if len(tv) > 1 else 0.0
    near = ((np.nan_to_num(nxt, nan=1e18) < pre_minutes + bar_min)
            | (np.nan_to_num(prv, nan=1e18) <= post_minutes)).astype(int)

    # surprise: only rows with both actual and forecast; z-scored by trailing std (past events only)
    sp = ev.dropna(subset=["actual", "forecast"])
    surprise = np.full(len(t), np.nan)
    if len(sp) >= 2:
        s_raw = (sp["actual"] - sp["forecast"]).to_numpy(float)
        sd = pd.Series(s_raw).rolling(surprise_window, min_periods=2).std().shift(1).to_numpy()
        z = s_raw / np.where(sd > 0, sd, np.nan)
        s_t = sp["datetime"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
        pos_s = np.searchsorted(s_t, tv, side="right") - 1
        ok = pos_s >= 0
        surprise[ok] = z[pos_s[ok]]
    return pd.DataFrame({"near": near, "mins_to_next": nxt, "mins_since_last": prv, "surprise": surprise},
                        index=index)
