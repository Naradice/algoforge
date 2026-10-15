"""
AI-agent risk overlay for the portfolio engine.

The agent never picks trades; it is asked, once per rebalance period, for a *risk multiplier* in
{0.0, 0.5, 1.0} given a point-in-time macro snapshot (rate differentials, VIX level/change, upcoming and
recent high-impact releases with surprise, measured broker swaps). The base strategy stays rule-based and
testable; the agent can only shrink exposure.

Leakage guard: a language model "remembers" history, so asking it about 2008 in a backtest is look-ahead.
By construction the overlay therefore only calls the model for dates >= ``min_date`` (default: ``2026-01-01``,
after the model's training horizon) and returns 1.0 before it, unless ``allow_historical`` is set explicitly —
in which case results must be treated as contaminated. Use it in paper/live mode and in forward evaluation.

Overlay spec (inside ``definition["overlays"]``):
    {"type": "agent_risk", "model": "gemini-2.0-flash", "min_date": "2026-01-01", "allow_historical": false,
     "events": "datasets/src_66/events.parquet", "series": {"VIX": "research/fx/rates/VIXCLS.csv"},
     "swaps": "datasets/src_67/swaps.parquet", "period": "W"}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SYSTEM = """You are the risk officer of a systematic FX carry portfolio. You do NOT pick trades.
Given the macro snapshot, answer with exactly one JSON object: {"multiplier": 1.0 | 0.5 | 0.0, "reason": "<=20 words"}.
1.0 = normal exposure; 0.5 = reduce (stress building, major event risk for funding/target currencies);
0.0 = flat (acute risk-off / carry-unwind conditions). Be conservative about cutting: only cut for concrete reasons."""

_cache: dict[str, float] = {}


def clear_cache() -> None:
    _cache.clear()


def build_snapshot(asof: pd.Timestamp, carry: pd.DataFrame | None, series: dict[str, pd.Series],
                   events: pd.DataFrame | None, swaps: pd.DataFrame | None, horizon_days: int = 7) -> dict:
    """Point-in-time snapshot: nothing dated after `asof` is included."""
    snap: dict = {"asof": asof.strftime("%Y-%m-%d")}
    if carry is not None and asof in carry.index:
        row = carry.loc[asof].dropna().sort_values(ascending=False)
        snap["rate_differentials_pct"] = {k: round(float(v) * 100, 2) for k, v in row.items()}
    for name, s in series.items():
        hist = s[s.index <= asof].dropna()
        if len(hist) > 21:
            snap[name] = {"last": round(float(hist.iloc[-1]), 2),
                          "chg_20d_pct": round(float(hist.iloc[-1] / hist.iloc[-21] - 1) * 100, 1),
                          "pctile_1y": round(float((hist.iloc[-250:] <= hist.iloc[-1]).mean()) * 100)}
    if events is not None and len(events):
        t = asof.tz_localize("UTC") if asof.tzinfo is None else asof
        ev = events[events["impact"].isin(["high", "medium"])]
        up = ev[(ev["datetime"] > t) & (ev["datetime"] <= t + pd.Timedelta(days=horizon_days))]
        snap["upcoming_events"] = [f"{r.datetime:%a %H:%M} {r.currency} {r.title} ({r.impact})"
                                   for r in up.head(15).itertuples()]
        past = ev[(ev["datetime"] <= t) & (ev["datetime"] > t - pd.Timedelta(days=horizon_days))
                  & ev["actual"].notna() & ev["forecast"].notna()]
        snap["recent_surprises"] = [f"{r.currency} {r.title}: actual {r.actual:g} vs forecast {r.forecast:g}"
                                    for r in past.head(10).itertuples()]
    if swaps is not None and len(swaps):
        last = swaps[swaps["datetime"] <= (asof.tz_localize("UTC") if asof.tzinfo is None else asof)]
        if len(last):
            cur = last.sort_values("datetime").groupby("symbol").tail(1)
            snap["broker_swap_ann_pct"] = {r.symbol: [None if pd.isna(r.swap_long_ann) else round(r.swap_long_ann * 100, 1),
                                                      None if pd.isna(r.swap_short_ann) else round(r.swap_short_ann * 100, 1)]
                                           for r in cur.itertuples()}
    return snap


def gemini_llm(model: str) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        try:
            from google import genai  # type: ignore
        except ImportError:
            logger.warning("google-genai not installed — agent_risk returns 1.0")
            return ""
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.warning("No GOOGLE_API_KEY — agent_risk returns 1.0")
            return ""
        try:
            return genai.Client(api_key=key).models.generate_content(model=model, contents=prompt).text or ""
        except Exception as exc:
            logger.warning("agent_risk: model call failed: %s", exc)
            return ""
    return _call


def parse_multiplier(text: str) -> float:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        m = float(json.loads(text[start:end])["multiplier"])
    except Exception:
        return 1.0
    return min([0.0, 0.5, 1.0], key=lambda c: abs(c - m))        # snap to the allowed set; default is "no cut"


def agent_multipliers(index: pd.DatetimeIndex, spec: dict, carry: pd.DataFrame | None,
                      series: dict[str, pd.Series], events: pd.DataFrame | None, swaps: pd.DataFrame | None,
                      llm: Callable[[str], str] | None = None) -> pd.Series:
    """Daily multiplier series. The model is called once per `period` and only for dates >= min_date."""
    llm = llm or gemini_llm(spec.get("model") or os.getenv("LLM_MODEL", "gemini-2.0-flash"))
    min_date = pd.Timestamp(spec.get("min_date", "2026-01-01"))
    allow_hist = bool(spec.get("allow_historical", False))
    period = {"W": "W", "M": "M", "D": "D"}.get(spec.get("period", "W"), "W")
    out = pd.Series(1.0, index=index)
    groups = pd.Series(index, index=index).groupby(index.to_period(period))
    current = 1.0
    for _, g in groups:
        first = g.iloc[0]
        if first >= min_date or allow_hist:
            snap = build_snapshot(first - pd.Timedelta(days=1) if first > index[0] else first,
                                  carry, series, events, swaps)
            prompt = f"{_SYSTEM}\n\nSnapshot:\n{json.dumps(snap, ensure_ascii=False)}"
            key = hashlib.md5(prompt.encode()).hexdigest()
            if key not in _cache:
                _cache[key] = parse_multiplier(llm(prompt))
            current = _cache[key]
        else:
            current = 1.0
        out.loc[g.index] = current
    return out                                  # each period's snapshot uses info up to the day before it starts
