"""Unit tests for Point-in-Time supply/demand features and indicator integration."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from strategy.engine.supply_features import supply_event_features, load_supply_events
from strategy.engine.indicators import apply_indicators


def test_supply_event_point_in_time_alignment():
    """Verify that events disclosed after market close on day D are ONLY visible on day D+1."""
    # 3 daily bars: 2026-11-02, 2026-11-03, 2026-11-04
    dates = pd.to_datetime(["2026-11-02 00:00:00+00:00", "2026-11-03 00:00:00+00:00", "2026-11-04 00:00:00+00:00"])
    
    # Event released on 2026-11-02 at 16:30 JST (07:30 UTC) -> after 2026-11-02 midnight UTC
    events = pd.DataFrame([
        {
            "code": "7735",
            "disclosure_time": "2026-11-02 07:30:00+00:00",  # 16:30 JST
            "event_type": "daily_margin",
            "ratio_change": 0.0,
            "holding_ratio": 0.0,
            "margin_ratio": 0.35,
            "is_squeeze_alert": True,
        }
    ])

    feats = supply_event_features(dates, events, code="7735", lookback_days=5)

    # On day 2026-11-02 00:00:00 UTC, the event had NOT been disclosed yet!
    assert feats.iloc[0]["squeeze_alert"] == 0
    assert feats.iloc[0]["event_active"] == 0

    # On day 2026-11-03 00:00:00 UTC, the event is now public and visible
    assert feats.iloc[1]["squeeze_alert"] == 1
    assert feats.iloc[1]["event_active"] == 1

    # On day 2026-11-04, still within lookback window
    assert feats.iloc[2]["squeeze_alert"] == 1


def test_supply_event_institutional_buy():
    """Verify that large holding report with ratio increase triggers institutional_buy."""
    dates = pd.to_datetime(["2026-11-01 00:00:00+00:00", "2026-11-02 00:00:00+00:00", "2026-11-03 00:00:00+00:00"])
    
    events = pd.DataFrame([
        {
            "code": "8031.T",
            "disclosure_time": "2026-11-01 06:00:00+00:00",  # 15:00 JST
            "event_type": "large_holding",
            "ratio_change": 1.45,
            "holding_ratio": 7.2,
            "margin_ratio": 0.0,
            "is_squeeze_alert": False,
        }
    ])

    feats = supply_event_features(dates, events, code="8031", lookback_days=5)

    # On 2026-11-01 00:00 UTC (09:00 JST), 15:00 JST event was NOT yet disclosed
    assert feats.iloc[0]["institutional_buy"] == 0
    # On 2026-11-02 00:00 UTC, visible
    assert feats.iloc[1]["institutional_buy"] == 1
    assert feats.iloc[1]["latest_ratio_change"] == 1.45


def test_apply_indicators_supply_event(tmp_path):
    """Test supply_event integration in apply_indicators."""
    parquet_path = tmp_path / "events.parquet"
    ev_df = pd.DataFrame([
        {
            "code": "6361",
            "disclosure_time": "2026-11-01 05:00:00+00:00",
            "event_type": "large_holding",
            "ratio_change": 2.1,
            "holding_ratio": 6.5,
            "margin_ratio": 0.0,
            "is_squeeze_alert": False,
        }
    ])
    ev_df.to_parquet(parquet_path)

    dates = pd.date_range("2026-11-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0, 102.0, 104.0, 103.0, 105.0],
        "high": [103.0, 105.0, 106.0, 104.0, 107.0],
        "low": [99.0, 101.0, 103.0, 102.0, 104.0],
        "close": [102.0, 104.0, 103.0, 105.0, 106.0],
        "volume": [1000, 1200, 1100, 1500, 1400],
    }, index=dates)

    indicators_config = [
        {
            "id": "supply",
            "type": "supply_event",
            "params": {
                "symbol": "6361",
                "events_path": str(parquet_path),
                "lookback_days": 3,
            }
        }
    ]

    out_df = apply_indicators(df, indicators_config)

    assert "supply_event_active" in out_df.columns
    assert "supply_institutional_buy" in out_df.columns
    assert "supply_squeeze_alert" in out_df.columns
    assert out_df.iloc[0]["supply_institutional_buy"] == 0  # not disclosed yet at 00:00 UTC
    assert out_df.iloc[1]["supply_institutional_buy"] == 1  # disclosed at 05:00 on 11-01, visible on 11-02
