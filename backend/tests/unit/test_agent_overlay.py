import json

import numpy as np
import pandas as pd

from strategy.portfolio import agent_overlay as ao
from strategy.portfolio.engine import run_portfolio_backtest


def _mk(tmp_path, n=400):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=n)
    (tmp_path / "rates").mkdir()
    months = pd.date_range("2014-12-01", periods=30, freq="MS")
    for sid, lvl in {"HI": 4.0, "LO": 0.0}.items():
        pd.DataFrame({"observation_date": months, sid: lvl}).to_csv(tmp_path / "rates" / f"{sid}.csv", index=False)
    pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))}, index=idx).to_parquet(tmp_path / "P.parquet")
    return {"universe": [{"symbol": "P", "artifact_path": str(tmp_path / "P.parquet"), "base": "HI", "quote": "LO"}],
            "rates": {"dir": str(tmp_path / "rates"), "series": {"HI": "HI", "LO": "LO"}},
            "sleeves": [{"type": "carry", "weight": 1.0}], "risk": {"target_vol": 0.10, "rebalance": "M"}}


def test_parse_multiplier_snaps_and_defaults():
    assert ao.parse_multiplier('{"multiplier": 0.4, "reason": "x"}') == 0.5
    assert ao.parse_multiplier("garbage") == 1.0
    assert ao.parse_multiplier('blah {"multiplier": 0} blah') == 0.0


def test_no_calls_before_min_date_and_snapshot_is_point_in_time():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return '{"multiplier": 0.0, "reason": "test"}'

    ao.clear_cache()
    idx = pd.bdate_range("2025-12-01", "2026-02-27")
    ev = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-15T13:30:00Z", "2026-03-01T13:30:00Z"], utc=True),
                       "currency": ["USD", "USD"], "title": ["NFP", "FUTURE"], "impact": ["high", "high"],
                       "actual": [1.0, 9.0], "forecast": [0.5, 9.5]})
    m = ao.agent_multipliers(idx, {"min_date": "2026-01-01", "period": "M"}, None, {}, ev, None, llm=fake)
    assert (m[:"2025-12-31"] == 1.0).all()                     # no model calls before min_date
    assert (m["2026-01-01":] == 0.0).all() and len(calls) == 2  # Jan + Feb periods
    assert all("FUTURE" not in c or "upcoming" in c for c in calls)
    assert "NFP: actual" not in calls[0]                       # Jan-15 release is not yet known on Dec-31


def test_overlay_in_portfolio_cuts_exposure(tmp_path):
    d = _mk(tmp_path)
    base = run_portfolio_backtest(d)["returns"]
    d["overlays"] = [{"type": "agent_risk", "min_date": "2015-01-01", "period": "M"}]
    d["_llm"] = lambda p: '{"multiplier": 0.0, "reason": "cut"}'
    cut = run_portfolio_backtest(d)["returns"]
    assert cut.abs().sum() == 0.0 and base.abs().sum() > 0
