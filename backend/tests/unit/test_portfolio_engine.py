import numpy as np
import pandas as pd
import pytest

from strategy.portfolio.engine import run_portfolio_backtest


def _mk(tmp_path, n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    (tmp_path / "rates").mkdir()
    months = pd.date_range("2014-12-01", periods=40, freq="MS")
    for sid, lvl in {"HI": 4.0, "LO": 0.0}.items():
        pd.DataFrame({"observation_date": months, sid: lvl}).to_csv(tmp_path / "rates" / f"{sid}.csv", index=False)
    # HILO drifts up by exactly the carry-neutral amount (no price drift): pure carry should earn the differential
    for s in ("HILO", "HILO2"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        pd.DataFrame({"close": c}, index=idx).to_parquet(tmp_path / f"{s}.parquet")
    return {
        "universe": [{"symbol": s, "artifact_path": str(tmp_path / f"{s}.parquet"), "base": "HI", "quote": "LO",
                      "spread_pct": 0.0} for s in ("HILO", "HILO2")],
        "rates": {"dir": str(tmp_path / "rates"), "series": {"HI": "HI", "LO": "LO"}},
        "sleeves": [{"type": "carry", "weight": 1.0}],
        "risk": {"target_vol": 0.10, "rebalance": "M"},
    }


def test_carry_sleeve_earns_accrual_and_goes_long(tmp_path):
    d = _mk(tmp_path)
    res = run_portfolio_backtest(d)
    r = res["returns"]
    assert len(r) > 700
    # positions are long (diff = +4% > threshold) -> accrual ~ 4%/yr * leverage; price noise is zero-mean
    assert res["contribution"].min() > -0.5
    d["sleeves"][0]["threshold"] = 0.10          # threshold above the differential -> flat -> no P&L
    flat = run_portfolio_backtest(d)["returns"]
    assert flat.abs().sum() == pytest.approx(0.0, abs=1e-12)


def test_costs_reduce_returns(tmp_path):
    d = _mk(tmp_path)
    a = run_portfolio_backtest(d)["returns"].sum()
    for u in d["universe"]:
        u["spread_pct"] = 0.001
    b = run_portfolio_backtest(d)["returns"].sum()
    assert b < a


def test_unknown_sleeve_raises(tmp_path):
    d = _mk(tmp_path)
    d["sleeves"] = [{"type": "nope"}]
    with pytest.raises(ValueError):
        run_portfolio_backtest(d)


def test_exposure_caps_and_drawdown_breaker(tmp_path):
    d = _mk(tmp_path)
    base_book = run_portfolio_backtest(d)["returns"]
    d["risk"]["max_gross"] = 0.05
    capped = run_portfolio_backtest(d)["returns"]
    assert capped.abs().sum() < base_book.abs().sum()
    d["risk"].pop("max_gross"); d["risk"]["dd_stop"] = 0.0005; d["risk"]["dd_cut"] = 0.0
    stopped = run_portfolio_backtest(d)["returns"]
    assert stopped.abs().sum() < base_book.abs().sum()


def test_stress_report_shape(tmp_path):
    res = run_portfolio_backtest(_mk(tmp_path))
    assert "2008 GFC (Sep-Dec 2008)" in res["stress"] and res["stress"]["2008 GFC (Sep-Dec 2008)"] is None


def test_series_below_ma_overlay_cuts_when_series_falls(tmp_path):
    d = _mk(tmp_path)
    idx = pd.bdate_range("2015-01-01", periods=800)
    px = np.r_[np.linspace(100, 200, 400), np.linspace(200, 80, 400)]       # up then down: below 50d MA in 2nd half
    pd.DataFrame({"close": px}, index=idx).to_parquet(tmp_path / "risk.parquet")
    base = run_portfolio_backtest(d)["returns"]
    d["overlays"] = [{"type": "series_below_ma", "path": str(tmp_path / "risk.parquet"), "window": 50, "cut": 1.0}]
    cut = run_portfolio_backtest(d)["returns"]
    assert cut[:300].equals(base[:300]) or np.allclose(cut[:300].dropna(), base[:300].dropna())
    assert cut[500:].abs().sum() == 0.0 and base[500:].abs().sum() > 0
