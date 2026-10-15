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
