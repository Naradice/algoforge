"""F1 (#889) R-09b (Stage C finding 2): the DDM simulation time axis and start state.

Fixed facts about `f1_ddm.simulate_long_path` / `cum_returns`, checked with a stand-in simulator whose price at trade i is a
known function of i, so every index is exact:
  * a candle is 20 consecutive trades and its close is the price of the last trade of the group;
  * the path has 24*60 + 1 closes; closes[0] is the close of the first candle (the forecast origin), no burn-in is dropped;
  * the horizon-h return spans exactly 60h candles = 1200h trades: log(closes[60h] / closes[0]);
  * the model is built with the frozen configuration (N=300, section 11 decayed shock, no spread feedback), the same as the
    earlier ensemble generator `ddm_horizon_analysis.simulate_long_path`.
The decayed-shock schedule is checked with the real model's own probability function."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
BACKEND = RESEARCH.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(BACKEND))

import f1_common as fc  # noqa: E402
import f1_ddm as fd  # noqa: E402


class FakeDDM:
    last_kwargs = None

    def __init__(self, **kw):
        FakeDDM.last_kwargs = kw

    def simulate(self, n_trades):
        import pandas as pd
        FakeDDM.n_trades = n_trades
        i = np.arange(n_trades, dtype=float)
        return pd.DataFrame({"price": 100.0 + 0.001 * i + 0.0005 * (i % 20)})   # known function of the trade index


@pytest.fixture()
def fake(monkeypatch):
    import data.collectors.ddm_simulator as sim
    monkeypatch.setattr(sim, "DDMv3", FakeDDM)
    return FakeDDM


# R-09b
def test_candles_are_twenty_trades_and_closes_are_the_last_trade_of_each_group(fake):
    closes = fd.simulate_long_path(0.0, 15, seed=1)
    assert fd.TRADES_PER_CANDLE == 20 and len(closes) == fc.MAX_H * fc.BLOCK + 1 == 1441
    assert fake.n_trades == 1441 * 20
    i = np.arange(1441 * 20, dtype=float)
    price = 100.0 + 0.001 * i + 0.0005 * (i % 20)
    np.testing.assert_array_equal(closes, price[19::20][:1441])           # trade 19, 39, ... : last trade of each group
    assert closes[0] == price[19]                                          # first candle's close is the origin; nothing dropped


# R-09b
def test_horizon_returns_span_exactly_sixty_h_candles_from_the_origin(fake):
    closes = fd.simulate_long_path(0.0, 15, seed=1)
    got = fd.cum_returns(closes)
    for hv, g in zip(fc.HORIZONS, got):
        assert g == pytest.approx(np.log(closes[60 * hv] / closes[0]))
        # trades between the two closes: (60 h) candles * 20 trades; the first window is not shortened
        t0, t1 = 19, 19 + 60 * hv * 20
        i = np.array([t0, t1], dtype=float)
        p = 100.0 + 0.001 * i + 0.0005 * (i % 20)
        assert g == pytest.approx(np.log(p[1] / p[0]))
    assert fc.HORIZONS == [1, 4, 12, 24] or list(fc.HORIZONS) == [1, 4, 12, 24]


# R-09b
def test_model_is_built_with_the_frozen_configuration(fake):
    fd.simulate_long_path(0.1, 7, seed=5)
    kw = fake.last_kwargs
    assert kw["num_agent"] == 300 and kw["wma"] == 7
    assert kw["dealer_sensitive_min"] == pytest.approx(-3.4) and kw["dealer_sensitive_max"] == pytest.approx(-1.4)
    assert kw["spread_feedback_a"] is None
    assert (kw["exogenous_shock_probability"], kw["exogenous_shock_size"], kw["exogenous_shock_decay_tau"],
            kw["exogenous_shock_floor_probability"]) == (0.002, 0.3, 300.0, 0.0)


# R-09b
def test_configuration_equals_the_earlier_ensemble_generator():
    """Same model construction as ddm_horizon_analysis.simulate_long_path run with the section-11 settings (the earlier
    ensembles): the only intended difference is the one extra candle that makes the horizon exactly 60h candles."""
    import inspect
    import ddm_horizon_analysis as dha
    src = inspect.getsource(dha.simulate_long_path)
    for token in ("num_agent=num_agent", "max_volatility=0.02", "min_volatility=0.01", "wma=wma",
                  "dealer_sensitive_min=-3.5 + delta", "dealer_sensitive_max=-1.5 + delta", "spread_feedback_window=SPREAD_FEEDBACK_WINDOW",
                  "exogenous_shock_decay_tau=SHOCK_DECAY_TAU"):
        assert token in src
    assert dha.SPREAD_FEEDBACK_WINDOW == 150 and dha.TRADES_PER_CANDLE == fd.TRADES_PER_CANDLE
    assert dha.CANDLES_PER_WINDOW == fc.BLOCK


# R-09b
def test_decayed_shock_schedule_is_concentrated_at_the_path_start():
    """The shock probability decays with the model's own step count from the path start (tau = 300 steps); the expected number
    of shocks is about P0*tau = 0.6 per path and almost all of them occur in the first window. This is a property of the frozen
    section-11 design, not of the forecast origin: the path start is the origin by construction."""
    from data.collectors.ddm_simulator import DDMv3
    m = DDMv3(num_agent=50, max_volatility=0.02, min_volatility=0.01, wma=15, dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5,
              spread_feedback_a=None, spread_feedback_window=150, **fd.SHOCK)
    probs = []
    for step in range(0, 20000):
        m._step_count = step
        probs.append(m._current_shock_probability())
    probs = np.array(probs)
    assert probs[0] == pytest.approx(0.002) and probs[300] == pytest.approx(0.002 * np.exp(-1))
    total = probs.sum()
    assert total == pytest.approx(0.6, rel=0.01)                          # P0 * tau
    steps_first_window = 60 * 20 * 1.45                                    # 60 candles x 20 trades x ~1.45 steps per trade
    assert probs[:int(steps_first_window)].sum() / total > 0.99
