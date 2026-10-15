import math
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from data.collectors import mt5_swap
from strategy.engine.backtest import _apply_swap


class FakeMT5:
    def initialize(self, **kw): return True
    def shutdown(self): pass
    def last_error(self): return (0, "ok")
    def symbol_info(self, s):
        return {"USDJPY": NS(swap_long=-7.5, swap_short=3.0, swap_mode=1, point=0.001, digits=3, spread=3),
                "EURUSD": NS(swap_long=-2.0, swap_short=0.5, swap_mode=3, point=0.00001, digits=5, spread=5),
                "XXX": None}.get(s)
    def symbol_info_tick(self, s): return NS(bid=150.0 if s == "USDJPY" else 1.1)


def test_swap_snapshot_and_annualise(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_swap, "ARTIFACT_STORE", tmp_path)
    r = mt5_swap.collect(7, {"symbols": ["USDJPY", "EURUSD", "XXX"]}, mt5_module=FakeMT5())
    df = pd.read_parquet(tmp_path / r.artifact_path)
    assert set(df.symbol) == {"USDJPY", "EURUSD"}
    u = df[df.symbol == "USDJPY"].iloc[0]
    assert u.swap_long_ann == pytest.approx(-7.5 * 0.001 / 150.0 * 365)
    assert df[df.symbol == "EURUSD"].iloc[0].swap_long_ann == pytest.approx(-0.02)
    mt5_swap.collect(7, {"symbols": ["USDJPY"]}, mt5_module=FakeMT5())      # appends a second snapshot
    assert len(pd.read_parquet(tmp_path / r.artifact_path)) == 3


def test_asymmetric_short_accrual():
    idx = pd.date_range("2020-01-01", periods=11, freq="D")
    trades = [{"direction": "sell", "volume": 1.0, "opened_at": idx[0], "closed_at": idx[10], "profit": 0.0}]
    eq = [{"timestamp": t.isoformat(), "equity": 1.0} for t in idx]
    _apply_swap(trades, eq, pd.Series(0.03, idx), pd.Series(-0.01, idx))
    assert trades[0]["profit"] == pytest.approx(-0.01 * 10 / 365, abs=1e-6)
