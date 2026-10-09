import pandas as pd
import pytest

from strategy.engine.indicators import apply_indicators


def _rates(tmp_path):
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    for sid, v in {"IR3TIB01USM156N": [2.0, 2.0, 2.0], "IR3TIB01JPM156N": [0.0, 0.0, 0.0]}.items():
        pd.DataFrame({"observation_date": idx, sid: v}).to_csv(tmp_path / f"{sid}.csv", index=False)
    pd.DataFrame({"observation_date": idx, "IRSTCB01JPM156N": [0.0] * 3}).to_csv(tmp_path / "IRSTCB01JPM156N.csv", index=False)


def test_carry_diff_and_weight(tmp_path):
    _rates(tmp_path)
    df = pd.DataFrame({"close": 1.0}, index=pd.date_range("2020-02-15", periods=3, freq="D"))
    out = apply_indicators(df, [{"id": "c", "type": "carry", "params": {"base": "USD", "quote": "JPY", "rates_dir": str(tmp_path)}}])
    assert out["c"].iloc[0] == pytest.approx(0.02)          # 1-month lag: Feb uses Jan values
    assert out["c_w"].iloc[0] == pytest.approx(0.02 / 0.03)


def test_carry_below_threshold_is_flat(tmp_path):
    _rates(tmp_path)
    df = pd.DataFrame({"close": 1.0}, index=pd.date_range("2020-02-15", periods=2, freq="D"))
    out = apply_indicators(df, [{"id": "c", "type": "carry", "params": {"base": "USD", "quote": "JPY", "rates_dir": str(tmp_path), "threshold": 0.05}}])
    assert (out["c_w"] == 0).all()


def test_swap_accrual_in_backtest(tmp_path, monkeypatch):
    from strategy.engine.backtest import _apply_swap
    idx = pd.date_range("2020-01-01", periods=11, freq="D")
    rate = pd.Series(0.0365, idx)                          # 3.65% p.a.
    trades = [{"direction": "buy", "volume": 2.0, "opened_at": idx[0], "closed_at": idx[10], "profit": 0.0}]
    eq = [{"timestamp": t.isoformat(), "equity": 1.0} for t in idx]
    _apply_swap(trades, eq, rate)
    assert trades[0]["profit"] == pytest.approx(2.0 * 0.0365 * 10 / 365)
    assert eq[-1]["equity"] == pytest.approx(1.0 + 2.0 * 0.0365 * 10 / 365) and eq[0]["equity"] == 1.0
