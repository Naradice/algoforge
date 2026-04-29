from __future__ import annotations

from datetime import datetime, timezone

from strategy.engine.execution import OpenPosition, RiskParams, check_sl_tp, close_trade


def test_check_sl_tp_closes_at_take_profit_without_extra_slippage():
    opened_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = OpenPosition(
        direction="buy",
        entry_price=100.0,
        sl_price=95.0,
        tp_price=110.0,
        volume=1.0,
        opened_at=opened_at,
    )
    trades: list[dict] = []

    closed, remaining = check_sl_tp(
        pos,
        high=111.0,
        low=99.0,
        bar_dt=opened_at,
        trades=trades,
        symbol="TEST",
        rp=RiskParams(commission_pct=0.0),
        phase="is",
    )

    assert closed is True
    assert remaining is None
    assert trades[0]["exit_price"] == 110.0
    assert trades[0]["exit_reason"] == "tp"
    assert trades[0]["phase"] == "is"


def test_close_trade_applies_commission_to_profit():
    opened_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    closed_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    pos = OpenPosition(
        direction="sell",
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        volume=2.0,
        opened_at=opened_at,
    )
    trades: list[dict] = []

    profit = close_trade(
        pos,
        exit_price=90.0,
        closed_at=closed_at,
        reason="signal",
        trades=trades,
        symbol="TEST",
        rp=RiskParams(commission_pct=0.01),
    )

    assert profit == 0.18
    assert trades[0]["profit"] == 0.18
    assert trades[0]["exit_reason"] == "signal"
