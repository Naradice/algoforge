"""Unit tests for Swing Trading Signals & Screening Engine."""

from __future__ import annotations

import datetime
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database import Base
from data.models import (
    DailyMarginRecord,
    LargeHoldingReportRecord,
    LargeHoldingTradeRecord,
    MarginRegulationRecord,
)
from strategy.engine.swing_signals import screen_swing_signals


@pytest.fixture
def test_db_session():
    """In-memory SQLite session with tables and populated test records."""
    engine = sa.create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    as_of = datetime.date(2026, 11, 6)

    # 1. Squeeze candidate: セキュアヴェイル (3042)
    session.add(
        DailyMarginRecord(
            as_of_date=as_of,
            code="3042",
            name="セキュアヴェイル",
            market="スタンダード",
            sell_shares=375500.0,
            sell_change=15000.0,
            sell_ratio=4.9,
            buy_shares=462800.0,
            buy_ratio=6.0,
            ratio=81.1,
        )
    )

    # 2. Overhang warning: ベースフード (2936)
    session.add(
        DailyMarginRecord(
            as_of_date=as_of,
            code="2936",
            name="ベースフード",
            market="グロース",
            sell_shares=100000.0,
            buy_shares=18400000.0,
            buy_change=25000.0,
            buy_ratio=33.6,
            ratio=3.4,
        )
    )

    # 3. Supply relief: イオレ (2334)
    session.add(
        DailyMarginRecord(
            as_of_date=as_of,
            code="2334",
            name="イオレ",
            market="グロース",
            sell_shares=50000.0,
            buy_shares=11600000.0,
            buy_change=-366000.0,
            buy_ratio=27.1,
            ratio=1.5,
        )
    )

    # 4. Lifted regulation: ブロードバンドセキュリティ (4398)
    session.add(
        MarginRegulationRecord(
            code="4398",
            name="ブロードバンドセキュリティ",
            event_date=datetime.date(2026, 11, 6),
            event_type="lifted",
        )
    )

    # 5. Institutional buy: ビープラッツ (4381)
    rep = LargeHoldingReportRecord(
        doc_id="S100TEST_INST_01",
        issuer_code="4381",
        issuer_name="ビープラッツ株式会社",
        filer_name="グロースパートナーズ株式会社",
        base_date=datetime.date(2026, 10, 27),
        holding_ratio=32.97,
        prev_holding_ratio=31.95,
        ratio_change=1.02,
        purpose_of_holding="業務資本提携",
    )
    session.add(rep)
    session.flush()

    session.add(
        LargeHoldingTradeRecord(
            report_id=rep.id,
            doc_id=rep.doc_id,
            issuer_code=rep.issuer_code,
            trade_date=datetime.date(2026, 9, 2),
            trade_type="取得",
            market_type="市場内",
            quantity=15000,
        )
    )

    session.commit()
    try:
        yield session
    finally:
        session.close()


def test_screen_squeeze_alert(test_db_session):
    """Test short squeeze signal generation (3042)."""
    signals = screen_swing_signals(test_db_session, target_date=datetime.date(2026, 11, 6))
    squeeze_sigs = [s for s in signals if s.signal_type == "squeeze_alert"]

    assert len(squeeze_sigs) >= 1
    sig = squeeze_sigs[0]
    assert sig.code == "3042"
    assert sig.direction == "LONG"
    assert sig.confidence_score >= 0.70
    assert any("取組比率" in r for r in sig.reasons)


def test_screen_supply_overhang(test_db_session):
    """Test excessive margin buy overhang warning (2936)."""
    signals = screen_swing_signals(test_db_session, target_date=datetime.date(2026, 11, 6))
    overhang_sigs = [s for s in signals if s.signal_type == "supply_overhang"]

    assert len(overhang_sigs) >= 1
    sig = overhang_sigs[0]
    assert sig.code == "2936"
    assert sig.direction == "AVOID"
    assert "33.6%" in sig.reasons[0]


def test_screen_supply_relief(test_db_session):
    """Test margin buy sharp drop relief signal (2334)."""
    signals = screen_swing_signals(test_db_session, target_date=datetime.date(2026, 11, 6))
    relief_sigs = [s for s in signals if s.signal_type == "supply_relief"]

    assert len(relief_sigs) >= 1
    sig = relief_sigs[0]
    assert sig.code == "2334"
    assert sig.direction == "LONG"
    assert sig.metrics["buy_change"] == -366000.0


def test_screen_daily_margin_lifted(test_db_session):
    """Test daily margin regulation lifted signal (4398)."""
    signals = screen_swing_signals(test_db_session, target_date=datetime.date(2026, 11, 6))
    lifted_sigs = [s for s in signals if s.signal_type == "daily_margin_lifted"]

    assert len(lifted_sigs) >= 1
    sig = lifted_sigs[0]
    assert sig.code == "4398"
    assert sig.direction == "LONG"
    assert "日々公表解除日" in sig.reasons[0]


def test_screen_institutional_buy(test_db_session):
    """Test institutional accumulation signal from 5% rule (4381)."""
    signals = screen_swing_signals(test_db_session, target_date=datetime.date(2026, 11, 6))
    inst_sigs = [s for s in signals if s.signal_type == "institutional_buy"]

    assert len(inst_sigs) >= 1
    sig = inst_sigs[0]
    assert sig.code == "4381"
    assert sig.direction == "LONG"
    assert "+1.02%" in sig.reasons[1]
