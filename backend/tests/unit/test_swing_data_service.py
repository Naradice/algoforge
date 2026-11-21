"""Unit tests for Swing Trading Data Service (Database models, persistence & queries)."""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database import Base
from data.extractors.edinet import (
    JointHolder,
    LargeHoldingReport,
    LargeHoldingTrade,
)
from data.extractors.margin import (
    DailyMarginIssue,
    MarginRegulationEvent,
    MarketMarginRecord,
)
from data.models import (
    DailyMarginRecord,
    LargeHoldingReportRecord,
    LargeHoldingTradeRecord,
    MarginRegulationRecord,
    MarketMarginWeeklyRecord,
)
from data.swing_data_service import (
    save_daily_margin_records,
    save_large_holding_report,
    save_margin_regulation_events,
    save_market_margin_records,
)


@pytest.fixture
def test_db_session():
    """In-memory SQLite session with all tables created."""
    engine = sa.create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_save_large_holding_report(test_db_session):
    """Verify storing and updating a LargeHoldingReport and child trades."""
    report = LargeHoldingReport(
        doc_id="S100TEST01",
        issuer_code="4381",
        issuer_name="ビープラッツ株式会社",
        filer_name="グロースパートナーズ株式会社",
        base_date="2026-10-27",
        holding_ratio=32.97,
        prev_holding_ratio=31.95,
        ratio_change=1.02,
        total_shares_held=2264500.0,
        total_outstanding_shares=4844099.0,
        purpose_of_holding="業務資本提携",
        joint_holders=[
            JointHolder(name="グロースパートナーズ", shares_held=1132250, holding_ratio=16.48),
            JointHolder(name="古川 徳厚", shares_held=1132250, holding_ratio=16.48),
        ],
        recent_trades=[
            LargeHoldingTrade(
                trade_date="2026-08-28",
                security_type="普通株式",
                quantity=20950,
                trade_type="処分",
                market_type="市場内",
            ),
            LargeHoldingTrade(
                trade_date="2026-09-02",
                security_type="普通株式",
                quantity=12200,
                trade_type="取得",
                market_type="市場外",
                unit_price=262.0,
            ),
        ],
    )

    rec = save_large_holding_report(report, session=test_db_session)
    assert rec.id is not None
    assert rec.doc_id == "S100TEST01"
    assert rec.issuer_code == "4381"
    assert rec.ratio_change == 1.02
    assert len(rec.trades) == 2
    assert rec.trades[0].quantity == 20950.0
    assert rec.trades[1].unit_price == 262.0

    # Upsert test (modify ratio)
    report.holding_ratio = 35.00
    report.ratio_change = 3.05
    rec_updated = save_large_holding_report(report, session=test_db_session)
    assert rec_updated.id == rec.id
    assert rec_updated.holding_ratio == 35.00
    assert rec_updated.ratio_change == 3.05


def test_save_daily_margin_records(test_db_session):
    """Verify upsert of daily margin balance records."""
    issues = [
        DailyMarginIssue(
            as_of_date="2026-11-06",
            code="3042",
            name="セキュアヴェイル",
            market="スタンダード",
            issue_type="貸",
            restrictions=["規", "株"],
            sell_shares=375500.0,
            sell_change=-28200.0,
            sell_ratio=4.9,
            buy_shares=462800.0,
            buy_change=-93300.0,
            buy_ratio=6.0,
            ratio=81.1,
        ),
        DailyMarginIssue(
            as_of_date="2026-11-06",
            code="3656",
            name="KLab",
            market="プライム",
            issue_type="貸",
            restrictions=["日"],
            sell_shares=4077800.0,
            buy_shares=14967000.0,
            buy_ratio=17.9,
            ratio=27.2,
        ),
    ]

    saved = save_daily_margin_records(issues, session=test_db_session)
    assert saved == 2

    # Query
    q = test_db_session.execute(
        sa.select(DailyMarginRecord).order_by(DailyMarginRecord.code)
    ).scalars().all()
    assert len(q) == 2
    assert q[0].code == "3042"
    assert q[0].buy_ratio == 6.0
    assert "規" in q[0].restrictions
    assert q[1].code == "3656"
    assert q[1].buy_ratio == 17.9


def test_save_margin_regulation_events(test_db_session):
    """Verify saving designation and lifting events with deduplication."""
    events = [
        MarginRegulationEvent(code="4498", name="サイバートラスト", event_date="2026-10-26", event_type="designated"),
        MarginRegulationEvent(code="4398", name="ブロードバンドセキュリティ", event_date="2026-11-06", event_type="lifted"),
        # Duplicate event
        MarginRegulationEvent(code="4498", name="サイバートラスト", event_date="2026-10-26", event_type="designated"),
    ]

    saved = save_margin_regulation_events(events, session=test_db_session)
    assert saved == 2  # 3rd is duplicate

    q = test_db_session.execute(sa.select(MarginRegulationRecord)).scalars().all()
    assert len(q) == 2


def test_save_market_margin_records(test_db_session):
    """Verify weekly market margin history upsert."""
    records = [
        MarketMarginRecord(
            date="2026-10-30",
            total_sell_shares=349228.0,
            total_buy_shares=3654853.0,
            margin_ratio=10.47,
        ),
        MarketMarginRecord(
            date="2026-10-23",
            total_sell_shares=375063.0,
            total_buy_shares=3660529.0,
            margin_ratio=9.76,
        ),
    ]

    saved = save_market_margin_records(records, session=test_db_session)
    assert saved == 2

    q = test_db_session.execute(sa.select(MarketMarginWeeklyRecord).order_by(MarketMarginWeeklyRecord.date)).scalars().all()
    assert len(q) == 2
    assert q[1].date == datetime.date(2026, 10, 30)
    assert q[1].margin_ratio == 10.47
