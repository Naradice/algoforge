"""Integration and route tests for swing trading data endpoints."""

from __future__ import annotations

import datetime
import pytest
from httpx import AsyncClient

from data.models import DailyMarginRecord, LargeHoldingReportRecord, MarginRegulationRecord


@pytest.mark.asyncio
async def test_get_daily_margin_records(client: AsyncClient, db_session):
    """Test GET /api/v1/margin/daily."""
    async with db_session() as session:
        session.add(
            DailyMarginRecord(
                as_of_date=datetime.date(2026, 11, 6),
                code="3042",
                name="セキュアヴェイル",
                market="スタンダード",
                issue_type="貸",
                restrictions=["規", "株"],
                sell_shares=375500.0,
                buy_shares=462800.0,
                buy_ratio=6.0,
                sell_ratio=4.9,
                ratio=81.1,
            )
        )
        session.add(
            DailyMarginRecord(
                as_of_date=datetime.date(2026, 11, 6),
                code="3656",
                name="KLab",
                market="プライム",
                issue_type="貸",
                restrictions=["日"],
                sell_shares=4077800.0,
                buy_shares=14967000.0,
                buy_ratio=17.9,
                sell_ratio=4.9,
                ratio=27.2,
            )
        )
        await session.commit()

    resp = await client.get("/api/v1/margin/daily?order_by=buy_ratio")
    assert resp.status_code == 200
    body = resp.json()
    data = body.get("data", [])
    assert len(data) >= 2
    # Check that KLab has higher buy_ratio than 3042
    codes = [d["code"] for d in data]
    assert "3656" in codes
    assert "3042" in codes
    klab = [d for d in data if d["code"] == "3656"][0]
    assert klab["buy_ratio"] == 17.9


@pytest.mark.asyncio
async def test_get_margin_regulations(client: AsyncClient, db_session):
    """Test GET /api/v1/margin/regulations."""
    async with db_session() as session:
        session.add(
            MarginRegulationRecord(
                code="4498",
                name="サイバートラスト",
                event_date=datetime.date(2026, 10, 26),
                event_type="designated",
            )
        )
        session.add(
            MarginRegulationRecord(
                code="4398",
                name="ブロードバンドセキュリティ",
                event_date=datetime.date(2026, 11, 6),
                event_type="lifted",
            )
        )
        await session.commit()

    resp = await client.get("/api/v1/margin/regulations?event_type=designated")
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    assert len(data) >= 1
    assert any(d["code"] == "4498" for d in data)


@pytest.mark.asyncio
async def test_get_large_holding_reports(client: AsyncClient, db_session):
    """Test GET /api/v1/large-holding/reports."""
    async with db_session() as session:
        session.add(
            LargeHoldingReportRecord(
                doc_id="S100TEST_ROUTE_01",
                issuer_code="4381",
                issuer_name="ビープラッツ株式会社",
                filer_name="グロースパートナーズ株式会社",
                base_date=datetime.date(2026, 10, 27),
                holding_ratio=32.97,
                prev_holding_ratio=31.95,
                ratio_change=1.02,
                total_shares_held=2264500.0,
                total_outstanding_shares=4844099.0,
                purpose_of_holding="業務資本提携",
                joint_holders_json=[{"name": "グロースパートナーズ", "shares_held": 1132250, "holding_ratio": 16.48}],
            )
        )
        await session.commit()

    resp = await client.get("/api/v1/large-holding/reports?issuer_code=4381")
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    assert len(data) >= 1
    target = [d for d in data if d["doc_id"] == "S100TEST_ROUTE_01"][0]
    assert target["ratio_change"] == 1.02
    assert "ビープラッツ" in target["issuer_name"]


@pytest.mark.asyncio
async def test_get_swing_signals(client: AsyncClient):
    """Test GET /api/v1/swing/signals with prepopulated records."""
    resp = await client.get("/api/v1/swing/signals?target_date=2026-11-06")
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    assert len(data) >= 1
    # 3042 should trigger squeeze_alert
    sig = [d for d in data if d["code"] == "3042"][0]
    assert sig["signal_type"] == "squeeze_alert"
    assert sig["direction"] == "LONG"
    assert sig["confidence_score"] >= 0.70
