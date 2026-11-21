"""Unit tests for EDINET Large Volume Holding Report (5% rule) Extractor."""

from __future__ import annotations

from pathlib import Path
import pytest

from data.extractors.edinet import (
    JointHolder,
    LargeHoldingReport,
    LargeHoldingTrade,
    extract_large_holding_from_html,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "edinet_html"


@pytest.fixture
def sample_0_html() -> str:
    path = FIXTURES_DIR / "large_holding_sample_0.html"
    assert path.exists(), f"Fixture missing: {path}"
    return path.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture
def sample_1_html() -> str:
    path = FIXTURES_DIR / "large_holding_sample_1.html"
    assert path.exists(), f"Fixture missing: {path}"
    return path.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture
def sample_2_html() -> str:
    path = FIXTURES_DIR / "large_holding_sample_2.html"
    assert path.exists(), f"Fixture missing: {path}"
    return path.read_text(encoding="utf-8", errors="ignore")


def test_extract_large_holding_sample_0(sample_0_html: str):
    """Test standard initial 5% holding report (Doc 350) with joint holders (UBS)."""
    report = extract_large_holding_from_html(sample_0_html, doc_id="S100ZC0B")

    assert report.doc_id == "S100ZC0B"
    assert report.issuer_code == "3963"
    assert "シンクロ" in report.issuer_name
    assert report.base_date == "2026-10-30"
    assert report.holding_ratio == 5.51
    assert report.prev_holding_ratio is None
    assert report.ratio_change is None
    assert report.total_shares_held == 1605411.0
    assert report.total_outstanding_shares == 29123000.0
    assert "ディーリング目的" in report.purpose_of_holding

    # Joint holders check
    assert len(report.joint_holders) == 2
    names = [h.name for h in report.joint_holders]
    assert any("ユービーエス" in n for n in names)
    assert any("UBS" in n.upper() for n in names)
    assert sum(h.shares_held for h in report.joint_holders) == 1605411.0


def test_extract_large_holding_sample_1(sample_1_html: str):
    """Test institutional investment fund initial holding report (Doc 350 - MFS)."""
    report = extract_large_holding_from_html(sample_1_html, doc_id="S100ZBJH")

    assert report.doc_id == "S100ZBJH"
    assert report.issuer_code == "4980"
    assert "デクセリアルズ" in report.issuer_name
    assert report.base_date == "2026-10-30"
    assert report.holding_ratio == 5.75
    assert report.total_shares_held == 10039400.0
    assert "純投資" in report.purpose_of_holding

    assert len(report.joint_holders) == 2
    mfs_us = [h for h in report.joint_holders if "Massachusetts" in h.name][0]
    assert mfs_us.holding_ratio == 5.53
    assert mfs_us.shares_held == 9664300.0


def test_extract_change_report_with_trades_sample_2(sample_2_html: str):
    """Test change report (Doc 360) with 60-day trading history and ratio increase."""
    report = extract_large_holding_from_html(sample_2_html, doc_id="S100ZC6M")

    assert report.doc_id == "S100ZC6M"
    assert report.issuer_code == "4381"
    assert "ビープラッツ" in report.issuer_name
    assert report.base_date == "2026-10-27"
    assert report.holding_ratio == 32.97
    assert report.prev_holding_ratio == 31.95
    assert report.ratio_change == 1.02
    assert report.total_shares_held == 2264500.0
    assert "業務資本提携" in report.purpose_of_holding

    # Joint holders check
    assert len(report.joint_holders) == 2
    assert report.joint_holders[0].holding_ratio == 16.48
    assert report.joint_holders[1].holding_ratio == 16.48

    # 60-day trades table check
    assert len(report.recent_trades) == 104
    # First trade
    t0 = report.recent_trades[0]
    assert t0.trade_date == "2026-08-28"
    assert "普通株式" in t0.security_type
    assert t0.quantity == 20950.0
    assert t0.trade_type == "処分"
    assert t0.market_type == "市場内"

    # Warrant execution trade with price
    warrant_trades = [t for t in report.recent_trades if t.unit_price is not None]
    assert len(warrant_trades) > 0
    assert warrant_trades[0].unit_price == 262.0
    assert warrant_trades[0].trade_type == "取得"
    assert warrant_trades[0].market_type == "市場外"


def test_fallback_html_without_ixbrl():
    """Test extractor resilience with pure HTML tables and no iXBRL namespaces."""
    raw_html = """
    <html>
      <body>
        <table>
          <tr><th>提出者及び共同保有者名</th><th>保有株券等の数（総数）</th><th>株券等保有割合（％）</th></tr>
          <tr><td>株式会社アクティビスト</td><td>500,000</td><td>5.10</td></tr>
          <tr><td>合計</td><td>500,000</td><td>5.10</td></tr>
        </table>
        <table>
          <tr><th>年月日</th><th>株券等の種類</th><th>数量</th><th>市場内外取引の別</th><th>取得又は処分の別</th></tr>
          <tr><td>2026年11月1日</td><td>株券（普通株式）</td><td>10,000</td><td>市場内</td><td>取得</td></tr>
        </table>
      </body>
    </html>
    """
    report = extract_large_holding_from_html(raw_html)

    assert report.holding_ratio == 5.10
    assert report.total_shares_held == 500000.0
    assert len(report.joint_holders) == 1
    assert "アクティビスト" in report.joint_holders[0].name
    assert len(report.recent_trades) == 1
    assert report.recent_trades[0].trade_date == "2026-11-01"
    assert report.recent_trades[0].trade_type == "取得"
    assert report.recent_trades[0].market_type == "市場内"
    assert report.recent_trades[0].quantity == 10000.0
