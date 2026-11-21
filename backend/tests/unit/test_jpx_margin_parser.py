"""Unit tests for JPX Margin Trading Data Extractors (日々公表銘柄等信用残高 & 市場推移)."""

from __future__ import annotations

from pathlib import Path
import re
import pytest

from data.extractors.margin import (
    DailyMarginIssue,
    MarginRegulationEvent,
    MarketMarginRecord,
    parse_daily_margin_announcements,
    parse_daily_margin_excel,
    parse_market_margin_excel,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "margin"


@pytest.fixture
def daily_excel_path() -> Path:
    p = FIXTURES_DIR / "20261106_mtdaily.xlsx"
    assert p.exists(), f"Fixture missing: {p}"
    return p


@pytest.fixture
def daily_html_path() -> Path:
    p = FIXTURES_DIR / "margin_daily_index.html"
    assert p.exists(), f"Fixture missing: {p}"
    return p


@pytest.fixture
def market_history_path() -> Path:
    p = FIXTURES_DIR / "market_margin_history.xlsx"
    assert p.exists(), f"Fixture missing: {p}"
    return p


def test_parse_daily_margin_excel(daily_excel_path: Path):
    """Test parsing mtdaily.xlsx with full issues and calculation columns."""
    issues = parse_daily_margin_excel(daily_excel_path)

    assert len(issues) > 10, f"Expected many issues, got {len(issues)}"

    # Check date extraction
    first = issues[0]
    assert first.as_of_date == "2026-11-06"

    # Test specific known issue from sample: セキュアヴェイル (3042)
    secure_issues = [i for i in issues if i.code == "3042"]
    assert len(secure_issues) == 1
    sec = secure_issues[0]
    assert "セキュアヴェイル" in sec.name
    assert sec.market == "スタンダード"
    assert sec.issue_type == "貸"
    assert "規" in sec.restrictions
    assert "株" in sec.restrictions
    assert sec.isin == "JP3420960001"
    assert sec.sell_shares == 375500.0
    assert sec.sell_change == -28200.0
    assert sec.sell_ratio == 4.9
    assert sec.buy_shares == 462800.0
    assert sec.buy_change == -93300.0
    assert sec.buy_ratio == 6.0
    assert sec.ratio == 81.1

    # Test high margin buy ratio issue: KLab (3656)
    klab_issues = [i for i in issues if i.code == "3656"]
    assert len(klab_issues) == 1
    klab = klab_issues[0]
    assert "ＫＬａｂ" in klab.name or "KLab" in klab.name
    assert klab.market == "プライム"
    assert "日" in klab.restrictions
    assert klab.buy_shares == 14967000.0
    assert klab.buy_change == -1116200.0
    assert klab.buy_ratio == 17.9


def test_parse_daily_margin_announcements(daily_html_path: Path):
    """Test parsing daily margin designated and lifted issues from HTML."""
    result = parse_daily_margin_announcements(daily_html_path)

    assert "designated" in result
    assert "lifted" in result

    designated = result["designated"]
    lifted = result["lifted"]

    assert len(designated) >= 20
    assert len(lifted) >= 3

    # Check top designated issues
    d0 = designated[0]
    assert d0.code == "4498"
    assert "サイバートラスト" in d0.name
    assert d0.event_date == "2026-10-26"
    assert d0.event_type == "designated"

    # Check alphanumeric symbol (e.g. 153A カウリス)
    caulis = [d for d in designated if d.code == "153A"]
    assert len(caulis) == 1
    assert "カウリス" in caulis[0].name
    assert caulis[0].event_date == "2026-10-08"

    # Check lifted issue (e.g. 4398 ブロードバンドセキュリティ)
    bb = [l for l in lifted if d0.code == "4398" or l.code == "4398"]
    assert len(bb) >= 1
    assert "ブロードバンドセキュリティ" in bb[0].name
    assert bb[0].event_date == "2026-11-06"
    assert bb[0].event_type == "lifted"


def test_parse_market_margin_excel(market_history_path: Path):
    """Test parsing weekly market-wide margin history Excel."""
    records = parse_market_margin_excel(market_history_path)

    assert len(records) > 0

    # Check first record structure
    r0 = records[0]
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", r0.date)
    assert r0.total_sell_shares > 0
    assert r0.total_buy_shares > 0
    assert r0.margin_ratio > 0
    assert r0.standardized_sell_shares > 0
    assert r0.standardized_buy_shares > 0
