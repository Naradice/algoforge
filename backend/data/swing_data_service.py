"""Swing Trading Data Service: Storage, Retrieval, and Automated Collection.

Manages database persistence and queries for:
1. EDINET Large Volume Holding Reports (大量保有報告書・変更報告書・60日間売買明細)
2. JPX Daily Margin Balances (日々公表銘柄等信用取引残高)
3. JPX Daily Margin Regulations (日々公表指定・解除イベント)
4. JPX Market-wide Weekly Margin History (信用取引現在高過去推移)
"""

from __future__ import annotations

import datetime
from datetime import date as dt_date
import io
import logging
from pathlib import Path
from typing import Any
import zipfile

import sqlalchemy as sa
from sqlalchemy.orm import Session

from database import SessionLocal
from data.collectors.jpx_margin import JPXMarginCollector
from data.extractors.edinet import (
    LargeHoldingReport,
    extract_large_holding_from_html,
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
from scripts.verify_edinet_html_parser import (
    download_document_zip,
    fetch_document_list,
    get_edinet_api_key,
)

logger = logging.getLogger(__name__)


def _to_date(val: Any) -> dt_date | None:
    """Helper to convert string YYYY-MM-DD or date object to datetime.date."""
    if val is None:
        return None
    if isinstance(val, dt_date):
        return val
    if isinstance(val, datetime.datetime):
        return val.date()
    try:
        parts = str(val).split("-")
        return dt_date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Database Persistence Functions
# ---------------------------------------------------------------------------

def save_large_holding_report(
    report: LargeHoldingReport,
    session: Session | None = None,
) -> LargeHoldingReportRecord:
    """Upsert a LargeHoldingReport and its 60-day trades into the database."""
    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    try:
        doc_id = report.doc_id or f"GEN_{report.issuer_code}_{report.base_date}_{report.filer_name}"

        # Check existing record
        existing = session.execute(
            sa.select(LargeHoldingReportRecord).where(LargeHoldingReportRecord.doc_id == doc_id)
        ).scalar_one_or_none()

        # Serialize joint holders
        holders_data = [
            {"name": jh.name, "shares_held": jh.shares_held, "holding_ratio": jh.holding_ratio}
            for jh in report.joint_holders
        ]

        base_dt = _to_date(report.base_date)

        if existing:
            rec = existing
            rec.issuer_code = report.issuer_code
            rec.issuer_name = report.issuer_name
            rec.filer_name = report.filer_name
            rec.base_date = base_dt
            rec.holding_ratio = report.holding_ratio
            rec.prev_holding_ratio = report.prev_holding_ratio
            rec.ratio_change = report.ratio_change
            rec.total_shares_held = report.total_shares_held
            rec.total_outstanding_shares = report.total_outstanding_shares
            rec.purpose_of_holding = report.purpose_of_holding
            rec.joint_holders_json = holders_data
            # Clear old trades for replacement
            rec.trades.clear()
        else:
            rec = LargeHoldingReportRecord(
                doc_id=doc_id,
                issuer_code=report.issuer_code,
                issuer_name=report.issuer_name,
                filer_name=report.filer_name,
                base_date=base_dt,
                holding_ratio=report.holding_ratio,
                prev_holding_ratio=report.prev_holding_ratio,
                ratio_change=report.ratio_change,
                total_shares_held=report.total_shares_held,
                total_outstanding_shares=report.total_outstanding_shares,
                purpose_of_holding=report.purpose_of_holding,
                joint_holders_json=holders_data,
            )
            session.add(rec)

        session.flush()

        # Add trades
        for t in report.recent_trades:
            tr_rec = LargeHoldingTradeRecord(
                report_id=rec.id,
                doc_id=doc_id,
                issuer_code=report.issuer_code,
                trade_date=_to_date(t.trade_date),
                security_type=t.security_type,
                quantity=t.quantity,
                ratio=t.ratio,
                market_type=t.market_type,
                trade_type=t.trade_type,
                unit_price=t.unit_price,
                description=t.description,
            )
            session.add(tr_rec)

        session.commit()
        session.refresh(rec)
        return rec
    finally:
        if should_close:
            session.close()


def save_daily_margin_records(
    issues: list[DailyMarginIssue],
    session: Session | None = None,
) -> int:
    """Upsert daily margin records into database."""
    if not issues:
        return 0

    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    saved_count = 0
    try:
        for iss in issues:
            as_of_dt = _to_date(iss.as_of_date)
            if not as_of_dt or not iss.code:
                continue

            existing = session.execute(
                sa.select(DailyMarginRecord).where(
                    DailyMarginRecord.as_of_date == as_of_dt,
                    DailyMarginRecord.code == iss.code,
                )
            ).scalar_one_or_none()

            if existing:
                rec = existing
                rec.name = iss.name
                rec.market = iss.market
                rec.issue_type = iss.issue_type
                rec.restrictions = iss.restrictions
                rec.isin = iss.isin
                rec.sell_shares = iss.sell_shares
                rec.sell_change = iss.sell_change
                rec.sell_ratio = iss.sell_ratio
                rec.buy_shares = iss.buy_shares
                rec.buy_change = iss.buy_change
                rec.buy_ratio = iss.buy_ratio
                rec.ratio = iss.ratio
                rec.negotiable_sell_shares = iss.negotiable_sell_shares
                rec.negotiable_sell_change = iss.negotiable_sell_change
                rec.standardized_sell_shares = iss.standardized_sell_shares
                rec.standardized_sell_change = iss.standardized_sell_change
                rec.negotiable_buy_shares = iss.negotiable_buy_shares
                rec.negotiable_buy_change = iss.negotiable_buy_change
                rec.standardized_buy_shares = iss.standardized_buy_shares
                rec.standardized_buy_change = iss.standardized_buy_change
            else:
                rec = DailyMarginRecord(
                    as_of_date=as_of_dt,
                    code=iss.code,
                    name=iss.name,
                    market=iss.market,
                    issue_type=iss.issue_type,
                    restrictions=iss.restrictions,
                    isin=iss.isin,
                    sell_shares=iss.sell_shares,
                    sell_change=iss.sell_change,
                    sell_ratio=iss.sell_ratio,
                    buy_shares=iss.buy_shares,
                    buy_change=iss.buy_change,
                    buy_ratio=iss.buy_ratio,
                    ratio=iss.ratio,
                    negotiable_sell_shares=iss.negotiable_sell_shares,
                    negotiable_sell_change=iss.negotiable_sell_change,
                    standardized_sell_shares=iss.standardized_sell_shares,
                    standardized_sell_change=iss.standardized_sell_change,
                    negotiable_buy_shares=iss.negotiable_buy_shares,
                    negotiable_buy_change=iss.negotiable_buy_change,
                    standardized_buy_shares=iss.standardized_buy_shares,
                    standardized_buy_change=iss.standardized_buy_change,
                )
                session.add(rec)
            saved_count += 1

        session.commit()
        return saved_count
    finally:
        if should_close:
            session.close()


def save_margin_regulation_events(
    events: list[MarginRegulationEvent],
    session: Session | None = None,
) -> int:
    """Save daily margin designation / lifting events without duplicates."""
    if not events:
        return 0

    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    saved_count = 0
    try:
        for ev in events:
            ev_dt = _to_date(ev.event_date)
            if not ev_dt or not ev.code:
                continue

            existing = session.execute(
                sa.select(MarginRegulationRecord).where(
                    MarginRegulationRecord.code == ev.code,
                    MarginRegulationRecord.event_date == ev_dt,
                    MarginRegulationRecord.event_type == ev.event_type,
                )
            ).scalar_one_or_none()

            if not existing:
                rec = MarginRegulationRecord(
                    code=ev.code,
                    name=ev.name,
                    event_date=ev_dt,
                    event_type=ev.event_type,
                )
                session.add(rec)
                saved_count += 1

        session.commit()
        return saved_count
    finally:
        if should_close:
            session.close()


def save_market_margin_records(
    records: list[MarketMarginRecord],
    session: Session | None = None,
) -> int:
    """Upsert market-wide weekly margin records into database."""
    if not records:
        return 0

    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    saved_count = 0
    try:
        for r in records:
            dt = _to_date(r.date)
            if not dt:
                continue

            existing = session.execute(
                sa.select(MarketMarginWeeklyRecord).where(MarketMarginWeeklyRecord.date == dt)
            ).scalar_one_or_none()

            if existing:
                existing.standardized_sell_shares = r.standardized_sell_shares
                existing.standardized_buy_shares = r.standardized_buy_shares
                existing.negotiable_sell_shares = r.negotiable_sell_shares
                existing.negotiable_buy_shares = r.negotiable_buy_shares
                existing.total_sell_shares = r.total_sell_shares
                existing.total_buy_shares = r.total_buy_shares
                existing.margin_ratio = r.margin_ratio
            else:
                rec = MarketMarginWeeklyRecord(
                    date=dt,
                    standardized_sell_shares=r.standardized_sell_shares,
                    standardized_buy_shares=r.standardized_buy_shares,
                    negotiable_sell_shares=r.negotiable_sell_shares,
                    negotiable_buy_shares=r.negotiable_buy_shares,
                    total_sell_shares=r.total_sell_shares,
                    total_buy_shares=r.total_buy_shares,
                    margin_ratio=r.margin_ratio,
                )
                session.add(rec)
            saved_count += 1

        session.commit()
        return saved_count
    finally:
        if should_close:
            session.close()


# ---------------------------------------------------------------------------
# Automated Collection Jobs
# ---------------------------------------------------------------------------

def run_daily_margin_collection() -> dict[str, Any]:
    """Execute scheduled collection of JPX Daily Margin Balances and Announcements."""
    collector = JPXMarginCollector()
    result: dict[str, Any] = {
        "status": "success",
        "announcements_saved": 0,
        "balances_saved": 0,
        "errors": [],
    }

    # 1. Announcements
    try:
        ann = collector.fetch_daily_margin_announcements()
        all_events = ann.get("designated", []) + ann.get("lifted", [])
        saved_ev = save_margin_regulation_events(all_events)
        result["announcements_saved"] = saved_ev
        logger.info("Saved %d margin regulation events", saved_ev)
    except Exception as e:
        logger.error("Error in daily margin announcements collection: %s", e)
        result["errors"].append(f"announcements: {e}")

    # 2. Daily Balances Excel
    try:
        url, issues = collector.fetch_latest_daily_margin_balances()
        saved_iss = save_daily_margin_records(issues)
        result["balances_saved"] = saved_iss
        result["source_url"] = url
        logger.info("Saved %d daily margin balance records from %s", saved_iss, url)
    except Exception as e:
        logger.error("Error in daily margin balances collection: %s", e)
        result["errors"].append(f"balances: {e}")

    if result["errors"]:
        result["status"] = "partial" if (result["announcements_saved"] > 0 or result["balances_saved"] > 0) else "error"

    return result


def run_large_holding_collection(
    target_date: dt_date | None = None,
    max_documents: int = 30,
) -> dict[str, Any]:
    """Execute collection of Large Volume Holding Reports from EDINET API."""
    result: dict[str, Any] = {
        "status": "success",
        "reports_processed": 0,
        "reports_saved": 0,
        "errors": [],
    }

    try:
        api_key = get_edinet_api_key()
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"API Key error: {e}")
        return result

    t_date = target_date or dt_date.today()
    date_str = t_date.strftime("%Y-%m-%d")

    try:
        docs = fetch_document_list(date_str, api_key)
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Document list fetch error: {e}")
        return result

    # Filter docTypeCode in [350, 360, 370]
    target_docs = [d for d in docs if d.get("docTypeCode") in ("350", "360", "370")]
    logger.info("Found %d large holding reports on %s", len(target_docs), date_str)

    for doc in target_docs[:max_documents]:
        doc_id = doc.get("docID")
        if not doc_id:
            continue

        result["reports_processed"] += 1
        zip_bytes = download_document_zip(doc_id, api_key)
        if not zip_bytes:
            continue

        honbun_bytes = None
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                for name in z.namelist():
                    if "honbun" in name and name.endswith((".htm", ".html")):
                        honbun_bytes = z.read(name)
                        break
        except Exception as e:
            logger.warning("[%s] Failed to extract ZIP: %s", doc_id, e)
            continue

        if not honbun_bytes:
            continue

        try:
            report = extract_large_holding_from_html(honbun_bytes, doc_id=doc_id)
            save_large_holding_report(report)
            result["reports_saved"] += 1
        except Exception as e:
            logger.warning("[%s] Failed to parse/save report: %s", doc_id, e)
            result["errors"].append(f"[{doc_id}]: {e}")

    return result


def run_market_margin_history_collection() -> dict[str, Any]:
    """Execute collection of weekly market-wide margin history."""
    collector = JPXMarginCollector()
    result: dict[str, Any] = {"status": "success", "records_saved": 0, "errors": []}
    try:
        records = collector.fetch_market_margin_history()
        saved = save_market_margin_records(records)
        result["records_saved"] = saved
        logger.info("Saved %d market margin weekly history records", saved)
    except Exception as e:
        logger.error("Error in market margin history collection: %s", e)
        result["status"] = "error"
        result["errors"].append(str(e))
    return result
