"""EDINET Large Volume Holding Report (5% Rule) Extractor.

Extracts structured information from Japanese Large Volume Holding Reports
(大量保有報告書 docTypeCode 350, 変更報告書 docTypeCode 360, 訂正報告書 docTypeCode 370).

Captures:
- Issuer info (Code, Name)
- Filer & Joint Holders (Holders, Shares, Ratios)
- Aggregate holding ratio & previous holding ratio (Change delta)
- Purpose of holding (純投資, 政策投資, 重要提案行為等)
- Recent 60 days acquisition and disposal transactions (Trading flow)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
import re
from typing import Any

import lxml.html
import numpy as np

import unicodedata

from data.extractors.edinet.financial_report import decode_html_bytes
from data.extractors.edinet.normalizers import (
    clean_annotations,
    convert_with_unit,
    replace_zenkaku,
)

logger = logging.getLogger(__name__)


def _to_float(val: Any) -> float | None:
    """Helper to convert raw text to float or None."""
    if val is None:
        return None
    res = convert_with_unit(val)
    return None if np.isnan(res) else float(res)


@dataclass
class LargeHoldingTrade:
    """Represents a single trade within the recent 60-day transaction history."""
    trade_date: str | None = None
    security_type: str | None = None
    quantity: float = 0.0
    ratio: float | None = None
    market_type: str | None = None  # 市場内 / 市場外
    trade_type: str | None = None   # 取得 / 処分
    unit_price: float | None = None
    description: str | None = None


@dataclass
class JointHolder:
    """Represents a filer or joint holder."""
    name: str
    shares_held: float = 0.0
    holding_ratio: float = 0.0


@dataclass
class LargeHoldingReport:
    """Comprehensive representation of a Large Volume Holding Report."""
    issuer_code: str | None = None
    issuer_name: str | None = None
    filer_name: str | None = None
    base_date: str | None = None  # 義務発生日 / 基準日 (YYYY-MM-DD)
    holding_ratio: float | None = None  # 今回保有割合 (%)
    prev_holding_ratio: float | None = None  # 前回保有割合 (%)
    ratio_change: float | None = None  # 変動幅 (%) = holding_ratio - prev_holding_ratio
    total_shares_held: float | None = None  # 保有株数
    total_outstanding_shares: float | None = None  # 発行済株式等総数
    purpose_of_holding: str | None = None
    joint_holders: list[JointHolder] = field(default_factory=list)
    recent_trades: list[LargeHoldingTrade] = field(default_factory=list)
    significant_contracts: str | None = None
    doc_id: str | None = None
    raw_tables_count: int = 0


def _parse_japanese_date(date_str: str) -> str | None:
    """Convert Japanese era date (e.g., 2026年10月30日 or 令和8年10月30日) to YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = replace_zenkaku(date_str)
    cleaned = re.sub(r"\s+", "", date_str)

    # Western year
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", cleaned)
    if m:
        try:
            return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        except Exception:
            pass

    # Reiwa
    m = re.search(r"令和(元|\d{1,2})年(\d{1,2})月(\d{1,2})日", cleaned)
    if m:
        year_val = 1 if m.group(1) == "元" else int(m.group(1))
        western_year = 2018 + year_val
        return f"{western_year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Heisei
    m = re.search(r"平成(元|\d{1,2})年(\d{1,2})月(\d{1,2})日", cleaned)
    if m:
        year_val = 1 if m.group(1) == "元" else int(m.group(1))
        western_year = 1988 + year_val
        return f"{western_year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Already ISO
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if m:
        return m.group(0)

    return None


def _clean_str(val: str | None) -> str | None:
    if val is None:
        return None
    normalized = unicodedata.normalize("NFKC", val.strip())
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    return cleaned if cleaned else None


def extract_large_holding_from_html(
    content: str | bytes,
    doc_id: str | None = None,
) -> LargeHoldingReport:
    """Parse EDINET Large Volume Holding Report HTML / iXBRL into structured LargeHoldingReport.

    Parameters
    ----------
    content : str | bytes
        HTML document string or raw bytes (e.g. from zip *_honbun_*.htm).
    doc_id : str | None
        Optional document ID (e.g. S100XXXX).

    Returns
    -------
    LargeHoldingReport
        Structured report data.
    """
    if isinstance(content, bytes):
        text = decode_html_bytes(content)
    else:
        text = content

    # Remove XML declaration if present to avoid lxml parser error
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.DOTALL)
    tree = lxml.html.fromstring(text)

    report = LargeHoldingReport(doc_id=doc_id)
    report.raw_tables_count = len(tree.xpath("//table"))

    # 1. Extract from iXBRL tags
    _extract_ixbrl_metadata(tree, report)

    # 2. Extract recent 60-day trades table
    _extract_recent_trades(tree, report)

    # 3. Fallback table inspection if essential fields are missing
    if report.holding_ratio is None or not report.joint_holders:
        _extract_from_summary_tables(tree, report)

    # Calculate ratio_change if both current and previous exist
    if report.holding_ratio is not None and report.prev_holding_ratio is not None:
        report.ratio_change = round(report.holding_ratio - report.prev_holding_ratio, 2)

    return report


def _extract_ixbrl_metadata(tree: lxml.html.HtmlElement, report: LargeHoldingReport) -> None:
    """Extract standard tags defined in the EDINET Large Volume Holding taxonomy (jplvh_cor)."""
    ix_nodes = tree.xpath("//*[starts-with(name(), 'ix:')]")
    if not ix_nodes:
        return

    # Categorize nodes by tag name and context
    # Structure: by_name[tag_name] -> list of (contextref, text)
    by_name: dict[str, list[tuple[str, str]]] = {}
    for node in ix_nodes:
        name = node.attrib.get("name")
        if not name:
            continue
        ctx = node.attrib.get("contextref", "")
        val = "".join(node.xpath(".//text()")).strip()
        by_name.setdefault(name, []).append((ctx, val))

    def get_first_val(tag_name: str) -> str | None:
        entries = by_name.get(tag_name, [])
        for _, v in entries:
            if v:
                return v
        return None

    # Issuer Info
    report.issuer_code = _clean_str(get_first_val("jplvh_cor:SecurityCodeOfIssuer"))
    report.issuer_name = _clean_str(get_first_val("jplvh_cor:NameOfIssuer"))

    # Base Date
    raw_base_date = get_first_val("jplvh_cor:BaseDate")
    if raw_base_date:
        report.base_date = _parse_japanese_date(raw_base_date)

    # Purpose of Holding
    report.purpose_of_holding = _clean_str(get_first_val("jplvh_cor:PurposeOfHolding"))

    # Significant contracts
    report.significant_contracts = _clean_str(
        get_first_val("jplvh_cor:SignificantContractsRelatedToSaidStocksEtcSuchAsCollateralAgreementsTextBlock")
    )

    # Holding Ratios & Shares (Overall total vs Member specific)
    # The overall total usually has contextref == 'FilingDateInstant' (without 'Member')
    def get_total_value(tag_name: str) -> float | None:
        entries = by_name.get(tag_name, [])
        # 1. Exact match contextref == 'FilingDateInstant'
        for ctx, v in entries:
            if ctx.lower() == "filingdateinstant" and v:
                return _to_float(v)
        # 2. Context without 'Member'
        for ctx, v in entries:
            if "member" not in ctx.lower() and v:
                return _to_float(v)
        # 3. Maximum value across entries if numbers
        vals = [_to_float(v) for _, v in entries if v]
        valid_vals = [x for x in vals if x is not None]
        return max(valid_vals) if valid_vals else None

    report.holding_ratio = get_total_value("jplvh_cor:HoldingRatioOfShareCertificatesEtc")
    report.prev_holding_ratio = get_total_value("jplvh_cor:HoldingRatioOfShareCertificatesEtcPerLastReport")
    report.total_shares_held = get_total_value("jplvh_cor:TotalNumberOfStocksEtcHeld")
    report.total_outstanding_shares = get_total_value("jplvh_cor:TotalNumberOfOutstandingStocksEtc")

    # Joint Holders & Filer Info
    # Group by contextref containing 'FilerLargeVolumeHolder' or similar
    member_contexts: set[str] = set()
    for name_tag in ["jplvh_cor:Name", "jplvh_cor:HoldingRatioOfShareCertificatesEtc"]:
        for ctx, _ in by_name.get(name_tag, []):
            if "member" in ctx.lower():
                member_contexts.add(ctx)

    joint_holders: list[JointHolder] = []
    for ctx in sorted(member_contexts):
        # Find name
        holder_name = None
        for c, v in by_name.get("jplvh_cor:Name", []):
            if c == ctx and v:
                holder_name = _clean_str(v)
                break
        if not holder_name:
            continue

        # Find shares
        shares = 0.0
        for c, v in by_name.get("jplvh_cor:TotalNumberOfStocksEtcHeld", []):
            if c == ctx and v:
                f_val = _to_float(v)
                if f_val is not None:
                    shares = f_val
                break

        # Find ratio
        ratio = 0.0
        for c, v in by_name.get("jplvh_cor:HoldingRatioOfShareCertificatesEtc", []):
            if c == ctx and v:
                f_val = _to_float(v)
                if f_val is not None:
                    ratio = f_val
                break

        joint_holders.append(JointHolder(name=holder_name, shares_held=shares, holding_ratio=ratio))

    if joint_holders:
        report.joint_holders = joint_holders
        # Primary filer is usually the first one
        report.filer_name = joint_holders[0].name
    else:
        report.filer_name = _clean_str(get_first_val("jplvh_cor:Name"))


def _extract_recent_trades(tree: lxml.html.HtmlElement, report: LargeHoldingReport) -> None:
    """Extract trading records from the 60-day transaction table."""
    tables = tree.xpath("//table")
    trades: list[LargeHoldingTrade] = []

    for table in tables:
        t_text = "".join(table.xpath(".//text()"))
        # Check if table is the 60-day trading table
        if not ("取得又は処分の別" in t_text or ("年月日" in t_text and "株券等の種類" in t_text)):
            continue

        rows = table.xpath(".//tr")
        if len(rows) < 2:
            continue

        # Header detection
        headers: list[str] = []
        header_row_idx = -1
        for idx, row in enumerate(rows[:3]):
            cells = [replace_zenkaku(c.text_content().strip().replace("\n", "")) for c in row.xpath(".//th | .//td")]
            if any("年月日" in c for c in cells) and any("数量" in c or "株券" in c for c in cells):
                headers = cells
                header_row_idx = idx
                break

        if header_row_idx == -1:
            continue

        # Column indices
        col_map: dict[str, int] = {}
        for idx, h in enumerate(headers):
            if "年月日" in h:
                col_map["date"] = idx
            elif "種類" in h:
                col_map["security_type"] = idx
            elif "数量" in h:
                col_map["quantity"] = idx
            elif "割合" in h:
                col_map["ratio"] = idx
            elif "市場内外" in h or "取引の別" in h:
                col_map["market_type"] = idx
            elif "取得又は処分の別" in h or "取得" in h and "処分" in h:
                col_map["trade_type"] = idx
            elif "単価" in h or "価格" in h:
                col_map["unit_price"] = idx

        for row in rows[header_row_idx + 1:]:
            cells = [replace_zenkaku(c.text_content().strip().replace("\n", " ")) for c in row.xpath(".//td | .//th")]
            if len(cells) < 3:
                continue

            # Extract fields
            raw_date = cells[col_map["date"]] if "date" in col_map and col_map["date"] < len(cells) else None
            parsed_date = _parse_japanese_date(raw_date) if raw_date else None
            if not parsed_date:
                continue

            sec_type = cells[col_map["security_type"]] if "security_type" in col_map and col_map["security_type"] < len(cells) else None
            raw_qty = cells[col_map["quantity"]] if "quantity" in col_map and col_map["quantity"] < len(cells) else "0"
            raw_ratio = cells[col_map["ratio"]] if "ratio" in col_map and col_map["ratio"] < len(cells) else None
            mkt_type = cells[col_map["market_type"]] if "market_type" in col_map and col_map["market_type"] < len(cells) else None
            tr_type = cells[col_map["trade_type"]] if "trade_type" in col_map and col_map["trade_type"] < len(cells) else None
            raw_price = cells[col_map["unit_price"]] if "unit_price" in col_map and col_map["unit_price"] < len(cells) else None

            # Clean quantity & ratio
            qty = _to_float(raw_qty) or 0.0
            ratio = _to_float(raw_ratio) if raw_ratio else None

            # Parse unit price & description
            price = None
            desc = None
            if raw_price:
                m_price = re.search(r"(\d[\d,]*\.?\d*)\s*円?", raw_price)
                if m_price:
                    price = _to_float(m_price.group(1))
                if not raw_price.replace(",", "").strip().replace("円", "").isdigit():
                    desc = raw_price.strip()

            # Trade type normalization
            normalized_trade_type = "取得" if tr_type and "取得" in tr_type else ("処分" if tr_type and "処分" in tr_type else tr_type)

            trades.append(
                LargeHoldingTrade(
                    trade_date=parsed_date,
                    security_type=_clean_str(sec_type),
                    quantity=qty,
                    ratio=ratio,
                    market_type=_clean_str(mkt_type),
                    trade_type=normalized_trade_type,
                    unit_price=price,
                    description=desc,
                )
            )

    report.recent_trades = trades


def _extract_from_summary_tables(tree: lxml.html.HtmlElement, report: LargeHoldingReport) -> None:
    """Fallback extraction from standard summary tables if iXBRL tags were incomplete."""
    tables = tree.xpath("//table")
    for table in tables:
        t_text = "".join(table.xpath(".//text()"))
        # Check for summary table of joint holders
        # Table typically has columns: 提出者及び共同保有者名 | 保有株券等の数（総数） | 株券等保有割合（％）
        if "提出者及び共同保有者名" in t_text and "株券等保有割合" in t_text:
            rows = table.xpath(".//tr")
            holders: list[JointHolder] = []
            for r in rows:
                cells = [replace_zenkaku(c.text_content().strip().replace("\n", " ")) for c in r.xpath(".//td | .//th")]
                if len(cells) >= 3:
                    name_cell = cells[0].strip()
                    shares_cell = cells[1].strip()
                    ratio_cell = cells[2].strip()

                    # Check for total row
                    if "合計" in name_cell:
                        tot_shares = _to_float(shares_cell)
                        tot_ratio = _to_float(ratio_cell)
                        if report.total_shares_held is None and tot_shares is not None:
                            report.total_shares_held = tot_shares
                        if report.holding_ratio is None and tot_ratio is not None:
                            report.holding_ratio = tot_ratio
                    elif name_cell and not any(k in name_cell for k in ["提出者", "共同保有者名", "項目"]):
                        s = _to_float(shares_cell) or 0.0
                        rt = _to_float(ratio_cell) or 0.0
                        holders.append(JointHolder(name=_clean_str(name_cell) or name_cell, shares_held=s, holding_ratio=rt))

            if holders and not report.joint_holders:
                report.joint_holders = holders
                if not report.filer_name:
                    report.filer_name = holders[0].name
