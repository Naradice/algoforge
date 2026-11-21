"""Parsers for JPX Daily Margin Publications (日々公表銘柄等信用取引残高 & 告知一覧)."""

from __future__ import annotations

import datetime
import io
from pathlib import Path
import re
from typing import Any
import unicodedata

from bs4 import BeautifulSoup
import openpyxl

from data.extractors.margin.models import DailyMarginIssue, MarginRegulationEvent


def _to_clean_str(val: Any) -> str:
    """Normalize string using NFKC, strip whitespace and fullwidth spaces."""
    if val is None:
        return ""
    norm = unicodedata.normalize("NFKC", str(val).strip())
    return re.sub(r"\s+", " ", norm).strip()


def _to_float(val: Any) -> float:
    """Safely convert cell value to float, handling formulas and strings."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = _to_clean_str(val)
    if not s or s in ("-", "--", "―", "ー", "*", "N/A"):
        return 0.0
    # Clean formula prefix '= ' or '=-'
    s = re.sub(r"^=\s*", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _normalize_symbol_code(raw_code: Any) -> str:
    """Convert JPX 5-digit code or string (e.g. '30420', '153A0') to standard symbol code."""
    code_str = _to_clean_str(raw_code)
    # If 5-digit ending in '0', strip trailing 0
    if len(code_str) == 5 and code_str.endswith("0"):
        return code_str[:-1]
    return code_str


def _parse_date(date_val: Any) -> str:
    """Convert date object or string into YYYY-MM-DD."""
    if isinstance(date_val, (datetime.datetime, datetime.date)):
        return date_val.strftime("%Y-%m-%d")
    s = _to_clean_str(date_val)
    # Match YYYY/M/D or YYYY-M-D
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return s


def parse_daily_margin_excel(content: bytes | str | Path) -> list[DailyMarginIssue]:
    """Parse JPX Daily Margin Trading Excel file (mtdaily.xlsx).

    Parameters
    ----------
    content : bytes | str | Path
        Raw bytes or file path to mtdaily.xlsx.

    Returns
    -------
    list[DailyMarginIssue]
        Parsed issues with balances and ratios.
    """
    if isinstance(content, (str, Path)):
        wb = openpyxl.load_workbook(content, data_only=True, read_only=True)
    else:
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 32:
        return []

    # 1. Parse as_of_date from header (around row 25)
    as_of_date = ""
    for r_idx in range(20, 28):
        if r_idx < len(rows):
            r = rows[r_idx]
            for cell in r:
                d = _parse_date(cell)
                if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                    as_of_date = d
                    break
            if as_of_date:
                break

    # 2. Find header row (usually contains 'コード' and '売残高')
    data_start_idx = -1
    for r_idx in range(25, 35):
        if r_idx < len(rows):
            row_str = " ".join([_to_clean_str(c) for c in rows[r_idx] if c is not None])
            if "コード" in row_str and ("売残高" in row_str or "Section" in row_str):
                # Data starts 2 rows after main header (or next row if subheader checked)
                data_start_idx = r_idx + 2
                break

    if data_start_idx == -1:
        data_start_idx = 31

    results: list[DailyMarginIssue] = []

    for row in rows[data_start_idx:]:
        if not row or len(row) < 14:
            continue

        raw_code = row[6]
        if raw_code is None:
            continue
        code = _normalize_symbol_code(raw_code)
        if not code:
            continue

        # Restrictions marks from columns 1 and 2
        restrictions = []
        for c_idx in (1, 2):
            if c_idx < len(row) and row[c_idx]:
                mark = _to_clean_str(row[c_idx])
                if mark:
                    restrictions.append(mark)

        raw_name = _to_clean_str(row[3])
        # Clean common stock suffixes for cleaner presentation
        name = re.sub(r"\s*普通株式$", "", raw_name).strip()

        market = _to_clean_str(row[4])
        issue_type = _to_clean_str(row[5])
        isin = _to_clean_str(row[7])
        unit = _to_clean_str(row[0])

        # Balances & Ratios
        sell_shares = _to_float(row[8])
        sell_change = _to_float(row[9])
        sell_ratio = _to_float(row[10])

        buy_shares = _to_float(row[11])
        buy_change = _to_float(row[12])
        buy_ratio = _to_float(row[13])

        ratio = _to_float(row[14]) if len(row) > 14 else 0.0

        # Detailed breakdowns if available
        neg_sell = _to_float(row[15]) if len(row) > 15 else 0.0
        neg_sell_chg = _to_float(row[16]) if len(row) > 16 else 0.0
        std_sell = _to_float(row[17]) if len(row) > 17 else 0.0
        std_sell_chg = _to_float(row[18]) if len(row) > 18 else 0.0

        neg_buy = _to_float(row[19]) if len(row) > 19 else 0.0
        neg_buy_chg = _to_float(row[20]) if len(row) > 20 else 0.0
        std_buy = _to_float(row[21]) if len(row) > 21 else 0.0
        std_buy_chg = _to_float(row[22]) if len(row) > 22 else 0.0

        issue = DailyMarginIssue(
            as_of_date=as_of_date,
            code=code,
            name=name,
            market=market,
            issue_type=issue_type,
            restrictions=restrictions,
            isin=isin,
            unit=unit,
            sell_shares=sell_shares,
            sell_change=sell_change,
            sell_ratio=sell_ratio,
            buy_shares=buy_shares,
            buy_change=buy_change,
            buy_ratio=buy_ratio,
            ratio=ratio,
            negotiable_sell_shares=neg_sell,
            negotiable_sell_change=neg_sell_chg,
            standardized_sell_shares=std_sell,
            standardized_sell_change=std_sell_chg,
            negotiable_buy_shares=neg_buy,
            negotiable_buy_change=neg_buy_chg,
            standardized_buy_shares=std_buy,
            standardized_buy_change=std_buy_chg,
        )
        results.append(issue)

    return results


def parse_daily_margin_announcements(content: bytes | str | Path) -> dict[str, list[MarginRegulationEvent]]:
    """Parse JPX daily margin designation & lifting announcements HTML.

    Parameters
    ----------
    content : bytes | str | Path
        Raw HTML bytes, string, or file path to margin-daily/index.html.

    Returns
    -------
    dict[str, list[MarginRegulationEvent]]
        Dictionary with keys 'designated' and 'lifted'.
    """
    if isinstance(content, Path) or (isinstance(content, str) and Path(content).is_file()):
        raw = Path(content).read_bytes()
    elif isinstance(content, str):
        raw = content.encode("utf-8")
    else:
        raw = content

    soup = BeautifulSoup(raw, "html.parser")
    tables = soup.find_all("table")

    results: dict[str, list[MarginRegulationEvent]] = {
        "designated": [],
        "lifted": [],
    }

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        # Determine table type by inspecting header row
        header_cells = [_to_clean_str(c.text) for c in rows[0].find_all(["th", "td"])]
        event_type = None
        if any("指定日" in c for c in header_cells):
            event_type = "designated"
        elif any("解除日" in c for c in header_cells):
            event_type = "lifted"

        if not event_type:
            continue

        # Parse data rows
        for row in rows[1:]:
            cells = [_to_clean_str(c.text) for c in row.find_all(["td", "th"])]
            if len(cells) < 3:
                continue

            raw_name = cells[0]
            # Clean marker like ※ and corporate types
            clean_name = re.sub(r"^[※*]\s*", "", raw_name)
            clean_name = re.sub(r"[（\(](株|社|有|合)[）\)]", "", clean_name).strip()

            raw_code = cells[1]
            code = _normalize_symbol_code(raw_code)

            raw_date = cells[2]
            event_date = _parse_date(raw_date)

            results[event_type].append(
                MarginRegulationEvent(
                    code=code,
                    name=clean_name,
                    event_date=event_date,
                    event_type=event_type,
                )
            )

    return results
