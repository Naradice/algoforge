"""Parser for JPX Market-wide Weekly Margin History (信用取引現在高過去推移表)."""

from __future__ import annotations

import datetime
import io
from pathlib import Path
import re
from typing import Any

import openpyxl

from data.extractors.margin.models import MarketMarginRecord
from data.extractors.margin.daily_parser import _parse_date, _to_clean_str, _to_float


def parse_market_margin_excel(content: bytes | str | Path) -> list[MarketMarginRecord]:
    """Parse JPX Market-wide Weekly Margin History Excel file.

    Parameters
    ----------
    content : bytes | str | Path
        Raw bytes or file path to market margin history Excel.

    Returns
    -------
    list[MarketMarginRecord]
        List of weekly market margin records sorted chronologically.
    """
    if isinstance(content, (str, Path)):
        wb = openpyxl.load_workbook(content, data_only=True, read_only=True)
    else:
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 11:
        return []

    records: list[MarketMarginRecord] = []

    # Find the data rows starting after header (row 10 onwards)
    for row in rows[9:]:
        if not row or row[0] is None:
            continue

        raw_date = row[0]
        # Check if valid date
        date_str = _parse_date(raw_date)
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            continue

        # Values are in thousand shares (千株)
        # Row layout:
        # col 1: 制度自己売, col 3: 制度自己買
        # col 5: 制度委託売, col 7: 制度委託買
        # col 9: 一般委託売, col 11: 一般委託買 (if present)
        std_prop_sell = _to_float(row[1]) if len(row) > 1 else 0.0
        std_prop_buy = _to_float(row[3]) if len(row) > 3 else 0.0

        std_cust_sell = _to_float(row[5]) if len(row) > 5 else 0.0
        std_cust_buy = _to_float(row[7]) if len(row) > 7 else 0.0

        neg_cust_sell = _to_float(row[9]) if len(row) > 9 else 0.0
        neg_cust_buy = _to_float(row[11]) if len(row) > 11 else 0.0

        std_sell_total = std_prop_sell + std_cust_sell
        std_buy_total = std_prop_buy + std_cust_buy

        total_sell = std_sell_total + neg_cust_sell
        total_buy = std_buy_total + neg_cust_buy

        ratio = round(total_buy / total_sell, 2) if total_sell > 0 else 0.0

        rec = MarketMarginRecord(
            date=date_str,
            standardized_sell_shares=std_sell_total,
            standardized_buy_shares=std_buy_total,
            negotiable_sell_shares=neg_cust_sell,
            negotiable_buy_shares=neg_cust_buy,
            total_sell_shares=total_sell,
            total_buy_shares=total_buy,
            margin_ratio=ratio,
        )
        records.append(rec)

    return records
