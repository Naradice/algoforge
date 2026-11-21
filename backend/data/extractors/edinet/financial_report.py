"""Extractor for EDINET financial report documents (有価証券報告書 / 四半期報告書).

Extracts key financial metrics (主要な経営指標等の推移) into structured
DataFrames and serializable dictionaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lxml.html
import numpy as np
import pandas as pd

from data.extractors.edinet.field_mapper import convert_label_to_key
from data.extractors.edinet.normalizers import (
    convert_to_month_end,
    convert_with_unit,
    is_unit_column,
    is_unit_value,
    parse_unit_multiplier,
    split_label_and_unit,
    split_symbol,
)
from data.extractors.edinet.table_parser import parse_html_table_to_grid

logger = logging.getLogger(__name__)


def parse_financial_metrics_table(table_element: Any) -> pd.DataFrame:
    """Parse an lxml table element representing '主要な経営指標等の推移' into normalized metrics.

    Handles:
      - 2-level headers (e.g. Period row + Date row)
      - Separate unit columns (e.g. Col 0: metric name, Col 1: unit like '(千円)')
      - Standard 1-column labels with inline units (e.g. '売上高 (百万円)')
      - Multi-row spans / cell alignments

    Returns:
        pd.DataFrame where:
          - index: standardized financial keys (e.g. 'net_sales', 'ordinary_income')
          - columns: datetime.date objects or period strings
          - values: float values normalized by unit multiplier
    """
    grid = parse_html_table_to_grid(table_element)
    if not grid or len(grid) < 2:
        return pd.DataFrame()

    # Filter out empty rows
    valid_rows = [r for r in grid if any(cell.strip() for cell in r)]
    if len(valid_rows) < 2:
        return pd.DataFrame()

    num_cols = len(valid_rows[0])
    if num_cols < 2:
        return pd.DataFrame()

    # Step 1: Detect header row(s) and date columns
    # Look for settlement date row (contains 決算年月 or multiple dates like 202X年)
    date_row_idx: int | None = None
    period_labels: list[str] = []

    for r_idx in range(min(4, len(valid_rows))):
        row = valid_rows[r_idx]
        row_text = " ".join(row)
        # Check if row looks like date row
        date_candidates = [convert_to_month_end(cell) for cell in row]
        valid_date_count = sum(1 for d in date_candidates if d is not None)

        if "決算年月" in row_text or "事業年度" in row_text or valid_date_count >= 2:
            date_row_idx = r_idx
            break

    # Determine data starting row
    if date_row_idx is not None:
        header_end_idx = date_row_idx + 1
        raw_periods = valid_rows[date_row_idx]
    else:
        # Default first row as header
        header_end_idx = 1
        raw_periods = valid_rows[0]

    data_rows = valid_rows[header_end_idx:]
    if not data_rows:
        return pd.DataFrame()

    # Step 2: Detect whether Column 1 is a dedicated "unit" column
    # E.g. [ '売上高', '（千円）', '43,075,732', ... ]
    has_unit_col = False
    if num_cols >= 3:
        col1_values = [r[1] for r in data_rows if len(r) > 1 and r[1].strip()]
        if col1_values:
            # Check how many look like units (e.g. 円, 千円, 百万円, %, ％, 株, 人, -)
            unit_like_count = 0
            for val in col1_values:
                val_clean = val.strip().replace("（", "").replace("）", "").replace("(", "").replace(")", "")
                if is_unit_value(val_clean) or val_clean in ("千円", "百万円", "百株", "千株", "株", "円銭", "円"):
                    unit_like_count += 1
            if unit_like_count / len(col1_values) >= 0.5:
                has_unit_col = True

    data_col_start = 2 if has_unit_col else 1

    # Build column headers (dates or period names)
    col_headers: list[Any] = []
    for c_idx in range(data_col_start, num_cols):
        date_val = None
        # Check date_row_idx first
        if date_row_idx is not None and c_idx < len(valid_rows[date_row_idx]):
            date_val = convert_to_month_end(valid_rows[date_row_idx][c_idx])

        # Fallback check row 0
        if date_val is None and c_idx < len(valid_rows[0]):
            date_val = convert_to_month_end(valid_rows[0][c_idx])

        if date_val is not None:
            col_headers.append(date_val)
        else:
            # Fallback to period name or column index
            name = ""
            if date_row_idx is not None and c_idx < len(valid_rows[date_row_idx]):
                name = valid_rows[date_row_idx][c_idx].strip()
            if not name and c_idx < len(valid_rows[0]):
                name = valid_rows[0][c_idx].strip()
            col_headers.append(name if name else f"period_{c_idx - data_col_start + 1}")

    # Step 3: Extract metrics row by row
    parsed_rows: dict[str, list[float]] = {}

    for row in data_rows:
        raw_label = row[0].strip()
        if not raw_label:
            continue

        # Skip section header rows that span across columns or label blocks
        unique_non_empty = set(c.strip() for c in row if c.strip())
        if len(unique_non_empty) == 1 and len(row) >= 3:
            continue
        if any(sec in raw_label for sec in ("連結経営指標等", "提出会社の状況", "主要な経営指標等の推移")):
            if not any(kw in raw_label for kw in ("売上", "収益", "利益", "資産", "株", "率")):
                continue

        index_key = convert_label_to_key(raw_label)
        if index_key == "settlement_date":
            # Another settlement date row inside data rows
            continue

        # Extract unit multiplier
        if has_unit_col and len(row) > 1 and row[1].strip():
            raw_unit = row[1].strip().replace("（", "").replace("）", "").replace("(", "").replace(")", "")
            unit_mult = parse_unit_multiplier(raw_unit)
        else:
            _, unit_str = split_label_and_unit(raw_label)
            if unit_str:
                unit_clean, _ = split_symbol(unit_str)
                unit_mult = parse_unit_multiplier(unit_clean)
            else:
                unit_mult = 1.0

        # Extract values
        numeric_vals: list[float] = []
        for c_idx in range(data_col_start, num_cols):
            cell_txt = row[c_idx] if c_idx < len(row) else ""
            numeric_vals.append(convert_with_unit(cell_txt, unit=unit_mult))

        # Avoid empty rows
        if any(not np.isnan(v) for v in numeric_vals):
            # If duplicated key, only keep the first occurrence or specific one
            if index_key not in parsed_rows:
                parsed_rows[index_key] = numeric_vals

    if not parsed_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(parsed_rows, index=col_headers).T
    result_df.index.name = "metric"
    return result_df


def decode_html_bytes(html_content: str | bytes) -> str:
    """Safely decode HTML bytes trying utf-8, cp932 (Shift-JIS), then fallback."""
    if isinstance(html_content, str):
        return html_content
    for enc in ("utf-8", "cp932", "shift_jis", "euc-jp"):
        try:
            return html_content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return html_content.decode("utf-8", errors="ignore")


def extract_financial_metrics_from_html(html_content: str | bytes) -> pd.DataFrame:
    """Extract financial metrics table from HTML content string or bytes."""
    decoded = decode_html_bytes(html_content)
    # Strip XML declaration if present in unicode string to prevent lxml ValueError
    if "<?xml" in decoded[:120]:
        decoded = re.sub(r"<\?xml.*?\?>", "", decoded, count=1).strip()

    tree = lxml.html.fromstring(decoded)

    if tree.tag == "table":
        tables = [tree]
    else:
        tables = tree.xpath(".//table")
    if not tables:
        return pd.DataFrame()

    for table in tables:
        txt = table.text_content()
        if any(kw in txt for kw in ("売上高", "営業収益", "経常利益", "当期純利益", "純資産額")):
            df = parse_financial_metrics_table(table)
            if not df.empty and len(df.columns) > 1:
                return df

    return parse_financial_metrics_table(tables[0])


def extract_financial_metrics_from_file(file_path: Path | str) -> pd.DataFrame:
    """Extract financial metrics table from a saved HTML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        html_content = fp.read()

    return extract_financial_metrics_from_html(html_content)
