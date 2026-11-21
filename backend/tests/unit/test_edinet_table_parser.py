"""Unit and regression tests for EDINET HTML table parser and financial report extractor.

Ensures parser handles various real-world irregularities across EDINET HTML disclosures:
- Standard metric tables
- Nested tables within cells (e.g. S100IX7L)
- Complex multi-row / multi-column spans (rowspan, colspan)
- Kanji / Zenkaku numbers, Era dates (平成/令和/元年), annotations (※, 注), and signs (▲, △, ▼)
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from data.extractors.edinet import (
    convert_label_to_key,
    convert_to_month_end,
    convert_with_unit,
    extract_financial_metrics_from_file,
    extract_financial_metrics_from_html,
    kanji_to_int,
    parse_unit_multiplier,
    replace_zenkaku,
    split_label_and_unit,
)
from data.extractors.edinet.table_parser import parse_html_table_to_grid
import lxml.html

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "edinet_html"


# ----------------------------------------------------------------------
# 1. Normalizer Unit Tests
# ----------------------------------------------------------------------

def test_kanji_to_int():
    assert kanji_to_int("百") == 100
    assert kanji_to_int("千") == 1000
    assert kanji_to_int("万") == 10000
    assert kanji_to_int("百万") == 1_000_000
    assert kanji_to_int("一億二千五百万") == 125_000_000
    assert kanji_to_int("100万") == 1_000_000


def test_replace_zenkaku():
    assert replace_zenkaku("１２３４５６７８９０") == "1234567890"


def test_convert_to_month_end():
    # Western dates
    assert convert_to_month_end("2023年3月") == datetime.date(2023, 3, 31)
    assert convert_to_month_end("2024年2月期") == datetime.date(2024, 2, 29)  # Leap year
    assert convert_to_month_end("2023-04-15") == datetime.date(2023, 4, 15)

    # Japanese Era dates
    assert convert_to_month_end("平成30年3月期") == datetime.date(2018, 3, 31)
    assert convert_to_month_end("平成元年3月期") == datetime.date(1989, 3, 31)
    assert convert_to_month_end("令和元年6月期") == datetime.date(2019, 6, 30)
    assert convert_to_month_end("令和5年3月31日") == datetime.date(2023, 3, 31)

    # With '至'
    assert convert_to_month_end("至 2024年3月31日") == datetime.date(2024, 3, 31)
    assert convert_to_month_end("至 2024年5月") == datetime.date(2024, 5, 31)


def test_split_label_and_unit():
    label, unit = split_label_and_unit("売上高 (百万円)")
    assert label == "売上高"
    assert unit == "百万円"

    label, unit = split_label_and_unit("１株当たり当期純利益（円）")
    assert label == "１株当たり当期純利益"
    assert unit == "円"

    label, unit = split_label_and_unit("自己資本比率")
    assert label == "自己資本比率"
    assert unit is None


def test_convert_with_unit():
    # Unit multiplier (百万円 -> x 1,000,000)
    assert convert_with_unit("100", unit="百万円") == 100_000_000.0
    assert convert_with_unit("1,250", unit=1_000) == 1_250_000.0

    # Negative signs: ▲, △, ▼, -
    assert convert_with_unit("▲500", unit=1) == -500.0
    assert convert_with_unit("△1,200", unit="千円") == -1_200_000.0
    assert convert_with_unit("▼50", unit=1) == -50.0

    # Dash / missing
    assert np.isnan(convert_with_unit("―", unit=1))
    assert np.isnan(convert_with_unit("-", unit=1))
    assert np.isnan(convert_with_unit("", unit=1))

    # Bracketed note or value
    assert convert_with_unit("〔120〕", unit=1) == 120.0
    assert convert_with_unit("1,500 ※1", unit=1) == 1500.0


# ----------------------------------------------------------------------
# 2. Field Mapper Unit Tests
# ----------------------------------------------------------------------

def test_convert_label_to_key():
    assert convert_label_to_key("売上高") == "net_sales"
    assert convert_label_to_key("営業収益") == "net_sales"
    assert convert_label_to_key("営業利益") == "operating_income"
    assert convert_label_to_key("経常利益") == "ordinary_income"
    assert convert_label_to_key("親会社株主に帰属する当期純利益") == "profit_attributable_to_owners"
    assert convert_label_to_key("親会社の所有者に帰属する当期利益") == "profit_attributable_to_owners"
    assert convert_label_to_key("１株当たり当期純利益") == "earnings_per_share"
    assert convert_label_to_key("自己資本比率") == "equity_ratio"
    assert convert_label_to_key("決算年月") == "settlement_date"


# ----------------------------------------------------------------------
# 3. HTML Table Parser Grid & Span Tests
# ----------------------------------------------------------------------

def test_parse_html_table_to_grid_rowspan_colspan():
    html = """
    <table>
      <tr>
        <th rowspan="2">Header A</th>
        <th colspan="2">Group B</th>
      </tr>
      <tr>
        <th>Sub B1</th>
        <th>Sub B2</th>
      </tr>
      <tr>
        <td>Val A</td>
        <td>Val B1</td>
        <td>Val B2</td>
      </tr>
    </table>
    """
    table = lxml.html.fromstring(html)
    grid = parse_html_table_to_grid(table)

    assert len(grid) == 3
    # Row 0
    assert grid[0] == ["Header A", "Group B", "Group B"]
    # Row 1 (Header A propagated via rowspan)
    assert grid[1] == ["Header A", "Sub B1", "Sub B2"]
    # Row 2
    assert grid[2] == ["Val A", "Val B1", "Val B2"]


# ----------------------------------------------------------------------
# 4. Fixture-based Regression Tests
# ----------------------------------------------------------------------

def test_standard_metrics_fixture():
    fixture_path = FIXTURES_DIR / "standard_metrics.html"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    df = extract_financial_metrics_from_file(fixture_path)
    assert not df.empty

    # Columns should be month-end dates
    expected_cols = [
        datetime.date(2022, 3, 31),
        datetime.date(2023, 3, 31),
        datetime.date(2024, 3, 31),
    ]
    assert list(df.columns) == expected_cols

    # Check key mappings and normalized values
    assert "net_sales" in df.index
    # 100,500 百万円 -> 100,500,000,000 JPY
    assert df.loc["net_sales", datetime.date(2022, 3, 31)] == 100_500_000_000.0

    assert "ordinary_income" in df.index
    assert df.loc["ordinary_income", datetime.date(2023, 3, 31)] == 9_800_000_000.0

    # Negative net income (▲1,500 百万円)
    assert "profit_attributable_to_owners" in df.index
    assert df.loc["profit_attributable_to_owners", datetime.date(2024, 3, 31)] == -1_500_000_000.0

    # Per-share metric with decimal and negative (△44.15 円)
    assert "earnings_per_share" in df.index
    assert df.loc["earnings_per_share", datetime.date(2024, 3, 31)] == -44.15


def test_nested_table_fixture():
    """Ensure parser does not break when encountering <table> inside a <td> cell (S100IX7L case)."""
    fixture_path = FIXTURES_DIR / "nested_table.html"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    df = extract_financial_metrics_from_file(fixture_path)
    assert not df.empty

    # Check net_sales
    assert "net_sales" in df.index
    assert df.loc["net_sales", datetime.date(2023, 3, 31)] == 250_000_000_000.0

    # Cell with inner table should still be parsed without crash
    assert "ordinary_income" in df.index
    assert df.loc["ordinary_income", datetime.date(2023, 3, 31)] == 25_000.0

    # Net income cell containing table
    assert "profit_attributable_to_owners" in df.index
    assert df.loc["profit_attributable_to_owners", datetime.date(2024, 3, 31)] == 18_000_000_000.0


def test_multi_row_col_span_fixture():
    """Ensure complex multi-span headers resolve correctly without misalignment."""
    fixture_path = FIXTURES_DIR / "multi_row_col_span.html"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    df = extract_financial_metrics_from_file(fixture_path)
    assert not df.empty
    assert len(df.columns) == 4
    assert "net_sales" in df.index
    assert "operating_income" in df.index
    assert "equity_ratio" in df.index


def test_irregular_values_fixture():
    """Test Zenkaku numbers, Era dates, annotations, and multiple negative sign types."""
    fixture_path = FIXTURES_DIR / "irregular_values.html"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    df = extract_financial_metrics_from_file(fixture_path)
    assert not df.empty

    cols = list(df.columns)
    # 平成30年3月期 -> 2018-03-31
    assert datetime.date(2018, 3, 31) in cols
    # 令和元年3月期 -> 2019-03-31
    assert datetime.date(2019, 3, 31) in cols
    # 至 2024年3月31日 -> 2024-03-31
    assert datetime.date(2024, 3, 31) in cols

    d2018 = datetime.date(2018, 3, 31)
    d2019 = datetime.date(2019, 3, 31)
    d2024 = datetime.date(2024, 3, 31)

    # Zenkaku '１２，３４５' 百万円
    assert df.loc["net_sales", d2018] == 12_345_000_000.0

    # Annotation '14,500 ※1' 百万円
    assert df.loc["net_sales", d2019] == 14_500_000_000.0

    # 千円 negative '▲500,000'
    assert df.loc["ordinary_income", d2018] == -500_000_000.0
    # 千円 negative '▼250,000'
    assert df.loc["ordinary_income", d2019] == -250_000_000.0

    # Dash '―' -> NaN
    assert np.isnan(df.loc["profit_attributable_to_owners", d2018])
    # Bracketed '〔120〕' 百万円
    assert df.loc["profit_attributable_to_owners", d2019] == 120_000_000.0
    # Delta '△450' 百万円
    assert df.loc["profit_attributable_to_owners", d2024] == -450_000_000.0

    # Headcount '1,234 (注)2' 人
    assert df.loc["employee_count", d2018] == 1234.0


def test_two_column_unit_table_fixture():
    """Test real EDINET table layout with separate unit column and dual header rows."""
    fixture_path = FIXTURES_DIR / "two_column_unit_table.html"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    df = extract_financial_metrics_from_file(fixture_path)
    assert not df.empty

    # Verify extracted month-end dates
    expected_dates = [
        datetime.date(2022, 4, 30),
        datetime.date(2023, 4, 30),
        datetime.date(2024, 4, 30),
        datetime.date(2025, 4, 30),
        datetime.date(2026, 4, 30),
    ]
    assert list(df.columns) == expected_dates

    # Net sales in 千円 (43,075,732 千円 -> 43,075,732,000 JPY)
    assert "net_sales" in df.index
    assert df.loc["net_sales", datetime.date(2022, 4, 30)] == 43_075_732_000.0
    assert df.loc["net_sales", datetime.date(2026, 4, 30)] == 62_742_030_000.0

    # Ordinary income in 千円
    assert "ordinary_income" in df.index
    assert df.loc["ordinary_income", datetime.date(2026, 4, 30)] == 6_619_124_000.0


def test_rowspan_multiline_labels_alignment():
    """Ensure multi-line labels inside a single rowspan cell are distributed to correct rows.

    Prevents classic misalignment where values on row 2 inherit the merged text from row 1.
    """
    html = """
    <table>
      <tr>
        <th>決算年月</th>
        <th>2023年3月</th>
        <th>2024年3月</th>
      </tr>
      <tr>
        <th colspan="3">（１）連結経営指標等</th>
      </tr>
      <tr>
        <th rowspan="2">
          売上高 (百万円)<br>
          経常利益 (百万円)
        </th>
        <td>100,000</td>
        <td>120,000</td>
      </tr>
      <tr>
        <td>10,000</td>
        <td>15,000</td>
      </tr>
    </table>
    """
    df = extract_financial_metrics_from_html(html)
    assert not df.empty

    # Section row should be skipped
    assert "（１）連結経営指標等" not in df.index

    # Net sales (row 1)
    assert "net_sales" in df.index
    assert df.loc["net_sales", datetime.date(2023, 3, 31)] == 100_000_000_000.0
    assert df.loc["net_sales", datetime.date(2024, 3, 31)] == 120_000_000_000.0

    # Ordinary income (row 2) correctly separated and aligned
    assert "ordinary_income" in df.index
    assert df.loc["ordinary_income", datetime.date(2023, 3, 31)] == 10_000_000_000.0
    assert df.loc["ordinary_income", datetime.date(2024, 3, 31)] == 15_000_000_000.0


