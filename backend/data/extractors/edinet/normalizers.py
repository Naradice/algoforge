"""Text and numeric normalizers for EDINET financial tables.

Handles conversion of Japanese imperial years (和暦), Kanji numbers,
currency/unit multipliers, negative sign markers (▲, △, ▼), and delimiters.
"""

from __future__ import annotations

import calendar
import datetime
import re
from typing import Any

import numpy as np
import pandas as pd

NUMBERS = "0123456789"

KANJI_NUM_MAP = {
    "〇": 0, "零": 0,
    "一": 1, "壱": 1, "弌": 1,
    "二": 2, "弐": 2, "貳": 2,
    "三": 3, "参": 3, "參": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

KANJI_UNIT_MAP = {
    "銭": 0.01,
    "十": 10, "拾": 10,
    "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
    "万": 10**4, "萬": 10**4,
    "億": 10**8,
    "兆": 10**12,
}


def kanji_to_int(text: str) -> int | float:
    """Convert Kanji number string (e.g. '一億二千三百四十五万' or '百万円') to int/float."""
    result = 0.0
    section = 0.0
    num = 0.0

    for c in text:
        if c in KANJI_NUM_MAP:
            num = KANJI_NUM_MAP[c]
        elif c in NUMBERS:
            num = num * 10 + int(c)
        elif c in KANJI_UNIT_MAP:
            unit = KANJI_UNIT_MAP[c]
            if unit >= 10000:
                if section == 0.0 and num == 0.0:
                    num = 1.0
                section = (section + num) * unit
                result += section
                section = 0.0
            else:
                if num == 0.0:
                    num = 1.0
                section += num * unit
            num = 0.0
        else:
            continue

    result += section + num
    if result.is_integer():
        return int(result)
    return result


def replace_zenkaku(text: str) -> str:
    """Convert full-width digits and symbols to half-width."""
    trans = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "７" if False else "7",
        "８": "8", "９": "9",
        "，": ",", "．": ".",
    })
    return text.translate(trans)


def replace_number(text: str) -> int | float:
    """Convert Zenkaku digits and Kanji representation to int/float."""
    return kanji_to_int(replace_zenkaku(text))


def convert_to_month_end(date_str: Any) -> datetime.date | None:
    """Convert Japanese date or period string to the month-end date."""
    if pd.isna(date_str) or date_str is None:
        return None

    date_str = str(date_str).strip()
    date_str = replace_zenkaku(date_str)

    # Era conversion
    if date_str.startswith("平成") or "平成" in date_str:
        m = re.search(r"平成(.*?)年", date_str)
        if m:
            jp_year_str = m.group(1).strip()
            jp_year = 1 if jp_year_str == "元" else int(jp_year_str)
            year = 1988 + jp_year
            date_str = date_str.replace(f"平成{jp_year_str}年", f"{year}年")

    if date_str.startswith("令和") or "令和" in date_str:
        m = re.search(r"令和(.*?)年", date_str)
        if m:
            jp_year_str = m.group(1).strip()
            jp_year = 1 if jp_year_str == "元" else int(jp_year_str)
            year = 2018 + jp_year
            date_str = date_str.replace(f"令和{jp_year_str}年", f"{year}年")

    # Handle '至' (until)
    if "至" in date_str:
        m = re.search(r"至\s*([0-9]{4})年\s*([0-9]{1,2})月", date_str)
        if m:
            year, month = m.groups()
            date_str = f"{year}年{month}月"
        else:
            m2 = re.search(r"至\s*([0-9]{4})[-/]([0-9]{1,2})", date_str)
            if m2:
                year, month = m2.groups()
                date_str = f"{year}年{month}月"

    if date_str.endswith("期"):
        date_str = date_str[:-1].strip()

    # Match year and month
    m_ym = re.search(r"([0-9]{4})年\s*([0-9]{1,2})月(?:\s*([0-9]{1,2})日)?", date_str)
    if m_ym:
        year = int(m_ym.group(1))
        month = int(m_ym.group(2))
        day = int(m_ym.group(3)) if m_ym.group(3) else calendar.monthrange(year, month)[1]
        try:
            return datetime.date(year, month, day)
        except ValueError:
            last_day = calendar.monthrange(year, month)[1]
            return datetime.date(year, month, last_day)

    # Standard YYYY-MM-DD or YYYY/MM/DD
    m_std = re.search(r"([0-9]{4})[-/]([0-9]{1,2})[-/]?([0-9]{1,2})?", date_str)
    if m_std:
        year = int(m_std.group(1))
        month = int(m_std.group(2))
        day = int(m_std.group(3)) if m_std.group(3) else calendar.monthrange(year, month)[1]
        return datetime.date(year, month, day)

    return None


def split_label_and_unit(text: str) -> tuple[str, str | None]:
    """Split label into name and unit (e.g., '売上高 (百万円)' -> ('売上高', '百万円'))."""
    match = re.match(r"^(.*?)\s*[（(](.*?)[）)]\s*$", text)
    if match:
        label = match.group(1).strip()
        unit = match.group(2).strip()
        return label, unit
    return text.strip(), None


def is_unit_value(value: Any) -> bool:
    """Check if value is a standalone unit token."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    pattern = r"^(\$|€|£|¥|円|％|%|倍|人|件|台|点|kg|g|m|km|cm|時間|日|年|〃)$"
    return bool(re.match(pattern, text))


def is_unit_column(series: pd.Series, threshold: float = 0.8) -> bool:
    """Determine whether a column represents unit labels rather than values."""
    valid = series.dropna().map(is_unit_value)
    if len(valid) == 0:
        return False
    return bool(valid.mean() >= threshold)


def split_symbol(text: str) -> tuple[str, str | None]:
    """Extract trailing currency/unit symbols (e.g. '150円' -> ('150', 'JPY'))."""
    match = re.match(r"^(.*?)([¥\\$€円％%倍人]+)$", text)
    if match:
        label = match.group(1).strip()
        symbol = match.group(2).strip()
        symbol_map = {
            "円": "JPY",
            "¥": "JPY",
            "$": "USD",
            "€": "EUR",
            "%": "PCT",
            "％": "PCT",
            "倍": "RTO",
            "人": "NoP",
        }
        return label, symbol_map.get(symbol, symbol)
    return text, None


def remove_delimiter(text: str) -> str:
    """Remove digit commas and whitespace."""
    return text.replace(",", "").replace("，", "").replace(" ", "").replace("　", "")


def split_sign(text: str) -> tuple[int, str]:
    """Extract negative sign (▲, ▼, △, －, -) from text."""
    if text.startswith(("▲", "▼", "△")):
        return -1, text[1:].strip()
    elif text.startswith(("-", "－")):
        return -1, text[1:].strip()
    return 1, text.strip()


def parse_unit_multiplier(unit: Any) -> float:
    """Parse unit string (e.g. '百万円', '千円', '1000', 1000) into numeric multiplier."""
    if unit is None:
        return 1.0
    if isinstance(unit, (int, float)):
        return float(unit) if unit != 0 else 1.0

    unit_str = str(unit).strip()
    unit_str, _ = split_symbol(unit_str)

    # Direct unit names
    multipliers = {
        "円": 1.0,
        "千円": 1_000.0,
        "百万円": 1_000_000.0,
        "十万円": 100_000.0,
        "億円": 100_000_000.0,
        "兆円": 1_000_000_000_000.0,
        "千": 1_000.0,
        "百万": 1_000_000.0,
        "十万": 100_000.0,
        "億": 100_000_000.0,
        "兆": 1_000_000_000_000.0,
    }
    if unit_str in multipliers:
        return multipliers[unit_str]

    val = kanji_to_int(unit_str)
    if val and val > 0:
        return float(val)

    return 1.0


def clean_annotations(text: str) -> str:
    """Remove footnote and reference markers like ※1, (注)2, 〔注1〕, etc."""
    # Remove ※\d*, ※ \d*
    cleaned = re.sub(r"※\s*\d*", "", text)
    # Remove (注)\d*, （注）\d*, (注 1), (注) 2
    cleaned = re.sub(r"[（\(]\s*注\s*\d*\s*[）\)]\s*\d*", "", cleaned)
    # Remove 〔注\d*〕\d*, [注\d*]\d*
    cleaned = re.sub(r"[〔\[]\s*注\s*\d*\s*[〕\]]\s*\d*", "", cleaned)
    return cleaned.strip()


def convert_with_unit(num_txt: Any, unit: Any = 1.0) -> float:
    """Convert raw text to numeric float value multiplied by unit."""
    if num_txt is None or (isinstance(num_txt, float) and np.isnan(num_txt)):
        return np.nan

    num_txt = replace_zenkaku(str(num_txt))
    num_txt = num_txt.replace("　", "").replace("\xa0", "").strip()
    if not num_txt:
        return np.nan

    # Clean annotations (※1, (注)1, etc.)
    num_txt = clean_annotations(num_txt)

    # Strip bracketed contents if it contains number, or strip annotation brackets
    bracket_m = re.search(r"[〔（(［\[](.*?)[］\])）〕]", num_txt)
    if bracket_m:
        inner = bracket_m.group(1).strip()
        if any(c.isdigit() for c in inner):
            num_txt = inner
        else:
            num_txt = re.sub(r"[〔（(［\[].*?[］\])）〕]", "", num_txt).strip()

    # Dash / empty equivalents
    if num_txt in ("-", "－", "―", "ー", "…", "・", "/", "N/A", "NA", ""):
        return np.nan

    num_txt, _ = split_symbol(num_txt)
    num_txt = remove_delimiter(num_txt)
    sign, num_txt = split_sign(num_txt)

    if not num_txt:
        return np.nan

    multiplier = parse_unit_multiplier(unit)

    try:
        if "." in num_txt:
            number = float(num_txt)
        else:
            # Handle standard integers or Kanji numbers
            if any(c in NUMBERS for c in num_txt):
                number = float(num_txt)
            else:
                number = float(kanji_to_int(num_txt))
        return number * sign * multiplier
    except (ValueError, TypeError):
        return np.nan
