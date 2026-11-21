"""Financial field mapper for EDINET reports.

Maps various Japanese financial terminology used across Japanese GAAP,
IFRS, and US-GAAP into standardized snake_case keys.
"""

from __future__ import annotations

import re


KEY_MAPPINGS: list[tuple[str, str]] = [
    # Settlement date
    (r"決算年月|決算期|事業年度|会計期間", "settlement_date"),

    # Per Share Metrics (Specific phrases with '株当たり' to avoid matching '株主')
    (r"潜在株式調整.*株.*当期純利益|希薄化後.*株.*利益", "diluted_earnings_per_share"),
    (r"(１|1|一)?株当たり.*純資産|BPS", "net_assets_per_share"),
    (r"(１|1|一)?株当たり.*当期(純)?利益|基本的(１|1)?株当たり|EPS", "earnings_per_share"),
    (r"(１|1|一)?株当たり.*配当|配当性向", "dividend_per_share"),

    # Financial Ratios
    (r"自己資本比率|親会社所有者帰属持分比率", "equity_ratio"),
    (r"自己資本利益率|株主資本利益率|ROE", "return_on_equity"),
    (r"株価収益率|PER", "price_earnings_ratio"),

    # Revenue / Sales
    (r"売上高|営業収益|売上収益|経常収益", "net_sales"),

    # Operating Income
    (r"営業利益|営業損失（△）|営業利益又は営業損失", "operating_income"),

    # Ordinary Income (Japanese GAAP specific)
    (r"経常利益|経常損失（△）|経常利益又は経常損失", "ordinary_income"),

    # Profit before tax (IFRS / US-GAAP / J-GAAP)
    (r"税引前利益|税引前当期(純)?利益|税金等調整前当期(純)?利益", "income_before_income_taxes"),

    # Profit attributable to owners / Net income (General)
    (r"(親会社.*当期(純)?利益|親会社の所有者に帰属する|当期純利益|当期利益)", "profit_attributable_to_owners"),

    # Comprehensive Income
    (r"包括利益", "comprehensive_income"),

    # Net Assets / Equity
    (r"純資産額|資本合計|自己資本額", "net_assets"),

    # Total Assets
    (r"総資産額|資産合計", "total_assets"),

    # Cash Flows
    (r"営業活動.*キャッシュ.*フロー", "cash_flow_operating"),
    (r"投資活動.*キャッシュ.*フロー", "cash_flow_investing"),
    (r"財務活動.*キャッシュ.*フロー", "cash_flow_financing"),
    (r"現金.*同等物.*期末残高|期末残高", "cash_equivalents_end"),

    # Headcount
    (r"平均.*臨時雇用|臨時雇用者数", "avg_temp_staff"),
    (r"従業員数|従業員", "employee_count"),
]


def convert_label_to_key(label: str) -> str:
    """Convert raw Japanese label from financial statements into a standardized key."""
    if not label:
        return ""

    label_clean = (
        label.strip()
        .replace("\u3000", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("（注）", "")
        .replace("(注)", "")
    )

    for pattern, key in KEY_MAPPINGS:
        if re.search(pattern, label_clean):
            return key

    return label_clean
