"""Data models for JPX Margin Trading Data (信用取引残高・日々公表銘柄)."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Any


@dataclass
class DailyMarginIssue:
    """Represents daily margin trading balance and ratios for a single issue (日々公表銘柄等)."""
    as_of_date: str  # 申込み現在日 (YYYY-MM-DD)
    code: str  # 銘柄コード (例: 3042, 153A)
    name: str  # 銘柄名 (例: セキュアヴェイル)
    market: str  # 市場 (プライム, スタンダード, グロース)
    issue_type: str  # 銘柄種別 (貸: 貸借銘柄, 制: 制度信用銘柄, 他: その他)
    restrictions: list[str] = field(default_factory=list)  # 規, 日, 監, 株, 喚 等
    isin: str | None = None  # 新証券コード (ISIN)
    unit: str | None = None  # 売買単位記号 (B=100株等)

    # 合計 (Total)
    sell_shares: float = 0.0  # 売残高 (株)
    sell_change: float = 0.0  # 売残前日比 (株)
    sell_ratio: float = 0.0   # 売残上場比 (%)
    buy_shares: float = 0.0   # 買残高 (株)
    buy_change: float = 0.0   # 買残前日比 (株)
    buy_ratio: float = 0.0    # 買残上場比 (%)
    ratio: float = 0.0        # 取組比率 (売残/買残 %)

    # 内訳 (一般信用 & 制度信用)
    negotiable_sell_shares: float = 0.0  # 一般信用 売残高
    negotiable_sell_change: float = 0.0  # 一般信用 売残前日比
    standardized_sell_shares: float = 0.0  # 制度信用 売残高
    standardized_sell_change: float = 0.0  # 制度信用 売残前日比
    negotiable_buy_shares: float = 0.0  # 一般信用 買残高
    negotiable_buy_change: float = 0.0  # 一般信用 買残前日比
    standardized_buy_shares: float = 0.0  # 制度信用 買残高
    standardized_buy_change: float = 0.0  # 制度信用 買残前日比


@dataclass
class MarginRegulationEvent:
    """Represents a regulatory designation or lifting event for daily publication."""
    code: str  # 銘柄コード (例: 4498, 153A)
    name: str  # 銘柄名
    event_date: str  # 指定日または解除日 (YYYY-MM-DD)
    event_type: str  # 'designated' (指定) または 'lifted' (解除)


@dataclass
class MarketMarginRecord:
    """Weekly aggregate margin trading outstanding for the entire market (市場全体推移)."""
    date: str  # 算出基準日 (YYYY-MM-DD)
    standardized_sell_shares: float = 0.0  # 制度信用 売残高 (千株)
    standardized_buy_shares: float = 0.0   # 制度信用 買残高 (千株)
    negotiable_sell_shares: float = 0.0    # 一般信用 売残高 (千株)
    negotiable_buy_shares: float = 0.0     # 一般信用 買残高 (千株)
    total_sell_shares: float = 0.0         # 合計 売残高 (千株)
    total_buy_shares: float = 0.0          # 合計 買残高 (千株)
    margin_ratio: float = 0.0              # 信用倍率 (買残 / 売残)
