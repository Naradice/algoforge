"""Swing Trading Signals & Screening Engine.

Generates actionable swing trading signals by combining:
1. Short Squeeze Pressure (踏み上げ / 空売り残高・取組比率)
2. Daily Margin Lifted Turnaround (日々公表解除リバウンド)
3. Institutional Accumulation (大量保有・変更報告書での市場内買い増し)
4. Supply Overhang / Dilution Warning (買残過多・行使処分売り警戒)
5. Supply Relief (信用買残の急速な整理・底打ち)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from datetime import date as dt_date
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from data.models import (
    DailyMarginRecord,
    LargeHoldingReportRecord,
    LargeHoldingTradeRecord,
    MarginRegulationRecord,
)


SignalType = Literal[
    "squeeze_alert",         # 踏み上げ急騰候補 (High short interest & buy/sell ratio)
    "daily_margin_lifted",   # 日々公表解除リバウンド (Regulation lifted + supply digestion)
    "institutional_buy",     # 機関・大口買い増し (5% rule ratio increase + market purchase)
    "supply_relief",         # 需給改善・底打ち (Sharp drop in margin buy balance)
    "supply_overhang",       # 需給悪化警戒 / 売り (Excessive margin buy or dilution sale)
]

SignalDirection = Literal["LONG", "SHORT", "AVOID"]


@dataclass
class SwingSignal:
    """Actionable swing trading signal for a specific stock."""
    code: str
    name: str
    signal_type: SignalType
    direction: SignalDirection
    confidence_score: float  # 0.0 to 1.0
    detected_date: str       # YYYY-MM-DD
    title: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def screen_swing_signals(
    session: Session,
    target_date: dt_date | None = None,
    min_confidence: float = 0.5,
) -> list[SwingSignal]:
    """Scan latest database records and return active swing trading signals.

    Parameters
    ----------
    session : Session
        Active database session.
    target_date : dt_date | None
        Target date to evaluate (defaults to latest available date).
    min_confidence : float
        Minimum threshold for signal confidence (0.0 - 1.0).

    Returns
    -------
    list[SwingSignal]
        Ranked swing trading signals sorted by confidence score descending.
    """
    signals: list[SwingSignal] = []

    # 1. Determine latest margin date if not given
    if target_date is None:
        latest_dt = session.execute(
            sa.select(sa.func.max(DailyMarginRecord.as_of_date))
        ).scalar_one_or_none()
        target_date = latest_dt or dt_date.today()

    date_str = target_date.isoformat()

    # Fetch daily margin records for target_date
    margin_records = session.execute(
        sa.select(DailyMarginRecord).where(DailyMarginRecord.as_of_date == target_date)
    ).scalars().all()

    margin_map = {r.code: r for r in margin_records}

    # Fetch recent regulation events (last 14 days)
    two_weeks_ago = target_date - datetime.timedelta(days=14)
    reg_events = session.execute(
        sa.select(MarginRegulationRecord).where(
            MarginRegulationRecord.event_date >= two_weeks_ago,
            MarginRegulationRecord.event_date <= target_date,
        )
    ).scalars().all()

    # Fetch recent large holding reports (last 30 days)
    month_ago = target_date - datetime.timedelta(days=30)
    lh_reports = session.execute(
        sa.select(LargeHoldingReportRecord).where(
            LargeHoldingReportRecord.base_date >= month_ago,
            LargeHoldingReportRecord.base_date <= target_date,
        )
    ).scalars().all()

    lh_map: dict[str, list[LargeHoldingReportRecord]] = {}
    for rep in lh_reports:
        if rep.issuer_code:
            lh_map.setdefault(rep.issuer_code, []).append(rep)

    # -----------------------------------------------------------------------
    # Rule 1: Short Squeeze Alert (踏み上げ急騰候補)
    # -----------------------------------------------------------------------
    for code, m in margin_map.items():
        # High short ratio (sell_ratio >= 3.0%) and high ratio (ratio >= 50%)
        # or extreme ratio (ratio >= 80%)
        if m.sell_ratio >= 2.5 and m.ratio >= 40.0:
            reasons = [
                f"取組比率(売残/買残): {m.ratio:.1f}%",
                f"売残上場比: {m.sell_ratio:.1f}% ({m.sell_shares:,.0f}株)",
                f"買残上場比: {m.buy_ratio:.1f}% ({m.buy_shares:,.0f}株)",
            ]
            if m.sell_change > 0:
                reasons.append(f"売残前日比増加: +{m.sell_change:,.0f}株 (空売り増加中)")

            confidence = min(0.6 + (m.ratio / 200.0) + (m.sell_ratio / 20.0), 0.95)

            signals.append(
                SwingSignal(
                    code=code,
                    name=m.name,
                    signal_type="squeeze_alert",
                    direction="LONG",
                    confidence_score=round(confidence, 2),
                    detected_date=date_str,
                    title=f"【踏み上げ警戒】取組比率{m.ratio:.0f}%・空売り急増",
                    reasons=reasons,
                    metrics={
                        "ratio": m.ratio,
                        "sell_ratio": m.sell_ratio,
                        "sell_shares": m.sell_shares,
                        "sell_change": m.sell_change,
                        "buy_ratio": m.buy_ratio,
                    },
                )
            )

    # -----------------------------------------------------------------------
    # Rule 2: Daily Margin Lifted Momentum (日々公表解除リバウンド)
    # -----------------------------------------------------------------------
    for ev in reg_events:
        if ev.event_type == "lifted":
            # Check if we have margin info
            m = margin_map.get(ev.code)
            reasons = [f"日々公表解除日: {ev.event_date.isoformat()}"]
            confidence = 0.70

            if m:
                reasons.append(f"現在買残上場比: {m.buy_ratio:.1f}%")
                if m.buy_change < 0:
                    reasons.append(f"買残前日比減少: {m.buy_change:,.0f}株 (信用整理進行)")
                    confidence += 0.10

            signals.append(
                SwingSignal(
                    code=ev.code,
                    name=ev.name,
                    signal_type="daily_margin_lifted",
                    direction="LONG",
                    confidence_score=round(min(confidence, 0.90), 2),
                    detected_date=date_str,
                    title=f"【日々公表解除】規制解除による売買代金復活・リバウンド狙い",
                    reasons=reasons,
                    metrics={
                        "lifted_date": ev.event_date.isoformat(),
                        "buy_ratio": m.buy_ratio if m else None,
                        "buy_change": m.buy_change if m else None,
                    },
                )
            )

    # -----------------------------------------------------------------------
    # Rule 3: Institutional Accumulation (機関・アクティビスト買い増し)
    # -----------------------------------------------------------------------
    for code, reps in lh_map.items():
        for rep in reps:
            if rep.ratio_change and rep.ratio_change >= 0.5:
                # Check trades for market purchase
                market_buys = [
                    t for t in rep.trades
                    if t.trade_type == "取得" and (t.market_type == "市場内" or t.market_type is None)
                ]

                has_market_buy = len(market_buys) > 0
                reasons = [
                    f"提出者: {rep.filer_name}",
                    f"保有割合増減: {rep.ratio_change:+.2f}% (現在 {rep.holding_ratio:.2f}%)",
                    f"保有目的: {rep.purpose_of_holding or '記載なし'}",
                ]
                if has_market_buy:
                    reasons.append(f"直近市場内買付あり ({len(market_buys)}回)")

                confidence = 0.65 + min(rep.ratio_change / 10.0, 0.20)
                if has_market_buy:
                    confidence += 0.10

                signals.append(
                    SwingSignal(
                        code=code,
                        name=rep.issuer_name or code,
                        signal_type="institutional_buy",
                        direction="LONG",
                        confidence_score=round(min(confidence, 0.95), 2),
                        detected_date=date_str,
                        title=f"【大口買い増し】{rep.filer_name} 保有比率{rep.ratio_change:+.2f}%",
                        reasons=reasons,
                        metrics={
                            "filer": rep.filer_name,
                            "holding_ratio": rep.holding_ratio,
                            "ratio_change": rep.ratio_change,
                            "market_buys_count": len(market_buys),
                        },
                    )
                )

    # -----------------------------------------------------------------------
    # Rule 4: Supply Overhang & Dilution Sale Warning (需給悪化警戒 / AVOID or SHORT)
    # -----------------------------------------------------------------------
    for code, m in margin_map.items():
        # High margin overhang (buy_ratio >= 25%)
        if m.buy_ratio >= 25.0:
            reasons = [
                f"買残上場比: {m.buy_ratio:.1f}% ({m.buy_shares:,.0f}株)",
                f"発行済株式総数に対する信用買い残が極めて大きく、戻り売り圧力が強大",
            ]
            if m.buy_change > 0:
                reasons.append(f"買残前日比増加: +{m.buy_change:,.0f}株 (高値掴み増加)")

            signals.append(
                SwingSignal(
                    code=code,
                    name=m.name,
                    signal_type="supply_overhang",
                    direction="AVOID",
                    confidence_score=round(min(0.65 + (m.buy_ratio - 25.0) / 50.0, 0.90), 2),
                    detected_date=date_str,
                    title=f"【上値しこり警戒】買残上場比{m.buy_ratio:.1f}%・膨大な戻り売り圧力",
                    reasons=reasons,
                    metrics={
                        "buy_ratio": m.buy_ratio,
                        "buy_shares": m.buy_shares,
                        "buy_change": m.buy_change,
                    },
                )
            )

    # -----------------------------------------------------------------------
    # Rule 5: Supply Relief (信用買残の急減・需給改善)
    # -----------------------------------------------------------------------
    for code, m in margin_map.items():
        # Margin buy dropped significantly (e.g. drop >= 200,000 shares and buy_change < 0)
        # and buy_ratio still moderate (10% - 25%)
        if m.buy_shares > 0 and m.buy_change <= -150000.0 and m.buy_ratio <= 30.0:
            drop_pct = abs(m.buy_change) / (m.buy_shares + abs(m.buy_change)) * 100.0
            reasons = [
                f"買残前日比: {m.buy_change:,.0f}株 (約{drop_pct:.1f}%の信用買い玉が決済・損切り)",
                f"現在買残上場比: {m.buy_ratio:.1f}%",
                "追証・損切りの一巡により需給の重しが急激に軽減",
            ]

            signals.append(
                SwingSignal(
                    code=code,
                    name=m.name,
                    signal_type="supply_relief",
                    direction="LONG",
                    confidence_score=round(min(0.60 + drop_pct / 50.0, 0.85), 2),
                    detected_date=date_str,
                    title=f"【需給整理】買残{abs(m.buy_change):,.0f}株急減・投げ売り一巡",
                    reasons=reasons,
                    metrics={
                        "buy_shares": m.buy_shares,
                        "buy_change": m.buy_change,
                        "drop_pct": round(drop_pct, 1),
                    },
                )
            )

    # Filter by confidence and sort descending
    filtered = [s for s in signals if s.confidence_score >= min_confidence]
    filtered.sort(key=lambda s: s.confidence_score, reverse=True)
    return filtered
