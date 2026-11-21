"""Verification runner for JPX Margin Collector against real live JPX endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from data.collectors.jpx_margin import JPXMarginCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_jpx_margin")
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    collector = JPXMarginCollector()

    print("\n==================================================")
    print("1. Fetching Live Daily Margin Announcements (日々公表指定・解除)")
    print("==================================================")
    announcements = collector.fetch_daily_margin_announcements()
    designated = announcements.get("designated", [])
    lifted = announcements.get("lifted", [])

    print(f"Total Currently Designated Issues: {len(designated)}")
    print("Sample Designated Issues (Top 5):")
    for d in designated[:5]:
        print(f"  [{d.code}] {d.name} | Designated Date: {d.event_date}")

    print(f"\nTotal Recently Lifted Issues: {len(lifted)}")
    print("Lifted Issues:")
    for l in lifted:
        print(f"  [{l.code}] {l.name} | Lifted Date: {l.event_date}")

    print("\n==================================================")
    print("2. Fetching Latest Daily Margin Balances (日々公表等残高 Excel)")
    print("==================================================")
    file_url, issues = collector.fetch_latest_daily_margin_balances()
    print(f"Downloaded from: {file_url}")
    print(f"Total Parsed Issues: {len(issues)}")

    if issues:
        print(f"As of Date: {issues[0].as_of_date}")
        # Top 5 highest buy ratio (しこり玉 / 過熱感)
        sorted_by_buy_ratio = sorted(issues, key=lambda x: x.buy_ratio, reverse=True)
        print("\nTop 5 Issues with Highest Buy Ratio (上場比: 買残/上場株式):")
        for iss in sorted_by_buy_ratio[:5]:
            chg_sign = f"{iss.buy_change:+,.0f}"
            print(f"  [{iss.code}] {iss.name:15} | 買残比: {iss.buy_ratio:4.1f}% | 買残: {iss.buy_shares:10,.0f}株 (前日比 {chg_sign:>10}) | 信用倍率/取組: {iss.ratio:5.1f}%")

        # Top 5 highest short ratio (踏み上げ / 売り残高)
        sorted_by_sell_ratio = sorted(issues, key=lambda x: x.sell_ratio, reverse=True)
        print("\nTop 5 Issues with Highest Sell Ratio (上場比: 売残/上場株式):")
        for iss in sorted_by_sell_ratio[:5]:
            chg_sign = f"{iss.sell_change:+,.0f}"
            print(f"  [{iss.code}] {iss.name:15} | 売残比: {iss.sell_ratio:4.1f}% | 売残: {iss.sell_shares:10,.0f}株 (前日比 {chg_sign:>10}) | 信用倍率/取組: {iss.ratio:5.1f}%")

    print("\n==================================================")
    print("3. Fetching Market-wide Margin History (市場全体週次推移)")
    print("==================================================")
    market_records = collector.fetch_market_margin_history()
    print(f"Total Weekly Records: {len(market_records)}")
    if market_records:
        print("Latest 3 Weeks Market Total:")
        for rec in market_records[-3:]:
            print(f"  {rec.date}: 合計買残 {rec.total_buy_shares:10,.0f}千株 | 合計売残 {rec.total_sell_shares:10,.0f}千株 | 信用倍率 {rec.margin_ratio:4.2f}倍")

    print("\n[SUCCESS] All JPX Margin collections and extractions completed cleanly.")


if __name__ == "__main__":
    main()
