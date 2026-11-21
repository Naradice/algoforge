"""Live screening demo using actual JPX & EDINET swing trading datasets."""

from __future__ import annotations

import datetime
from pathlib import Path
import sys

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from database import SessionLocal
from data.swing_data_service import (
    run_daily_margin_collection,
    run_large_holding_collection,
)
from strategy.engine.swing_signals import screen_swing_signals

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("==================================================")
    print("1. Running automated data collection into database")
    print("==================================================")
    margin_res = run_daily_margin_collection()
    print(f"Daily Margin Collection: {margin_res['balances_saved']} balances, {margin_res['announcements_saved']} regulations")

    print("\n==================================================")
    print("2. Running Swing Trading Screening Engine")
    print("==================================================")
    with SessionLocal() as db:
        signals = screen_swing_signals(db, min_confidence=0.55)

    print(f"Total Actionable Signals Generated: {len(signals)}")

    # Group by signal type
    by_type = {}
    for s in signals:
        by_type.setdefault(s.signal_type, []).append(s)

    for stype, sigs in by_type.items():
        print(f"\n>>> [{stype.upper()}] ({len(sigs)} issues):")
        for s in sigs[:5]:
            print(f"  [{s.direction}] {s.code:6} {s.name:15} | Score: {s.confidence_score*100:.0f}%")
            print(f"      Title: {s.title}")
            for r in s.reasons:
                print(f"      - {r}")


if __name__ == "__main__":
    main()
