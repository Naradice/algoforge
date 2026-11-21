"""Analyze timing of breakout and reversion entries against report publication times using M30 data."""

import datetime
from pathlib import Path
import pandas as pd
import numpy as np

def run_timing_analysis():
    # 1. Load M30 candles
    m30_path = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\artifacts\datasets\USDJPY_M30_latest.parquet")
    df_m30 = pd.read_parquet(m30_path)
    if not isinstance(df_m30.index, pd.DatetimeIndex):
        df_m30["time"] = pd.to_datetime(df_m30["time"], utc=True)
        df_m30.set_index("time", inplace=True)
    df_m30.sort_index(inplace=True)

    # 2. Load cached report forecasts
    csv_path = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\data\reports_cache\backtest_detailed_results.csv")
    df_reports = pd.read_csv(csv_path)
    df_reports["date"] = pd.to_datetime(df_reports["date"]).dt.date

    print(f"Loaded {len(df_m30)} M30 bars and {len(df_reports)} report rows.")

    # Analysis results
    # Assumed publication times in JST (and UTC):
    # Sony: 09:30 JST (00:30 UTC)
    # SMBC: 09:30 JST (00:30 UTC)
    # Minna: 08:30 JST (23:30 UTC of previous day / 00:00 UTC)
    # Standard conservative assumption: 09:30 JST (00:30 UTC)
    PUB_TIME_UTC = datetime.time(0, 30)  # 09:30 JST

    total_daily_breakouts = 0
    valid_after_pub = 0
    invalid_premature_only = 0
    premature_but_continued = 0

    premature_trades = []

    for _, r in df_reports.iterrows():
        r_date = r["date"]
        exp_low = float(r["expected_low"])
        exp_high = float(r["expected_high"])
        source = r.get("source", "Report")

        # Get all M30 bars for that day
        # In FX, trading day typically starts at 00:00 UTC or 21:00 UTC
        # Let's inspect bars for r_date
        day_bars = df_m30[df_m30.index.date == r_date]
        if day_bars.empty:
            continue

        # Separate pre-publication bars (before 00:30 UTC) and post-publication bars (00:30 UTC onwards)
        pre_pub_bars = day_bars[day_bars.index.time < PUB_TIME_UTC]
        post_pub_bars = day_bars[day_bars.index.time >= PUB_TIME_UTC]

        # Daily full range
        daily_high = day_bars["high"].max()
        daily_low = day_bars["low"].min()

        has_high_break_daily = daily_high > exp_high
        has_low_break_daily = daily_low < exp_low

        for break_type, daily_break, trigger_val, cmp_op in [
            ("HIGH_BREAK", has_high_break_daily, exp_high, lambda b, v: b["high"] > v),
            ("LOW_BREAK", has_low_break_daily, exp_low, lambda b, v: b["low"] < v),
        ]:
            if not daily_break:
                continue

            total_daily_breakouts += 1

            pre_broke = any(cmp_op(b, trigger_val) for _, b in pre_pub_bars.iterrows()) if not pre_pub_bars.empty else False
            post_broke = any(cmp_op(b, trigger_val) for _, b in post_pub_bars.iterrows()) if not post_pub_bars.empty else False

            # Find first break bar
            first_break_time = None
            for bar_time, bar in day_bars.iterrows():
                if cmp_op(bar, trigger_val):
                    first_break_time = bar_time
                    break

            first_break_jst = first_break_time + datetime.timedelta(hours=9) if first_break_time else None

            if pre_broke and not post_broke:
                # BREAK ONLY HAPPENED BEFORE 09:30 JST! (Invalid / Lookahead)
                invalid_premature_only += 1
                premature_trades.append({
                    "date": r_date,
                    "source": source,
                    "type": break_type,
                    "level": trigger_val,
                    "first_break_jst": str(first_break_jst),
                    "status": "Lookahead Error: Broke ONLY before 09:30 JST",
                })
            elif pre_broke and post_broke:
                # Started before 09:30 JST, but also broke after 09:30 JST
                premature_but_continued += 1
            else:
                # Clean break: Happened after 09:30 JST
                valid_after_pub += 1

    # Also test Range Reversion (Fade High / Fade Low)
    total_reversion = 0
    valid_rev_after_pub = 0
    premature_rev_only = 0
    premature_rev_persisted = 0

    for _, r in df_reports.iterrows():
        r_date = r["date"]
        exp_low = float(r["expected_low"])
        exp_high = float(r["expected_high"])
        source = r.get("source", "Report")

        day_bars = df_m30[df_m30.index.date == r_date]
        if day_bars.empty:
            continue

        pre_pub_bars = day_bars[day_bars.index.time < PUB_TIME_UTC]
        post_pub_bars = day_bars[day_bars.index.time >= PUB_TIME_UTC]

        daily_high = day_bars["high"].max()
        daily_low = day_bars["low"].min()

        # Buy fade (low dipped to or below exp_low)
        # Sell fade (high reached or exceeded exp_high)
        for rev_type, daily_touch, trigger_val, cmp_op in [
            ("FADE_LOW", daily_low <= exp_low, exp_low, lambda b, v: b["low"] <= v),
            ("FADE_HIGH", daily_high >= exp_high, exp_high, lambda b, v: b["high"] >= v),
        ]:
            if not daily_touch:
                continue

            total_reversion += 1

            pre_touched = any(cmp_op(b, trigger_val) for _, b in pre_pub_bars.iterrows()) if not pre_pub_bars.empty else False
            post_touched = any(cmp_op(b, trigger_val) for _, b in post_pub_bars.iterrows()) if not post_pub_bars.empty else False

            if pre_touched and not post_touched:
                premature_rev_only += 1
            elif pre_touched and post_touched:
                premature_rev_persisted += 1
            else:
                valid_rev_after_pub += 1

    print("\n=======================================================")
    print("=== MEAN-REVERSION TIMING VERIFICATION (M30 Analysis) ===")
    print("=======================================================")
    print(f"Total Daily Reversion Signals: {total_reversion}")
    print(f"1. Valid (First touched AFTER 09:30 JST): {valid_rev_after_pub} ({valid_rev_after_pub/total_reversion*100:.1f}%)")
    print(f"2. Early touch and ALSO touched AFTER 09:30 JST: {premature_rev_persisted} ({premature_rev_persisted/total_reversion*100:.1f}%)")
    print(f"3. Premature ONLY (Touched BEFORE 09:30 JST, never touched after): {premature_rev_only} ({premature_rev_only/total_reversion*100:.1f}%)")
    print("=======================================================\n")

    print("=======================================================")
    print(f"Total Daily Breakout Signals: {total_daily_breakouts}")
    print(f"1. Valid (First broke AFTER 09:30 JST): {valid_after_pub} ({valid_after_pub/total_daily_breakouts*100:.1f}%)")
    print(f"2. Early break but persisted AFTER 09:30 JST: {premature_but_continued} ({premature_but_continued/total_daily_breakouts*100:.1f}%)")
    print(f"3. Premature ONLY (Broke BEFORE 09:30 JST, never broke after): {invalid_premature_only} ({invalid_premature_only/total_daily_breakouts*100:.1f}%)")
    print("=======================================================\n")

    if premature_trades:
        print(f"Sample of Invalid Lookahead Trades ({len(premature_trades)} total):")
        for pt in premature_trades[:10]:
            print(f"  {pt['date']} | {pt['source']} | {pt['type']} @ {pt['level']} | First broke: {pt['first_break_jst']}")

if __name__ == "__main__":
    run_timing_analysis()
