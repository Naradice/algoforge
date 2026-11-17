"""
Analyze holding time, recovery rate, and evaluate Time-Stop exits for mean-reversion trades.
Part of EXP-3 (Node 674).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from strategy.engine.analyst_strategy import get_forecast_records
from strategy.engine.rebound_features import compute_m30_indicators, predict_rebound_prob

output_dir = backend_dir / "artifacts" / "models" / "timestop_analysis"
output_dir.mkdir(parents=True, exist_ok=True)


def simulate_trajectories(
    symbol: str = "USDJPY",
    source: str = "FX168",
    sl_pips: float = 15.0,
    tp_pips: float = 40.0,
    tolerance_pips: float = 10.0,
    use_ml_filter: bool = True,
    ml_threshold: float = 0.50,
) -> list[dict]:
    m30_store_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"
    raw_df = pd.read_parquet(m30_store_path)
    if not isinstance(raw_df.index, pd.DatetimeIndex) and "time" in raw_df.columns:
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        raw_df.set_index("time", inplace=True)
    raw_df.rename(columns={c: str(c).lower().strip() for c in raw_df.columns}, inplace=True)

    df_ohlc = compute_m30_indicators(raw_df)
    df_ohlc["date_key"] = df_ohlc.index.date

    daily_ohlc = df_ohlc.groupby("date_key").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    df_fc = get_forecast_records(symbol, source=source)

    sl_yen = sl_pips * 0.01
    tp_yen = tp_pips * 0.01
    tol_yen = tolerance_pips * 0.01
    lot_multiplier = 100_000.0

    pub_time = datetime.strptime("00:30", "%H:%M").time()
    trajectories = []

    for idx, fc_row in df_fc.iterrows():
        fc_date = fc_row["date"]
        exp_low = float(fc_row["expected_low"])
        exp_high = float(fc_row["expected_high"])
        fc_source = str(fc_row.get("source") or "Report")

        if fc_date not in daily_ohlc.index:
            candidates = [d for d in daily_ohlc.index if fc_date <= d <= fc_date + timedelta(days=4)]
            if not candidates:
                continue
            act_date = candidates[0]
        else:
            act_date = fc_date

        day_bars = df_ohlc[df_ohlc["date_key"] == act_date]
        if day_bars.empty:
            continue

        last_bar_time = day_bars.index[-1]
        last_close_p = float(day_bars.iloc[-1]["close"])
        post_pub_bars = day_bars[day_bars.index.time >= pub_time]
        if post_pub_bars.empty:
            continue

        buy_entered = False
        sell_entered = False

        for b_idx, (b_time, b) in enumerate(post_pub_bars.iterrows()):
            b_high = float(b["high"])
            b_low = float(b["low"])

            # Buy Fade
            if not buy_entered and b_low <= (exp_low + tol_yen):
                if use_ml_filter:
                    prob = predict_rebound_prob(
                        bar_series=b,
                        direction="buy",
                        exp_low=exp_low,
                        exp_high=exp_high,
                        touch_price=b_low,
                        bar_time=b_time,
                    )
                    if prob < ml_threshold:
                        continue
                else:
                    prob = 0.5

                buy_entered = True
                entry_p = exp_low
                remaining = post_pub_bars.iloc[b_idx:]

                # Record bar-by-bar progression
                bar_count = 0
                hit_exit = False
                final_exit_p = min(exp_high, last_close_p)
                final_reason = "close"
                pnl_path = []

                for _, rb in remaining.iterrows():
                    bar_count += 1
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    rb_close = float(rb["close"])
                    cur_pnl = (rb_close - entry_p) * lot_multiplier
                    pnl_path.append(cur_pnl)

                    if rb_low <= (entry_p - sl_yen):
                        final_exit_p = entry_p - sl_yen
                        final_reason = "sl"
                        hit_exit = True
                        break
                    elif rb_high >= (entry_p + tp_yen):
                        final_exit_p = entry_p + tp_yen
                        final_reason = "tp"
                        hit_exit = True
                        break
                    elif rb_high >= exp_high:
                        final_exit_p = exp_high
                        final_reason = "tp_boundary"
                        hit_exit = True
                        break

                pnl = (final_exit_p - entry_p) * lot_multiplier
                trajectories.append({
                    "time": b_time,
                    "direction": "buy",
                    "entry_price": entry_p,
                    "final_pnl": pnl,
                    "final_reason": final_reason,
                    "bars_to_exit": bar_count,
                    "holding_hours": bar_count * 0.5,
                    "pnl_path": pnl_path,
                    "is_win": 1 if pnl > 0 else 0,
                    "ml_prob": prob,
                })

            # Sell Fade
            if not sell_entered and b_high >= (exp_high - tol_yen):
                if use_ml_filter:
                    prob = predict_rebound_prob(
                        bar_series=b,
                        direction="sell",
                        exp_low=exp_low,
                        exp_high=exp_high,
                        touch_price=b_high,
                        bar_time=b_time,
                    )
                    if prob < ml_threshold:
                        continue
                else:
                    prob = 0.5

                sell_entered = True
                entry_p = exp_high
                remaining = post_pub_bars.iloc[b_idx:]

                bar_count = 0
                hit_exit = False
                final_exit_p = max(exp_low, last_close_p)
                final_reason = "close"
                pnl_path = []

                for _, rb in remaining.iterrows():
                    bar_count += 1
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    rb_close = float(rb["close"])
                    cur_pnl = (entry_p - rb_close) * lot_multiplier
                    pnl_path.append(cur_pnl)

                    if rb_high >= (entry_p + sl_yen):
                        final_exit_p = entry_p + sl_yen
                        final_reason = "sl"
                        hit_exit = True
                        break
                    elif rb_low <= (entry_p - tp_yen):
                        final_exit_p = entry_p - tp_yen
                        final_reason = "tp"
                        hit_exit = True
                        break
                    elif rb_low <= exp_low:
                        final_exit_p = exp_low
                        final_reason = "tp_boundary"
                        hit_exit = True
                        break

                pnl = (entry_p - final_exit_p) * lot_multiplier
                trajectories.append({
                    "time": b_time,
                    "direction": "sell",
                    "entry_price": entry_p,
                    "final_pnl": pnl,
                    "final_reason": final_reason,
                    "bars_to_exit": bar_count,
                    "holding_hours": bar_count * 0.5,
                    "pnl_path": pnl_path,
                    "is_win": 1 if pnl > 0 else 0,
                    "ml_prob": prob,
                })

    return trajectories


def run_timestop_experiments():
    print("Simulating trade trajectories for FX168 (with and without ML Filter)...")
    base_trades = simulate_trajectories(use_ml_filter=False)
    ml_trades = simulate_trajectories(use_ml_filter=True, ml_threshold=0.50)

    print(f"\n--- 1. Holding Time Distribution (Base: {len(base_trades)} trades) ---")
    wins_base = [t for t in base_trades if t["is_win"]]
    losses_base = [t for t in base_trades if not t["is_win"]]
    print(f"Wins: {len(wins_base)}, Mean Holding Time: {np.mean([t['holding_hours'] for t in wins_base]):.1f}h (Median: {np.median([t['holding_hours'] for t in wins_base]):.1f}h)")
    print(f"Losses: {len(losses_base)}, Mean Holding Time: {np.mean([t['holding_hours'] for t in losses_base]):.1f}h (Median: {np.median([t['holding_hours'] for t in losses_base]):.1f}h)")

    print(f"\n--- 2. Holding Time Distribution (ML Filtered: {len(ml_trades)} trades) ---")
    wins_ml = [t for t in ml_trades if t["is_win"]]
    losses_ml = [t for t in ml_trades if not t["is_win"]]
    print(f"Wins: {len(wins_ml)}, Mean Holding Time: {np.mean([t['holding_hours'] for t in wins_ml]):.1f}h (Median: {np.median([t['holding_hours'] for t in wins_ml]):.1f}h)")
    if losses_ml:
        print(f"Losses: {len(losses_ml)}, Mean Holding Time: {np.mean([t['holding_hours'] for t in losses_ml]):.1f}h (Median: {np.median([t['holding_hours'] for t in losses_ml]):.1f}h)")

    # 3. Analyze recovery rate if negative at bar K
    print("\n--- 3. Recovery Rate when trade is underwater at Bar K ---")
    print(f"{'Bars elapsed':>12} | {'Time elapsed':>12} | {'Underwater count':>16} | {'Eventually Won':>14} | {'Recovery Rate%':>14}")
    print("-" * 75)
    for k in [2, 4, 6, 8, 10, 12]:
        underwater = [t for t in base_trades if len(t["pnl_path"]) >= k and t["pnl_path"][k-1] < 0]
        if underwater:
            won = [t for t in underwater if t["is_win"]]
            rec_rate = (len(won) / len(underwater)) * 100
            print(f"{k:>12} | {k*0.5:>10.1f}h | {len(underwater):>16} | {len(won):>14} | {rec_rate:>13.1f}%")

    # 4. Simulate Time-Stop Policies:
    # If a trade is underwater (pnl < 0) after max_bars, close it immediately at that bar's close price
    def test_timestop(trades_list: list[dict], max_bars: int | None):
        starting_capital = 1_000_000.0
        pnl_results = []
        hours_results = []

        for t in trades_list:
            path = t["pnl_path"]
            if max_bars is not None and len(path) >= max_bars and path[max_bars - 1] < 0:
                # Time Stop triggered: exit at bar max_bars close
                exit_pnl = path[max_bars - 1]
                pnl_results.append(exit_pnl)
                hours_results.append(max_bars * 0.5)
            else:
                pnl_results.append(t["final_pnl"])
                hours_results.append(t["holding_hours"])

        pnl_arr = np.array(pnl_results)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr < 0]
        wr = len(wins) / len(pnl_arr) if len(pnl_arr) > 0 else 0
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        pf = (gp / gl) if gl > 0 else 99.0

        cum = starting_capital + np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        max_dd = np.max(dd) * 100 if len(dd) > 0 else 0

        ret = pnl_arr / starting_capital
        sharpe = (np.mean(ret) / (np.std(ret) + 1e-9)) * np.sqrt(252)

        worst_loss = float(np.min(pnl_arr)) if len(pnl_arr) > 0 else 0

        return {
            "max_bars": max_bars if max_bars else "None (Hold to SL/TP)",
            "max_hours": f"{max_bars*0.5:.1f}h" if max_bars else "None",
            "trades": len(pnl_arr),
            "win_rate": round(wr * 100, 1),
            "pf": round(pf, 2),
            "total_pnl": round(float(pnl_arr.sum())),
            "worst_loss": round(worst_loss),
            "max_dd_pct": round(float(max_dd), 2),
            "mean_hold_h": round(float(np.mean(hours_results)), 1),
            "sharpe": round(float(sharpe), 2),
        }

    results = []
    # Baseline policies
    for mb in [None, 12, 8, 6, 4]:
        r = test_timestop(base_trades, mb)
        r["strategy"] = "FX168 Baseline (No ML)"
        results.append(r)

    # ML Filtered policies
    for mb in [None, 12, 8, 6, 4]:
        r = test_timestop(ml_trades, mb)
        r["strategy"] = "FX168 + ML Rebound Filter"
        results.append(r)

    df_res = pd.DataFrame(results)
    cols = ["strategy", "max_bars", "max_hours", "win_rate", "pf", "total_pnl", "worst_loss", "max_dd_pct", "mean_hold_h", "sharpe"]
    df_res = df_res[cols]

    print("\n==========================================================================================")
    print("TIME-STOP PERFORMANCE COMPARISON")
    print("==========================================================================================")
    print(df_res.to_string(index=False))
    print("==========================================================================================")

    res_file = output_dir / "timestop_results.json"
    df_res.to_json(res_file, orient="records", indent=2, force_ascii=False)
    print(f"Saved results to {res_file}")

    return df_res


if __name__ == "__main__":
    run_timestop_experiments()
