"""
Verify Standalone Mathematical Pivot vs FX168 Report and run Long-Term Multi-Year Backtest.
Part of EXP-4 (Node 678) and Goal 675.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from strategy.engine.analyst_strategy import get_forecast_records, compute_metrics
from strategy.engine.rebound_features import compute_m30_indicators, predict_rebound_prob
from strategy.engine.standalone_pivot import calculate_daily_pivots

output_dir = backend_dir / "artifacts" / "models" / "standalone_pivot"
output_dir.mkdir(parents=True, exist_ok=True)


def run_standalone_backtest(
    df_ohlc: pd.DataFrame,
    df_pivots: pd.DataFrame,
    use_ml_filter: bool = True,
    ml_threshold: float = 0.50,
    sl_pips: float = 15.0,
    tp_pips: float = 40.0,
    tolerance_pips: float = 10.0,
) -> tuple[list[dict], dict]:
    sl_yen = sl_pips * 0.01
    tp_yen = tp_pips * 0.01
    tol_yen = tolerance_pips * 0.01
    lot_multiplier = 100_000.0

    df_pivots_map = {row["date"]: row for _, row in df_pivots.iterrows()}
    trades = []
    pnl_list = []

    for date_key, day_bars in df_ohlc.groupby("date_key"):
        if date_key not in df_pivots_map:
            continue
        p_row = df_pivots_map[date_key]
        exp_low = float(p_row["expected_low"])
        exp_high = float(p_row["expected_high"])

        # Filter post 00:30 UTC
        post_pub_bars = day_bars[day_bars["hour"] >= 0]
        if post_pub_bars.empty:
            continue

        last_bar_time = day_bars.index[-1]
        last_close_p = float(day_bars.iloc[-1]["close"])

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
                exit_p = min(exp_high, last_close_p)
                exit_reason = "close"
                exit_time = last_bar_time

                for _, rb in remaining.iterrows():
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    if rb_low <= (entry_p - sl_yen):
                        exit_p = entry_p - sl_yen
                        exit_reason = "sl"
                        exit_time = rb.name
                        break
                    elif rb_high >= (entry_p + tp_yen):
                        exit_p = entry_p + tp_yen
                        exit_reason = "tp"
                        exit_time = rb.name
                        break
                    elif rb_high >= exp_high:
                        exit_p = exp_high
                        exit_reason = "tp_boundary"
                        exit_time = rb.name
                        break

                pnl = (exit_p - entry_p) * lot_multiplier
                pnl_list.append(pnl)
                trades.append({
                    "time": b_time,
                    "direction": "buy",
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "exit_reason": exit_reason,
                    "profit": pnl,
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
                exit_p = max(exp_low, last_close_p)
                exit_reason = "close"
                exit_time = last_bar_time

                for _, rb in remaining.iterrows():
                    rb_low = float(rb["low"])
                    rb_high = float(rb["high"])
                    if rb_high >= (entry_p + sl_yen):
                        exit_p = entry_p + sl_yen
                        exit_reason = "sl"
                        exit_time = rb.name
                        break
                    elif rb_low <= (entry_p - tp_yen):
                        exit_p = entry_p - tp_yen
                        exit_reason = "tp"
                        exit_time = rb.name
                        break
                    elif rb_low <= exp_low:
                        exit_p = exp_low
                        exit_reason = "tp_boundary"
                        exit_time = rb.name
                        break

                pnl = (entry_p - exit_p) * lot_multiplier
                pnl_list.append(pnl)
                trades.append({
                    "time": b_time,
                    "direction": "sell",
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "exit_reason": exit_reason,
                    "profit": pnl,
                    "ml_prob": prob,
                })

    metrics = compute_metrics(pnl_list)
    return trades, metrics


def main():
    print("Loading USDJPY M30 dataset...")
    m30_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"
    raw_df = pd.read_parquet(m30_path)
    if not isinstance(raw_df.index, pd.DatetimeIndex) and "time" in raw_df.columns:
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        raw_df.set_index("time", inplace=True)
    raw_df.rename(columns={c: str(c).lower().strip() for c in raw_df.columns}, inplace=True)

    print("Computing technical indicators on M30...")
    df_ohlc = compute_m30_indicators(raw_df)
    df_ohlc["date_key"] = df_ohlc.index.date
    df_ohlc["hour"] = df_ohlc.index.hour

    # 1. Compute Standalone Pivots
    print("Computing Standalone Daily Pivots from historical bars...")
    df_pivots_all = calculate_daily_pivots(raw_df, pivot_type="classic")
    print(f"Total Standalone Pivot Days computed: {len(df_pivots_all)}")

    # 2. Check Match against FX168 Report (65 days)
    df_fc_fx168 = get_forecast_records("USDJPY", source="FX168")
    fx_dates = set(df_fc_fx168["date"])
    df_pivots_fx_period = df_pivots_all[df_pivots_all["date"].isin(fx_dates)].copy()

    merged = pd.merge(df_fc_fx168, df_pivots_fx_period, on="date", how="inner", suffixes=("_report", "_calc"))
    mae_s1 = ((merged["expected_low_report"] - merged["S1"]).abs() / 0.01).mean()
    mae_r1 = ((merged["expected_high_report"] - merged["R1"]).abs() / 0.01).mean()
    print(f"\n--- Equivalence Check (FX168 Report vs Standalone Classic Pivot) ---")
    print(f"Matching days: {len(merged)}")
    print(f"MAE Support 1:    {mae_s1:.4f} pips")
    print(f"MAE Resistance 1: {mae_r1:.4f} pips")

    # 3. Backtest on the 65-day Report Period
    print("\n--- Running Backtest on Report Period (65 days) ---")
    _, m_fx_base = run_standalone_backtest(df_ohlc, df_pivots_fx_period, use_ml_filter=False)
    _, m_fx_ml = run_standalone_backtest(df_ohlc, df_pivots_fx_period, use_ml_filter=True, ml_threshold=0.50)

    # 4. Long-Term Multi-Year Backtest on Full Dataset (677 days)
    print("\n--- Running Multi-Year Backtest on Full Intraday History (2.5 Years / 677 days) ---")
    trades_long_base, m_long_base = run_standalone_backtest(df_ohlc, df_pivots_all, use_ml_filter=False)
    trades_long_ml, m_long_ml = run_standalone_backtest(df_ohlc, df_pivots_all, use_ml_filter=True, ml_threshold=0.50)

    # Also test Fibonacci and Camarilla pivots for comparison
    df_fib = calculate_daily_pivots(raw_df, pivot_type="fibonacci")
    _, m_fib_ml = run_standalone_backtest(df_ohlc, df_fib, use_ml_filter=True, ml_threshold=0.50)

    df_cam = calculate_daily_pivots(raw_df, pivot_type="camarilla")
    _, m_cam_ml = run_standalone_backtest(df_ohlc, df_cam, use_ml_filter=True, ml_threshold=0.50)

    summary_rows = [
        {
            "scope": "Report Period (65d)",
            "strategy": "FX168 Report (Strategy 7 Baseline)",
            "trades": 62, "win_rate": 51.6, "pf": 2.80, "total_pnl": 781900, "max_dd_pct": 3.17, "sharpe": 7.49
        },
        {
            "scope": "Report Period (65d)",
            "strategy": "Standalone Classic Pivot (No ML)",
            "trades": int(m_fx_base["total_trades"]), "win_rate": round(m_fx_base["win_rate"]*100, 1),
            "pf": round(m_fx_base["profit_factor"], 2), "total_pnl": round(m_fx_base["total_pnl"]),
            "max_dd_pct": round(m_fx_base["max_drawdown"]*100, 2), "sharpe": round(m_fx_base["sharpe_ratio"], 2)
        },
        {
            "scope": "Report Period (65d)",
            "strategy": "Standalone Classic Pivot + ML Filter (prob >= 0.50)",
            "trades": int(m_fx_ml["total_trades"]), "win_rate": round(m_fx_ml["win_rate"]*100, 1),
            "pf": round(m_fx_ml["profit_factor"], 2), "total_pnl": round(m_fx_ml["total_pnl"]),
            "max_dd_pct": round(m_fx_ml["max_drawdown"]*100, 2), "sharpe": round(m_fx_ml["sharpe_ratio"], 2)
        },
        {
            "scope": "Full 2.5 Years (677d)",
            "strategy": "Standalone Classic Pivot (No ML)",
            "trades": int(m_long_base["total_trades"]), "win_rate": round(m_long_base["win_rate"]*100, 1),
            "pf": round(m_long_base["profit_factor"], 2), "total_pnl": round(m_long_base["total_pnl"]),
            "max_dd_pct": round(m_long_base["max_drawdown"]*100, 2), "sharpe": round(m_long_base["sharpe_ratio"], 2)
        },
        {
            "scope": "Full 2.5 Years (677d)",
            "strategy": "Standalone Classic Pivot + ML Filter (prob >= 0.50)",
            "trades": int(m_long_ml["total_trades"]), "win_rate": round(m_long_ml["win_rate"]*100, 1),
            "pf": round(m_long_ml["profit_factor"], 2), "total_pnl": round(m_long_ml["total_pnl"]),
            "max_dd_pct": round(m_long_ml["max_drawdown"]*100, 2), "sharpe": round(m_long_ml["sharpe_ratio"], 2)
        },
        {
            "scope": "Full 2.5 Years (677d)",
            "strategy": "Standalone Fibonacci Pivot + ML Filter",
            "trades": int(m_fib_ml["total_trades"]), "win_rate": round(m_fib_ml["win_rate"]*100, 1),
            "pf": round(m_fib_ml["profit_factor"], 2), "total_pnl": round(m_fib_ml["total_pnl"]),
            "max_dd_pct": round(m_fib_ml["max_drawdown"]*100, 2), "sharpe": round(m_fib_ml["sharpe_ratio"], 2)
        },
        {
            "scope": "Full 2.5 Years (677d)",
            "strategy": "Standalone Camarilla Pivot + ML Filter",
            "trades": int(m_cam_ml["total_trades"]), "win_rate": round(m_cam_ml["win_rate"]*100, 1),
            "pf": round(m_cam_ml["profit_factor"], 2), "total_pnl": round(m_cam_ml["total_pnl"]),
            "max_dd_pct": round(m_cam_ml["max_drawdown"]*100, 2), "sharpe": round(m_cam_ml["sharpe_ratio"], 2)
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    print("\n==========================================================================================")
    print("STANDALONE PIVOT STRATEGY: EQUIVALENCE & 2.5-YEAR LONG-TERM PERFORMANCE")
    print("==========================================================================================")
    print(df_summary.to_string(index=False))
    print("==========================================================================================")

    res_file = output_dir / "standalone_pivot_results.json"
    df_summary.to_json(res_file, orient="records", indent=2, force_ascii=False)
    print(f"Saved results to {res_file}")

    return df_summary


if __name__ == "__main__":
    main()
