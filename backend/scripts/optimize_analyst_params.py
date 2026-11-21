"""
Analyst Strategy Parameter Optimizer (Grid Search & Visualization).

Searches optimal parameters for Analyst Range Reversion / Breakout:
- Stop Loss (SL pips): None, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150
- Take Profit (TP pips): Opposite boundary or fixed pips
- Tolerance pips: 0, 5, 10
- Report Source: SMBC, all, minfx

Outputs:
1. Top 3 Best Parameter Combinations (by Total PnL, Profit Factor, Sharpe).
2. ASCII / Markdown Chart: Stop Loss Pips vs Total PnL & Win Rate.
3. Generative UI HTML visualization report.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from database import SessionLocal
from data.models import ReportForecast


def load_data(symbol="USDJPY"):
    # Load MT5 parquet
    parquet_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "datasets" / f"{symbol}_D1_latest.parquet"
    if not parquet_path.exists():
        parquet_path = Path(__file__).resolve().parent.parent / "artifacts" / "datasets" / f"{symbol}_D1_latest.parquet"
    
    df_ohlc = pd.read_parquet(parquet_path)
    df_ohlc.index = pd.to_datetime(df_ohlc.index, utc=True)
    df_ohlc["date_key"] = df_ohlc.index.date
    daily_ohlc = df_ohlc.groupby("date_key").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    # Load ReportForecast
    forecast_rows = []
    with SessionLocal() as db:
        q = db.query(ReportForecast).filter(
            ReportForecast.symbol == symbol,
            ReportForecast.expected_low.isnot(None),
            ReportForecast.expected_high.isnot(None),
        ).order_by(ReportForecast.report_date.asc())
        for rf in q.all():
            forecast_rows.append({
                "date": rf.report_date,
                "expected_low": float(rf.expected_low),
                "expected_high": float(rf.expected_high),
                "source": rf.source or "SMBC",
                "horizon": rf.horizon or "Daily",
            })
    df_fc = pd.DataFrame(forecast_rows)
    return daily_ohlc, df_fc


def simulate_strategy(
    daily_ohlc: pd.DataFrame,
    df_fc: pd.DataFrame,
    sl_pips: float | None = None,
    tp_pips: float | None = None,
    tolerance_pips: float = 0.0,
    source: str = "SMBC",
    starting_capital: float = 1_000_000.0,
) -> dict:
    lot_multiplier = 100_000.0
    tol_yen = tolerance_pips * 0.01
    sl_yen = (sl_pips * 0.01) if sl_pips is not None else None
    tp_yen = (tp_pips * 0.01) if tp_pips is not None else None

    fc_subset = df_fc.copy()
    if source != "all":
        fc_subset = fc_subset[fc_subset["source"].str.contains(source, case=False, na=False)]
    fc_subset = fc_subset.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    pnl_list = []
    trades_count = 0

    for _, fc in fc_subset.iterrows():
        fc_date = fc["date"]
        exp_low = float(fc["expected_low"])
        exp_high = float(fc["expected_high"])

        if fc_date not in daily_ohlc.index:
            candidates = [d for d in daily_ohlc.index if fc_date <= d <= fc_date + timedelta(days=4)]
            if not candidates:
                continue
            act_date = candidates[0]
        else:
            act_date = fc_date

        bar = daily_ohlc.loc[act_date]
        low_p = float(bar["low"])
        high_p = float(bar["high"])
        close_p = float(bar["close"])

        # 1. Buy at exp_low (Fade lower bound)
        if low_p <= (exp_low + tol_yen):
            entry_p = exp_low
            trades_count += 1
            # Check Stop Loss
            if sl_yen is not None and (entry_p - low_p) >= sl_yen:
                exit_p = entry_p - sl_yen
            else:
                target_p = min(exp_high, entry_p + tp_yen) if tp_yen is not None else exp_high
                exit_p = min(target_p, close_p)
            pnl = (exit_p - entry_p) * lot_multiplier
            pnl_list.append(pnl)

        # 2. Sell at exp_high (Fade upper bound)
        if high_p >= (exp_high - tol_yen):
            entry_p = exp_high
            trades_count += 1
            # Check Stop Loss
            if sl_yen is not None and (high_p - entry_p) >= sl_yen:
                exit_p = entry_p + sl_yen
            else:
                target_p = max(exp_low, entry_p - tp_yen) if tp_yen is not None else exp_low
                exit_p = max(target_p, close_p)
            pnl = (entry_p - exit_p) * lot_multiplier
            pnl_list.append(pnl)

    if not pnl_list:
        return {
            "trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "pf": 0.0,
            "max_dd": 0.0, "sharpe": 0.0,
        }

    arr = np.array(pnl_list)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = len(wins) / len(arr) if len(arr) > 0 else 0.0
    gp = float(wins.sum()) if len(wins) > 0 else 0.0
    gl = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    total_pnl = float(arr.sum())

    cum = starting_capital + np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    ret = arr / starting_capital
    mean_r = np.mean(ret)
    std_r = np.std(ret) if len(ret) > 1 else 0.0
    sharpe = float((mean_r / std_r) * math.sqrt(252)) if std_r > 0 else 0.0

    return {
        "trades": len(arr),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "pf": pf,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


def optimize():
    print("Loading market & forecast data...")
    daily_ohlc, df_fc = load_data("USDJPY")
    print(f"Loaded {len(daily_ohlc)} daily OHLC bars and {len(df_fc)} forecasts.")

    sl_options = [None, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150]
    tp_options = [None, 50, 80, 100, 150]  # None = Exit at opposite boundary / close
    sources = ["SMBC", "all"]

    results = []

    print("\nRunning Parameter Grid Search...")
    for src in sources:
        for sl in sl_options:
            for tp in tp_options:
                res = simulate_strategy(daily_ohlc, df_fc, sl_pips=sl, tp_pips=tp, source=src)
                results.append({
                    "source": src,
                    "sl_pips": "None" if sl is None else sl,
                    "tp_pips": "Opposite" if tp is None else tp,
                    "trades": res["trades"],
                    "win_rate": res["win_rate"],
                    "total_pnl": res["total_pnl"],
                    "pf": res["pf"],
                    "max_dd": res["max_dd"],
                    "sharpe": res["sharpe"],
                })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

    print("\n=======================================================")
    print("[TOP 3 BEST PARAMETER COMBINATIONS (By Total PnL)]")
    print("=======================================================")
    for rank in range(min(3, len(res_df))):
        r = res_df.iloc[rank]
        print(f"\n[Rank {rank + 1}]")
        print(f"  Source:      {r['source']}")
        print(f"  Stop Loss:   {r['sl_pips']} pips")
        print(f"  Take Profit: {r['tp_pips']}")
        print(f"  Trades:      {r['trades']}")
        print(f"  Win Rate:    {r['win_rate']*100:.1f}%")
        print(f"  Total PnL:   {r['total_pnl']:+,.0f} JPY (+{r['total_pnl']/100000.0:+.2f} Yen)")
        print(f"  Profit Fac:  {r['pf']:.2f}")
        print(f"  Max DD:      {r['max_dd']*100:.1f}%")
        print(f"  Sharpe:      {r['sharpe']:.2f}")

    # Generate SL vs PnL chart table for SMBC (with TP=Opposite)
    smbc_sl_curve = res_df[(res_df["source"] == "SMBC") & (res_df["tp_pips"] == "Opposite")].copy()
    # Sort by numerical SL (None at end)
    def sl_sort_key(val):
        return 9999 if val == "None" else int(val)
    smbc_sl_curve["sl_num"] = smbc_sl_curve["sl_pips"].map(sl_sort_key)
    smbc_sl_curve = smbc_sl_curve.sort_values("sl_num")

    print("\n=======================================================")
    print("[Parameter Curve: Stop Loss (SL) vs Total PnL]")
    print("=======================================================")
    print(f"{'SL (pips)':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Total PnL (JPY)':<16} | {'Profit Factor':<14} | {'Visual PnL Bar'}")
    print("-" * 80)
    for _, row in smbc_sl_curve.iterrows():
        sl_str = f"{row['sl_pips']} pips" if row['sl_pips'] != "None" else "No SL"
        wr_str = f"{row['win_rate']*100:.1f}%"
        pnl_val = row['total_pnl']
        pnl_str = f"{pnl_val:+,.0f} JPY"
        pf_str = f"{row['pf']:.2f}"
        
        # ASCII Visual Bar
        bar_len = int(abs(pnl_val) / 50000)
        if pnl_val >= 0:
            bar = "[+] " + "#" * min(bar_len, 25)
        else:
            bar = "[-] " + "#" * min(bar_len, 25)
        print(f"{sl_str:<10} | {row['trades']:<8} | {wr_str:<10} | {pnl_str:<16} | {pf_str:<14} | {bar}")

    # Save results to CSV for UI/investigation
    out_csv = backend_root / "data" / "reports_cache" / "param_optimization_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved optimization full results to: {out_csv}")


if __name__ == "__main__":
    optimize()
