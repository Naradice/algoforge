"""Enhanced Supply/Demand Swing Strategy with Trend Alignment Filter."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.run_nikkei_swing_optimization import (
    SELECTED_SYMBOLS,
    download_3y_data,
)


def backtest_enhanced(
    df: pd.DataFrame,
    sl_pct: float = 0.03,
    tp_pct: float = 0.08,
    max_bars: int = 15,
    vol_mult: float = 1.3,
    use_trend_filter: bool = True,
    slippage_pct: float = 0.001,
    initial_capital: float = 1_000_000.0,
) -> dict:
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_ = df["Open"].values
    volume = df["Volume"].values
    dates = df.index
    n = len(df)

    sma20 = pd.Series(close).rolling(20).mean().values
    sma50 = pd.Series(close).rolling(50).mean().values
    sma5 = pd.Series(close).rolling(5).mean().values
    vol_sma20 = pd.Series(volume).rolling(20).mean().values

    # RSI
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    trades = []
    equity = initial_capital
    position = None

    for i in range(55, n):
        cur_date = dates[i]
        c_open = open_[i]
        c_high = high[i]
        c_low = low[i]
        c_close = close[i]

        if position is not None:
            bars_held = i - position["entry_bar"]
            exit_price = None
            exit_reason = None

            if c_low <= position["sl"]:
                exit_price = position["sl"] * (1 - slippage_pct)
                exit_reason = "SL"
            elif c_high >= position["tp"]:
                exit_price = position["tp"] * (1 - slippage_pct)
                exit_reason = "TP"
            elif bars_held >= 3 and c_close < sma5[i] and c_close < position["entry_price"]:
                exit_price = c_close * (1 - slippage_pct)
                exit_reason = "SMA5 Cut"
            elif bars_held >= max_bars:
                exit_price = c_close * (1 - slippage_pct)
                exit_reason = "Time Stop"

            if exit_price is not None:
                profit = (exit_price - position["entry_price"]) * position["shares"]
                profit_pct = (exit_price / position["entry_price"] - 1.0) * 100.0
                equity += profit
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": cur_date,
                    "profit": round(profit, 1),
                    "profit_pct": round(profit_pct, 2),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                position = None

        if position is None:
            prev_c = close[i - 1]
            prev_c2 = close[i - 2]
            prev_sma20 = sma20[i - 1]
            prev_sma50 = sma50[i - 1]
            prev_v = volume[i - 1]
            prev_vsma = vol_sma20[i - 1]
            prev_rsi = rsi[i - 1]

            # Inflow volume spike
            has_vol = prev_v >= (vol_mult * prev_vsma)
            # Breakout
            price_break = (prev_c > prev_sma20) and (prev_c2 <= sma20[i - 2] or prev_c > close[i-15:i-1].max() * 0.99)
            # Healthy RSI
            rsi_ok = 45.0 <= prev_rsi <= 68.0

            # Trend alignment: Stock must be above upward-sloping 50-day SMA (eliminates downtrend traps)
            trend_ok = True
            if use_trend_filter:
                sma20_slope_up = sma20[i - 1] > sma20[i - 5]
                above_50 = prev_c > prev_sma50
                trend_ok = sma20_slope_up and above_50

            if has_vol and price_break and rsi_ok and trend_ok:
                fill = c_open * (1 + slippage_pct)
                shares = math.floor(equity / fill)
                if shares > 0:
                    position = {
                        "entry_price": fill,
                        "entry_bar": i,
                        "entry_date": cur_date,
                        "shares": shares,
                        "sl": fill * (1.0 - sl_pct),
                        "tp": fill * (1.0 + tp_pct),
                    }

    total_trades = len(trades)
    wins = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = equity - initial_capital
    roi = (total_pnl / initial_capital) * 100.0
    gw = sum(t["profit"] for t in wins)
    gl = abs(sum(t["profit"] for t in losses))
    pf = (gw / gl) if gl > 0 else 0.0

    return {
        "trades": trades,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "roi_pct": round(roi, 2),
        "total_pnl": round(total_pnl, 0),
    }


def main():
    data = download_3y_data(SELECTED_SYMBOLS)

    # Grid search with trend filter
    best_pf = 0
    best_res = None
    best_cfg = ()

    for sl in [0.025, 0.03, 0.04]:
        for tp in [0.06, 0.08, 0.10, 0.12]:
            for mb in [10, 15, 20]:
                port_roi = 0
                gw_tot = 0
                gl_tot = 0
                tr_count = 0
                for c, df in data.items():
                    res = backtest_enhanced(df, sl_pct=sl, tp_pct=tp, max_bars=mb, vol_mult=1.3)
                    port_roi += res["roi_pct"]
                    tr_count += res["total_trades"]
                    gw_tot += sum(t["profit"] for t in res["trades"] if t["profit"] > 0)
                    gl_tot += abs(sum(t["profit"] for t in res["trades"] if t["profit"] <= 0))

                pf = (gw_tot / gl_tot) if gl_tot > 0 else 0
                if pf > best_pf and tr_count >= 40:
                    best_pf = pf
                    best_cfg = (sl, tp, mb)

    sl, tp, mb = best_cfg
    print(f"\n========================================================")
    print(f"ENHANCED MODEL (Trend-Filtered Supply/Demand Swing):")
    print(f"Best Config: SL={sl*100:.1f}%, TP={tp*100:.1f}%, MaxBars={mb}")
    print(f"========================================================")

    tot_pnl = 0
    tot_trades = 0
    tot_wins = 0

    print(f"{'Code':<6} {'Name':<24} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'ROI (%)':<12} {'Total PnL':<15}")
    print("-" * 90)
    for c, df in data.items():
        res = backtest_enhanced(df, sl_pct=sl, tp_pct=tp, max_bars=mb, vol_mult=1.3)
        name = SELECTED_SYMBOLS.get(c, c)
        tot_pnl += res["total_pnl"]
        tot_trades += res["total_trades"]
        tot_wins += len([t for t in res["trades"] if t["profit"] > 0])
        print(f"{c:<6} {name:<24} {res['total_trades']:<8} {res['win_rate']:<9.1f}% {res['profit_factor']:<8.2f} {res['roi_pct']:+10.2f}% {res['total_pnl']:+13,.0f}円")

    port_ret = tot_pnl / 10_000_000 * 100
    print("=" * 90)
    print(f"PORTFOLIO TOTAL: {tot_pnl:+,.0f}円 ({port_ret:+.2f}%) | Trades: {tot_trades} | Win Rate: {tot_wins/tot_trades*100:.1f}%")


if __name__ == "__main__":
    main()
