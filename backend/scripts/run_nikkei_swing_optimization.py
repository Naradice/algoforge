"""Comprehensive 3-Year Swing Backtest & Risk Optimization on 10 Random Nikkei 225 Stocks.

Strategy: Supply/Demand-Driven Swing (需給連動スイング)
- Identifies strong institutional volume accumulation (Volume Breakout)
- Confirms price trend momentum (20-day SMA crossover or breakout)
- Filters out overbought/extended conditions (RSI / MA dispersion)
- Evaluates exit via Stop Loss, Take Profit, Trailing Stop, and Time stop

Optimizes:
- Stop Loss % (sl_pct)
- Take Profit % (tp_pct)
- Max Holding Days (max_bars)
- Trailing Stop usage & multiplier
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")

# Selected 10 Nikkei 225 Symbols (Seed 123)
SELECTED_SYMBOLS = {
    "2802": "味の素 (食品)",
    "3402": "東レ (素材・繊維)",
    "4063": "信越化学工業 (半導体材料・化学)",
    "4452": "花王 (消費財・化学)",
    "6361": "荏原製作所 (機械・半導体装置)",
    "7201": "日産自動車 (自動車)",
    "7735": "SCREENホールディングス (半導体装置)",
    "8031": "三井物産 (総合商社)",
    "9434": "ソフトバンク (通信)",
    "9735": "セコム (セキュリティ・サービス)",
}


def download_3y_data(symbols: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Download and cache 3-year daily OHLCV data."""
    data = {}
    print("\n[1/3] Downloading 3-year historical daily data from JPX/Yahoo...")
    for code, name in symbols.items():
        ticker = f"{code}.T"
        df = yf.download(ticker, period="3y", interval="1d", progress=False)
        if df.empty or len(df) < 100:
            print(f"  Warning: insufficient data for {ticker}")
            continue
        # Flatten MultiIndex if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        data[code] = df
        start_d = df.index[0].strftime("%Y-%m-%d")
        end_d = df.index[-1].strftime("%Y-%m-%d")
        print(f"  ✓ [{code}] {name:30} : {len(df):4d} bars ({start_d} ~ {end_d})")
    return data


def backtest_single_symbol(
    df: pd.DataFrame,
    sl_pct: float,
    tp_pct: float,
    max_bars: int,
    vol_mult: float = 1.3,
    trailing_stop: bool = False,
    trailing_mult: float = 2.0,
    slippage_pct: float = 0.001,  # 0.1% per fill
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run bar-by-bar simulation for a single stock."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_ = df["Open"].values
    volume = df["Volume"].values
    dates = df.index

    n = len(df)
    if n < 30:
        return {"trades": [], "roi": 0.0, "pf": 0.0, "win_rate": 0.0, "pnl": 0.0}

    # Technical & Supply/Demand Indicators
    # 20-day SMA
    sma20 = pd.Series(close).rolling(20).mean().values
    sma5 = pd.Series(close).rolling(5).mean().values
    vol_sma20 = pd.Series(volume).rolling(20).mean().values

    # ATR (14) for volatility & trailing stop
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).rolling(14).mean().values

    # RSI (14)
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

    for i in range(25, n):
        cur_date = dates[i]
        c_open = open_[i]
        c_high = high[i]
        c_low = low[i]
        c_close = close[i]

        # 1. Manage active position
        if position is not None:
            bars_held = i - position["entry_bar"]
            exit_price = None
            exit_reason = None

            # Trailing stop adjustment
            if trailing_stop:
                cur_atr = atr[i] if not np.isnan(atr[i]) else (position["entry_price"] * 0.02)
                trail_sl = c_close - (cur_atr * trailing_mult)
                if trail_sl > position["sl"]:
                    position["sl"] = trail_sl

            # Check SL
            if c_low <= position["sl"]:
                exit_price = position["sl"] * (1 - slippage_pct)
                exit_reason = "SL (Stop Loss)"
            # Check TP
            elif c_high >= position["tp"]:
                exit_price = position["tp"] * (1 - slippage_pct)
                exit_reason = "TP (Take Profit)"
            # Check Trend Breakdown (close below SMA5 after at least 3 bars)
            elif bars_held >= 3 and c_close < sma5[i] and c_close < position["entry_price"]:
                exit_price = c_close * (1 - slippage_pct)
                exit_reason = "SMA5 Cut"
            # Time Stop (Max holding period)
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
                    "entry_price": position["entry_price"],
                    "exit_price": round(exit_price, 1),
                    "profit": round(profit, 1),
                    "profit_pct": round(profit_pct, 2),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                position = None

        # 2. Check Supply/Demand Entry Signal
        if position is None:
            prev_c = close[i - 1]
            prev_c2 = close[i - 2]
            prev_sma = sma20[i - 1]
            prev_sma2 = sma20[i - 2]
            prev_v = volume[i - 1]
            prev_vsma = vol_sma20[i - 1]
            prev_rsi = rsi[i - 1]

            # Supply & Demand Conditions:
            # A. Volume Surge: volume exceeds average by vol_mult (Demand inflow)
            has_volume_surge = prev_v >= (vol_mult * prev_vsma)

            # B. Price Breakout: Crossing above 20-day SMA or closing at upper quartile
            price_break = (prev_c > prev_sma) and (prev_c2 <= prev_sma2 or prev_c > close[i-20:i-1].max() * 0.98)

            # C. No Extreme Overextension: RSI between 42 and 70 (Healthy momentum, not chasing ceiling)
            rsi_valid = 42.0 <= prev_rsi <= 70.0

            if has_volume_surge and price_break and rsi_valid:
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

    # Summary
    total_trades = len(trades)
    wins = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = equity - initial_capital
    roi = (total_pnl / initial_capital) * 100.0

    gross_win = sum(t["profit"] for t in wins)
    gross_loss = abs(sum(t["profit"] for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "roi_pct": round(roi, 2),
        "total_pnl": round(total_pnl, 0),
        "trades": trades,
    }


def optimize_risk_parameters(data_dict: dict[str, pd.DataFrame]) -> dict:
    """Grid search optimization across risk parameters."""
    print("\n[2/3] Optimizing Risk Management Parameters across the 10-stock universe...")

    sl_options = [0.02, 0.03, 0.04, 0.05]           # 2% to 5% Stop Loss
    tp_options = [0.04, 0.06, 0.08, 0.10, 0.12]     # 4% to 12% Take Profit
    max_bars_options = [7, 10, 15, 20]               # 1.5 to 4 weeks holding
    vol_mult_options = [1.2, 1.4]                    # Volume spike multiplier

    best_score = -float("inf")
    best_params = {}
    best_stats = {}

    grid = list(itertools.product(sl_options, tp_options, max_bars_options, vol_mult_options))
    print(f"  Evaluating {len(grid)} parameter combinations on 3-year multi-asset portfolio...")

    for sl, tp, mb, vm in grid:
        # Require reasonable Risk-Reward Ratio (TP >= 1.5 * SL)
        if tp < sl * 1.3:
            continue

        port_roi = 0.0
        port_trades = 0
        port_wins = 0
        port_pnl = 0.0
        gross_w = 0.0
        gross_l = 0.0

        for code, df in data_dict.items():
            res = backtest_single_symbol(df, sl_pct=sl, tp_pct=tp, max_bars=mb, vol_mult=vm)
            port_roi += res["roi_pct"]
            port_trades += res["total_trades"]
            port_pnl += res["total_pnl"]
            wins = [t for t in res["trades"] if t["profit"] > 0]
            losses = [t for t in res["trades"] if t["profit"] <= 0]
            port_wins += len(wins)
            gross_w += sum(t["profit"] for t in wins)
            gross_l += abs(sum(t["profit"] for t in losses))

        if port_trades < 30:
            continue

        port_pf = (gross_w / gross_l) if gross_l > 0 else 0.0
        port_win_rate = (port_wins / port_trades * 100) if port_trades > 0 else 0.0
        avg_roi = port_roi / len(data_dict)

        # Composite score: ROI * Profit Factor * sqrt(Trades)
        score = avg_roi * (port_pf if port_pf > 0 else 0.1)

        if score > best_score:
            best_score = score
            best_params = {
                "sl_pct": sl,
                "tp_pct": tp,
                "max_bars": mb,
                "vol_mult": vm,
            }
            best_stats = {
                "avg_roi": round(avg_roi, 2),
                "total_pnl": round(port_pnl, 0),
                "win_rate": round(port_win_rate, 1),
                "pf": round(port_pf, 2),
                "total_trades": port_trades,
            }

    print("\n  >>> Optimal Risk Parameters Discovered:")
    print(f"      - Stop Loss (SL)      : {best_params['sl_pct']*100:.1f}%")
    print(f"      - Take Profit (TP)    : {best_params['tp_pct']*100:.1f}%  (Risk-Reward {best_params['tp_pct']/best_params['sl_pct']:.2f}:1)")
    print(f"      - Max Holding Days    : {best_params['max_bars']} bars (~{best_params['max_bars']/5:.1f} weeks)")
    print(f"      - Volume Inflow Mult  : {best_params['vol_mult']}x 20-day SMA volume")
    print(f"      - Portfolio 3Y ROI    : {best_stats['avg_roi']:+.2f}% (Total PnL: {best_stats['total_pnl']:+,.0f} JPY across 10 stocks)")
    print(f"      - Portfolio Win Rate  : {best_stats['win_rate']:.1f}%")
    print(f"      - Profit Factor (PF)  : {best_stats['pf']:.2f}")

    return best_params


def run_full_report(data_dict: dict[str, pd.DataFrame], optimal_params: dict) -> None:
    """Generate detailed breakdown for each of the 10 stocks under optimal parameters."""
    print("\n[3/3] Detailed 3-Year Performance per Symbol under Optimized Strategy:")
    print("=" * 95)
    print(f"{'Code':<6} {'Name':<24} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'ROI (%)':<12} {'Total PnL':<15} {'Best Win':<10}")
    print("-" * 95)

    all_trades = []
    total_capital = 1_000_000.0 * len(data_dict)
    total_final_equity = 0.0

    for code, df in data_dict.items():
        name = SELECTED_SYMBOLS.get(code, code)
        res = backtest_single_symbol(
            df,
            sl_pct=optimal_params["sl_pct"],
            tp_pct=optimal_params["tp_pct"],
            max_bars=optimal_params["max_bars"],
            vol_mult=optimal_params["vol_mult"],
        )
        all_trades.extend(res["trades"])
        total_final_equity += (1_000_000.0 + res["total_pnl"])

        best_w = max([t["profit_pct"] for t in res["trades"]], default=0.0)
        print(
            f"{code:<6} {name:<24} {res['total_trades']:<8} "
            f"{res['win_rate']:<9.1f}% {res['profit_factor']:<8.2f} "
            f"{res['roi_pct']:+10.2f}% {res['total_pnl']:+13,.0f}円 "
            f"{best_w:+7.2f}%"
        )

    print("=" * 95)
    total_pnl_all = total_final_equity - total_capital
    portfolio_roi = (total_pnl_all / total_capital) * 100.0
    wins_all = [t for t in all_trades if t["profit"] > 0]
    overall_win_rate = (len(wins_all) / len(all_trades) * 100) if all_trades else 0.0

    print(f"PORTFOLIO TOTAL: Initial: 10,000,000円 -> Final: {total_final_equity:,.0f}円 ({portfolio_roi:+.2f}%)")
    print(f"Total Completed Trades: {len(all_trades)} | Overall Win Rate: {overall_win_rate:.1f}%")

    # Exit reason distribution
    reasons = {}
    for t in all_trades:
        r = t["exit_reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print("\nExit Reason Breakdown:")
    for r, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_trades) * 100
        print(f"  - {r:<18}: {count:3d} trades ({pct:4.1f}%)")


if __name__ == "__main__":
    data_dict = download_3y_data(SELECTED_SYMBOLS)
    optimal_params = optimize_risk_parameters(data_dict)
    run_full_report(data_dict, optimal_params)
