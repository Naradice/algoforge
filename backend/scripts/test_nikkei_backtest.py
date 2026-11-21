"""Demonstration script for backtesting swing strategy on random Nikkei 225 stocks."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# Load Nikkei 225 codes
codes_path = Path(r"c:\Users\ksato\workspace\finance\agentic_trade\nikkei225_codes.json")
with open(codes_path, encoding="utf-8") as f:
    nikkei_codes = json.load(f)

print(f"Total Nikkei 225 pool: {len(nikkei_codes)} symbols")

# Randomly select 5 symbols
random.seed(42)  # For reproducible demo
sample_codes = random.sample(nikkei_codes, 5)
print(f"Randomly selected 5 symbols: {sample_codes}")


class SwingBacktester:
    """Simple and realistic bar-by-bar swing trading backtester."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        sl_pct: float = 0.03,        # 3% Stop Loss
        tp_pct: float = 0.06,        # 6% Take Profit
        slippage_pct: float = 0.001, # 0.1% slippage
        max_holding_bars: int = 15,  # Max 15 trading days (~3 weeks)
    ):
        self.initial_capital = initial_capital
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.slippage_pct = slippage_pct
        self.max_holding_bars = max_holding_bars

    def run_on_dataframe(self, df: pd.DataFrame, symbol: str) -> dict:
        """Run swing strategy:
        Entry Condition:
          - Price crosses above 20-day SMA
          - Volume > 1.2 * 20-day Volume SMA (Volume spike)
          - RSI(14) between 45 and 65 (Not overbought)
        Exit Condition:
          - Stop Loss (3%) or Take Profit (6%) hit intraday
          - Cross below 5-day SMA or Max holding period (15 days) reached
        """
        # Indicators
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        open_ = df["Open"].squeeze()
        volume = df["Volume"].squeeze()

        sma20 = close.rolling(20).mean()
        sma5 = close.rolling(5).mean()
        vol_sma20 = volume.rolling(20).mean()

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        trades = []
        equity = self.initial_capital
        position = None  # {'entry_price', 'entry_bar', 'entry_date', 'shares', 'sl', 'tp'}

        for i in range(25, len(df)):
            current_bar_date = df.index[i]
            prev_close = close.iloc[i - 1]
            prev_sma20 = sma20.iloc[i - 1]
            prev_sma20_2 = sma20.iloc[i - 2]
            prev_close_2 = close.iloc[i - 2]
            prev_vol = volume.iloc[i - 1]
            prev_vol_sma = vol_sma20.iloc[i - 1]
            prev_rsi = rsi.iloc[i - 1]

            cur_open = open_.iloc[i]
            cur_high = high.iloc[i]
            cur_low = low.iloc[i]
            cur_close = close.iloc[i]

            # 1. Manage Open Position
            if position is not None:
                bars_held = i - position["entry_bar"]
                exit_price = None
                exit_reason = None

                # Check SL
                if cur_low <= position["sl"]:
                    exit_price = position["sl"] * (1 - self.slippage_pct)
                    exit_reason = "SL (Stop Loss)"
                # Check TP
                elif cur_high >= position["tp"]:
                    exit_price = position["tp"] * (1 - self.slippage_pct)
                    exit_reason = "TP (Take Profit)"
                # Check SMA5 breakdown after 2 days
                elif bars_held >= 2 and cur_close < sma5.iloc[i]:
                    exit_price = cur_close * (1 - self.slippage_pct)
                    exit_reason = "SMA5 Exit"
                # Check Time Stop
                elif bars_held >= self.max_holding_bars:
                    exit_price = cur_close * (1 - self.slippage_pct)
                    exit_reason = "Max Bars Held"

                if exit_price is not None:
                    profit = (exit_price - position["entry_price"]) * position["shares"]
                    profit_pct = (exit_price / position["entry_price"] - 1.0) * 100
                    equity += profit
                    trades.append({
                        "symbol": symbol,
                        "entry_date": position["entry_date"],
                        "exit_date": current_bar_date,
                        "entry_price": position["entry_price"],
                        "exit_price": round(exit_price, 1),
                        "profit": round(profit, 1),
                        "profit_pct": round(profit_pct, 2),
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    })
                    position = None

            # 2. Check Entry Condition (Signal on bar i-1 -> fill at Open of bar i)
            if position is None:
                # Golden cross over SMA20 with Volume expansion
                crossover = (prev_close_2 <= prev_sma20_2) and (prev_close > prev_sma20)
                vol_spike = prev_vol > (1.1 * prev_vol_sma)
                rsi_ok = 40 <= prev_rsi <= 68

                if crossover and vol_spike and rsi_ok:
                    fill = cur_open * (1 + self.slippage_pct)
                    shares = math.floor(equity / fill)
                    if shares > 0:
                        position = {
                            "entry_price": fill,
                            "entry_bar": i,
                            "entry_date": current_bar_date,
                            "shares": shares,
                            "sl": fill * (1 - self.sl_pct),
                            "tp": fill * (1 + self.tp_pct),
                        }

        # Calculate metrics
        total_trades = len(trades)
        wins = [t for t in trades if t["profit"] > 0]
        losses = [t for t in trades if t["profit"] <= 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = equity - self.initial_capital
        roi = (total_pnl / self.initial_capital) * 100

        gross_win = sum(t["profit"] for t in wins)
        gross_loss = abs(sum(t["profit"] for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        return {
            "symbol": symbol,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "roi_pct": round(roi, 2),
            "total_pnl": round(total_pnl, 0),
            "final_equity": round(equity, 0),
            "trades": trades,
        }


import math

bt = SwingBacktester(sl_pct=0.03, tp_pct=0.06)
results = []

for code in sample_codes:
    ticker = f"{code}.T"
    print(f"Fetching {ticker} (1 year daily data)...")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty or len(df) < 50:
        continue

    # Flatten MultiIndex columns if present in newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    res = bt.run_on_dataframe(df, symbol=code)
    results.append(res)
    print(f"  -> {code}: Trades={res['total_trades']}, WinRate={res['win_rate']}%, PF={res['profit_factor']}, ROI={res['roi_pct']:+}%")

print("\n" + "=" * 60)
print("BACKTEST SUMMARY ON 5 RANDOM NIKKEI 225 STOCKS")
print("=" * 60)
for r in results:
    print(f"Code: {r['symbol']} | Trades: {r['total_trades']:2d} | Win: {r['win_rate']:4.1f}% | PF: {r['profit_factor']:4.2f} | Return: {r['roi_pct']:+6.2f}% ({r['total_pnl']:+,.0f} JPY)")
    if r["trades"]:
        sample_trade = r["trades"][-1]
        print(f"    Last Trade: {sample_trade['entry_date'].strftime('%Y-%m-%d')} -> {sample_trade['exit_date'].strftime('%Y-%m-%d')} | PnL: {sample_trade['profit_pct']:+.2f}% ({sample_trade['exit_reason']})")
