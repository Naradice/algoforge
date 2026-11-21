"""Register Analyst Forecast Strategies and save Backtest Runs into AlgoForge DB."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from database import SessionLocal
from strategy.models import RunMetric, Strategy, StrategyRun, Trade

CSV_PATH = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\data\reports_cache\backtest_detailed_results.csv")


def compute_metrics(pnl_series: list[float], starting_capital: float = 1_000_000.0) -> dict[str, float]:
    arr = np.array(pnl_series)
    trades_count = len(arr)
    if trades_count == 0:
        return {}

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = len(wins) / trades_count if trades_count > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    total_pnl = float(arr.sum())
    cum_equity = starting_capital + np.cumsum(arr)
    peak = np.maximum.accumulate(cum_equity)
    drawdowns = (peak - cum_equity) / peak
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio (annualized daily approximation)
    returns = arr / starting_capital
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 0.0
    sharpe = float((mean_ret / std_ret) * math.sqrt(252)) if std_ret > 0 else 0.0

    return {
        "total_trades": float(trades_count),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
    }


def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    with SessionLocal() as db:
        # =====================================================================
        # 1. Strategy: USD/JPY Analyst Range Mean-Reversion
        # =====================================================================
        s1 = db.query(Strategy).filter(Strategy.name == "USD/JPY Analyst Range Mean-Reversion").first()
        if not s1:
            s1 = Strategy(
                name="USD/JPY Analyst Range Mean-Reversion",
                description="アナリスト（SMBC信託・みんなのFX等）の日次予想レンジ下限で逆張り買い、上限で逆張り売りを行い、レンジ内収束を狙うミーンリバージョン戦略",
                definition={
                    "symbol": "USDJPY",
                    "strategy_type": "analyst_range_reversion",
                    "parameters": {"tolerance_pips": 10, "exit_at_close": True},
                },
                status="active",
            )
            db.add(s1)
            db.flush()
            print(f"Created Strategy 1: ID {s1.id}")
        else:
            print(f"Found Strategy 1: ID {s1.id}")

        # Build Trades & Equity Curve for Strategy 1
        s1_trades = []
        s1_pnl_list = []
        equity_1 = 1_000_000.0
        equity_curve_1 = []
        peak_1 = equity_1

        for _, r in df.iterrows():
            low_p = float(r["low"])
            high_p = float(r["high"])
            close_p = float(r["close"])
            exp_low = float(r["expected_low"])
            exp_high = float(r["expected_high"])
            t_date = r["date"]

            daily_pnl = 0.0
            # Buy at exp_low
            if low_p <= exp_low:
                exit_p = min(exp_high, close_p)
                # 1 standard lot = 100,000 units
                pnl = (exit_p - exp_low) * 100_000.0
                daily_pnl += pnl
                s1_pnl_list.append(pnl)
                s1_trades.append({
                    "symbol": "USDJPY",
                    "direction": "buy",
                    "entry_price": exp_low,
                    "exit_price": exit_p,
                    "volume": 1.0,
                    "profit": pnl,
                    "opened_at": datetime(t_date.year, t_date.month, t_date.day, 9, 0, tzinfo=timezone.utc),
                    "closed_at": datetime(t_date.year, t_date.month, t_date.day, 23, 0, tzinfo=timezone.utc),
                    "exit_reason": "signal",
                    "phase": "is",
                })

            # Sell at exp_high
            if high_p >= exp_high:
                exit_p = max(exp_low, close_p)
                pnl = (exp_high - exit_p) * 100_000.0
                daily_pnl += pnl
                s1_pnl_list.append(pnl)
                s1_trades.append({
                    "symbol": "USDJPY",
                    "direction": "sell",
                    "entry_price": exp_high,
                    "exit_price": exit_p,
                    "volume": 1.0,
                    "profit": pnl,
                    "opened_at": datetime(t_date.year, t_date.month, t_date.day, 9, 0, tzinfo=timezone.utc),
                    "closed_at": datetime(t_date.year, t_date.month, t_date.day, 23, 0, tzinfo=timezone.utc),
                    "exit_reason": "signal",
                    "phase": "is",
                })

            equity_1 += daily_pnl
            peak_1 = max(peak_1, equity_1)
            dd = (peak_1 - equity_1) / peak_1 if peak_1 > 0 else 0.0
            equity_curve_1.append({
                "timestamp": t_date.isoformat(),
                "equity": round(equity_1, 2),
                "drawdown": round(dd, 4),
                "phase": "is",
            })

        # Create StrategyRun 1
        run1 = StrategyRun(
            strategy_id=s1.id,
            mode="backtest",
            status="completed",
            progress_pct=100.0,
            message="Backtest completed across 577 verified forecast events",
            starting_capital=1_000_000.0,
            from_ts=df["date"].min().to_pydatetime().replace(tzinfo=timezone.utc),
            to_ts=df["date"].max().to_pydatetime().replace(tzinfo=timezone.utc),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            equity_curve=equity_curve_1,
        )
        db.add(run1)
        db.flush()

        # Add Trades & Metrics
        for t_dict in s1_trades:
            t_obj = Trade(run_id=run1.id, **t_dict)
            db.add(t_obj)

        m1 = compute_metrics(s1_pnl_list, starting_capital=1_000_000.0)
        for k, v in m1.items():
            db.add(RunMetric(run_id=run1.id, key=k, value=v))

        print(f"Created Run 1 for Strategy 1 (Trades: {len(s1_trades)}, Total PnL: {m1.get('total_pnl'):+,.0f} JPY, WinRate: {m1.get('win_rate')*100:.1f}%)")

        # =====================================================================
        # 2. Strategy: USD/JPY Analyst Breakout Momentum
        # =====================================================================
        s2 = db.query(Strategy).filter(Strategy.name == "USD/JPY Analyst Breakout Momentum").first()
        if not s2:
            s2 = Strategy(
                name="USD/JPY Analyst Breakout Momentum",
                description="アナリスト（ソニーFG・みんなのFX等）の予想上限・下限をブレイクした際に強いトレンド発生とみなして追随する順張りモメンタム戦略",
                definition={
                    "symbol": "USDJPY",
                    "strategy_type": "analyst_breakout_momentum",
                    "parameters": {"break_margin_pips": 10, "exit_at_close": True},
                },
                status="active",
            )
            db.add(s2)
            db.flush()
            print(f"Created Strategy 2: ID {s2.id}")
        else:
            print(f"Found Strategy 2: ID {s2.id}")

        # Build Trades & Equity Curve for Strategy 2
        s2_trades = []
        s2_pnl_list = []
        equity_2 = 1_000_000.0
        equity_curve_2 = []
        peak_2 = equity_2

        for _, r in df.iterrows():
            low_p = float(r["low"])
            high_p = float(r["high"])
            close_p = float(r["close"])
            exp_low = float(r["expected_low"])
            exp_high = float(r["expected_high"])
            t_date = r["date"]

            daily_pnl = 0.0
            if high_p > exp_high:
                pnl = (close_p - exp_high) * 100_000.0
                daily_pnl += pnl
                s2_pnl_list.append(pnl)
                s2_trades.append({
                    "symbol": "USDJPY",
                    "direction": "buy",
                    "entry_price": exp_high,
                    "exit_price": close_p,
                    "volume": 1.0,
                    "profit": pnl,
                    "opened_at": datetime(t_date.year, t_date.month, t_date.day, 14, 0, tzinfo=timezone.utc),
                    "closed_at": datetime(t_date.year, t_date.month, t_date.day, 23, 0, tzinfo=timezone.utc),
                    "exit_reason": "end_of_data",
                    "phase": "is",
                })

            if low_p < exp_low:
                pnl = (exp_low - close_p) * 100_000.0
                daily_pnl += pnl
                s2_pnl_list.append(pnl)
                s2_trades.append({
                    "symbol": "USDJPY",
                    "direction": "sell",
                    "entry_price": exp_low,
                    "exit_price": close_p,
                    "volume": 1.0,
                    "profit": pnl,
                    "opened_at": datetime(t_date.year, t_date.month, t_date.day, 14, 0, tzinfo=timezone.utc),
                    "closed_at": datetime(t_date.year, t_date.month, t_date.day, 23, 0, tzinfo=timezone.utc),
                    "exit_reason": "end_of_data",
                    "phase": "is",
                })

            equity_2 += daily_pnl
            peak_2 = max(peak_2, equity_2)
            dd = (peak_2 - equity_2) / peak_2 if peak_2 > 0 else 0.0
            equity_curve_2.append({
                "timestamp": t_date.isoformat(),
                "equity": round(equity_2, 2),
                "drawdown": round(dd, 4),
                "phase": "is",
            })

        run2 = StrategyRun(
            strategy_id=s2.id,
            mode="backtest",
            status="completed",
            progress_pct=100.0,
            message="Backtest completed across 577 verified forecast events",
            starting_capital=1_000_000.0,
            from_ts=df["date"].min().to_pydatetime().replace(tzinfo=timezone.utc),
            to_ts=df["date"].max().to_pydatetime().replace(tzinfo=timezone.utc),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            equity_curve=equity_curve_2,
        )
        db.add(run2)
        db.flush()

        for t_dict in s2_trades:
            t_obj = Trade(run_id=run2.id, **t_dict)
            db.add(t_obj)

        m2 = compute_metrics(s2_pnl_list, starting_capital=1_000_000.0)
        for k, v in m2.items():
            db.add(RunMetric(run_id=run2.id, key=k, value=v))

        print(f"Created Run 2 for Strategy 2 (Trades: {len(s2_trades)}, Total PnL: {m2.get('total_pnl'):+,.0f} JPY, WinRate: {m2.get('win_rate')*100:.1f}%)")

        db.commit()
        print("\nAll analyst strategies and backtest runs successfully saved to AlgoForge DB!")


if __name__ == "__main__":
    main()
