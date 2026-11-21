"""Register M30 Strict Backtest Runs for Strategy 3 and 4 into AlgoForge DB."""

import datetime
from pathlib import Path
import pandas as pd
from database import SessionLocal
from strategy.models import Strategy, StrategyRun, Trade, RunMetric
from strategy.engine.analyst_strategy import run_analyst_backtest

def main():
    m30_path = Path("artifacts/datasets/USDJPY_M30_latest.parquet")
    m30_df = pd.read_parquet(m30_path)

    with SessionLocal() as db:
        # =====================================================================
        # 1. Strategy 3: Range Mean-Reversion (M30 Strict Timing)
        # =====================================================================
        s3 = db.query(Strategy).filter(Strategy.id == 3).first()
        if s3:
            def_3 = {
                "strategy_type": "analyst_range_reversion",
                "symbol": "USDJPY",
                "parameters": {"source": "SMBC", "tolerance_pips": 10, "sl_pips": 50, "tp_pips": 80}
            }
            trades_3, metrics_3, curve_3 = run_analyst_backtest(def_3, m30_df)

            from_ts = m30_df.index.min().to_pydatetime()
            to_ts = m30_df.index.max().to_pydatetime()

            run3 = StrategyRun(
                strategy_id=3,
                mode="backtest",
                status="completed",
                progress_pct=100.0,
                message="M30 Strict Timing Backtest (Filtered by 09:30 JST Release)",
                starting_capital=1_000_000.0,
                from_ts=from_ts,
                to_ts=to_ts,
                started_at=datetime.datetime.now(datetime.timezone.utc),
                ended_at=datetime.datetime.now(datetime.timezone.utc),
                equity_curve=curve_3,
            )
            db.add(run3)
            db.flush()

            for t_dict in trades_3:
                t_obj = Trade(run_id=run3.id, **t_dict)
                db.add(t_obj)

            for k, v in metrics_3.items():
                db.add(RunMetric(run_id=run3.id, key=k, value=float(v)))

            db.commit()
            print(f"Created Run {run3.id} for Strategy 3: {len(trades_3)} trades, PnL: {metrics_3.get('total_pnl', 0):+,.0f} JPY, PF: {metrics_3.get('profit_factor', 0):.2f}, WinRate: {metrics_3.get('win_rate', 0)*100:.1f}%")

        # =====================================================================
        # 2. Strategy 4: Breakout Momentum (M30 Strict Timing)
        # =====================================================================
        s4 = db.query(Strategy).filter(Strategy.id == 4).first()
        if s4:
            def_4 = {
                "strategy_type": "analyst_breakout_momentum",
                "symbol": "USDJPY",
                "parameters": {"source": "all", "sl_pips": 15, "tp_pips": 50}
            }
            trades_4, metrics_4, curve_4 = run_analyst_backtest(def_4, m30_df)

            from_ts = m30_df.index.min().to_pydatetime()
            to_ts = m30_df.index.max().to_pydatetime()

            run4 = StrategyRun(
                strategy_id=4,
                mode="backtest",
                status="completed",
                progress_pct=100.0,
                message="M30 Strict Timing Backtest (Filtered by 09:30 JST Release)",
                starting_capital=1_000_000.0,
                from_ts=from_ts,
                to_ts=to_ts,
                started_at=datetime.datetime.now(datetime.timezone.utc),
                ended_at=datetime.datetime.now(datetime.timezone.utc),
                equity_curve=curve_4,
            )
            db.add(run4)
            db.flush()

            for t_dict in trades_4:
                t_obj = Trade(run_id=run4.id, **t_dict)
                db.add(t_obj)

            for k, v in metrics_4.items():
                db.add(RunMetric(run_id=run4.id, key=k, value=float(v)))

            db.commit()
            print(f"Created Run {run4.id} for Strategy 4: {len(trades_4)} trades, PnL: {metrics_4.get('total_pnl', 0):+,.0f} JPY, PF: {metrics_4.get('profit_factor', 0):.2f}, WinRate: {metrics_4.get('win_rate', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
