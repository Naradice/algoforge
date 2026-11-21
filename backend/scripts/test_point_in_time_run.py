"""Test AlgoForge Strategy with Point-in-Time Supply/Demand Event Condition."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import select

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from database import async_session_factory
from strategy.models import Strategy, StrategyRun, Trade, RunMetric
from strategy.executor import execute_strategy_run
from data.models import Dataset


async def test_supply_conditioned_strategy():
    print("\n--- Testing Strategy with Direct Point-in-Time Supply Filter in AlgoForge ---")
    
    # Update strategy definition with supply_event_active condition
    strategy_definition = {
        "symbol": "7735",
        "indicators": [
            {"id": "sma20", "type": "sma", "params": {"period": 20, "column": "close"}},
            {"id": "sma50", "type": "sma", "params": {"period": 50, "column": "close"}},
            {"id": "vol_sma20", "type": "rolling", "params": {"column": "volume", "window": 20, "function": "mean"}},
            {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
            {
                "id": "supply",
                "type": "supply_event",
                "params": {
                    "symbol": "7735",
                    "events_path": "artifacts/supply_events.parquet",
                    "lookback_days": 10
                }
            }
        ],
        "long": {
            "entry": {
                "logic": "and",
                "conditions": [
                    {"left": "close", "op": ">", "right": "sma20"},
                    {"left": "close", "op": ">", "right": "sma50"},
                    {"left": "supply_event_active", "op": "==", "right": 1}
                ]
            },
            "exit": {
                "logic": "or",
                "conditions": [
                    {"left": "close", "op": "<", "right": "sma20"}
                ]
            }
        },
        "risk": {
            "risk_type": "fixed",
            "position_size": 1.0,
            "sl_pct": 0.025,
            "tp_pct": 0.10,
            "max_positions": 1,
            "slippage_pct": 0.001,
            "commission_pct": 0.001
        }
    }

    async with async_session_factory() as db:
        res = await db.execute(select(Dataset).where(Dataset.symbol == "7735").order_by(Dataset.id.desc()))
        ds = res.scalars().first()
        assert ds is not None, "Dataset not found"

        strat = Strategy(
            name="需給Point-in-Time厳密連動戦略 (7735)",
            description="開示日時Point-in-Time需給イベント（大量保有・信用規制）発生後10営業日以内かつ上昇トレンドでのみエントリーするAlgoForgeネイティブ戦略",
            definition=strategy_definition,
            status="active",
        )
        db.add(strat)
        await db.commit()
        await db.refresh(strat)
        
        run = StrategyRun(
            strategy_id=strat.id,
            dataset_id=ds.id,
            mode="backtest",
            status="pending",
            starting_capital=1_000_000.0,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        run_id = run.id

    print(f"Created StrategyRun id={run_id} for Strategy id={strat.id}")
    exec_res = await execute_strategy_run(run_id)
    print(f"execute_strategy_run completed: {exec_res}")

    async with async_session_factory() as db:
        res_run = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))
        run_db = res_run.scalar_one()
        res_trades = await db.execute(select(Trade).where(Trade.run_id == run_id))
        trades = res_trades.scalars().all()
        res_metrics = await db.execute(select(RunMetric).where(RunMetric.run_id == run_id))
        metrics = res_metrics.scalars().all()

    print(f"\nFinal Run Status: {run_db.status}")
    print(f"Trades Count: {len(trades)}")
    for m in metrics:
        if m.key in ("total_trades", "win_rate", "profit_factor", "total_pnl", "sharpe_ratio", "max_drawdown_pct"):
            print(f"  {m.key:20}: {m.value}")

    if trades:
        print("\nExecuted Trades:")
        for t in trades:
            print(f"  [{t.opened_at.strftime('%Y-%m-%d')} ~ {t.closed_at.strftime('%Y-%m-%d')}] "
                  f"Entry: {t.entry_price:.1f}, Exit: {t.exit_price:.1f}, PnL: {t.profit:+.1f} yen ({t.exit_reason})")


if __name__ == "__main__":
    asyncio.run(test_supply_conditioned_strategy())
