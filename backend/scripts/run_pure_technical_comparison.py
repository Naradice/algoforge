"""Run Pure Technical Backtest on all 10 Nikkei 225 Stocks via AlgoForge Engine.

Compare with Supply/Demand Swing Strategy:
1. Pure Technical (SMA20 + SMA50 + RSI14, without volume/supply)
2. MACD Momentum (MACD line > signal + SMA50 + RSI14)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import select

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from database import async_session_factory
from data.models import Dataset
from strategy.models import Strategy, StrategyRun, Trade, RunMetric
from strategy.executor import execute_strategy_run

SELECTED_SYMBOLS = {
    "2802": "味の素",
    "3402": "東レ",
    "4063": "信越化学工業",
    "4452": "花王",
    "6361": "荏原製作所",
    "7201": "日産自動車",
    "7735": "SCREENホールディングス",
    "8031": "三井物産",
    "9434": "ソフトバンク",
    "9735": "セコム",
}


async def run_pure_technical_comparison():
    print("=" * 70)
    print(" Running AlgoForge Native Backtests: PURE TECHNICAL STRATEGIES")
    print("=" * 70)

    results_pure_tech = []
    results_macd = []

    for code, name in SELECTED_SYMBOLS.items():
        print(f"\nProcessing [{code}] {name}...")

        # 1. Fetch Dataset from DB
        async with async_session_factory() as db:
            res_ds = await db.execute(select(Dataset).where(Dataset.symbol == code).order_by(Dataset.id.desc()))
            ds = res_ds.scalars().first()
            assert ds is not None, f"Dataset for {code} not found!"
            ds_id = ds.id

        # =====================================================================
        # Strategy 1: Pure Technical Trend (No Volume, No Supply)
        # =====================================================================
        strat1_def = {
            "symbol": code,
            "indicators": [
                {"id": "sma20", "type": "sma", "params": {"period": 20, "column": "close"}},
                {"id": "sma50", "type": "sma", "params": {"period": 50, "column": "close"}},
                {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
            ],
            "long": {
                "entry": {
                    "logic": "and",
                    "conditions": [
                        {"left": "close", "op": ">", "right": "sma20"},
                        {"left": "close", "op": ">", "right": "sma50"},
                        {"left": "rsi14", "op": "<", "right": 70},
                    ],
                },
                "exit": {
                    "logic": "or",
                    "conditions": [
                        {"left": "close", "op": "<", "right": "sma20"},
                    ],
                },
            },
            "risk": {
                "risk_type": "fixed",
                "position_size": 1.0,
                "sl_pct": 0.025,
                "tp_pct": 0.10,
                "max_positions": 1,
                "slippage_pct": 0.001,
                "commission_pct": 0.001,
            },
        }

        async with async_session_factory() as db:
            s1_name = f"純テクニカル_{name}_{code}"
            res_s1 = await db.execute(select(Strategy).where(Strategy.name == s1_name))
            s1 = res_s1.scalar_one_or_none()
            if s1 is None:
                s1 = Strategy(
                    name=s1_name,
                    description=f"{name}({code}) 純テクニカルスイング（出来高・需給なし、SMA20/50+RSI）",
                    definition=strat1_def,
                    status="active",
                )
                db.add(s1)
                await db.commit()
                await db.refresh(s1)
            else:
                s1.definition = strat1_def
                await db.commit()

            run1 = StrategyRun(
                strategy_id=s1.id,
                dataset_id=ds_id,
                mode="backtest",
                status="pending",
                starting_capital=1_000_000.0,
            )
            db.add(run1)
            await db.commit()
            await db.refresh(run1)
            run1_id = run1.id

        await execute_strategy_run(run1_id)

        async with async_session_factory() as db:
            res_m1 = await db.execute(select(RunMetric).where(RunMetric.run_id == run1_id))
            m1 = {m.key: m.value for m in res_m1.scalars().all()}
            res_t1 = await db.execute(select(Trade).where(Trade.run_id == run1_id))
            t1 = res_t1.scalars().all()

        trades1 = int(m1.get("total_trades", len(t1)))
        win1 = m1.get("win_rate", 0.0) * 100
        pf1 = m1.get("profit_factor", 0.0)
        pnl1 = m1.get("total_pnl", 0.0)
        roi1 = (pnl1 / 1_000_000.0) * 100

        print(f"  [純テクニカル] Run #{run1_id:2d}: Trades={trades1:2d}, Win={win1:4.1f}%, PnL={pnl1:+10.1f}円 (ROI {roi1:+5.2f}%), PF={pf1:.2f}")

        results_pure_tech.append({
            "code": code,
            "name": name,
            "run_id": run1_id,
            "trades": trades1,
            "win_rate": round(win1, 1),
            "pnl": round(pnl1, 1),
            "roi": round(roi1, 2),
            "pf": round(pf1, 2),
        })

        # =====================================================================
        # Strategy 2: MACD Golden Cross Momentum
        # =====================================================================
        strat2_def = {
            "symbol": code,
            "indicators": [
                {"id": "macd", "type": "macd", "params": {"fast": 12, "slow": 26, "signal_period": 9}},
                {"id": "sma50", "type": "sma", "params": {"period": 50, "column": "close"}},
                {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
            ],
            "long": {
                "entry": {
                    "logic": "and",
                    "conditions": [
                        {"left": "macd_line", "op": ">", "right": "macd_signal"},
                        {"left": "close", "op": ">", "right": "sma50"},
                        {"left": "rsi14", "op": "<", "right": 70},
                    ],
                },
                "exit": {
                    "logic": "or",
                    "conditions": [
                        {"left": "macd_line", "op": "<", "right": "macd_signal"},
                    ],
                },
            },
            "risk": {
                "risk_type": "fixed",
                "position_size": 1.0,
                "sl_pct": 0.025,
                "tp_pct": 0.10,
                "max_positions": 1,
                "slippage_pct": 0.001,
                "commission_pct": 0.001,
            },
        }

        async with async_session_factory() as db:
            s2_name = f"MACDモメンタム_{name}_{code}"
            res_s2 = await db.execute(select(Strategy).where(Strategy.name == s2_name))
            s2 = res_s2.scalar_one_or_none()
            if s2 is None:
                s2 = Strategy(
                    name=s2_name,
                    description=f"{name}({code}) MACDゴールデンクロススイング（MACD line>signal + SMA50 + RSI）",
                    definition=strat2_def,
                    status="active",
                )
                db.add(s2)
                await db.commit()
                await db.refresh(s2)
            else:
                s2.definition = strat2_def
                await db.commit()

            run2 = StrategyRun(
                strategy_id=s2.id,
                dataset_id=ds_id,
                mode="backtest",
                status="pending",
                starting_capital=1_000_000.0,
            )
            db.add(run2)
            await db.commit()
            await db.refresh(run2)
            run2_id = run2.id

        await execute_strategy_run(run2_id)

        async with async_session_factory() as db:
            res_m2 = await db.execute(select(RunMetric).where(RunMetric.run_id == run2_id))
            m2 = {m.key: m.value for m in res_m2.scalars().all()}
            res_t2 = await db.execute(select(Trade).where(Trade.run_id == run2_id))
            t2 = res_t2.scalars().all()

        trades2 = int(m2.get("total_trades", len(t2)))
        win2 = m2.get("win_rate", 0.0) * 100
        pf2 = m2.get("profit_factor", 0.0)
        pnl2 = m2.get("total_pnl", 0.0)
        roi2 = (pnl2 / 1_000_000.0) * 100

        print(f"  [MACDモメンタム] Run #{run2_id:2d}: Trades={trades2:2d}, Win={win2:4.1f}%, PnL={pnl2:+10.1f}円 (ROI {roi2:+5.2f}%), PF={pf2:.2f}")

        results_macd.append({
            "code": code,
            "name": name,
            "run_id": run2_id,
            "trades": trades2,
            "win_rate": round(win2, 1),
            "pnl": round(pnl2, 1),
            "roi": round(roi2, 2),
            "pf": round(pf2, 2),
        })

    # Summary Output
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY: PURE TECHNICAL vs MACD")
    print("=" * 70)
    pnl_tech = sum(r["pnl"] for r in results_pure_tech)
    pnl_macd = sum(r["pnl"] for r in results_macd)
    cap = 10_000_000.0
    print(f"1. 純テクニカル (SMA20/50+RSI, 出来高なし): PnL = {pnl_tech:+,.1f} 円 (ROI {pnl_tech/cap*100:+.2f}%)")
    print(f"2. MACDモメンタム (MACD+SMA50+RSI):       PnL = {pnl_macd:+,.1f} 円 (ROI {pnl_macd/cap*100:+.2f}%)")

    # Save to JSON
    with open(backend_root / "artifacts" / "pure_technical_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "pure_technical": {
                "total_pnl": pnl_tech,
                "roi_pct": round(pnl_tech / cap * 100, 2),
                "symbols": results_pure_tech,
            },
            "macd_momentum": {
                "total_pnl": pnl_macd,
                "roi_pct": round(pnl_macd / cap * 100, 2),
                "symbols": results_macd,
            },
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(run_pure_technical_comparison())
