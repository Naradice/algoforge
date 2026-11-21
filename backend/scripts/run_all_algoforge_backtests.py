"""Run AlgoForge Native Backtests on all 10 Nikkei 225 Stocks with Point-in-Time Events.

Persists Datasets, Strategy, StrategyRuns, Trades, and RunMetrics in AlgoForge database.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timezone
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf
from sqlalchemy import select

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from database import async_session_factory
from data.models import Dataset, LargeHoldingReportRecord, DailyMarginRecord
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


async def run_all_algoforge_backtests():
    art_datasets_dir = backend_root / "artifacts" / "datasets"
    art_datasets_dir.mkdir(parents=True, exist_ok=True)
    events_parquet_path = backend_root / "artifacts" / "supply_events.parquet"

    print("=" * 70)
    print(" Running AlgoForge Native Backtest across all 10 Nikkei 225 stocks")
    print(" (With Point-in-Time supply events & exact execution timing)")
    print("=" * 70)

    # 1. Ensure supply events parquet exists
    assert events_parquet_path.exists(), "supply_events.parquet not found"

    results = []

    for code, name in SELECTED_SYMBOLS.items():
        print(f"\nProcessing [{code}] {name}...")
        ticker = f"{code}.T"
        parquet_fname = f"JP_{code}_D1.parquet"
        full_p = art_datasets_dir / parquet_fname
        rel_path = f"datasets/{parquet_fname}"

        # Download or use existing parquet
        if not full_p.exists():
            df = yf.download(ticker, period="3y", interval="1d", progress=False)
            if df.empty:
                print(f"  Warning: no data for {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.columns = ["open", "high", "low", "close", "volume"]
            df["spread"] = 0.0
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.to_parquet(full_p)
        else:
            df = pd.read_parquet(full_p)

        # 2. Register/Get Dataset in DB
        async with async_session_factory() as db:
            res_ds = await db.execute(select(Dataset).where(Dataset.symbol == code).order_by(Dataset.id.desc()))
            ds = res_ds.scalars().first()
            if ds is None:
                ds = Dataset(
                    name=f"{name} ({code}) D1",
                    symbol=code,
                    timeframe="D1",
                    from_ts=df.index[0].to_pydatetime(),
                    to_ts=df.index[-1].to_pydatetime(),
                    row_count=len(df),
                    artifact_path=rel_path,
                    status="ready",
                )
                db.add(ds)
                await db.commit()
                await db.refresh(ds)
            ds_id = ds.id

        # 3. Register Strategy in DB
        strategy_def = {
            "symbol": code,
            "indicators": [
                {"id": "sma20", "type": "sma", "params": {"period": 20, "column": "close"}},
                {"id": "sma50", "type": "sma", "params": {"period": 50, "column": "close"}},
                {"id": "vol_sma20", "type": "rolling", "params": {"column": "volume", "window": 20, "function": "mean"}},
                {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
                {
                    "id": "supply",
                    "type": "supply_event",
                    "params": {
                        "symbol": code,
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
                        {"left": "volume", "op": ">", "right": "vol_sma20"},
                        {"left": "rsi14", "op": "<", "right": 70}
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
            strat_name = f"需給スイング_{name}_{code}"
            res_strat = await db.execute(select(Strategy).where(Strategy.name == strat_name))
            strat = res_strat.scalar_one_or_none()
            if strat is None:
                strat = Strategy(
                    name=strat_name,
                    description=f"{name}({code}) 需給連動・Point-in-Timeスイング戦略 (SL 2.5%, TP 10.0%)",
                    definition=strategy_def,
                    status="active",
                )
                db.add(strat)
                await db.commit()
                await db.refresh(strat)
            else:
                strat.definition = strategy_def
                await db.commit()
            strat_id = strat.id

            # Create Run
            run = StrategyRun(
                strategy_id=strat_id,
                dataset_id=ds_id,
                mode="backtest",
                status="pending",
                starting_capital=1_000_000.0,
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)
            run_id = run.id

        # 4. Execute backtest via AlgoForge engine
        await execute_strategy_run(run_id)

        # 5. Fetch metrics
        async with async_session_factory() as db:
            res_m = await db.execute(select(RunMetric).where(RunMetric.run_id == run_id))
            metrics = {m.key: m.value for m in res_m.scalars().all()}
            res_t = await db.execute(select(Trade).where(Trade.run_id == run_id))
            trades = res_t.scalars().all()

        total_trades = int(metrics.get("total_trades", len(trades)))
        win_rate = metrics.get("win_rate", 0.0) * 100
        pf = metrics.get("profit_factor", 0.0)
        pnl = metrics.get("total_pnl", 0.0)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        roi = (pnl / 1_000_000.0) * 100

        print(f"  ✓ Run ID {run_id:2d}: Trades={total_trades:2d}, WinRate={win_rate:4.1f}%, PnL={pnl:+10.1f}円 (ROI {roi:+5.2f}%), PF={pf:.2f}")

        results.append({
            "code": code,
            "name": name,
            "run_id": run_id,
            "trades": total_trades,
            "win_rate": round(win_rate, 1),
            "pnl": round(pnl, 1),
            "roi": round(roi, 2),
            "pf": round(pf, 2),
            "sharpe": round(sharpe, 2) if abs(sharpe) < 100 else 0.0,
        })

    print("\n" + "=" * 70)
    print(" SUMMARY OF ALGOFORGE NATIVE BACKTESTS (Point-in-Time)")
    print("=" * 70)
    total_capital = 1_000_000.0 * len(results)
    total_pnl = sum(r["pnl"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    overall_roi = (total_pnl / total_capital) * 100

    print(f"Total Initial Capital: {total_capital:,.0f} yen")
    print(f"Total Net Profit:     {total_pnl:+,.1f} yen (ROI: {overall_roi:+.2f}%)")
    print(f"Total Trades:         {total_trades}")
    print("-" * 70)
    for r in results:
        print(f"[{r['code']}] {r['name']:15} | Run #{r['run_id']:2d} | Trades: {r['trades']:2d} | Win: {r['win_rate']:4.1f}% | PnL: {r['pnl']:+9.0f}円 ({r['roi']:+5.2f}%) | PF: {r['pf']:4.2f}")

    # Output JSON summary for artifact update
    with open(backend_root / "artifacts" / "algoforge_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall": {
                "total_capital": total_capital,
                "total_pnl": total_pnl,
                "roi_pct": round(overall_roi, 2),
                "total_trades": total_trades,
            },
            "symbols": results,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(run_all_algoforge_backtests())
