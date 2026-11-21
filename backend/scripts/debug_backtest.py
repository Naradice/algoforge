import pandas as pd
from strategy.engine.indicators import apply_indicators
from strategy.engine.conditions import evaluate_conditions
from strategy.engine.backtest import run_backtest

# Load 7735 parquet
df = pd.read_parquet("artifacts/datasets/JP_7735_D1.parquet")
print("Original DF shape:", df.shape)
print("DF columns:", df.columns.tolist())

# Load strategy definition
from sqlalchemy import select, create_engine
import json

definition = {
    "symbol": "7735",
    "indicators": [
        {"id": "sma20", "type": "sma", "params": {"period": 20, "column": "close"}},
        {"id": "sma50", "type": "sma", "params": {"period": 50, "column": "close"}},
        {"id": "vol_sma20", "type": "rolling", "params": {"column": "volume", "window": 20, "op": "mean"}},
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

ind_df = apply_indicators(df, definition["indicators"])
print("Indicator DF columns:", ind_df.columns.tolist())
print("Null count in columns:")
print(ind_df[["close", "sma20", "sma50", "vol_sma20", "rsi14"]].isna().sum())

# Test evaluate_conditions on rows
matches = 0
for i in range(len(ind_df)):
    row = ind_df.iloc[i]
    if evaluate_conditions(row, definition["long"]["entry"]):
        matches += 1
print(f"Total entry condition matches across {len(ind_df)} bars: {matches}")

# Run backtest directly
trades, metrics, eq = run_backtest(definition, "artifacts/datasets/JP_7735_D1.parquet", starting_capital=1_000_000.0)
print(f"Direct run_backtest trades: {len(trades)}, metrics: {metrics}")
