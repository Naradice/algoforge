"""Run the carry indicator through AlgoForge's own backtest engine on MT5 D1 data and compare with the vectorised result."""
import os, sys, pathlib, numpy as np, pandas as pd
os.environ.setdefault("ARTIFACT_STORE_PATH", str(pathlib.Path(__file__).resolve().parents[3] / "artifacts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from strategy.engine.backtest import run_backtest
store = pathlib.Path(os.environ["ARTIFACT_STORE_PATH"]); out = store / "research" / "fx" / "d1"; out.mkdir(parents=True, exist_ok=True)
PAIRS = ["USDJPY","EURUSD","GBPUSD","AUDUSD","USDCHF","EURJPY","GBPJPY","AUDJPY","CHFJPY"]
def defn(sym):
    ind = [{"id": "carry", "type": "carry", "params": {"base": sym[:3], "quote": sym[3:]}}]
    return {"symbol": sym, "indicators": ind,
        "long":  {"entry": {"conditions": [{"left": "carry_w", "op": ">", "right": 0}], "logic": "and"},
                  "exit":  {"conditions": [{"left": "carry_w", "op": "<=", "right": 0}], "logic": "or"}},
        "short": {"entry": {"conditions": [{"left": "carry_w", "op": "<", "right": 0}], "logic": "and"},
                  "exit":  {"conditions": [{"left": "carry_w", "op": ">=", "right": 0}], "logic": "or"}},
        "risk": {"risk_type": "fixed", "position_size": 1.0, "sl_pct": 0.0, "tp_pct": 0.0, "swap_col": "carry",
                 "slippage_pct": 0.00003, "commission_pct": 0.0, "max_positions": 1}}
for s in PAIRS:
    d = pd.read_csv(f"L:/data/fx/OANDA-Japan MT5 Live/mt5_{s}_d1.csv", parse_dates=["time"]).set_index("time")
    d = d[d.index >= "2000-01-01"][["open","high","low","close"]]; d.to_parquet(out / f"{s}.parquet")
    trades, m, eq = run_backtest(defn(s), f"research/fx/d1/{s}.parquet")
    print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in ("total_trades","win_rate","sharpe_ratio","max_drawdown","total_pnl","profit_factor")})
