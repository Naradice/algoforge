"""Run the productionised portfolio engine on the 9 majors (yfinance D1 + MT5 history) and compare with d2 research."""
import os, sys, pathlib, json
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault("ARTIFACT_STORE_PATH", str(ROOT / "artifacts"))
sys.path.insert(0, str(ROOT / "backend"))
from strategy.portfolio.engine import run_portfolio_backtest
PIPS = {"USDJPY":0.6,"EURUSD":0.7,"GBPUSD":1.0,"AUDUSD":1.0,"USDCHF":1.3,"EURJPY":1.2,"GBPJPY":2.0,"AUDJPY":1.6,"CHFJPY":2.5}
store = ROOT / "artifacts"
# merged price history: MT5 D1 (to 2025-05) + AlgoForge yfinance D1 after it
outdir = store / "research" / "fx" / "d1_merged"; outdir.mkdir(parents=True, exist_ok=True)
uni = []
for s, pips in PIPS.items():
    m = pd.read_parquet(store / "research" / "fx" / "d1" / f"{s}.parquet")[["close"]]
    m.index = pd.to_datetime(m.index).tz_localize(None).normalize()
    f = next((store / "datasets").glob(f"src_*/{s}=X_D1.parquet"), None)
    if f is not None:
        y = pd.read_parquet(f)[["close"]]; y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
        m = pd.concat([m, y[y.index > m.index[-1]]])
    m.to_parquet(outdir / f"{s}.parquet")
    pip = 0.01 if s.endswith("JPY") else 0.0001
    uni.append({"symbol": s, "artifact_path": f"research/fx/d1_merged/{s}.parquet", "base": s[:3], "quote": s[3:],
                "spread_pct": pips * pip / float(m.close.iloc[-1])})
ids = {"USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "GBP": "IR3TIB01GBM156N", "JPY": "IR3TIB01JPM156N",
       "AUD": "IR3TIB01AUM156N", "CHF": "IR3TIB01CHM156N"}
defn = {"universe": uni, "rates": {"dir": "research/fx/rates", "series": ids},
        "sleeves": [{"type": "carry", "weight": 1.0}],
        "overlays": [{"type": "series_jump", "path": "research/fx/rates/VIXCLS.csv", "column": "VIXCLS", "ratio": 1.3,
                      "lookback": 20, "cut": 0.7}],
        "risk": {"target_vol": 0.10, "rebalance": "M"}}
if __name__ == "__main__":
    res = run_portfolio_backtest(defn)
    m = res["metrics"]; print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "yearly"}))
    print({k: v for k, v in m["yearly"].items() if k >= 2020})
    print(res["contribution"].round(4).to_dict())
    json.dump(defn, open(ROOT / "artifacts" / "research" / "fx" / "carry9_portfolio.json", "w"), indent=1)
