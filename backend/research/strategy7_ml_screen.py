"""Walk-forward comparison of fixed and learned Strategy 7 entry filters."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

module = runpy.run_path(str(Path(__file__).with_name("sweep_strategy7_execution.py")))
root = module["ROOT"]
from strategy.engine.standalone_pivot import calculate_daily_pivots

bars = pd.read_parquet(root / "artifacts/datasets/HistData_USDJPY_30min.parquet")
bars.index = pd.to_datetime(bars.index, utc=True)
bars = bars.sort_index().loc["2010":"2022"]
bars["day"] = bars.index.strftime("%Y-%m-%d")
bars["minute"] = bars.index.hour * 60 + bars.index.minute
pivots = calculate_daily_pivots(bars).set_index("date")
bars = bars.join(pivots[["expected_low", "expected_high"]], on=bars.index.date)
change = bars.close.diff()
bars["efficiency"] = change.rolling(24).sum().abs() / change.abs().rolling(24).sum()
bars["vol_ratio"] = change.abs().rolling(12).mean() / change.abs().rolling(96).mean()
bars["atr"] = (bars.high - bars.low).rolling(24).mean()
bars["drift"] = change.rolling(12).sum() / bars.atr
bars["width"] = (bars.expected_high - bars.expected_low) / bars.atr
bars["location"] = (bars.close - bars.expected_low) / (bars.expected_high - bars.expected_low)

ledger = module["replay"](bars, "market", 0, "all", 10, 15, 40, 1.2, details=True)
columns = ["efficiency", "vol_ratio", "atr", "drift", "width", "location"]
# The previous bar is the completed trigger bar. Never use entry-bar OHLC.
features = bars[columns].shift().reindex(ledger.entry_time).reset_index(drop=True)
data = pd.concat([ledger, features], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
data["drift"] *= data.side
data["location"] = np.where(data.side == 1, data.location, 1 - data.location)
rows = []
for year in range(2015, 2023):
    cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
    train = data[data.exit_time < cutoff]
    test = data[data.entry_time.dt.year == year].copy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train[columns], train.pnl)
    score = model.predict(test[columns])
    for name, chosen in [
        ("all", test),
        ("low_efficiency", test[test.efficiency <= .30]),
        ("ridge_expected_net_positive", test[score > 0]),
    ]:
        rows.append({"year": year, "rule": name, "trades": len(chosen),
                     "net": chosen.pnl.sum(), "avg": chosen.pnl.mean()})
result = pd.DataFrame(rows)
result.to_csv(root.parent.parent / "algoforge-live-research/strategy7-ml-screen.csv", index=False)
print(result.groupby("rule")[["trades", "net"]].sum().to_string())
