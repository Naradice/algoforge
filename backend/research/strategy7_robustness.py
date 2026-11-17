"""Strategy 7 robustness comparison with independent pivot data and execution modes."""
import json
from pathlib import Path
import runpy

import pandas as pd

from strategy.engine.backtest import run_backtest
from strategy.engine import forecast_levels
from strategy.engine.standalone_pivot import calculate_daily_pivots

module = runpy.run_path(str(Path(__file__).with_name("sweep_strategy7_execution.py")))
replay = module["replay"]
root = module["ROOT"]
out = root.parent.parent / "algoforge-live-research"
feed = pd.read_csv(out / "strategy7-fx168-levels.csv")
feed = feed[feed.horizon.eq("Daily")].rename(columns={"report_date": "date"})
feed["date"] = pd.to_datetime(feed.date).dt.date
feed["source"] = "FX168"
forecast_levels.get_forecast_records = lambda **_: feed
bars = module["load"]().loc["2025-08-04":"2025-10-31"]
definition = {
    "strategy_type": "analyst_range_reversion", "symbol": "USDJPY",
    "parameters": {"source": "FX168", "sl_pips": 15, "tp_pips": 40,
                   "tolerance_pips": 10, "exit_at_close": True},
    "risk": {"slippage_pct": 0, "commission_pct": 0},
}
trades, _, _ = run_backtest(definition, str(root / "artifacts/datasets/USDJPY_M30_latest.parquet"),
                            from_ts=bars.index[0], to_ts=bars.index[-1])
standard = pd.DataFrame(trades)
manual = replay(bars, "market", 0, "all", 10, 15, 40, 0, details=True, standard=True)
actual = {(t["opened_at"], t["direction"]): (t["closed_at"], t["entry_price"], t["exit_price"])
          for t in trades}
replayed = {(r.entry_time, "buy" if r.side == 1 else "sell"): (r.exit_time, r.entry, r.exit)
            for r in manual.itertuples()}
common = actual.keys() & replayed.keys()
print("Ledger", len(actual), len(replayed), len(common),
      sum(actual[k] == replayed[k] for k in common), flush=True)
standard.to_csv(out / "strategy7-standard-ledger.csv", index=False)
manual.to_csv(out / "strategy7-replay-ledger.csv", index=False)

full = pd.read_parquet(root / "artifacts/datasets/HistData_USDJPY_30min.parquet")
full.index = pd.to_datetime(full.index, utc=True)
full = full.sort_index().loc["2010":"2022"]
pivots = calculate_daily_pivots(full).set_index("date")
full["day"] = full.index.strftime("%Y-%m-%d")
full = full.join(pivots[["expected_low", "expected_high"]], on=full.index.date)
full["minute"] = full.index.hour * 60 + full.index.minute
d = full.close.diff()
full["efficiency"] = full.close.diff(24).abs() / d.abs().rolling(24).sum()
full["vol_ratio"] = d.abs().rolling(12).mean() / d.abs().rolling(96).mean()
cases = [
    ("baseline", "market", 0, "all", 10, 15, 40),
    ("at_pivot_market", "market", 0, "all", 0, 15, 40),
    ("limit_inside", "limit", 0, "all", 10, 15, 40),
    ("limit_pivot", "limit", 0, "all", 0, 15, 40),
    ("limit_outside", "limit", 0, "all", -5, 15, 40),
    ("reclaim", "reclaim", 0, "all", 0, 15, 40),
    ("reclaim_range", "reclaim", 0, "low_efficiency", 0, 15, 40),
    ("retest", "retest", 0, "all", 0, 15, 40),
    ("48h", "market", 2, "all", 10, 15, 40),
    ("inside_at_cutoff", "market", "inside", "all", 10, 15, 40),
    ("profit_at_cutoff", "market", "profit", "all", 10, 15, 40),
    ("room_at_cutoff", "market", "room", "all", 10, 15, 40),
    ("wider_stop", "market", 0, "all", 0, 30, 40),
    ("wider_stop_48h", "market", 2, "all", 0, 30, 40),
]
rows = []
for label, frame in [("FX168_2025", bars)] + [
    (f"Classic_{a}_{b}", full.loc[str(a):str(b)])
    for a, b in [(2010, 2013), (2014, 2017), (2018, 2020), (2021, 2022)]
]:
    for name, *spec in cases:
        gross = replay(frame, *spec, 0, details=True)
        for cost in (0.6, 1.2, 2.4):
            net = gross.pnl - cost
            curve = pd.concat([pd.Series([0.0]), net.cumsum()], ignore_index=True)
            rows.append(dict(period=label, variant=name, cost=cost, trades=len(net),
                             net=net.sum(), avg=net.mean(), dd=(curve.cummax()-curve).max()))
    print(label, "completed", flush=True)
pd.DataFrame(rows).to_csv(out / "strategy7-robustness.csv", index=False)
