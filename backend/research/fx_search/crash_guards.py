"""Candidate crash guards for the carry book from freely available cross-asset indicators (few fixed-parameter rules)."""
import sys, copy, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../.."); sys.path.insert(0, ".")
import portfolio_check as pc
from strategy.portfolio.engine import run_portfolio_backtest
d = copy.deepcopy(pc.defn); d["overlays"] = []; d["risk"].update(swap_markup=0.015); d["sleeves"][0]["threshold"] = 0.02
book = run_portfolio_backtest(d)["returns"].fillna(0)
X = pd.read_parquet(pc.ROOT / "artifacts/research/fx/stress_indicators.parquet")
X.index = pd.to_datetime(X.index).tz_localize(None).normalize()
X = X.reindex(book.index).ffill()
sr = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else float("nan")
dd = lambda x: (x.cumsum() - x.cumsum().cummax()).min()
rules = {
    "VIX > 30": X["^VIX"] > 30,
    "VIX 20d jump > 30%": X["^VIX"] / X["^VIX"].shift(20) > 1.3,
    "VIX > 1y 80th pct": X["^VIX"] > X["^VIX"].rolling(250, min_periods=120).quantile(0.8),
    "MOVE > 1y 80th pct": X["^MOVE"] > X["^MOVE"].rolling(250, min_periods=120).quantile(0.8),
    "HYG-LQD 20d < -2%": (np.log(X["HYG"]).diff(20) - np.log(X["LQD"]).diff(20)) < -0.02,
    "S&P 20d < -7%": np.log(X["^GSPC"]).diff(20) < -0.07,
    "S&P below 200d MA": X["^GSPC"] < X["^GSPC"].rolling(200).mean(),
    "Nikkei 20d < -8%": np.log(X["^N225"]).diff(20) < -0.08,
    "DXY 20d > +4%": np.log(X["DX-Y.NYB"]).diff(20) > 0.04,
}
s = slice("2007-06", None)
print(f"{'base (no guard)':26s} SR={sr(book[s]):5.2f} DD={dd(book[s]):.3f} 2008={book['2008'].sum():.3f}")
for name, cond in rules.items():
    m = 1.0 - 0.7 * cond.astype(float).shift(1).fillna(0)
    adj = book * m
    print(f"{name:26s} SR={sr(adj[s]):5.2f} DD={dd(adj[s]):.3f} 2008={adj['2008'].sum():.3f}  active={cond[s].mean():.2f}")
combo = (rules["VIX 20d jump > 30%"] | rules["S&P below 200d MA"]).astype(float).shift(1).fillna(0)
adj = book * (1 - 0.7 * combo); print(f"{'VIX jump OR S&P<200d':26s} SR={sr(adj[s]):5.2f} DD={dd(adj[s]):.3f} 2008={adj['2008'].sum():.3f}")
