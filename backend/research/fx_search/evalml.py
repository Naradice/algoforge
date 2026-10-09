import sys, numpy as np, pandas as pd
from core import *
H = int(sys.argv[1]); P = pd.read_parquet(DATA / f"ml_preds_H{H}.parquet")
def ev(P, thr, cost=True):
    out = []
    for s, g in P.groupby("sym"):
        g = g.iloc[::H]                       # rebalance every H bars (non-overlapping)
        pos = np.where(g.pred.abs() > thr, np.sign(g.pred), 0) * g.scale
        gross = pos * g.raw_fwd
        turn = pd.Series(pos, g.index).diff().abs().fillna(0)
        c = turn * SPREAD[s] * PIP(s) / g.close if cost else 0
        out.append((gross - c).rename(s))
    r = pd.concat(out, axis=1).fillna(0).mean(axis=1).sort_index()
    return r
for thr in (0, 0.1, 0.2, 0.3):
    for cost in (False, True):
        r = ev(P, thr, cost); n = 24 * 252 / H
        print(f"thr={thr} cost={cost} SR={r.mean()/r.std()*np.sqrt(252*24/H):.2f} ann={r.mean()*n:.3f} years+={(r.groupby(r.index.year).sum()>0).sum()}/{r.index.year.nunique()}")
