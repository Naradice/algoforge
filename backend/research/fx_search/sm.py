"""Spread mean-reversion basket with rolling (expanding-window beta, annual) selection -> walk-forward."""
import numpy as np, pandas as pd, itertools, warnings; warnings.filterwarnings("ignore")
from d2 import *
LP = np.log(C.ffill())
def spread_pnl(a, b, win=60, ent=2.0, ex=0.5):
    d = pd.concat([LP[a], LP[b]], axis=1).dropna()
    # rolling 500d beta (no look-ahead), shifted
    cov = d.iloc[:, 0].rolling(500).cov(d.iloc[:, 1]); var = d.iloc[:, 1].rolling(500).var(); beta = (cov / var).shift(1)
    sp = d.iloc[:, 0] - beta * d.iloc[:, 1]
    z = (sp - sp.rolling(win).mean()) / sp.rolling(win).std()
    ret = LR[a].reindex(d.index) - beta * LR[b].reindex(d.index)
    raw = pd.Series(np.nan, d.index); raw[z > ent] = -1; raw[z < -ent] = 1; raw[z.abs() < ex] = 0
    pos = raw.ffill().fillna(0).shift(1)
    gross = (1 + beta.abs()).fillna(1)
    cost = pos.diff().abs().fillna(0) * (spr[a] / C[a] + beta.abs() * spr[b] / C[b]).reindex(d.index)
    # scale to ~10% vol on spread
    v = ret.rolling(60).std().shift(1); sc = (0.10 / np.sqrt(252) / v).clip(upper=5)
    return (pos * sc * ret - (pos.diff().abs().fillna(0) * sc * (spr[a] / C[a] + beta.abs() * spr[b] / C[b]).reindex(d.index))).dropna()
P = {}
for a, b in itertools.combinations(PAIRS, 2):
    if len({a[:3], a[3:], b[:3], b[3:]}) == 4: continue
    r = spread_pnl(a, b)
    if len(r) > 2500: P[(a, b)] = r
D = pd.DataFrame(P).fillna(0)
print("spreads:", D.shape[1], "mean SR all %.2f" % np.mean([sharpe(D[c], 252) for c in D]))
out = []
for y in range(2013, 2026):
    trn = D[f"{y-5}-01-01":f"{y-1}-12-31"]; srs = trn.apply(lambda x: sharpe(x, 252))
    pick = srs[srs > 0.4].index
    if len(pick) == 0: continue
    out.append(D.loc[str(y), pick].mean(axis=1))
r = pd.concat(out); rep = lambda n, x: print(f"{n} SR={sharpe(x,252):.2f} ann={x.mean()*252:.3f} yrs+={(x.groupby(x.index.year).sum()>0).sum()}/{x.index.year.nunique()} 23+={sharpe(x['2023':],252):.2f}")
rep("WF top-train spreads (SR>0.4 prior 5y)", r)
rep("all spreads equal-weight", D.mean(axis=1)["2013":])
print({y: round(v, 3) for y, v in r.groupby(r.index.year).sum().items()})
