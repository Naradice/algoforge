import pickle, pandas as pd, numpy as np, sys
from core import *
res = pickle.load(open(DATA / "screen_daily.pkl", "rb"))
keys = list(res["USDJPY"].keys())
D = {k: pd.concat({s: res[s][k] for s in PAIRS}, axis=1).fillna(0) for k in keys}   # daily net ret per pair
P = {k: v.mean(axis=1) for k, v in D.items()}                                       # equal-weight across pairs

def wf(fam, train_years=3, first=2004, last=2022):
    ks = [k for k in keys if k[0] == fam]; out = []
    for y in range(first, last + 1):
        tr = slice(f"{y-train_years}-01-01", f"{y-1}-12-31")
        best = max(ks, key=lambda k: sharpe(P[k][tr], 252))
        out.append(P[best][str(y)])
    return pd.concat(out)

if __name__ == "__main__":
    print("== full-sample portfolio Sharpe by family (mean/max over configs) ==")
    t = pd.DataFrame([(k[0], sharpe(P[k], 252)) for k in keys], columns=["fam", "sr"])
    print(t.groupby("fam").sr.agg(["mean", "max", "min"]).round(2))
    print("== walk-forward OOS 2004-2022 (equal-weight 9 pairs, net of cost) ==")
    W = {f: wf(f) for f in FAMILIES}
    for f, r in W.items():
        yr = r.groupby(r.index.year).sum()
        print(f"{f:9s} SR={sharpe(r,252):5.2f} ann={r.mean()*252:6.3f} pos_years={(yr>0).sum()}/{len(yr)}")
    comb = pd.concat(W, axis=1).fillna(0).mean(axis=1)
    print("COMBINED  SR=%.2f ann=%.3f" % (sharpe(comb, 252), comb.mean() * 252))
    print(comb.groupby(comb.index.year).sum().round(3).to_dict())
