"""Train(<=2015)/test(>=2016) scans: cross-asset lead-lag, calendar effects, pair-spread mean reversion."""
import numpy as np, pandas as pd, glob, itertools, warnings; warnings.filterwarnings("ignore")
from d2 import *
tr = lambda x: x[:"2015"]; te = lambda x: x["2016":]
# ---------- B: cross-asset lead-lag (asset return today -> pair return tomorrow, 1d and 5d)
others = {}
for f in glob.glob(str(MT5 / "mt5_*_d1.csv")):
    n = pathlib.Path(f).stem[4:-3]
    if n in PAIRS: continue
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time").close; d = d[d.index >= "2000-01-01"]
    if len(d) > 3000: others[n] = d
O = pd.DataFrame(others).reindex(C.index).ffill(limit=3); OR = np.log(O).diff()
res = []
for H in (1, 5):
    tgt = LR.rolling(H).sum().shift(-H) if H > 1 else LR.shift(-1)
    for a in OR.columns:
        x = OR[a].rolling(H).sum() if H > 1 else OR[a]
        for p in PAIRS:
            df = pd.concat([x, tgt[p]], axis=1).dropna(); df.columns = ["x", "y"]
            if len(tr(df)) < 800 or len(te(df)) < 400: continue
            a_, b_ = tr(df).corr(method="spearman").iloc[0, 1], te(df).corr(method="spearman").iloc[0, 1]
            res.append((H, a, p, a_, b_, len(tr(df)), len(te(df))))
R = pd.DataFrame(res, columns=["H","asset","pair","ic_tr","ic_te","ntr","nte"])
R["t_tr"] = R.ic_tr * np.sqrt(R.ntr / (H if False else 1))
sel = R[(R.ic_tr.abs() * np.sqrt(R.ntr) > 3.0)]
print("lead-lag tests:", len(R), " significant in train (|t|>3):", len(sel), " same-sign in test:", int((np.sign(sel.ic_tr) == np.sign(sel.ic_te)).sum()),
      " test |t|>2 & same sign:", int(((np.sign(sel.ic_tr) == np.sign(sel.ic_te)) & (sel.ic_te.abs() * np.sqrt(sel.nte) > 2)).sum()))
print(sel.assign(t_te=sel.ic_te * np.sqrt(sel.nte)).sort_values("t_te", key=abs, ascending=False).head(10).round(3).to_string())
# ---------- C: calendar effects (day of week, day of month bucket) per pair, 1-day returns
cal = []
for p in PAIRS:
    r = LR[p]
    for name, key in (("dow", r.index.dayofweek), ("dom", np.minimum(r.index.day // 5, 5)), ("moy", r.index.month)):
        g_tr = tr(r).groupby(key[:len(tr(r))]).mean(); g_te = te(r).groupby(key[len(tr(r)):]).mean()
        sd = r.std()
        for k in g_tr.index:
            t_tr = g_tr[k] / (sd / np.sqrt((key[:len(tr(r))] == k).sum()))
            if abs(t_tr) > 3.0: cal.append((p, name, k, round(t_tr, 2), round(g_te.get(k, np.nan) / (sd / np.sqrt((key[len(tr(r)):] == k).sum())), 2)))
cal = pd.DataFrame(cal, columns=["pair","type","k","t_tr","t_te"])
print("calendar: train-sig", len(cal), "same-sign test", int((np.sign(cal.t_tr) == np.sign(cal.t_te)).sum()), "test |t|>2 same sign", int(((np.sign(cal.t_tr) == np.sign(cal.t_te)) & (cal.t_te.abs() > 2)).sum()))
print(cal.head(12).to_string())
# ---------- D: pair-spread mean reversion (log price spread of two pairs, z-score), train select, test trade
LP = np.log(C.ffill())
cand = []
for a, b in itertools.combinations(PAIRS, 2):
    if len({a[:3], a[3:], b[:3], b[3:]}) == 4: continue    # require shared currency
    # orient: regress a on b over train
    d = pd.concat([LP[a], LP[b]], axis=1).dropna(); dtr = tr(d)
    if len(dtr) < 800: continue
    beta = np.polyfit(dtr.iloc[:, 1], dtr.iloc[:, 0], 1)[0]
    sp = d.iloc[:, 0] - beta * d.iloc[:, 1]
    z = (sp - sp.rolling(60).mean()) / sp.rolling(60).std()
    ret = (LR[a] - beta * LR[b]).reindex(z.index)
    pos = (-np.sign(z) * (z.abs() > 2)).where(z.abs() > 0.5).ffill().fillna(0).shift(1)    # enter |z|>2, exit |z|<0.5
    cost = pos.diff().abs().fillna(0) * (spr[a] / C[a] + abs(beta) * spr[b] / C[b]).reindex(z.index)
    pnl = (pos * ret - cost).dropna(); v = pnl.std()
    cand.append((a, b, sharpe(tr(pnl), 252), sharpe(te(pnl), 252), pnl))
cd = pd.DataFrame([(a, b, s1, s2) for a, b, s1, s2, _ in cand], columns=["a","b","sr_tr","sr_te"])
print("spread MR: n=", len(cd), "mean SR tr/te: %.2f/%.2f" % (cd.sr_tr.mean(), cd.sr_te.mean()), " corr(tr,te)=%.2f" % cd.sr_tr.corr(cd.sr_te))
top = cd.sort_values("sr_tr", ascending=False).head(10); print(top.round(2).to_string())
