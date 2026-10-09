from d1 import *
from sklearn.ensemble import HistGradientBoostingRegressor
# 1) rate-change momentum: sign of 3m / 6m change in differential
for n in (63, 126):
    ch = CARRY - CARRY.shift(n)
    rep(f"ratechg{n}", book(np.sign(ch).where(ch.abs() > 0.002, 0)).mean(axis=1))
# 2) daily GBM walk-forward, weekly horizon 21d, features carry/chg/mom/vol/ccy-scores
H = 21; rows = []
sc = {L: ccy_scores(L) for L in (20, 60, 250)}
for s in PAIRS:
    f = pd.DataFrame(index=C.index)
    f["carry"] = CARRY[s]; f["chg63"] = CARRY[s] - CARRY[s].shift(63); f["chg126"] = CARRY[s] - CARRY[s].shift(126)
    v = VOL[s]
    for n in (5, 20, 60, 120, 250): f[f"r{n}"] = np.log(C[s]).diff(n) / (v * np.sqrt(n))
    for L in (20, 60, 250): f[f"cs{L}"] = sc[L][s[:3]] - sc[L][s[3:]]
    f["vol"] = v * np.sqrt(252); f["volr"] = LR[s].rolling(10).std().shift(1) / v
    f["y"] = np.log(C[s]).shift(-H).sub(np.log(C[s])) / (v * np.sqrt(H)); f["sym"] = s
    f["raw"] = np.log(C[s]).shift(-H) - np.log(C[s]); f["scale"] = scale(s); f["px"] = C[s]
    rows.append(f)
X = pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "carry", "r250"]); feats = [c for c in X.columns if c not in ("y","sym","raw","scale","px")]
preds = []
for yr in range(2008, 2026):
    tr = X[X.index < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=H*2)]; te = X[(X.index >= f"{yr}-01-01") & (X.index <= f"{yr}-12-31")]
    if len(te) == 0: continue
    m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.04, max_depth=3, min_samples_leaf=200, l2_regularization=5, random_state=0).fit(tr[feats], tr.y.clip(-4, 4))
    p = te[["sym","raw","scale","px","y"]].copy(); p["pred"] = m.predict(te[feats]); preds.append(p)
P = pd.concat(preds); print("GBM IC", round(np.corrcoef(P.pred, P.y)[0,1], 4), "yearly:", {y: round(np.corrcoef(g.pred, g.y)[0,1],3) for y, g in P.groupby(P.index.year)})
out = {}
for s, g in P.groupby("sym"):
    g = g.iloc[::H]; pos = np.sign(g.pred) * g.scale
    out[s] = pos * g.raw - pos.diff().abs().fillna(0) * SPREAD[s] * PIP(s) / g.px
r = pd.DataFrame(out).fillna(0).mean(axis=1); n = 252 / H
print(f"GBM net: SR={r.mean()/r.std()*np.sqrt(n):.2f} 15+={r['2015':].mean()/r['2015':].std()*np.sqrt(n):.2f} OOS23+={r['2023':].mean()/r['2023':].std()*np.sqrt(n):.2f} (price only, excludes carry accrual)")
