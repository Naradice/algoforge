import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from d2 import *
from sklearn.ensemble import HistGradientBoostingRegressor
# 5) momentum on carry-adjusted (total) returns
TR = LR + CARRY / 252
def ccy_scores_tr(L):
    sc = pd.DataFrame(0.0, index=C.index, columns=CCY); R = TR.rolling(L).sum()
    for s in PAIRS: sc[s[:3]] += R[s].fillna(0); sc[s[3:]] -= R[s].fillna(0)
    return sc
for L in (60, 250):
    rep(f"mom TR{L}", agg(book(pair_from_scores(ccy_scores_tr(L)), "W")))
    rep(f"mom px{L}", agg(book(pair_from_scores(ccy_scores(L)), "W")))
# 7) monthly pooled GBM
H = 21; ME = C.index.to_series().groupby(C.index.to_period("M")).last().values
rows = []
for s in PAIRS:
    f = pd.DataFrame(index=C.index); v = VOL[s]
    f["carry"] = CARRY[s]; f["chg63"] = CARRY[s] - CARRY[s].shift(63)
    for n in (21, 63, 126, 252): f[f"r{n}"] = np.log(C[s]).diff(n) / (v * np.sqrt(n))
    f["vol"] = v * np.sqrt(252); f["vix"] = VIX; f["vixchg"] = VIX / VIX.shift(21)
    f["y"] = np.log(C[s]).shift(-H).sub(np.log(C[s])) / (v * np.sqrt(H)); f["raw"] = np.log(C[s]).shift(-H) - np.log(C[s])
    f["sym"] = s; f["sc"] = scale(s); f["px"] = C[s]; f["carry_acc"] = CARRY[s] * H / 252
    rows.append(f.loc[f.index.isin(ME)])
X = pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "r252", "vix"]); feats = ["carry","chg63","r21","r63","r126","r252","vol","vix","vixchg"]; X = X.sort_index()
preds = []
for yr in range(2012, 2026):
    trn = X[X.index < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=35)]; tst = X.loc[str(yr)]
    if len(tst) == 0: continue
    m = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.04, max_depth=2, min_samples_leaf=100, l2_regularization=10, random_state=0).fit(trn[feats], trn.y.clip(-4, 4))
    p = tst.copy(); p["pred"] = m.predict(tst[feats]); preds.append(p)
P = pd.concat(preds)
print("monthly GBM IC %.3f" % np.corrcoef(P.pred, P.y)[0, 1], {y: round(np.corrcoef(g.pred, g.y)[0, 1], 2) for y, g in P.groupby(P.index.year)})
for name, pos in (("sign(pred)", np.sign(P.pred)), ("carry-sign", np.sign(P.carry))):
    pnl = (pos * P.sc * (P.raw + P.carry_acc) - pos.groupby(P.sym).diff().abs().fillna(0) * P.sc * 0)  # cost added below
    cost = (pos.groupby(P.sym).diff().abs().fillna(0) * P.sc * pd.Series([spr[s] for s in P.sym], P.index) / P.px)
    r = (pnl - cost).groupby(level=0).mean()
    print(name, "SR=%.2f 23+=%.2f" % (r.mean() / r.std() * np.sqrt(12), r['2023':].mean() / r['2023':].std() * np.sqrt(12)))
