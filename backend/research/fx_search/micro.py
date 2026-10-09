import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
d = pd.read_parquet("../../../artifacts/datasets/src_42/USDJPY_ticks_1min.parquet"); d.index = pd.to_datetime(d.index)
d = d.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","spread":"mean","tick_count":"sum","bid_size":"sum","ask_size":"sum","imbalance":"mean","ofi":"sum"}).dropna(subset=["close"])
d = d[d.tick_count > 0]; print(len(d), d.index[0], d.index[-1], "median spread", d.spread.median())
r = np.log(d.close).diff()
f = pd.DataFrame(index=d.index)
for n in (1, 3, 10, 30): f[f"r{n}"] = r.rolling(n).sum() / (r.rolling(300).std() * np.sqrt(n))
for n in (1, 5, 15): f[f"ofi{n}"] = d.ofi.rolling(n).sum() / d.tick_count.rolling(n).sum().clip(lower=1); f[f"imb{n}"] = d.imbalance.rolling(n).mean()
f["spread"] = d.spread; f["tc"] = np.log1p(d.tick_count); f["hr"] = d.index.hour
for H in (1, 5, 15):
    y = (np.log(d.close).shift(-H) - np.log(d.close)); sd = r.rolling(300).std() * np.sqrt(H)
    X = f.assign(y=y / sd, raw=y, px=d.close, sp=d.spread).replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    for yr in range(2019, 2024):
        trn = X[X.index < f"{yr}-01-01"].iloc[::5]; tst = X.loc[str(yr)]
        if len(tst) == 0: continue
        m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=4, min_samples_leaf=500, random_state=0).fit(trn[list(f.columns)], trn.y.clip(-5, 5))
        p = tst.copy(); p["pred"] = m.predict(tst[list(f.columns)]); out.append(p)
    P = pd.concat(out); ic = np.corrcoef(P.pred, P.y)[0, 1]
    # trade top/bottom decile predictions, hold H minutes (non-overlapping approx by taking all, report per-trade expectation)
    q = P.pred.abs().quantile(0.9); T = P[P.pred.abs() >= q]
    gross = (np.sign(T.pred) * T.raw).mean() * 1e4        # in bp
    cost_bp = (T.sp.mean()) / T.px.mean() * 1e4
    print(f"H={H}m IC={ic:.4f} top-decile trades={len(T)} gross={gross:.3f}bp/trade spread_cost~{cost_bp:.3f}bp")
