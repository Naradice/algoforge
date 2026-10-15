"""Walk-forward unsupervised regime model (Gaussian mixture) used to scale the carry book."""
import sys, copy, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../.."); sys.path.insert(0, ".")
import portfolio_check as pc
from strategy.portfolio.engine import run_portfolio_backtest, _load_series
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
d = copy.deepcopy(pc.defn); d["overlays"] = []; d["risk"].update(swap_markup=0.015); d["sleeves"][0]["threshold"] = 0.02
book = run_portfolio_backtest(d)["returns"].fillna(0)
vix = _load_series("research/fx/rates/VIXCLS.csv", "VIXCLS", None).reindex(book.index).ffill()
F = pd.DataFrame({"vix": vix, "vix_chg20": vix / vix.shift(20) - 1, "bvol60": book.rolling(60).std() * np.sqrt(252),
                  "bret20": book.rolling(20).sum(), "bdd": (book.cumsum() - book.cumsum().rolling(250).max())}).dropna()
nxt = book.shift(-1).reindex(F.index)
def sr(x): return x.mean() / x.std() * np.sqrt(252)
for K in (2, 3, 4):
    mult = pd.Series(1.0, index=F.index)
    for yr in range(2006, 2027):
        tr = F[F.index < f"{yr}-01-01"]; te = F.index[(F.index >= f"{yr}-01-01") & (F.index <= f"{yr}-12-31")]
        if len(tr) < 750 or len(te) == 0: continue
        sc = StandardScaler().fit(tr); gm = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=3).fit(sc.transform(tr))
        lab_tr = gm.predict(sc.transform(tr)); y = nxt.loc[tr.index]
        mu = pd.Series({k: y[lab_tr == k].mean() for k in range(K)}); cnt = pd.Series({k: (lab_tr == k).sum() for k in range(K)})
        good = mu > 0
        lab_te = gm.predict(sc.transform(F.loc[te]))
        mult.loc[te] = np.where(good.reindex(lab_te).values, 1.0, 0.3)
    adj = (book.reindex(F.index) * mult.shift(1).fillna(1.0))
    base = book.reindex(F.index)
    s = slice("2006", None)
    print(f"K={K}: base SR={sr(base[s]):.2f} DD={((base[s].cumsum()-base[s].cumsum().cummax()).min()):.3f} | regime-scaled SR={sr(adj[s]):.2f} DD={((adj[s].cumsum()-adj[s].cumsum().cummax()).min()):.3f}  (share at 0.3x: {(mult[s]<1).mean():.2f})")
