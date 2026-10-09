"""Wide-universe daily framework: 10 currencies, 30 MT5 D1 pairs, real FRED rates, VIX."""
import numpy as np, pandas as pd, pathlib, warnings; warnings.filterwarnings("ignore")
from core import sharpe
MT5 = pathlib.Path("L:/data/fx/OANDA-Japan MT5 Live")
RD_ = pathlib.Path(__file__).resolve().parents[3] / "artifacts" / "research" / "fx" / "rates"
PAIRS = "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY NZDUSD USDCAD USDCHF USDJPY USDNOK USDSEK".split()
CCY = ["USD","EUR","GBP","JPY","AUD","CHF","CAD","NZD","NOK","SEK"]
IDS = dict(USD="US", EUR="EZ", GBP="GB", JPY="JP", AUD="AU", CHF="CH", CAD="CA", NZD="NZ", NOK="NO", SEK="SE")
def fred(i):
    d = pd.read_csv(RD_ / f"{i}.csv", index_col=0, parse_dates=True).iloc[:, 0]; return pd.to_numeric(d, errors="coerce").dropna()
closes, spr = {}, {}
for s in PAIRS:
    d = pd.read_csv(MT5 / f"mt5_{s}_d1.csv", parse_dates=["time"]).set_index("time")
    closes[s] = d.close[d.index >= "2000-01-01"]
    pt = 0.001 if s.endswith("JPY") else 0.00001
    spr[s] = max(2 * d.spread[-750:].median(), 10) * pt      # price units, 2x recent median, floor 1 pip
C = pd.DataFrame(closes).dropna(how="all"); C = C[C.index.dayofweek < 5]
C = C.dropna(thresh=20)
LR = np.log(C).diff().fillna(0)
VOL = LR.rolling(60, min_periods=20).std().shift(1)
rm = pd.DataFrame({c: fred(f"IR3TIB01{IDS[c]}M156N") / 100 for c in CCY})
rm["JPY"] = rm["JPY"].fillna(fred("IRSTCB01JPM156N") / 100)
RD = rm.shift(1).reindex(C.index.union(rm.index)).ffill().reindex(C.index).ffill().bfill()
CARRY = pd.DataFrame({s: RD[s[:3]] - RD[s[3:]] for s in PAIRS})
VIX = fred("VIXCLS").reindex(C.index).ffill()
AVAIL = C.notna()
def scale(s, tgt=0.10): return (tgt / np.sqrt(252) / VOL[s]).clip(upper=5).fillna(0)
def book(W, reb="M", carry_on=True):
    out = {}
    for s in PAIRS:
        key = C.index.to_period(reb); g = W[s].groupby(key).last()
        g = pd.Series(g.shift(1).reindex(key).values, C.index).fillna(0) * AVAIL[s]
        pos = g * scale(s)
        cost = pos.diff().abs().fillna(0) * spr[s] / C[s]
        out[s] = pos * LR[s] + (pos * CARRY[s] / 252 if carry_on else 0) - cost
    return pd.DataFrame(out)
def agg(B): return B.sum(axis=1) / AVAIL.sum(axis=1).clip(lower=1)    # equal weight over available pairs
def ccy_scores(L, sign=1):
    sc = pd.DataFrame(0.0, index=C.index, columns=CCY); R = np.log(C).diff(L).fillna(0)
    for s in PAIRS: sc[s[:3]] += R[s]; sc[s[3:]] -= R[s]
    return sc * sign
def pair_from_scores(sc, thr=0): return pd.DataFrame({s: np.sign(sc[s[:3]] - sc[s[3:]]) for s in PAIRS})
def w_carry(thr=0.005, sc_=0.03, cap=1.5): return (CARRY / sc_).clip(-cap, cap).where(CARRY.abs() > thr, 0)
def guard(r, cap=1.0):
    bv = r.rolling(20).std().shift(1); med = r.rolling(500, min_periods=120).std().shift(1)
    return r * (cap * med / bv).clip(upper=1).fillna(1)
def rep(n, r):
    f = lambda x: sharpe(x, 252) if len(x) > 50 else float("nan")
    print(f"{n:30s} SR all={f(r):5.2f} 09+={f(r['2009':]):5.2f} 15+={f(r['2015':]):5.2f} 23+={f(r['2023':]):5.2f} ann={r.mean()*252:.3f} DD={(r.cumsum()-r.cumsum().cummax()).min():.3f}")
if __name__ == "__main__":
    print(C.shape, C.index[0].date(), C.index[-1].date())
    S = {}
    S["carry diffw"] = agg(book(w_carry())); S["carry*"] = guard(S["carry diffw"])
    S["carry sign"] = agg(book(np.sign(CARRY) * (CARRY.abs() > 0.005)))
    for L in (60, 250): S[f"mom{L}"] = agg(book(pair_from_scores(ccy_scores(L)), "W"))
    S["mom60/250"] = (S["mom60"] + S["mom250"]) / 2
    for L in (756, 1260): S[f"value rev{L}"] = agg(book(pair_from_scores(ccy_scores(L, -1)), "M"))
    # VIX guard on carry: halve when VIX above its trailing 1y median, or 20d change > +30%
    hi = (VIX > VIX.rolling(250, min_periods=60).median().shift(1)).astype(float)
    S["carry* vixmed"] = S["carry*"] * (1 - 0.5 * hi.shift(1).fillna(0))
    jump = (VIX / VIX.shift(20) > 1.3).astype(float)
    S["carry* vixjump"] = S["carry*"] * (1 - 0.7 * jump.shift(1).fillna(0))
    for k, v in S.items(): rep(k, v)
    import pickle; pickle.dump(S, open(RD_.parent / "d2_sleeves.pkl", "wb"))
