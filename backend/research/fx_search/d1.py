"""Daily-bar framework on MT5 D1 (2000-2025-05). Sleeves: carry, ccy momentum; overlays."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from core import PAIRS, SPREAD, PIP, sharpe
import carry as cy
cy.R["USD"] = pd.concat([cy.R["USD"], pd.Series({2023:5.1,2024:5.2,2025:4.35})/100]) if False else cy.R["USD"]
EXT = {"USD":[5.1,5.2,4.35],"EUR":[3.8,3.8,2.3],"GBP":[4.7,5.1,4.3],"JPY":[0.0,0.2,0.5],"AUD":[4.1,4.35,3.9],"CHF":[1.5,1.2,0.3]}  # APPROX
RATES = {k: pd.concat([cy.R[k], pd.Series(np.array(EXT[k])/100, index=[2023,2024,2025])]) for k in cy.R}
CCY = list(RATES)
def load_c():
    out = {}
    for s in PAIRS:
        d = pd.read_csv(f"L:/data/fx/OANDA-Japan MT5 Live/mt5_{s}_d1.csv", parse_dates=["time"]).set_index("time")
        d = d[(d.index >= "2000-01-01") & (d.close > 0)]
        out[s] = d.close
    C = pd.DataFrame(out).dropna(how="any")
    return C
C = load_c(); LR = np.log(C).diff().fillna(0)
VOL = LR.rolling(60, min_periods=20).std().shift(1)
CARRY = pd.DataFrame({s: (RATES[s[:3]] - RATES[s[3:]]).reindex(C.index.year).values for s in PAIRS}, index=C.index)
def scale(s, tgt=0.10): return (tgt / np.sqrt(252) / VOL[s]).clip(upper=5).fillna(0)
def book(sig, reb="M", carry_on=True):
    """sig: DataFrame of desired direction per pair; rebalance at period end, applied next day."""
    out = {}
    for s in PAIRS:
        g = sig[s]; key = C.index.to_period(reb) if reb != "W" else C.index.to_period("W")
        last = g.groupby(key).last(); g = pd.Series(last.shift(1).reindex(key).values, C.index).fillna(0)
        pos = g * scale(s)
        cost = pos.diff().abs().fillna(0) * SPREAD[s] * PIP(s) / C[s]
        out[s] = pos * LR[s] + (pos * CARRY[s] / 252 if carry_on else 0) - cost
    return pd.DataFrame(out)
def ccy_scores(L):
    sc = pd.DataFrame(0.0, index=C.index, columns=CCY); R = np.log(C).diff(L).fillna(0)
    for s in PAIRS: sc[s[:3]] += R[s]; sc[s[3:]] -= R[s]
    return sc
def sig_mom(L): sc = ccy_scores(L); return pd.DataFrame({s: np.sign(sc[s[:3]] - sc[s[3:]]) for s in PAIRS})
def sig_carry(thr=0.005): return np.sign(CARRY) * (CARRY.abs() > thr)
def sig_xcarry(k=2):
    """Cross-sectional: long top-k, short bottom-k currencies by rate."""
    rk = pd.DataFrame({c: RATES[c].reindex(C.index.year).values for c in CCY}, index=C.index)
    r = rk.rank(axis=1); w = (r > len(CCY) - k).astype(float) - (r <= k).astype(float)
    return pd.DataFrame({s: np.sign(w[s[:3]] - w[s[3:]]) for s in PAIRS})
def rep(n, r):
    f = lambda x: sharpe(x, 252)
    y = r.groupby(r.index.year).sum()
    print(f"{n:28s} SR all={f(r):5.2f} 09+={f(r['2009':]):5.2f} 15+={f(r['2015':]):5.2f} OOS23+={f(r['2023':]):5.2f} ann={r.mean()*252:.3f} DD={(r.cumsum()-r.cumsum().cummax()).min():.3f}")
if __name__ == "__main__":
    S = {}
    S["carry"] = book(sig_carry()).mean(axis=1)
    S["xcarry2"] = book(sig_xcarry(2)).mean(axis=1)
    S["xcarry1"] = book(sig_xcarry(1)).mean(axis=1)
    for L in (5, 60, 120, 250): S[f"mom{L}"] = book(sig_mom(L), "W").mean(axis=1)
    S["mombasket"] = pd.concat([S[f"mom{L}"] for L in (5,60,120,250)], axis=1).mean(axis=1)
    for k, v in S.items(): rep(k, v)
    for c in ("carry", "xcarry2"):
        comb = (S[c] / S[c].std() + S["mombasket"] / S["mombasket"].std()) / 2 * S[c].std()
        rep(f"{c}+mombasket", comb)
