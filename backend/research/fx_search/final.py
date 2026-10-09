"""Candidate: diff-weighted carry + vol guard (+ optional mom60/250 sleeve). Rules fixed before looking at 2023+."""
from d1 import *
W = (CARRY / 0.03).clip(-1.5, 1.5).where(CARRY.abs() > 0.005, 0)
raw = book(W).mean(axis=1)
bv = raw.rolling(20).std().shift(1); med = raw.rolling(500, min_periods=120).std().shift(1)
carry = raw * (med / bv).clip(upper=1).fillna(1)
m = pd.concat([book(sig_mom(L), "W").mean(axis=1) for L in (60, 250)], axis=1).mean(axis=1)
rep("carry*", carry); rep("mom60/250", m)
for w in (0.2, 0.35):
    c = (1-w) * carry / carry.std() + w * m / m.std(); c *= carry.std()
    rep(f"carry*+{w} mom", c)
# robustness: cost x2, rate perturbation (+-0.5% noise per ccy-year), ex-JPY pairs
import d1
rng = np.random.default_rng(0); srs = []
for i in range(20):
    noise = {k: pd.Series(rng.normal(0, 0.005, len(v)), v.index) for k, v in RATES.items()}
    d1.CARRY = pd.DataFrame({s: ((RATES[s[:3]] + noise[s[:3]]) - (RATES[s[3:]] + noise[s[3:]])).reindex(C.index.year).values for s in PAIRS}, index=C.index)
    Wn = (d1.CARRY / 0.03).clip(-1.5, 1.5).where(d1.CARRY.abs() > 0.005, 0); r = d1.book(Wn).mean(axis=1)
    r *= (r.rolling(500, min_periods=120).std().shift(1) / r.rolling(20).std().shift(1)).clip(upper=1).fillna(1)
    srs.append((sharpe(r, 252), sharpe(r["2009":], 252)))
print("rate-noise robustness SR all / 09+: mean %.2f/%.2f min %.2f/%.2f" % (np.mean([a for a,_ in srs]), np.mean([b for _,b in srs]), min(a for a,_ in srs), min(b for _,b in srs)))
print({y: round(v, 3) for y, v in carry.groupby(carry.index.year).sum().items()})
