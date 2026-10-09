import d2
from d2 import *
nine = "USDJPY EURUSD GBPUSD AUDUSD USDCHF EURJPY GBPJPY AUDJPY CHFJPY".split()
jump = (VIX / VIX.shift(20) > 1.3).astype(float).shift(1).fillna(0)
W = w_carry().copy()
for s in PAIRS:
    if s not in nine: W[s] = 0
for name, v in (("vol60", LR.rolling(60, min_periods=20).std()),
                ("max(20,60,250)", pd.concat([LR.rolling(n, min_periods=20).std() for n in (20, 60, 250)]).groupby(level=0).max()),
                ("max(20,250)", pd.concat([LR.rolling(n, min_periods=20).std() for n in (20, 250)]).groupby(level=0).max())):
    d2.VOL = v.shift(1)
    r = book(W).sum(axis=1) / AVAIL[nine].sum(axis=1)
    rep(name, r); rep(name + " +vixjump", r * (1 - 0.7 * jump))
    x = r * (1 - 0.7 * jump); t = x * (0.10 / (x.std() * np.sqrt(252)))
    print("   2007-09 @10%vol:", {y: round(val, 3) for y, val in t.groupby(t.index.year).sum().items() if y in (2007, 2008, 2009)})
