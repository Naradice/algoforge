from d2 import *
nine = "USDJPY EURUSD GBPUSD AUDUSD USDCHF EURJPY GBPJPY AUDJPY CHFJPY".split()
jump = (VIX / VIX.shift(20) > 1.3).astype(float).shift(1).fillna(0)
hi = (VIX > VIX.rolling(250, min_periods=60).median().shift(1)).astype(float).shift(1).fillna(0)
res = {}
for name, U in (("9 pairs", nine), ("30 pairs", PAIRS)):
    W = w_carry().copy()
    for s in PAIRS:
        if s not in U: W[s] = 0
    r = book(W).sum(axis=1) / AVAIL[U].sum(axis=1)
    for gname, g in (("none", 1), ("volguard", None), ("vixjump", 1 - 0.7 * jump), ("volguard+vixjump", None), ("volguard+vixjump+vixmed", None)):
        x = r
        if "volguard" in gname: x = guard(x)
        if "vixjump" in gname: x = x * (1 - 0.7 * jump)
        if "vixmed" in gname: x = x * (1 - 0.3 * hi)
        res[(name, gname)] = x; rep(f"{name} {gname}", x)
x = res[("9 pairs", "volguard+vixjump")]; v = x.std() * np.sqrt(252)
t = x * (0.10 / v); print("scaled to 10% vol (full-sample constant):", {y: round(val, 3) for y, val in t.groupby(t.index.year).sum().items()})
