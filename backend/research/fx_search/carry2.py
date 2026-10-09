from d1 import *
base = sig_carry()
def book_w(W, reb="M"):
    """W: signed weights (not just sign), applied like book()."""
    return book(W, reb)
res = {}
res["carry sign"] = book(base).mean(axis=1)
# magnitude-weighted: carry/vol-ish, capped
mag = (CARRY / (VOL * np.sqrt(252))).clip(-1.5, 1.5)
mag = mag.where(CARRY.abs() > 0.005, 0)
res["carry/vol weight"] = book(mag).mean(axis=1)
# proportional to diff only
res["carry diff weight"] = book((CARRY / 0.03).clip(-1.5, 1.5).where(CARRY.abs() > 0.005, 0)).mean(axis=1)
# crash guard: basket vol of sign-carry book relative to its 1y median
raw = book(base).mean(axis=1)
bv = raw.rolling(20).std().shift(1); med = raw.rolling(500, min_periods=120).std().shift(1)
for cap in (1.0, 1.5):
    g = (cap * med / bv).clip(upper=1).fillna(1)
    res[f"carry volguard{cap}"] = raw * g
# pair stop: drop pair when 20d return against position exceeds 1.5 sigma_20
r20 = np.log(C).diff(20)
stop = (np.sign(base) * r20 / (VOL * np.sqrt(20)) < -1.5)
res["carry pairstop"] = book(base.mask(stop, 0)).mean(axis=1)
res["carry diffw+volguard"] = book((CARRY / 0.03).clip(-1.5, 1.5).where(CARRY.abs() > 0.005, 0)).mean(axis=1) * (1.0 * med / bv).clip(upper=1).fillna(1)
for k, v in res.items(): rep(k, v)
