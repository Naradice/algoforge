import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from core import *
CCY = ["USD","EUR","GBP","JPY","AUD","CHF"]
D = {}
for s in PAIRS:
    df = load(s); D[s] = df.close.resample("1D").last()
C = pd.DataFrame(D).dropna(how="all"); LR = np.log(C).diff()
VOL = LR.rolling(60, min_periods=20).std().shift(1)
def ccy_scores(L):
    sc = pd.DataFrame(0.0, index=C.index, columns=CCY); n = pd.Series(0.0, index=CCY)
    R = np.log(C).diff(L)
    for s in PAIRS:
        b, q = s[:3], s[3:]
        sc[b] += R[s].fillna(0); sc[q] -= R[s].fillna(0)
    return sc
def run(L, sign, reb=5, cost=True):
    sc = ccy_scores(L); out = {}
    for s in PAIRS:
        sig = np.sign(sc[s[:3]] - sc[s[3:]]) * sign
        sig = sig.iloc[::reb].reindex(C.index).ffill().shift(1).fillna(0)
        pos = sig * (0.10 / np.sqrt(252) / VOL[s]).clip(upper=5).fillna(0)
        c = pos.diff().abs().fillna(0) * SPREAD[s] * PIP(s) / C[s] if cost else 0
        out[s] = pos * LR[s].fillna(0) - c
    return pd.DataFrame(out).mean(axis=1)
