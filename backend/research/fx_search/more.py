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
print("== currency-strength momentum(+1)/reversal(-1), weekly rebalance, net ==")
for L in (5, 20, 60, 120, 250):
    for sg in (1, -1):
        r = run(L, sg); h = r["2010":]
        print(f"L={L:3d} sign={sg:+d} SR={sharpe(r,252):5.2f} SR2010+={sharpe(h,252):5.2f}")

print("== hour-of-day drift with t-stat gate (H1, net) ==")
for sym in ("USDJPY","EURUSD","GBPUSD","EURJPY","GBPJPY","AUDUSD"):
    df = load(sym); lr = np.log(df.close).diff().fillna(0); h = df.index.hour
    for days, tt in ((500, 2.0), (500, 3.0)):
        mu = lr.groupby(h).transform(lambda x: x.rolling(days, min_periods=250).mean()).groupby(h).shift(1)
        sd = lr.groupby(h).transform(lambda x: x.rolling(days, min_periods=250).std()).groupby(h).shift(1)
        t = mu / sd * np.sqrt(days)
        sig = (np.sign(mu) * (t.abs() > tt)).shift(-1).fillna(0)   # known at close of t for bar t+1 (stats exclude bar t+1)
        # NOTE: stats at bar t+1's hour are from bars strictly before t+1 -> need mu at t+1: use mu.shift(-1)
        sig = (np.sign(mu.shift(-1)) * (t.shift(-1).abs() > tt)).fillna(0)
        pos = sig.shift(1).fillna(0)  # held during bar t+1 -> sig at index t+1 position known -> pos[t+1]=sig[t]
        sc = vol_scale(df.close)
        gross = (sig.shift(1).fillna(0) * lr)
        # sig[t] relates to bar t+1; position over bar t+1 = sig[t] => pos_series[t+1]=sig[t]
        P_ = sig.shift(1).fillna(0) * sc
        cost = P_.diff().abs().fillna(0) * SPREAD[sym] * PIP(sym) / df.close
        net = P_ * lr - cost; g = P_ * lr
        print(f"{sym} days={days} t>{tt}: SR gross={sharpe(g):5.2f} net={sharpe(net):5.2f} on_frac={(P_!=0).mean():.2f} net2010+={sharpe(net['2010':]):5.2f}")
