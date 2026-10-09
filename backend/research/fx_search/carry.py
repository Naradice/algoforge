"""Carry (+ trend filter) across 9 pairs. Policy rates are APPROXIMATE annual averages (%), from memory."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from core import *
yrs = list(range(2000, 2023))
R = {
 "USD": [6.2,3.9,1.7,1.1,1.4,3.2,5.0,5.0,1.9,.25,.25,.25,.25,.25,.25,.3,.5,1.1,1.9,2.2,.5,.1,1.7],
 "EUR": [3.7,4.3,3.3,2.3,2.0,2.0,2.8,3.9,3.8,1.0,1.0,1.25,.75,.5,.15,.05,0,0,0,0,0,0,.6],
 "GBP": [5.9,5.0,4.0,3.7,4.5,4.7,4.6,5.5,4.6,.6,.5,.5,.5,.5,.5,.5,.35,.3,.6,.75,.25,.1,1.4],
 "JPY": [.05,.05,0,0,0,0,.1,.5,.4,.1,.1,.1,.1,.1,.1,.1,-.05,-.1,-.1,-.1,-.1,-.1,-.1],
 "AUD": [5.9,5.1,4.7,4.8,5.25,5.5,5.8,6.4,6.0,3.1,4.4,4.7,3.5,2.75,2.5,2.1,1.7,1.5,1.5,1.1,.2,.1,1.9],
 "CHF": [3.2,2.8,1.2,.3,.5,.75,1.5,2.5,2.1,.25,.25,.1,.1,.1,0,-.5,-.75,-.75,-.75,-.75,-.75,-.75,-.2],
}
R = {k: pd.Series(v, index=yrs) / 100 for k, v in R.items()}
def diff(pair, idx):
    b, q = pair[:3], pair[3:]
    return ((R[b] - R[q]).reindex(idx.year).values)

def pair_daily(sym, trend_days=120, use_trend=True, thr=0.005, rebal="M"):
    df = load(sym); d = df.close.resample("1D").last().dropna()
    lr = np.log(d).diff().fillna(0)
    carry = pd.Series(diff(sym, d.index), d.index)
    sig = np.sign(carry) * (carry.abs() > thr)
    if use_trend:
        tr = np.sign(np.log(d).diff(trend_days))
        sig = sig.where(sig == tr, 0)          # hold only when trend agrees with carry
    # monthly rebalance: take month-end signal, hold through next month
    if rebal == "M":
        me = sig.groupby([d.index.year, d.index.month]).transform("last")
        sig = me.shift(1).ffill().fillna(0)    # previous month's last signal
        # shift by month: use previous month's value for the whole current month
        key = d.index.to_period("M"); last = sig.groupby(key).last()
        sig = pd.Series(last.shift(1).reindex(key).values, d.index).fillna(0)
    vol = lr.rolling(60, min_periods=20).std().shift(1)
    sc = (0.10 / np.sqrt(252) / vol).clip(upper=5).fillna(0)
    pos = sig * sc
    pos_l = pos.shift(0)
    accr = pos_l * carry / 252
    cost = pos_l.diff().abs().fillna(0) * SPREAD[sym] * PIP(sym) / d
    return pos_l * lr + accr - cost, pos_l * lr - cost

if __name__ == "__main__":
    for ut in (False, True):
        net = pd.concat({s: pair_daily(s, use_trend=ut)[0] for s in PAIRS}, axis=1).fillna(0)
        px = pd.concat({s: pair_daily(s, use_trend=ut)[1] for s in PAIRS}, axis=1).fillna(0)
        p, q = net.mean(axis=1), px.mean(axis=1)
        print(f"trend_filter={ut}: SR(total)={sharpe(p,252):.2f} ann={p.mean()*252:.3f} | SR(price only)={sharpe(q,252):.2f} ann={q.mean()*252:.3f}")
        yr = p.groupby(p.index.year).sum()
        print("   yearly:", yr.round(3).to_dict())
