"""Reconstructed-schedule event study: US Non-Farm Payrolls (first Friday, 08:30 New York time) on HistData H1.
HistData timestamps are fixed EST (UTC-5, no DST); 08:30 New York time is 08:30 EST in winter, 07:30 EST in summer."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from core import load, SPREAD, PIP, sharpe
USD_PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCHF"]
SIGN = {"USDJPY": 1, "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "USDCHF": 1}   # + = USD up

def us_dst(ts):
    y = ts.year
    start = pd.Timestamp(y, 3, 8) + pd.offsets.Week(weekday=6)      # second Sunday of March
    end = pd.Timestamp(y, 11, 1) + pd.offsets.Week(weekday=6) if pd.Timestamp(y, 11, 1).weekday() != 6 else pd.Timestamp(y, 11, 1)
    return start <= ts < end

def nfp_dates(y0=2001, y1=2022):
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            d = pd.Timestamp(y, m, 1)
            fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
            out.append(fri)
    return out

rows = []
for s in USD_PAIRS:
    df = load(s); c = df.close; lr = np.log(c).diff()
    for d in nfp_dates():
        t0 = d + pd.Timedelta(hours=7 if us_dst(d) else 8)          # bar covering 08:30 NY = the bar starting at 08:00/07:00 EST
        if t0 not in c.index:
            continue
        i = c.index.get_loc(t0)
        if i < 30 or i + 8 >= len(c): continue
        pre = float(np.log(c.iloc[i]) - np.log(c.iloc[i - 1]))         # return of the bar before release
        r0 = float(np.log(c.iloc[i + 1] / c.iloc[i]) )                 # return of the release hour (close of release bar vs its open ~ c[i]) 
        # decision at close of release bar (i): trade next k hours in direction of the release-bar move
        move = float(np.log(c.iloc[i] / c.iloc[i - 1]))
        for k in (1, 2, 4, 8):
            fwd = float(np.log(c.iloc[i + k] / c.iloc[i]))
            rows.append((s, d, k, SIGN[s] * move, SIGN[s] * fwd, np.sign(move) * fwd, SPREAD[s] * PIP(s) / c.iloc[i], float(lr.iloc[i - 24:i].std())))
R = pd.DataFrame(rows, columns=["pair", "date", "k", "move", "fwd", "follow", "cost", "vol"])
R["follow_net"] = R.follow - R.cost
print("n events per pair:", R[R.k == 1].groupby("pair").size().to_dict())
for k in (1, 2, 4, 8):
    g = R[R.k == k]
    t = g.follow.mean() / g.follow.std() * np.sqrt(len(g)); tn = g.follow_net.mean() / g.follow_net.std() * np.sqrt(len(g))
    print(f"follow release-bar move, hold {k}h: mean gross {g.follow.mean()*1e4:6.2f}bp (t={t:5.2f})  net {g.follow_net.mean()*1e4:6.2f}bp (t={tn:5.2f})  hit {np.mean(g.follow>0):.2f}  n={len(g)}")
    # fade version
    print(f"     fade: net {(-g.follow - g.cost).mean()*1e4:6.2f}bp")
big = R[(R.k == 4)]
big = big[big.move.abs() > big.vol * 2]      # large release-hour moves only
print("large-move events k=4: n", len(big), "follow gross %.2fbp net %.2fbp" % (big.follow.mean()*1e4, big.follow_net.mean()*1e4))
