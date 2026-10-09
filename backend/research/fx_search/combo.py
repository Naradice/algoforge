import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from core import *
import more_lib as m, carry as cy
mom = {L: m.run(L, +1) for L in (5, 60, 120, 250)}
M = pd.concat(mom, axis=1).mean(axis=1)
Cy = pd.concat({s: cy.pair_daily(s, use_trend=False)[0] for s in PAIRS}, axis=1).fillna(0).mean(axis=1)
idx = M.index.intersection(Cy.index); M, Cy = M[idx], Cy[idx]
def rep(n, r):
    y = r.groupby(r.index.year).sum()
    print(f"{n:22s} SR={sharpe(r,252):5.2f} SR09+={sharpe(r['2009':],252):5.2f} ann={r.mean()*252:.3f} maxDD={(r.cumsum()-r.cumsum().cummax()).min():.3f} yrs+={(y>0).sum()}/{len(y)}")
rep("momentum basket", M); rep("carry", Cy)
for w in (0.5, ):
    rep("mom+carry 50/50 (inv-vol)", (M/M.std() + Cy/Cy.std()) / 2 * (M.std()+Cy.std())/2)
print("corr", np.corrcoef(M, Cy)[0,1])
