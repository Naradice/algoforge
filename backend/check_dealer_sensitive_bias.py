"""Throwaway: screen dealer_sensitive bias (shifting both dealer_sensitive_min/max bounds by a
constant delta from the default [-3.5,-1.5]) for (a) stability (no monotonic runaway, same
Spearman-rho check as the num_agent sweep) and (b) whether it actually produces meaningfully
different volatility/trend character -- the two properties needed for it to work as a
regime-defining parameter."""
import random
import numpy as np
from scipy.stats import spearmanr

from data.collectors.ddm_simulator import DDMv3

# delta shifts both bounds; default is min=-3.5, max=-1.5 (delta=0)
DELTAS = [-2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5]


def run_one(delta, n_trades, seed):
    random.seed(seed)
    np.random.seed(seed)
    dmin, dmax = -3.5 + delta, -1.5 + delta
    model = DDMv3(
        num_agent=300,
        max_volatility=0.02,
        min_volatility=0.01,
        wma=5,
        dealer_sensitive_min=dmin,
        dealer_sensitive_max=dmax,
    )
    df = model.simulate(n_trades=n_trades)
    prices = df["price"].values
    rho, _ = spearmanr(prices, np.arange(len(prices)))
    net_change_pct = (prices[-1] - prices[0]) / prices[0] * 100
    returns = np.diff(np.log(prices))
    return rho, net_change_pct, prices.min(), prices.max(), returns.std()


print("=== Stage 1: short screen (8,000 trades, 5 seeds) ===")
for delta in DELTAS:
    rhos, changes, vols = [], [], []
    for seed in range(5):
        rho, chg, pmin, pmax, vol = run_one(delta, 8000, seed)
        rhos.append(rho)
        changes.append(chg)
        vols.append(vol)
    dmin, dmax = -3.5 + delta, -1.5 + delta
    print(f"delta={delta:+.1f} [{dmin:.1f},{dmax:.1f}]: rho={[round(r,2) for r in rhos]} "
          f"net_chg%={[round(c,1) for c in changes]} ret_std={[round(v,5) for v in vols]}")

print("\n=== Stage 2: long confirm (50,000 trades, 3 seeds) ===")
for delta in DELTAS:
    rhos, changes, mins, maxs, vols = [], [], [], [], []
    for seed in range(3):
        rho, chg, pmin, pmax, vol = run_one(delta, 50000, seed)
        rhos.append(rho)
        changes.append(chg)
        mins.append(pmin)
        maxs.append(pmax)
        vols.append(vol)
    dmin, dmax = -3.5 + delta, -1.5 + delta
    print(f"delta={delta:+.1f} [{dmin:.1f},{dmax:.1f}]: rho={[round(r,2) for r in rhos]} "
          f"net_chg%={[round(c,1) for c in changes]} price_range=[{min(mins):.1f},{max(maxs):.1f}] "
          f"ret_std_mean={np.mean(vols):.5f}")
