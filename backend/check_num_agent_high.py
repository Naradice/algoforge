"""Throwaway: verify DDMv3 stability at HIGHER num_agent values (300 default up to 5000), using
the same two-stage methodology as the original low-end sensitivity sweep (see memory
project_ddm_parameter_sensitivity_sweep) -- short screen (8,000 trades, 5 seeds) via
Spearman rho(price, tick index), then a long confirm (50,000 trades, 3 seeds) on anything that
looks risky, since short windows show spurious "trend" even in genuinely stable processes."""
import random
import numpy as np
from scipy.stats import spearmanr

from data.collectors.ddm_simulator import DDMv3

NUM_AGENT_VALUES = [300, 500, 800, 1500, 3000, 5000]


def run_one(num_agent, n_trades, seed):
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=num_agent,
        max_volatility=0.02,
        min_volatility=0.01,
        wma=5,
        dealer_sensitive_min=-3.5,
        dealer_sensitive_max=-1.5,
    )
    df = model.simulate(n_trades=n_trades)
    prices = df["price"].values
    rho, _ = spearmanr(prices, np.arange(len(prices)))
    net_change_pct = (prices[-1] - prices[0]) / prices[0] * 100
    return rho, net_change_pct, prices.min(), prices.max()


print("=== Stage 1: short screen (8,000 trades, 5 seeds) ===")
stage1_flagged = []
for na in NUM_AGENT_VALUES:
    rhos, changes = [], []
    for seed in range(5):
        rho, chg, pmin, pmax = run_one(na, 8000, seed)
        rhos.append(rho)
        changes.append(chg)
    print(f"num_agent={na}: rho={[round(r,3) for r in rhos]} net_chg%={[round(c,1) for c in changes]}")
    if any(abs(r) > 0.3 for r in rhos) or any(abs(c) > 15 for c in changes):
        stage1_flagged.append(na)

print(f"\nFlagged for stage 2: {stage1_flagged if stage1_flagged else 'none -- testing all anyway for a real long-run check'}")

print("\n=== Stage 2: long confirm (50,000 trades, 3 seeds) ===")
to_confirm = stage1_flagged if stage1_flagged else NUM_AGENT_VALUES
for na in to_confirm:
    rhos, changes, mins, maxs = [], [], [], []
    for seed in range(3):
        rho, chg, pmin, pmax = run_one(na, 50000, seed)
        rhos.append(rho)
        changes.append(chg)
        mins.append(pmin)
        maxs.append(pmax)
    print(f"num_agent={na}: rho={[round(r,3) for r in rhos]} net_chg%={[round(c,1) for c in changes]} "
          f"price_range=[{min(mins):.1f},{max(maxs):.1f}]")
