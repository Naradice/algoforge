"""Replicate the one promising cell from ddm_loss_limit_experiment.py (delta=-0.8, wma=7, the
p20-calibrated threshold ~1.404) across many more seeds and a longer run, to check whether its
5.4x post-cascade |return| amplification (measured on only ~8 cascades across 3 seeds) is a real
effect or noise from a thin sample -- per the "an especially clean result deserves more
scrutiny, not less" discipline (CLAUDE.md's model-comparison section, generalized here).

Reports the per-seed ratio distribution, not just a pooled mean, since a single dominant outlier
cascade could otherwise masquerade as a consistent effect.
"""
from __future__ import annotations

import random

import numpy as np

from data.collectors.ddm_simulator import DDMv3
from ddm_loss_limit_experiment import CASCADE_GAP_TRADES, POST_WINDOW

DELTA, WMA = -0.8, 7
LOSS_LIMIT = 1.404  # the p20-calibrated value found in ddm_loss_limit_experiment.py
LOSS_LIMIT_C = 0.02
N_TRADES = 100_000
N_SEEDS = 20


def run_one(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=WMA,
        dealer_sensitive_min=-3.5 + DELTA, dealer_sensitive_max=-1.5 + DELTA,
        loss_limit=LOSS_LIMIT, loss_limit_c=LOSS_LIMIT_C,
    )
    prices = np.array([p for p, _ in model.simulate_stream(n_trades=N_TRADES)])
    returns = np.diff(np.log(prices))
    sigma = returns.std()
    kurtosis = float(((returns - returns.mean()) ** 4).mean() / sigma**4 - 3.0) if sigma > 0 else float("nan")
    p2 = float((np.abs(returns) > 2 * sigma).mean())
    p3 = float((np.abs(returns) > 3 * sigma).mean())

    triggers = sorted(t for t, _, kind in model._liq_events if kind == "trigger")
    cascades: list[list[int]] = []
    for t in triggers:
        if cascades and t - cascades[-1][-1] <= CASCADE_GAP_TRADES:
            cascades[-1].append(t)
        else:
            cascades.append([t])
    cascade_sizes = [len(c) for c in cascades]

    cond_abs_rets = []
    for c in cascades:
        end_trade = c[-1]
        if end_trade + POST_WINDOW < len(returns):
            cond_abs_rets.append(np.abs(returns[end_trade : end_trade + POST_WINDOW]).mean())
    uncond_abs_ret = float(np.abs(returns).mean())

    return {
        "seed": seed, "n_triggers": len(triggers), "n_cascades": len(cascades),
        "mean_cascade_size": float(np.mean(cascade_sizes)) if cascade_sizes else float("nan"),
        "max_cascade_size": max(cascade_sizes) if cascade_sizes else 0,
        "kurtosis": kurtosis, "p_gt_2sigma": p2, "p_gt_3sigma": p3,
        "cond_abs_rets": cond_abs_rets, "uncond_abs_ret": uncond_abs_ret,
    }


def main():
    results = [run_one(s) for s in range(N_SEEDS)]

    print(f"{'seed':>4} {'triggers':>8} {'cascades':>8} {'mean_sz':>7} {'max_sz':>6} "
          f"{'kurt':>7} {'P>2s':>7} {'P>3s':>7} {'n_postobs':>9} {'ratio_mean':>10}")
    all_ratios = []
    for r in results:
        ratios = [c / r["uncond_abs_ret"] for c in r["cond_abs_rets"]]
        all_ratios.extend(ratios)
        ratio_mean = np.mean(ratios) if ratios else float("nan")
        print(f"{r['seed']:>4} {r['n_triggers']:>8} {r['n_cascades']:>8} "
              f"{r['mean_cascade_size']:>7.1f} {r['max_cascade_size']:>6} "
              f"{r['kurtosis']:>7.2f} {r['p_gt_2sigma']:>7.4f} {r['p_gt_3sigma']:>7.4f} "
              f"{len(ratios):>9} {ratio_mean:>10.2f}")

    print(f"\nTotal cascades across {N_SEEDS} seeds: {sum(r['n_cascades'] for r in results)}")
    print(f"Total post-cascade observations: {len(all_ratios)}")
    if all_ratios:
        arr = np.array(all_ratios)
        print(f"Per-cascade |ret|-ratio (post-cascade / unconditional): "
              f"mean={arr.mean():.2f} median={np.median(arr):.2f} "
              f"std={arr.std():.2f} min={arr.min():.2f} max={arr.max():.2f}")
        print(f"Fraction of cascades with ratio > 1.5: {(arr > 1.5).mean():.2f}")
        print(f"Fraction of cascades with ratio > 1.0: {(arr > 1.0).mean():.2f}")
    print(f"\nKurtosis across seeds: mean={np.mean([r['kurtosis'] for r in results]):.2f} "
          f"std={np.std([r['kurtosis'] for r in results]):.2f}")
    print(f"P(>2sigma) across seeds: mean={np.mean([r['p_gt_2sigma'] for r in results]):.4f} "
          f"std={np.std([r['p_gt_2sigma'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
