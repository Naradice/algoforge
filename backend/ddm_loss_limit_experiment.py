"""Step 3, verification items 2-3: characterize DDMv3's loss-limit / forced-liquidation cascade
mechanism ON ITS OWN TERMS -- trigger frequency, cascade size, and whether cascades actually
coincide with larger moves -- BEFORE touching the calibration pipeline. Per the user's specified
order: "loss-limitなし -> baseline / loss-limitあり / thresholdを変える / 発生頻度・cascade
sizeを確認 / その後に calibration", so that any later tail-risk change can be attributed to the
cascade mechanism specifically, not just "more noise."

Reuses a representative subset of the Step 3 ensemble candidates (wma in {7,10,15,20}, delta in
{-0.8,-0.5,-0.1,0.0} -- the region ddm_ensemble_selection.py actually selected from, never the
unstable wma<=3 region) rather than the full 20-quantile selection, matching how the spread-
feedback amplification characterization (Step 2) was also done against a representative subset,
not the full pipeline.

Thresholds are PER-CANDIDATE PERCENTILES of that candidate's own baseline unrealized-pnl
distribution, not one shared absolute value. A first attempt at a shared absolute sweep
(loss_limit in {1.0, 2.0, 3.0} for all candidates) found triggers were near-zero for 3 of 4
candidates: those candidates were deliberately selected (Step 3 ensemble selection) from DDM's
STABLE region, where agents' unrealized pnl rarely wanders as far as it does for the one
strongly trend-following candidate (delta=-0.8, wma=7) that the shared thresholds were
effectively tuned against. Percentile-relative thresholds make the trigger RATE comparable
across candidates with very different intrinsic volatility, which a shared absolute value cannot.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from data.collectors.ddm_simulator import DDMv3

CANDIDATES = [(-0.8, 7), (-0.5, 10), (-0.1, 15), (0.0, 20)]
PERCENTILE_LEVELS = [20, 5, 1]  # % of baseline instantaneous pnl snapshots already beyond -loss_limit
CALIB_TRADES = 50_000
CALIB_SAMPLE_EVERY = 50  # steps between pnl snapshots during calibration
LOSS_LIMIT_C = 0.02
N_TRADES = 50_000
N_SEEDS = 3
CASCADE_GAP_TRADES = 10  # triggers within this many trades of each other belong to one cascade
POST_WINDOW = 20  # trades, for the before/after-cascade return comparison


def calibrate_thresholds(delta: float, wma: int) -> dict[int, float]:
    """Run one baseline (loss_limit disabled) simulation, sample instantaneous unrealized pnl,
    and derive a loss_limit value per PERCENTILE_LEVELS entry: the magnitude such that `level`%
    of baseline snapshots already exceed it. Returns {level: loss_limit_value}."""
    random.seed(0)
    np.random.seed(0)
    model = DDMv3(
        num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
    )
    samples = []
    step = 0
    for _price, _tick in model.simulate_stream(n_trades=CALIB_TRADES):
        step += 1
        if step % CALIB_SAMPLE_EVERY == 0:
            pnl = model.position * (model.entry_price - model.market_price)
            samples.append(pnl.copy())
    arr = np.concatenate(samples)
    return {level: float(-np.percentile(arr, level)) for level in PERCENTILE_LEVELS}


@dataclass
class RunStats:
    n_trigger_events: int
    n_cascades: int
    cascade_sizes: list[int]
    kurtosis: float
    p_gt_2sigma: float
    p_gt_3sigma: float
    cond_abs_ret_after_cascade: float
    uncond_abs_ret: float


def run_one(delta: float, wma: int, loss_limit: float | None, seed: int) -> RunStats:
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
        loss_limit=loss_limit, loss_limit_c=LOSS_LIMIT_C,
    )
    prices = np.array([p for p, _ in model.simulate_stream(n_trades=N_TRADES)])
    returns = np.diff(np.log(prices))
    sigma = returns.std()

    kurtosis = float(
        ((returns - returns.mean()) ** 4).mean() / sigma**4 - 3.0
    ) if sigma > 0 else float("nan")
    p2 = float((np.abs(returns) > 2 * sigma).mean())
    p3 = float((np.abs(returns) > 3 * sigma).mean())

    triggers = sorted(t for t, _, kind in model._liq_events if kind == "trigger")
    n_trigger_events = len(triggers)

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
    cond_abs_ret = float(np.mean(cond_abs_rets)) if cond_abs_rets else float("nan")
    uncond_abs_ret = float(np.abs(returns).mean())

    return RunStats(
        n_trigger_events=n_trigger_events,
        n_cascades=len(cascades),
        cascade_sizes=cascade_sizes,
        kurtosis=kurtosis,
        p_gt_2sigma=p2,
        p_gt_3sigma=p3,
        cond_abs_ret_after_cascade=cond_abs_ret,
        uncond_abs_ret=uncond_abs_ret,
    )


def print_row(delta, wma, label, row):
    print(
        f"delta={delta:+.1f} wma={wma:>2} {label:>16}  "
        f"triggers={row['n_trigger_events']:>8.0f}  cascades={row['n_cascades']:>6.1f}  "
        f"cascade_size(mean/median/max)={row['mean_cascade_size']:>5.1f}/"
        f"{row['median_cascade_size']:>5.1f}/{row['max_cascade_size']:>4.0f}  "
        f"kurt={row['kurtosis']:>7.2f}  P(>2s)={row['p_gt_2sigma']:.4f}  "
        f"P(>3s)={row['p_gt_3sigma']:.4f}  "
        f"|ret| after-cascade/uncond={row['cond_abs_ret_after_cascade']:.6f}/"
        f"{row['uncond_abs_ret']:.6f}"
    )


def aggregate(per_seed: list[RunStats]) -> dict:
    all_cascade_sizes = [s for r in per_seed for s in r.cascade_sizes]
    return {
        "n_trigger_events": np.mean([r.n_trigger_events for r in per_seed]),
        "n_cascades": np.mean([r.n_cascades for r in per_seed]),
        "mean_cascade_size": np.mean(all_cascade_sizes) if all_cascade_sizes else float("nan"),
        "median_cascade_size": np.median(all_cascade_sizes) if all_cascade_sizes else float("nan"),
        "max_cascade_size": np.max(all_cascade_sizes) if all_cascade_sizes else 0,
        "kurtosis": np.mean([r.kurtosis for r in per_seed]),
        "p_gt_2sigma": np.mean([r.p_gt_2sigma for r in per_seed]),
        "p_gt_3sigma": np.mean([r.p_gt_3sigma for r in per_seed]),
        "cond_abs_ret_after_cascade": np.nanmean([r.cond_abs_ret_after_cascade for r in per_seed]),
        "uncond_abs_ret": np.mean([r.uncond_abs_ret for r in per_seed]),
    }


def main():
    rows = []
    for delta, wma in CANDIDATES:
        thresholds = calibrate_thresholds(delta, wma)
        print(f"\n=== delta={delta:+.1f} wma={wma} -- calibrated thresholds: "
              + ", ".join(f"p{lvl}={v:.3f}" for lvl, v in thresholds.items()))

        baseline_row = aggregate([run_one(delta, wma, None, seed) for seed in range(N_SEEDS)])
        print_row(delta, wma, "baseline", baseline_row)
        rows.append({"delta": delta, "wma": wma, "label": "baseline", "loss_limit": None, **baseline_row})

        for level, loss_limit in thresholds.items():
            per_seed = [run_one(delta, wma, loss_limit, seed) for seed in range(N_SEEDS)]
            row = aggregate(per_seed)
            label = f"p{level}={loss_limit:.2f}"
            print_row(delta, wma, label, row)
            rows.append({"delta": delta, "wma": wma, "label": label, "loss_limit": loss_limit, **row})
    return rows


if __name__ == "__main__":
    main()
