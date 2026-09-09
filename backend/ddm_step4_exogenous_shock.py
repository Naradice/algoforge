"""Step 4: exogenous order-flow shock -- single-mechanism characterization and a coarse
(shock_probability x shock_size) grid, per the user's spec (numbered items below match that
spec's own numbering).

Item 8 scope: Bernoulli timing + random +-direction + fixed magnitude only. No news
type/sentiment/regime-dependent rate/clustering/heavy-tailed size/multiple simultaneous shocks.

Item 9 (backward compatibility) is verified separately, not in this script: bit-for-bit identity
of DDMv3 output with exogenous_shock_probability=0.0 against the pre-Step-4 committed code was
checked directly (git-stash A/B comparison) before this script was written.

Item 12 (stability check) reuses the Step 3 ensemble candidates AS-IS -- no re-selection here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from data.collectors.ddm_simulator import DDMv3

CANDIDATES = [(-0.8, 7), (-0.5, 10), (-0.1, 15), (0.0, 20)]

# Item 11: coarse 2-axis grid. Sized relative to the SAME price-unit scale loss_limit was
# calibrated against (Step 3 found unrealized-pnl swings of O(0.5-3.5) price units for this
# candidate family) -- shock_size values span well below and up to that range.
SHOCK_PROBABILITIES = {"very_low": 0.0005, "low": 0.002, "medium": 0.01}
SHOCK_SIZES = {"small": 0.1, "medium": 0.3, "large": 0.8}

N_TRADES = 50_000
N_SEEDS = 5


@dataclass
class ShockRunStats:
    n_shocks_drawn: int
    n_shocks_executed: int
    n_buy: int
    n_sell: int
    mean_shock_interval: float
    kurtosis: float
    return_std: float
    p_gt_2sigma: float
    p_gt_3sigma: float
    max_abs_return: float
    max_drawdown: float
    max_drawup: float
    recovery_time: float  # trades from max-drawdown trough back to the pre-drawdown peak level; nan if never


def _drawdown_drawup_recovery(log_prices: np.ndarray) -> tuple[float, float, float]:
    running_max = np.maximum.accumulate(log_prices)
    drawdown = log_prices - running_max  # <= 0
    trough_idx = int(np.argmin(drawdown))
    max_drawdown = float(-drawdown[trough_idx])
    peak_level = running_max[trough_idx]
    after = log_prices[trough_idx:]
    recovered = np.where(after >= peak_level)[0]
    recovery_time = float(recovered[0]) if len(recovered) else float("nan")

    running_min = np.minimum.accumulate(log_prices)
    drawup = log_prices - running_min  # >= 0
    max_drawup = float(drawup.max())

    return max_drawdown, max_drawup, recovery_time


def run_one(delta: float, wma: int, shock_probability: float, shock_size: float, seed: int) -> ShockRunStats:
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
        exogenous_shock_probability=shock_probability, exogenous_shock_size=shock_size,
    )
    prices = np.array([p for p, _ in model.simulate_stream(n_trades=N_TRADES)])
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    sigma = returns.std()

    kurtosis = float(((returns - returns.mean()) ** 4).mean() / sigma**4 - 3.0) if sigma > 0 else float("nan")
    p2 = float((np.abs(returns) > 2 * sigma).mean())
    p3 = float((np.abs(returns) > 3 * sigma).mean())
    max_dd, max_du, recovery = _drawdown_drawup_recovery(log_prices)

    steps = np.array([e[0] for e in model._shock_events])
    directions = np.array([e[1] for e in model._shock_events])
    executed = np.array([e[2] for e in model._shock_events])
    mean_interval = float(np.mean(np.diff(steps))) if len(steps) > 1 else float("nan")

    return ShockRunStats(
        n_shocks_drawn=len(model._shock_events),
        n_shocks_executed=int(executed.sum()),
        n_buy=int((directions == 1).sum()),
        n_sell=int((directions == -1).sum()),
        mean_shock_interval=mean_interval,
        kurtosis=kurtosis,
        return_std=float(sigma),
        p_gt_2sigma=p2,
        p_gt_3sigma=p3,
        max_abs_return=float(np.abs(returns).max()),
        max_drawdown=max_dd,
        max_drawup=max_du,
        recovery_time=recovery,
    )


def aggregate(runs: list[ShockRunStats]) -> dict:
    def m(attr):
        vals = [getattr(r, attr) for r in runs]
        return float(np.nanmean(vals))
    return {
        "n_shocks_drawn": m("n_shocks_drawn"), "n_shocks_executed": m("n_shocks_executed"),
        "n_buy": m("n_buy"), "n_sell": m("n_sell"), "mean_shock_interval": m("mean_shock_interval"),
        "kurtosis": m("kurtosis"), "return_std": m("return_std"),
        "p_gt_2sigma": m("p_gt_2sigma"), "p_gt_3sigma": m("p_gt_3sigma"),
        "max_abs_return": m("max_abs_return"), "max_drawdown": m("max_drawdown"),
        "max_drawup": m("max_drawup"), "recovery_time": m("recovery_time"),
    }


def print_row(delta, wma, label, row):
    print(
        f"delta={delta:+.1f} wma={wma:>2} {label:>18}  "
        f"shocks(drawn/exec)={row['n_shocks_drawn']:>6.0f}/{row['n_shocks_executed']:>6.0f}  "
        f"buy/sell={row['n_buy']:>5.0f}/{row['n_sell']:>5.0f}  interval={row['mean_shock_interval']:>8.0f}  "
        f"std={row['return_std']:.6f}  kurt={row['kurtosis']:>7.2f}  "
        f"P(>2s)={row['p_gt_2sigma']:.4f}  P(>3s)={row['p_gt_3sigma']:.4f}  "
        f"maxabs={row['max_abs_return']:.5f}  DD={row['max_drawdown']:.4f}  DU={row['max_drawup']:.4f}  "
        f"recov={row['recovery_time']:.0f}"
    )


def main():
    results = []

    print("=" * 100)
    print("Item 10: single-mechanism characterization, shock disabled vs. one representative setting")
    print("=" * 100)
    for delta, wma in CANDIDATES:
        baseline = aggregate([run_one(delta, wma, 0.0, 0.0, s) for s in range(N_SEEDS)])
        print_row(delta, wma, "no shock", baseline)
        results.append({"delta": delta, "wma": wma, "shock_probability": 0.0, "shock_size": 0.0,
                         "prob_label": "none", "size_label": "none", **baseline})

        rep_prob, rep_size = SHOCK_PROBABILITIES["low"], SHOCK_SIZES["medium"]
        shocked = aggregate([run_one(delta, wma, rep_prob, rep_size, s) for s in range(N_SEEDS)])
        print_row(delta, wma, f"p={rep_prob},sz={rep_size}", shocked)
        results.append({"delta": delta, "wma": wma, "shock_probability": rep_prob, "shock_size": rep_size,
                         "prob_label": "low", "size_label": "medium", **shocked})

    print()
    print("=" * 100)
    print("Item 11-12: coarse grid (probability x size) + stability, all Step-3 candidates")
    print("=" * 100)
    for delta, wma in CANDIDATES:
        for prob_label, prob in SHOCK_PROBABILITIES.items():
            for size_label, size in SHOCK_SIZES.items():
                runs = [run_one(delta, wma, prob, size, s) for s in range(N_SEEDS)]
                row = aggregate(runs)
                print_row(delta, wma, f"{prob_label}/{size_label}", row)
                results.append({"delta": delta, "wma": wma, "shock_probability": prob, "shock_size": size,
                                 "prob_label": prob_label, "size_label": size_label, **row})

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    results = main()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ddm_step4_exogenous_shock.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
