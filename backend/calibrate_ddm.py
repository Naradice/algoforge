"""DDM parameter calibration via grid search + summary-statistic (moment) matching.

Given a target window's summary statistics (real or synthetic), searches a grid of
(delta, wma) -- delta being the dealer_sensitive bias from the regime-regression experiment,
shifting both dealer_sensitive_min/max from the default [-3.5,-1.5] -- and returns whichever
grid point's own simulated statistics land closest, normalized, to the target's.

num_agent and the other DDMv3 constructor parameters are held fixed at defaults: num_agent was
already shown (this investigation's earlier sensitivity sweep) to barely move any observable
statistic across 300-5000, so searching over it adds noise without identifying power.

Reuses the existing characteristics registry (data/characteristics.py) for the statistics
themselves rather than reimplementing them -- this is exactly the toolkit the data layer already
maintains and validates for return distribution shape and autocorrelation structure.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.characteristics import compute_acf, compute_return_dist, compute_vol_clustering
from data.collectors.ddm_simulator import DDMv3

NUM_AGENT = 300
TRADES_PER_CANDLE = 20
ACF_LAGS = [1, 5, 20]
VOL_CLUSTER_LAG = 50

# Stat vector layout, fixed order: [skewness, kurtosis, acf_lag1]
# "std" is deliberately excluded from the MATCHING vector (though characterize() still reports it):
# it's scale-dependent, and DDM's internal simulation units (initial_price=100, trade_unit=0.001,
# spread=1.0, ...) have no calibrated correspondence to a real instrument's actual pip scale.
# Comparing raw std across the two domains isn't comparing like with like -- confirmed live on
# USDJPY, where its real return std (~1.3e-4) sits entirely below DDM's own achievable range at
# these settings (~1e-3+), which pinned every real window's calibrated delta to the grid's most
# extreme boundary value regardless of the window's own actual character.
#
# acf_5, acf_20, and vol_clustering (lag 50) are ALSO excluded, despite being scale-free: checked
# directly against the grid's own spread and they sit near zero across the ENTIRE delta range
# (e.g. acf_20 spans only -0.07 to 0.03 grid-wide) -- almost no delta-driven signal, just sample
# noise, which z-score normalization (dividing by each dimension's small grid-wide spread) then
# amplifies to dominate the total distance and drown out the genuinely informative dimensions.
# skewness/kurtosis/acf_1 all showed real, wide grid-wide spread in the same check.
STAT_NAMES = ["skewness", "kurtosis", "acf_1"]


def simulate_ohlc(delta: float, wma: int, n_trades: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=NUM_AGENT, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
    )
    prices = model.simulate(n_trades=n_trades)["price"].values
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        # Some (delta, wma) combinations -- e.g. very low wma (little price-setting smoothing)
        # combined with strongly negative delta -- push the simulation into the same kind of
        # divergence the original num_agent sweep guarded against, just along a different axis.
        # Not a bug to patch in DDMv3 itself here; the calibration grid just needs to skip these
        # points rather than let a corrupted (NaN/negative-price) statistic silently poison the
        # distance metric.
        raise ValueError(f"simulate_ohlc: non-finite or non-positive price at delta={delta}, wma={wma}")
    n = (len(prices) // TRADES_PER_CANDLE) * TRADES_PER_CANDLE
    grouped = prices[:n].reshape(-1, TRADES_PER_CANDLE)
    return pd.DataFrame({"close": grouped[:, -1]})


def characterize(df: pd.DataFrame) -> dict:
    rd = compute_return_dist(df)["stats"]
    acf = compute_acf(df)["returns"]
    vc = compute_vol_clustering(df)["values"]
    out = {"std": rd["std"], "skewness": rd["skewness"], "kurtosis": rd["kurtosis"]}
    for lag in ACF_LAGS:
        out[f"acf_{lag}"] = acf[lag] if lag < len(acf) else np.nan
    out[f"volclust_{VOL_CLUSTER_LAG}"] = vc[VOL_CLUSTER_LAG] if VOL_CLUSTER_LAG < len(vc) else np.nan
    return out


def stat_vector(stats: dict) -> np.ndarray:
    return np.array([stats[name] for name in STAT_NAMES], dtype=np.float64)


def simulate_and_characterize(delta: float, wma: int, n_trades: int, seed: int) -> dict:
    return characterize(simulate_ohlc(delta, wma, n_trades, seed))


def build_grid(delta_values, wma_values, n_trades: int, n_seeds: int, base_seed: int = 1000) -> list[dict]:
    grid = []
    skipped = []
    for delta in delta_values:
        for wma in wma_values:
            seed_stats = []
            for s in range(n_seeds):
                try:
                    seed_stats.append(simulate_and_characterize(delta, wma, n_trades, seed=base_seed + s))
                except (ValueError, RuntimeError):
                    # ValueError: simulate_ohlc's own finite/positive-price check.
                    # RuntimeError: DDMv3.simulate_stream's internal 10M-step stall guard fires
                    # first in most divergent cases -- this seed just never produced a trade.
                    continue
            if not seed_stats:
                skipped.append((delta, wma))
                continue
            avg_stats = {name: float(np.nanmean([s[name] for s in seed_stats])) for name in STAT_NAMES}
            grid.append({"delta": delta, "wma": wma, "stats": avg_stats})
    if skipped:
        print(f"build_grid: skipped {len(skipped)} grid point(s) where every seed diverged: {skipped}")
    return grid


def calibrate(target_stats: dict, grid_results: list[dict], top_k: int = 5) -> dict:
    grid_matrix = np.stack([stat_vector(g["stats"]) for g in grid_results])  # [n_grid, n_stats]
    target_vec = stat_vector(target_stats)

    # Normalize each statistic's scale using its spread ACROSS THE GRID (not the target alone),
    # so a raw-scale statistic like std (~1e-4) doesn't get swamped by kurtosis (~O(1-10)).
    mu = grid_matrix.mean(axis=0)
    sigma = grid_matrix.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    grid_z = (grid_matrix - mu) / sigma
    target_z = (target_vec - mu) / sigma

    dists = np.linalg.norm(grid_z - target_z, axis=1)
    order = np.argsort(dists)
    best_idx = order[0]
    top = [
        {"delta": grid_results[i]["delta"], "wma": grid_results[i]["wma"], "distance": float(dists[i])}
        for i in order[:top_k]
    ]
    return {
        "best_delta": grid_results[best_idx]["delta"],
        "best_wma": grid_results[best_idx]["wma"],
        "best_distance": float(dists[best_idx]),
        "top_k": top,
    }


if __name__ == "__main__":
    import time

    delta_values = np.round(np.linspace(-1.5, 0.0, 16), 3).tolist()
    wma_values = [1, 3, 5, 10, 20, 40]
    n_trades = 20_000  # matches generate_regime_datasets.py's per-regime trade count

    print(f"Building grid: {len(delta_values)} delta x {len(wma_values)} wma = "
          f"{len(delta_values) * len(wma_values)} points, 3 seeds each...")
    t0 = time.time()
    grid = build_grid(delta_values, wma_values, n_trades=n_trades, n_seeds=3)
    print(f"Grid built in {time.time()-t0:.1f}s")

    print("\n=== Synthetic recovery test ===")
    test_points = [(-1.2, 15), (-0.8, 7), (-0.4, 25), (-1.45, 2), (-0.05, 30)]
    errors_delta, errors_wma = [], []
    for true_delta, true_wma in test_points:
        try:
            target = simulate_and_characterize(true_delta, true_wma, n_trades=n_trades, seed=9999)
        except (ValueError, RuntimeError):
            print(f"true=(delta={true_delta:+.2f}, wma={true_wma:2d})  SKIPPED (diverged)")
            continue
        result = calibrate(target, grid)
        d_err = abs(result["best_delta"] - true_delta)
        w_err = abs(result["best_wma"] - true_wma)
        errors_delta.append(d_err)
        errors_wma.append(w_err)
        print(f"true=(delta={true_delta:+.2f}, wma={true_wma:2d})  "
              f"recovered=(delta={result['best_delta']:+.2f}, wma={result['best_wma']:2d})  "
              f"err=(delta={d_err:.3f}, wma={w_err:.1f})  dist={result['best_distance']:.3f}")

    print(f"\nMean |delta error|: {np.mean(errors_delta):.3f}  Mean |wma error|: {np.mean(errors_wma):.2f}")
