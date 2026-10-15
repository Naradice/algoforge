"""Step 4: DDM ensemble future-distribution generation and out-of-sample backtest.

4-A (calibration): for each USDJPY quantile bin's DDM candidate set (from ddm_ensemble_selection.py),
run many simulations PER CANDIDATE, pool them all together, and report the resulting future-return
distribution (mean, std, percentiles). Uncertainty across candidates is included in the pool, not
averaged away -- the ensemble's spread reflects both within-candidate simulation noise and
across-candidate parameter uncertainty.

4-B (strict out-of-sample backtest): using the SAME calibration-period quantile boundaries as
delta_forward_validation.py (never re-fit on test data), for each test-period window: look up its
quantile, take that quantile's pooled future-return distribution, and check whether the window's
ACTUAL realized future return (24 windows / ~1 day ahead, matching the horizon with the strongest
already-established real-data correlation) falls inside the ensemble's prediction interval.

Primary success criterion (deliberately NOT about direction, since delta showed ~0 correlation
with signed future return): does ensemble WIDTH increase for high-vol quantiles and shrink for
low-vol quantiles, and does empirical interval coverage roughly match nominal coverage?
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_ddm import TRADES_PER_CANDLE, simulate_ohlc
from ddm_ensemble_selection import (
    CALIB_FRAC, CHAR_WINDOW, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT,
    predict_all_windows,
)
from real_data_transfer_test import load_models

HORIZON_CANDLES = 24  # ~1 day ahead -- matches the strongest already-validated real correlation
N_SIMS_PER_CANDIDATE = 100
PRED_INTERVAL = (5, 95)  # percentile interval checked for coverage


def simulate_future_return(delta: float, wma: int, horizon_candles: int, seed: int):
    n_trades = horizon_candles * TRADES_PER_CANDLE
    try:
        df = simulate_ohlc(delta, wma, n_trades=n_trades, seed=seed)
    except (ValueError, RuntimeError):
        return None
    prices = df["close"].values
    if len(prices) < 2:
        return None
    return float(np.log(prices[-1] / prices[0]))


def build_ensemble_distributions(quantile_results: list[dict]) -> dict[int, np.ndarray]:
    """Returns {quantile: array of pooled future log-returns across all candidates x sims}."""
    distributions = {}
    for r in quantile_results:
        q = r["q"]
        pooled = []
        for _, cand in r["candidates"].iterrows():
            for s in range(N_SIMS_PER_CANDIDATE):
                seed = 90000 + q * 10000 + int(cand.name) * 1000 + s
                ret = simulate_future_return(cand["delta"], int(cand["wma"]), HORIZON_CANDLES, seed)
                if ret is not None:
                    pooled.append(ret)
        distributions[q] = np.array(pooled)
        print(f"  q={q}: pooled {len(pooled)} simulated future returns "
              f"(mean={np.mean(pooled):+.5f}, std={np.std(pooled):.5f})")
    return distributions


def get_real_future_returns_by_quantile():
    """Reproduces the exact out-of-sample quantile assignment + realized future returns from
    delta_forward_validation.py, at HORIZON_CANDLES ahead."""
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes_extended = df["close"].values[START_ROW : START_ROW + N_ROWS + CHAR_WINDOW].astype(np.float64)
    closes = closes_extended[:N_ROWS]
    log_returns = np.diff(np.log(closes))
    mu, sigma = log_returns.mean(), log_returns.std()

    models = asyncio.run(load_models())
    delta_sliding = predict_all_windows(models, log_returns, mu, sigma)

    n_windows = len(log_returns) // OBS_LEN
    window_start_idx = np.arange(n_windows) * OBS_LEN
    delta_per_window = delta_sliding[window_start_idx]

    n_calib = int(n_windows * CALIB_FRAC)
    calib_deltas = delta_per_window[:n_calib]
    quantile_edges = np.quantile(calib_deltas, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])

    test_idx = np.arange(n_calib, n_windows - HORIZON_CANDLES)
    test_delta = delta_per_window[test_idx]
    test_quantile = np.digitize(test_delta, quantile_edges)

    real_future_ret = {}
    for q in range(N_QUANTILES):
        mask = test_quantile == q
        rets = []
        for i in test_idx[mask]:
            fwd_start = window_start_idx[i + HORIZON_CANDLES]
            cur_start = window_start_idx[i]
            rets.append(np.log(closes[fwd_start] / closes[cur_start]))
        real_future_ret[q] = np.array(rets)
    return real_future_ret


DISTRIBUTIONS_CACHE = Path(__file__).resolve().parent / "ddm_ensemble_distributions.npz"


def run_step3_and_4a():
    from ddm_ensemble_selection import main as step3_main
    print("=== Re-running Step 3 to get candidate sets ===")
    step3_results, real_df, atlas = step3_main()

    print(f"\n=== 4-A: calibration -- pooling {N_SIMS_PER_CANDIDATE} sims/candidate, "
          f"horizon={HORIZON_CANDLES} candles ===")
    distributions = build_ensemble_distributions(step3_results)

    print("\n=== Ensemble future-return distribution summary per quantile ===")
    print(f"{'q':>3} {'n':>5} {'mean':>9} {'std':>9} {'p5':>9} {'p25':>9} {'p50':>9} {'p75':>9} {'p95':>9}")
    for q in sorted(distributions):
        d = distributions[q]
        if len(d) == 0:
            continue
        p = np.percentile(d, [5, 25, 50, 75, 95])
        print(f"{q:>3} {len(d):>5} {d.mean():>+9.5f} {d.std():>9.5f} "
              f"{p[0]:>+9.5f} {p[1]:>+9.5f} {p[2]:>+9.5f} {p[3]:>+9.5f} {p[4]:>+9.5f}")

    # Save to disk so 4-B can run as a FRESH process -- calling asyncio.run(load_models()) a
    # second time in the same process reuses a DB connection pool bound to the first (now closed)
    # event loop and crashes (same failure mode hit repeatedly earlier this session).
    np.savez(DISTRIBUTIONS_CACHE, **{f"q{q}": d for q, d in distributions.items()})
    print(f"\nSaved ensemble distributions to {DISTRIBUTIONS_CACHE}")
    return distributions


def run_4b(distributions: dict[int, np.ndarray]):
    print("\n=== 4-B: strict out-of-sample backtest ===")
    print("Computing real out-of-sample future returns per quantile (this reruns the model)...")
    real_future = get_real_future_returns_by_quantile()

    print(f"\n{'q':>3} {'n_real':>7} {'ens_std':>9} {'real_std':>9} {'coverage%':>10} {'nominal%':>9}")
    total_covered, total_n = 0, 0
    for q in sorted(distributions):
        ens = distributions[q]
        real = real_future.get(q, np.array([]))
        if len(ens) == 0 or len(real) == 0:
            continue
        lo, hi = np.percentile(ens, PRED_INTERVAL)
        covered = np.sum((real >= lo) & (real <= hi))
        coverage_pct = 100 * covered / len(real)
        total_covered += covered
        total_n += len(real)
        print(f"{q:>3} {len(real):>7} {ens.std():>9.5f} {real.std():>9.5f} "
              f"{coverage_pct:>10.1f} {PRED_INTERVAL[1]-PRED_INTERVAL[0]:>9}")

    print(f"\nOverall coverage: {100*total_covered/total_n:.1f}% "
          f"(nominal target: {PRED_INTERVAL[1]-PRED_INTERVAL[0]}%)")

    # Width-scaling check: does ensemble std correlate with quantile (high-vol Q -> wide, low-vol -> narrow)?
    from scipy.stats import spearmanr
    qs = sorted(distributions)
    ens_stds = [distributions[q].std() for q in qs]
    real_stds = [real_future[q].std() if len(real_future.get(q, [])) > 1 else np.nan for q in qs]
    r_ens, p_ens = spearmanr(qs, ens_stds)
    valid = ~np.isnan(real_stds)
    r_real, p_real = spearmanr(np.array(qs)[valid], np.array(real_stds)[valid])
    print(f"\nSpearman(quantile, ensemble future std) = {r_ens:+.4f} (p={p_ens:.3e})")
    print(f"Spearman(quantile, REAL future std)      = {r_real:+.4f} (p={p_real:.3e})")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "4a"):
        dists = run_step3_and_4a()
    if mode in ("all",):
        run_4b(dists)
    elif mode == "4b":
        cached = np.load(DISTRIBUTIONS_CACHE)
        dists = {int(k[1:]): cached[k] for k in cached.files}
        run_4b(dists)
