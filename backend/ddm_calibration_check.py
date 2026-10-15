"""Proper probabilistic calibration check for the DDM ensemble's future-return distribution --
not just "does the dispersion scale match" (already shown, ddm_horizon_analysis.py) but "does the
predicted DISTRIBUTION actually behave like a calibrated probabilistic forecast."

A. Interval coverage: for nominal levels {50, 80, 90, 95}%, does the realized return fall inside
   the DDM ensemble's own [lo, hi] percentile interval at approximately the nominal rate? Checked
   both pooled (statistical power) and per-quantile (the regime-conditioned claim actually made).

B. Tail-probability calibration: for several |return| thresholds, does predicted P(|r|>x) from the
   DDM ensemble match the realized frequency, per quantile?

Uses the cached DDM horizon stats (ddm_horizon_stats.npz, from ddm_horizon_analysis.py phase1) and
a fresh real-data computation (own process, same event-loop-reuse constraint as everywhere else
this session).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ddm_ensemble_selection import (
    CALIB_FRAC, CHAR_WINDOW, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT,
    predict_all_windows,
)
from ddm_horizon_analysis import DDM_STATS_CACHE, HORIZONS_WINDOWS, MAX_HORIZON, _unflatten_ddm_stats
from real_data_transfer_test import load_models

NOMINAL_LEVELS = [50, 80, 90, 95]
CHECK_HORIZONS = [1, 24]


def get_real_cum_returns_by_quantile() -> dict:
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

    test_idx = np.arange(n_calib, n_windows - MAX_HORIZON)
    test_delta = delta_per_window[test_idx]
    test_quantile = np.digitize(test_delta, quantile_edges)

    out = {q: {h: [] for h in HORIZONS_WINDOWS} for q in range(N_QUANTILES)}
    for i, q in zip(test_idx, test_quantile):
        cur_start = window_start_idx[i]
        for h in HORIZONS_WINDOWS:
            fwd_start = window_start_idx[i + h]
            out[q][h].append(np.log(closes[fwd_start] / closes[cur_start]))
    return {q: {h: np.array(v) for h, v in hd.items()} for q, hd in out.items()}


def main(cache_path: Path = DDM_STATS_CACHE):
    print(f"Loading cached DDM ensemble distributions from {cache_path}...")
    ddm_stats = _unflatten_ddm_stats(np.load(cache_path))

    print("Computing real out-of-sample cumulative returns (fresh process)...")
    real_ret = get_real_cum_returns_by_quantile()

    for h in CHECK_HORIZONS:
        print(f"\n{'='*70}\nHORIZON = {h} windows\n{'='*70}")

        # ---- A. Interval coverage, pooled across quantiles ----
        print(f"\n--- A. Interval coverage (pooled across all 20 quantiles) ---")
        print(f"{'nominal%':>9} {'empirical%':>11} {'n':>6}")
        all_covered = {lvl: [] for lvl in NOMINAL_LEVELS}
        for q in range(N_QUANTILES):
            ddm_dist = ddm_stats[q][h]["cum_return"]
            real_dist = real_ret[q][h]
            if len(ddm_dist) == 0 or len(real_dist) == 0:
                continue
            for lvl in NOMINAL_LEVELS:
                lo_p, hi_p = (100 - lvl) / 2, 100 - (100 - lvl) / 2
                lo, hi = np.percentile(ddm_dist, [lo_p, hi_p])
                covered = (real_dist >= lo) & (real_dist <= hi)
                all_covered[lvl].extend(covered.tolist())
        for lvl in NOMINAL_LEVELS:
            arr = np.array(all_covered[lvl])
            print(f"{lvl:>9} {100*arr.mean():>11.1f} {len(arr):>6}")

        # ---- A2. Per-quantile coverage (90% interval only, for readability) ----
        print(f"\n--- A2. Per-quantile 90% interval coverage ---")
        print(f"{'q':>3} {'n_real':>7} {'coverage%':>10}")
        for q in range(N_QUANTILES):
            ddm_dist = ddm_stats[q][h]["cum_return"]
            real_dist = real_ret[q][h]
            if len(ddm_dist) == 0 or len(real_dist) == 0:
                continue
            lo, hi = np.percentile(ddm_dist, [5, 95])
            covered = np.mean((real_dist >= lo) & (real_dist <= hi)) * 100
            print(f"{q:>3} {len(real_dist):>7} {covered:>10.1f}")

        # ---- B. Tail-probability calibration ----
        print(f"\n--- B. Tail probability calibration: P(|return| > x) ---")
        all_real_abs = np.concatenate([real_ret[q][h] for q in range(N_QUANTILES)])
        thresholds = np.percentile(np.abs(all_real_abs), [50, 70, 85, 95])
        print(f"Thresholds (from real |return| percentiles 50/70/85/95): "
              f"{[f'{t:.5f}' for t in thresholds]}")
        print(f"{'q':>3}" + "".join(f" {'pred_x'+str(i):>10} {'real_x'+str(i):>10}" for i in range(len(thresholds))))
        for q in range(N_QUANTILES):
            ddm_dist = ddm_stats[q][h]["cum_return"]
            real_dist = real_ret[q][h]
            if len(ddm_dist) == 0 or len(real_dist) == 0:
                continue
            row = f"{q:>3}"
            for x in thresholds:
                pred_p = 100 * np.mean(np.abs(ddm_dist) > x)
                real_p = 100 * np.mean(np.abs(real_dist) > x)
                row += f" {pred_p:>10.1f} {real_p:>10.1f}"
            print(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DDM_STATS_CACHE,
                         help="path to a ddm_horizon_stats*.npz produced by ddm_horizon_analysis.py phase1")
    args = parser.parse_args()
    main(cache_path=args.cache)
