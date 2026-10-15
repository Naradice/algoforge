"""Apply the validated 1D delta calibration (wma fixed at 5 -- see calibrate_ddm.py's synthetic
recovery test, mean |delta error|~0.19 on ~1000-candle windows) to real USDJPY data.

Sanity check (mirrors the earlier neural-model transfer test's methodology, since real data has
no ground-truth delta to compare against directly): split USDJPY into many non-overlapping
~1000-candle windows, calibrate delta for each, and check that calibrated delta correlates with
the window's OWN realized volatility in the expected direction (more negative delta <-> higher
vol, per the established dealer_sensitive relationship) -- an internal-consistency check on the
calibration method itself, not proof the absolute delta value is correct.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from calibrate_ddm import build_grid, calibrate, characterize

USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
WINDOW_LEN = 1000  # candles per window, matching the grid's own simulated window size
WMA_FIXED = 5
N_TRADES = 20_000


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()

    print("Building delta grid (wma fixed=5)...")
    delta_values = np.round(np.linspace(-1.5, 0.0, 16), 3).tolist()
    grid = build_grid(delta_values, [WMA_FIXED], n_trades=N_TRADES, n_seeds=3)
    print(f"Grid: {len(grid)} points\n")

    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes = df["close"].values
    n_windows = len(closes) // WINDOW_LEN
    print(f"USDJPY rows: {len(closes)}, non-overlapping {WINDOW_LEN}-candle windows: {n_windows}")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_windows, size=min(200, n_windows), replace=False)

    calibrated_deltas = []
    real_stds = []
    for i in sample_idx:
        window = pd.DataFrame({"close": closes[i * WINDOW_LEN : (i + 1) * WINDOW_LEN]})
        stats = characterize(window)
        result = calibrate(stats, grid)
        calibrated_deltas.append(result["best_delta"])
        real_stds.append(stats["std"])

    calibrated_deltas = np.array(calibrated_deltas)
    real_stds = np.array(real_stds)

    print(f"\ncalibrated delta: mean={calibrated_deltas.mean():.3f} std={calibrated_deltas.std():.3f} "
          f"range=[{calibrated_deltas.min():.3f}, {calibrated_deltas.max():.3f}]")
    print(f"real ret_std: mean={real_stds.mean():.6f} range=[{real_stds.min():.6f}, {real_stds.max():.6f}]")

    pear_r, pear_p = pearsonr(calibrated_deltas, real_stds)
    spear_r, spear_p = spearmanr(calibrated_deltas, real_stds)
    print(f"\nPearson  r={pear_r:.4f}  p={pear_p:.3e}")
    print(f"Spearman r={spear_r:.4f}  p={spear_p:.3e}")

    # Distribution of calibrated delta across grid values -- flags a degenerate fit
    # (e.g. everything landing on one boundary value) vs. real spread across the range.
    print("\ndelta value -> count:")
    unique, counts = np.unique(calibrated_deltas, return_counts=True)
    for v, c in zip(unique, counts):
        print(f"  {v:+.2f}: {'#' * c} ({c})")


if __name__ == "__main__":
    main()
