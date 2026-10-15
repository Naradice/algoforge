"""Out-of-sample, forward-looking validation of predicted delta as a real-market regime signal.

Earlier correlation checks (real_data_transfer_test.py) were CONCURRENT: delta predicted from a
window vs. that SAME window's own realized volatility. That's a weaker claim than what a usable
regime signal needs -- this script checks whether today's delta-quantile predicts TOMORROW's (or
next-hour's / next-few-hours') market statistics, using a genuine time-ordered train/test split so
the quantile boundaries themselves aren't fit and evaluated on the same data.

Design:
  1. Non-overlapping 60-minute windows over a long contiguous USDJPY span, in time order.
  2. First CALIB_FRAC of windows -> compute delta quantile boundaries (20 bins) from THEIR deltas.
  3. Remaining windows (out-of-sample) -> assign each window's delta to a quantile bin using the
     CALIBRATION boundaries (never re-fit on test data), then look FORWARD 1/4/24 windows (~1h/4h/
     1 day) to that window's realized vol/skew/kurtosis/acf_1 and forward log-return.
  4. Report per-quantile-bin average forward stats (monotonicity check) and the overall Spearman
     correlation between today's delta and each forward horizon's realized volatility.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from calibrate_ddm import characterize
from real_data_transfer_test import load_models

USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
OBS_LEN = 60
N_ROWS = 300_000
START_ROW = 300_000
CALIB_FRAC = 0.7
N_QUANTILES = 20
HORIZONS = [1, 4, 24]  # in window units (60 min each) -- ~1h, ~4h, ~1 day


def predict_all_windows(models, log_returns: np.ndarray, mu: float, sigma: float, batch_size: int = 2000) -> np.ndarray:
    n = len(log_returns) - OBS_LEN + 1
    windows = np.stack([log_returns[i : i + OBS_LEN] for i in range(n)])
    z = ((windows - mu) / sigma).astype(np.float32)

    all_preds = [[] for _ in models]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = torch.tensor(z[start : start + batch_size, :, None], dtype=torch.float32)
            tgt = torch.zeros(chunk.size(0), 0, 1)
            for mi, model in enumerate(models):
                all_preds[mi].append(model(chunk, tgt)[:, -1, 0].numpy())
    preds = [np.concatenate(p) for p in all_preds]
    return np.mean(preds, axis=0)


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes = df["close"].values[START_ROW : START_ROW + N_ROWS].astype(np.float64)
    log_returns = np.diff(np.log(closes))
    mu, sigma = log_returns.mean(), log_returns.std()

    models = asyncio.run(load_models())
    delta_sliding = predict_all_windows(models, log_returns, mu, sigma)

    n_windows = len(log_returns) // OBS_LEN
    window_start_idx = np.arange(n_windows) * OBS_LEN  # index into log_returns/closes
    delta_per_window = delta_sliding[window_start_idx]  # delta predicted AT each window's start

    n_calib = int(n_windows * CALIB_FRAC)
    print(f"Total windows: {n_windows} (calib: {n_calib}, test: {n_windows - n_calib})")

    calib_deltas = delta_per_window[:n_calib]
    quantile_edges = np.quantile(calib_deltas, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])
    print(f"Quantile edges (from calibration period only): "
          f"min={calib_deltas.min():.3f} max={calib_deltas.max():.3f}")

    max_horizon = max(HORIZONS)
    test_idx = np.arange(n_calib, n_windows - max_horizon)  # leave room for the longest horizon
    print(f"Usable out-of-sample windows (with room for {max_horizon}-window forward horizon): {len(test_idx)}")

    test_delta = delta_per_window[test_idx]
    test_quantile = np.digitize(test_delta, quantile_edges)  # 0..N_QUANTILES-1, OOS assignment

    # Forward stats per horizon
    forward_vol = {h: [] for h in HORIZONS}
    forward_ret = {h: [] for h in HORIZONS}
    forward_kurt = {h: [] for h in HORIZONS}
    for i in test_idx:
        for h in HORIZONS:
            fwd_start = window_start_idx[i + h]
            fwd_closes = closes[fwd_start : fwd_start + OBS_LEN + 1]
            fwd_window_df = pd.DataFrame({"close": fwd_closes})
            stats = characterize(fwd_window_df)
            forward_vol[h].append(stats["std"])
            forward_kurt[h].append(stats["kurtosis"])
            forward_ret[h].append(np.log(fwd_closes[-1] / fwd_closes[0]))

    print("\n=== Spearman correlation: delta_t vs. forward realized vol ===")
    for h in HORIZONS:
        r, p = spearmanr(test_delta, forward_vol[h])
        print(f"  horizon={h:2d} windows (~{h}h if h<24 else '1 day'):  r={r:+.4f}  p={p:.3e}")

    print("\n=== Spearman correlation: delta_t vs. forward |return| (directionless vol proxy) ===")
    for h in HORIZONS:
        r, p = spearmanr(test_delta, np.abs(forward_ret[h]))
        print(f"  horizon={h:2d}:  r={r:+.4f}  p={p:.3e}")

    print("\n=== Spearman correlation: delta_t vs. forward RAW return (directional predictability) ===")
    for h in HORIZONS:
        r, p = spearmanr(test_delta, forward_ret[h])
        print(f"  horizon={h:2d}:  r={r:+.4f}  p={p:.3e}")

    print(f"\n=== Per-quantile-bin average forward realized vol (horizon={HORIZONS[0]}) ===")
    print(f"{'bin':>4} {'n':>5} {'delta_range':>20} {'fwd_vol':>10} {'fwd_kurt':>10}")
    h0 = HORIZONS[0]
    for q in range(N_QUANTILES):
        mask = test_quantile == q
        if mask.sum() == 0:
            continue
        d_lo = calib_deltas.min() if q == 0 else quantile_edges[q - 1]
        d_hi = calib_deltas.max() if q == N_QUANTILES - 1 else quantile_edges[q]
        vol_mean = np.mean(np.array(forward_vol[h0])[mask])
        kurt_mean = np.nanmean(np.array(forward_kurt[h0])[mask])
        print(f"{q:>4} {mask.sum():>5} [{d_lo:+.3f},{d_hi:+.3f}]{'':>3} {vol_mean:>10.6f} {kurt_mean:>10.2f}")


if __name__ == "__main__":
    main()
