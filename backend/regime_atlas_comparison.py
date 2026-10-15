"""Step 1+2 of the ensemble-regime plan: does DDM's delta axis and USDJPY's predicted-delta
quantile axis move THROUGH THE SAME STATISTICAL LANDSCAPE IN THE SAME DIRECTION?

Explicitly NOT another moment-matching attempt -- no absolute value needs to line up. This checks
ordinal correspondence only: as DDM's delta goes from -1.5 to 0 (or equivalently, in whichever
direction its own volatility decreases), do skewness/kurtosis/acf_1/acf_5/acf_20/vol_clustering
move in the same direction they do across USDJPY's Q0 (highest predicted-vol quantile, from the
out-of-sample validation) to Q19 (lowest)?

Reuses:
  - calibrate_ddm.py's build_grid/characterize for the DDM side (full 7-stat vector, not just
    the 3-stat matching subset -- this is a landscape comparison, not a distance calculation).
  - The exact same quantile assignment machinery as delta_forward_validation.py for the real side.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from calibrate_ddm import build_grid, characterize
from real_data_transfer_test import load_models

USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
OBS_LEN = 60
N_ROWS = 300_000
START_ROW = 300_000
CALIB_FRAC = 0.7
N_QUANTILES = 20
WMA_FIXED = 5

FULL_STAT_NAMES = ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]


def full_characterize(df: pd.DataFrame) -> dict:
    from calibrate_ddm import ACF_LAGS, VOL_CLUSTER_LAG
    from data.characteristics import compute_acf, compute_return_dist, compute_vol_clustering
    rd = compute_return_dist(df)["stats"]
    acf = compute_acf(df)["returns"]
    vc = compute_vol_clustering(df)["values"]
    out = {"std": rd["std"], "skewness": rd["skewness"], "kurtosis": rd["kurtosis"]}
    for lag in ACF_LAGS:
        out[f"acf_{lag}"] = acf[lag] if lag < len(acf) else np.nan
    out[f"volclust_{VOL_CLUSTER_LAG}"] = vc[VOL_CLUSTER_LAG] if VOL_CLUSTER_LAG < len(vc) else np.nan
    return out


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
    return np.mean([np.concatenate(p) for p in all_preds], axis=0)


def main():
    # ---- DDM side: full-stat trajectory across delta, wma=5 ----
    delta_values = np.round(np.linspace(-1.5, 0.0, 16), 3).tolist()
    print("Building DDM atlas slice (wma=5, full 7-stat vector)...")
    ddm_rows = []
    from calibrate_ddm import NUM_AGENT, TRADES_PER_CANDLE, simulate_ohlc
    for delta in delta_values:
        seed_stats = []
        for seed in range(3):
            try:
                df = simulate_ohlc(delta, WMA_FIXED, n_trades=20000, seed=1000 + seed)
                seed_stats.append(full_characterize(df))
            except (ValueError, RuntimeError):
                continue
        if seed_stats:
            avg = {name: float(np.nanmean([s[name] for s in seed_stats])) for name in FULL_STAT_NAMES}
            ddm_rows.append({"delta": delta, **avg})
    ddm_df = pd.DataFrame(ddm_rows).sort_values("delta").reset_index(drop=True)
    print(ddm_df.to_string(index=False))

    # ---- Real side: reproduce the same quantile assignment as delta_forward_validation.py ----
    CHAR_WINDOW = 1000  # characterization window length, matching the DDM atlas's own 1000-candle
    # regimes -- separate from OBS_LEN=60 (the delta PREDICTION window). vol_clustering needs
    # lag=50, which a 60-row window can't support at all (only ~29 lags available); a short
    # window also makes kurtosis/acf_20 unreliable, which is what produced the sign-mismatch and
    # NaN in the first pass.
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

    test_delta = delta_per_window[n_calib:]
    test_quantile = np.digitize(test_delta, quantile_edges)
    test_window_starts = window_start_idx[n_calib:]

    print("\nComputing full-stat vector per USDJPY quantile bin...")
    real_rows = []
    for q in range(N_QUANTILES):
        idx = np.where(test_quantile == q)[0]
        if len(idx) == 0:
            continue
        stats_list = []
        for i in idx:
            fwd_start = test_window_starts[i]
            window_closes = closes_extended[fwd_start : fwd_start + CHAR_WINDOW + 1]
            if len(window_closes) < CHAR_WINDOW // 2:
                continue  # too close to the end of the loaded data
            stats_list.append(full_characterize(pd.DataFrame({"close": window_closes})))
        avg = {name: float(np.nanmean([s[name] for s in stats_list])) for name in FULL_STAT_NAMES}
        d_lo = calib_deltas.min() if q == 0 else quantile_edges[q - 1]
        d_hi = calib_deltas.max() if q == N_QUANTILES - 1 else quantile_edges[q]
        real_rows.append({"q": q, "delta_mid": (d_lo + d_hi) / 2, "n": len(idx), **avg})
    real_df = pd.DataFrame(real_rows)
    print(real_df.to_string(index=False))

    # ---- Ordinal correspondence check ----
    # DDM side: as delta increases (-1.5 -> 0), does each stat trend up or down? (Spearman vs delta)
    # Real side: as quantile increases (Q0 -> Q19, i.e. delta increasing per the earlier finding
    # that Q0=most negative delta/highest vol), does each stat trend up or down? (Spearman vs delta_mid)
    print("\n=== Ordinal correspondence: Spearman(stat, delta) on each side ===")
    print(f"{'stat':>12} {'ddm_r':>10} {'real_r':>10} {'same_sign?':>12}")
    for name in FULL_STAT_NAMES:
        ddm_r, _ = spearmanr(ddm_df["delta"], ddm_df[name])
        real_r, _ = spearmanr(real_df["delta_mid"], real_df[name])
        same = "YES" if np.sign(ddm_r) == np.sign(real_r) else "no"
        print(f"{name:>12} {ddm_r:>10.3f} {real_r:>10.3f} {same:>12}")


if __name__ == "__main__":
    main()
