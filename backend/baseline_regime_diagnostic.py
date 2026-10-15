"""Throwaway diagnostic: before blaming the neural architecture for the ~0 R^2 on
ddm_regime_delta_regression, check whether delta is recoverable at all from a 60-candle window
using the simplest possible statistic -- the window's own realized volatility (std of log-returns
of close), the same "ret_std" measure that cleanly showed delta's effect in
check_dealer_sensitive_bias.py (there measured over 50,000-trade/regime samples, ~40x more trades
per estimate than a single 60-candle*20-trades/candle=1200-trade window here).

Fits a plain linear regression ret_std -> delta on TRAIN windows only, evaluates R^2 against
Var(delta) on HELD-OUT windows -- exactly mirroring the neural model's task/split, so the two R^2
numbers are directly comparable. Also reports the same fit aggregated per-regime (averaging many
windows' ret_std together) as a reference ceiling for how much a much larger effective sample size
per estimate would help.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OBS_LEN = 60
ARTIFACT = "datasets/derived/ddm_regime_delta_regression.parquet"
REGIME_MAP_PATH = "datasets/derived/ddm_regime_delta_regression_regime_map.json"


def window_ret_stds(closes: np.ndarray, obs_len: int) -> np.ndarray:
    """ret_std for every obs_len-length window in a single regime's close-price array
    (regime blocks are internally contiguous by construction, so no gap-masking needed here)."""
    log_returns = np.diff(np.log(closes))
    stds = np.empty(len(closes) - obs_len + 1)
    for i in range(len(stds)):
        stds[i] = log_returns[i : i + obs_len - 1].std()
    return stds


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / ARTIFACT)
    regime_map = json.load(open(store / REGIME_MAP_PATH))["regimes"]

    train_x, train_y, heldout_x, heldout_y = [], [], [], []
    heldout_per_regime: dict[float, list[float]] = {}

    for r in regime_map:
        block = df.iloc[r["start_row"] : r["start_row"] + r["n_rows"]]
        closes = block["close"].values
        stds = window_ret_stds(closes, OBS_LEN)
        delta = r["delta"]
        if r["split"] == "train":
            train_x.append(stds)
            train_y.append(np.full(len(stds), delta))
        else:
            heldout_x.append(stds)
            heldout_y.append(np.full(len(stds), delta))
            heldout_per_regime[delta] = stds.tolist()

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    heldout_x = np.concatenate(heldout_x)
    heldout_y = np.concatenate(heldout_y)

    print(f"n_train_windows={len(train_x)} n_heldout_windows={len(heldout_x)}")
    print(f"train ret_std range: [{train_x.min():.6f}, {train_x.max():.6f}]  "
          f"heldout ret_std range: [{heldout_x.min():.6f}, {heldout_x.max():.6f}]")

    # Plain OLS: delta ~ a + b * ret_std, fit on train only.
    A = np.vstack([train_x, np.ones_like(train_x)]).T
    (b, a), *_ = np.linalg.lstsq(A, train_y, rcond=None)
    print(f"\nFit on TRAIN: delta = {a:.4f} + {b:.4f} * ret_std")

    train_pred = a + b * train_x
    heldout_pred = a + b * heldout_x

    var_train = train_y.var()
    var_heldout = heldout_y.var()
    r2_train = 1 - np.mean((train_pred - train_y) ** 2) / var_train
    r2_heldout = 1 - np.mean((heldout_pred - heldout_y) ** 2) / var_heldout
    print(f"\nPer-window (obs_len={OBS_LEN}) R^2:")
    print(f"  TRAIN:    R2={r2_train:.4f}  (Var={var_train:.6f})")
    print(f"  HELD-OUT: R2={r2_heldout:.4f}  (Var={var_heldout:.6f})")

    # Reference ceiling: aggregate ret_std per held-out regime (averaging ~940 windows'
    # worth of overlapping-window noise away) before applying the same fitted line.
    print(f"\nPer-regime aggregated (mean ret_std over each regime's ~{len(heldout_x)//len(heldout_per_regime)} windows):")
    regime_deltas = np.array(sorted(heldout_per_regime))
    regime_mean_std = np.array([np.mean(heldout_per_regime[d]) for d in regime_deltas])
    regime_pred = a + b * regime_mean_std
    r2_regime = 1 - np.mean((regime_pred - regime_deltas) ** 2) / regime_deltas.var()
    print(f"  HELD-OUT (per-regime mean): R2={r2_regime:.4f}")
    print(f"\n{'true_delta':>12} {'mean_ret_std':>14} {'pred_delta':>12} {'abs_err':>10}")
    for d, s, p in zip(regime_deltas, regime_mean_std, regime_pred):
        print(f"{d:12.4f} {s:14.6f} {p:12.4f} {abs(p-d):10.4f}")


if __name__ == "__main__":
    main()
