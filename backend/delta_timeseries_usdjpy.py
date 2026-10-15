"""How does the neural model's predicted delta evolve over real USDJPY time, and is it stable
(persistent, like a genuine regime signal should be) or does it jump around noisily between
adjacent windows?

Two views over the same contiguous stretch of real data:
  1. Sliding, stride=1: predicted delta for every consecutive 60-bar window -- adjacent windows
     share 59 of 60 bars, so genuine regime persistence should show up as a SMOOTH trajectory
     (high lag-1 autocorrelation); a noise-fit would look choppy even at lag 1.
  2. Non-overlapping, stride=60 (window units): predicted delta once per fresh 60-bar window --
     the "coarser" view a regime-conditioned downstream model would actually consume.

Reuses the same 3-seed ensemble (ml_model 115, runs 1417-1419) and returns_zscore preprocessing
already validated in real_data_transfer_test.py.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf as _acf

USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
OBS_LEN = 60
N_ROWS = 20_000       # contiguous stretch analyzed
START_ROW = 500_000   # arbitrary offset into the series, away from the very edges


async def load_models():
    from real_data_transfer_test import load_models as _load
    return await _load()


def predict_sliding(models, log_returns: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Prediction for every stride=1 window -- returns array of length len(log_returns)-OBS_LEN+1."""
    n = len(log_returns) - OBS_LEN + 1
    windows = np.stack([log_returns[i : i + OBS_LEN] for i in range(n)])
    z = (windows - mu) / sigma
    x = torch.tensor(z[:, :, None], dtype=torch.float32)
    tgt = torch.zeros(x.size(0), 0, 1)
    preds = []
    with torch.no_grad():
        for model in models:
            preds.append(model(x, tgt)[:, -1, 0].numpy())
    return np.mean(preds, axis=0)


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes = df["close"].values[START_ROW : START_ROW + N_ROWS].astype(np.float64)
    log_returns = np.diff(np.log(closes))
    mu, sigma = log_returns.mean(), log_returns.std()
    print(f"Analyzing rows [{START_ROW}, {START_ROW + N_ROWS}) -- {len(log_returns)} returns, "
          f"stride-1 windows: {len(log_returns) - OBS_LEN + 1}")

    models = asyncio.run(load_models())

    print("\nComputing sliding (stride=1) delta trajectory...")
    delta_sliding = predict_sliding(models, log_returns, mu, sigma)

    print("Computing non-overlapping (stride=60) delta trajectory...")
    n_windows_stride60 = len(log_returns) // OBS_LEN
    stride60_idx = np.arange(n_windows_stride60) * OBS_LEN
    delta_stride60 = predict_sliding(models, log_returns, mu, sigma)[stride60_idx]

    def report(name, series):
        print(f"\n--- {name} (n={len(series)}) ---")
        print(f"mean={series.mean():.4f} std={series.std():.4f} range=[{series.min():.4f}, {series.max():.4f}]")
        max_lag = min(60, len(series) // 4)
        acf_vals = _acf(series, nlags=max_lag, fft=True)
        for lag in [1, 5, 20, 60]:
            if lag <= max_lag:
                print(f"  autocorr(delta, lag={lag:3d}) = {acf_vals[lag]:+.4f}")
        # Step-to-step change size, for the sliding view -- how much does delta typically move
        # between ADJACENT (59-bar-overlapping) windows, relative to its own overall spread?
        if name.startswith("sliding"):
            step_changes = np.abs(np.diff(series))
            print(f"  |step-to-step change|: mean={step_changes.mean():.4f} "
                  f"(vs. overall std={series.std():.4f} -- ratio={step_changes.mean()/series.std():.3f})")

    report("sliding (stride=1)", delta_sliding)
    report("non-overlapping (stride=60)", delta_stride60)

    # Save the sliding trajectory (subsampled for size) for potential later visualization.
    out_path = Path(__file__).resolve().parent / "delta_timeseries_sliding.csv"
    pd.DataFrame({"idx": np.arange(len(delta_sliding)), "delta": delta_sliding}).to_csv(out_path, index=False)
    print(f"\nFull sliding trajectory saved to {out_path}")


if __name__ == "__main__":
    main()
