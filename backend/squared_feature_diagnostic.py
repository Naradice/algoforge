"""Throwaway diagnostic: does handing the network (return, return^2) per position -- instead of
just return, requiring it to discover the squaring itself -- unlock learning?

Builds the same 60-length sliding windows per regime as the main pipeline (obs_len=60 log-returns,
one window per regime-internal position, never crossing a regime boundary), but with 2 input
channels per position: the raw log-return and its square. Two tiny, attention-free models test
this directly:
  1. mean-pool the 2-channel per-position projection, then a small head -- if handing the
     network return^2 makes "mean-pool it" already close to sufficient (since mean(return^2) is
     essentially the window's realized variance), this should work even with NO attention at all.
  2. same but WITHOUT the return^2 channel (return only) -- the control, expected to fail the
     same way the full DecoderOnlyTransformer did, confirming the squared channel specifically is
     what's missing rather than some other difference between this script and the real pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OBS_LEN = 60
ARTIFACT = "datasets/derived/ddm_regime_delta_regression.parquet"
REGIME_MAP_PATH = "datasets/derived/ddm_regime_delta_regression_regime_map.json"


def window_returns(closes: np.ndarray, obs_len: int) -> np.ndarray:
    """[n_windows, obs_len] log-returns, sliding by 1 close price at a time (matches the main
    pipeline's window convention: obs_len returns per window <- obs_len+1 raw closes)."""
    log_returns = np.diff(np.log(closes))
    n_windows = len(log_returns) - obs_len + 1
    out = np.empty((n_windows, obs_len), dtype=np.float64)
    for i in range(n_windows):
        out[i] = log_returns[i : i + obs_len]
    return out


def load_windows():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / ARTIFACT)
    regime_map = json.load(open(store / REGIME_MAP_PATH))["regimes"]

    train_x, train_y, heldout_x, heldout_y = [], [], [], []
    for r in regime_map:
        block = df.iloc[r["start_row"] : r["start_row"] + r["n_rows"]]
        rets = window_returns(block["close"].values, OBS_LEN)  # [n_windows, obs_len]
        delta = r["delta"]
        if r["split"] == "train":
            train_x.append(rets); train_y.append(np.full(len(rets), delta))
        else:
            heldout_x.append(rets); heldout_y.append(np.full(len(rets), delta))
    return (np.concatenate(train_x), np.concatenate(train_y),
            np.concatenate(heldout_x), np.concatenate(heldout_y))


class MeanPoolRegressor(nn.Module):
    """Per-position Linear+ReLU projection, mean-pooled across all obs_len positions, then a
    small head -- no attention, no positional encoding; a minimal permutation-invariant model."""

    def __init__(self, in_channels: int, d_model: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, obs_len, in_channels]
        h = self.proj(x)          # [batch, obs_len, d_model]
        pooled = h.mean(dim=1)    # [batch, d_model]
        return self.head(pooled)  # [batch, 1]


def train_and_eval(model, train_x, train_y, heldout_x, heldout_y,
                    epochs=15, lr=0.005, batch_size=256, seed=42, label=""):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    xt = torch.tensor(train_x, dtype=torch.float32)
    yt = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    xh = torch.tensor(heldout_x, dtype=torch.float32)
    yh = torch.tensor(heldout_y, dtype=torch.float32).unsqueeze(1)

    n = len(xt)
    var_heldout = float(np.var(heldout_y))
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(n)
        losses = []
        for i in range(0, n - batch_size, batch_size):
            idx = order[i : i + batch_size]
            pred = model(xt[idx])
            loss = criterion(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(xh)
            val_mse = criterion(val_pred, yh).item()
        r2 = 1 - val_mse / var_heldout
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"  [{label}] epoch={epoch+1} train_loss={np.mean(losses):.6f} "
                  f"val_mse={val_mse:.6f} val_R2={r2:.4f} pred_std={val_pred.std().item():.5f}")
    return r2


def main():
    train_x, train_y, heldout_x, heldout_y = load_windows()  # [n, 60] raw returns
    print(f"n_train={len(train_x)} n_heldout={len(heldout_x)}\n")

    # Standardize returns (same fix as returns_zscore) -- isolate the squared-channel question
    # from the earlier (largely ruled out, but still applied for good measure) scale question.
    mu, sigma = train_x.mean(), train_x.std()
    train_r = (train_x - mu) / sigma
    heldout_r = (heldout_x - mu) / sigma

    print("=== Control: return only (1 channel), mean-pool, no attention ===")
    x_train_1ch = train_r[:, :, None]
    x_heldout_1ch = heldout_r[:, :, None]
    model1 = MeanPoolRegressor(in_channels=1)
    train_and_eval(model1, x_train_1ch, train_y, x_heldout_1ch, heldout_y, label="return_only")

    print("\n=== return + return^2 (2 channels), mean-pool, no attention ===")
    train_r2 = train_r ** 2
    heldout_r2 = heldout_r ** 2
    x_train_2ch = np.stack([train_r, train_r2], axis=-1)
    x_heldout_2ch = np.stack([heldout_r, heldout_r2], axis=-1)
    model2 = MeanPoolRegressor(in_channels=2)
    train_and_eval(model2, x_train_2ch, train_y, x_heldout_2ch, heldout_y, label="return_and_sq")


if __name__ == "__main__":
    main()
