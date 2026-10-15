"""Throwaway diagnostic: can plain gradient descent (Adam + MSE, same optimizer/loss convention
as the main pipeline) recover the known linear relationship ret_std -> delta at all, when given
the feature directly (bypassing quantize_diff tokenization/embedding entirely)?

baseline_regime_diagnostic.py already showed a closed-form OLS fit (np.linalg.lstsq) gets
R^2=0.149 (train) / 0.043 (held-out) from this single feature. If a tiny FFN trained with SGD
can't match even the OLS line, the bottleneck is the training loop itself (loss scale, LR,
optimizer) rather than tokenization or architecture -- if it CAN match/beat OLS easily, the
bottleneck is specific to how information survives the quantize_diff -> embedding -> attention
pipeline in the full model.

Reuses window_ret_stds/regime-split loading from baseline_regime_diagnostic.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baseline_regime_diagnostic import window_ret_stds, OBS_LEN, ARTIFACT, REGIME_MAP_PATH


def load_windows():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / ARTIFACT)
    regime_map = json.load(open(store / REGIME_MAP_PATH))["regimes"]

    train_x, train_y, heldout_x, heldout_y = [], [], [], []
    for r in regime_map:
        block = df.iloc[r["start_row"] : r["start_row"] + r["n_rows"]]
        stds = window_ret_stds(block["close"].values, OBS_LEN)
        delta = r["delta"]
        if r["split"] == "train":
            train_x.append(stds); train_y.append(np.full(len(stds), delta))
        else:
            heldout_x.append(stds); heldout_y.append(np.full(len(stds), delta))
    return (np.concatenate(train_x), np.concatenate(train_y),
            np.concatenate(heldout_x), np.concatenate(heldout_y))


def train_and_eval(model, x_mean, x_std, train_x, train_y, heldout_x, heldout_y,
                    epochs=30, lr=0.005, batch_size=256, seed=42, label=""):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    xt = torch.tensor(((train_x - x_mean) / x_std).astype(np.float32)).unsqueeze(1)
    yt = torch.tensor(train_y.astype(np.float32)).unsqueeze(1)
    xh = torch.tensor(((heldout_x - x_mean) / x_std).astype(np.float32)).unsqueeze(1)
    yh = torch.tensor(heldout_y.astype(np.float32)).unsqueeze(1)

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
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f"  [{label}] epoch={epoch+1} train_loss={np.mean(losses):.6f} "
                  f"val_mse={val_mse:.6f} val_R2={r2:.4f} pred_std={val_pred.std().item():.5f}")
    return r2, val_pred.numpy().ravel()


def main():
    train_x, train_y, heldout_x, heldout_y = load_windows()
    x_mean, x_std = train_x.mean(), train_x.std()
    print(f"n_train={len(train_x)} n_heldout={len(heldout_x)} "
          f"ret_std mean/std(train)={x_mean:.6f}/{x_std:.6f}\n")

    print("=== Linear(1,1) via Adam+MSE (should match closed-form OLS R^2=0.043 held-out) ===")
    linear = nn.Linear(1, 1)
    train_and_eval(linear, x_mean, x_std, train_x, train_y, heldout_x, heldout_y,
                   epochs=30, lr=0.01, label="linear")

    print("\n=== Small MLP (1->32->32->1, ReLU) ===")
    mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    train_and_eval(mlp, x_mean, x_std, train_x, train_y, heldout_x, heldout_y,
                   epochs=30, lr=0.005, label="mlp")


if __name__ == "__main__":
    main()
