"""Throwaway diagnostic, follow-up to squared_feature_diagnostic.py: that script's minimal
mean-pool model (per-position Linear+ReLU, no attention, NO LayerNorm) recovered R^2~0.30 on
held-out regimes from raw returns alone -- squaring wasn't the missing ingredient. The one
structural difference left between that model and the real (failing) DecoderOnlyTransformer is
LayerNorm: the real model applies it twice per block (post-attention, post-FFN), each time
renormalizing every position's feature vector to unit scale -- which would repeatedly erase
exactly the magnitude information a volatility-like target depends on.

This adds ONE self-attention layer WITH a residual connection but WITHOUT LayerNorm to the same
minimal model, to isolate the two candidate culprits: if this still recovers a comparable R^2,
attention itself is not the problem (LayerNorm is); if it collapses back to a near-constant
prediction like the real model, attention itself carries some of the blame too.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from squared_feature_diagnostic import load_windows, train_and_eval


class AttentionNoNormRegressor(nn.Module):
    """Per-position projection -> one self-attention layer (residual, NO LayerNorm) -> mean-pool
    -> head. Isolates attention's own effect from LayerNorm's, unlike the real
    DecoderOnlyTransformer where the two are always bundled together."""

    def __init__(self, in_channels: int, d_model: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU())
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                                  # [batch, obs_len, d_model]
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out                                   # residual, no LayerNorm anywhere
        pooled = h.mean(dim=1)                              # [batch, d_model]
        return self.head(pooled)


def main():
    train_x, train_y, heldout_x, heldout_y = load_windows()
    mu, sigma = train_x.mean(), train_x.std()
    train_r = (train_x - mu) / sigma
    heldout_r = (heldout_x - mu) / sigma
    print(f"n_train={len(train_x)} n_heldout={len(heldout_x)}\n")

    print("=== return only (1 channel), projection + 1 self-attention layer (residual, NO LayerNorm), mean-pool ===")
    x_train = train_r[:, :, None]
    x_heldout = heldout_r[:, :, None]
    model = AttentionNoNormRegressor(in_channels=1)
    train_and_eval(model, x_train, train_y, x_heldout, heldout_y, epochs=15, lr=0.005, label="attn_no_norm")


if __name__ == "__main__":
    main()
