"""
PairLagModel — tests whether volatility-relevant information sits in the relationship between
two tokens a fixed distance `lag` apart, independent of any embedding, positional encoding, or
attention.

z_i(lag) = phi([token_i, token_{i+lag}])   -- phi sees the two raw (ordinal) token values
                                               directly, no embedding lookup
z(lag)   = mean over a fixed-size random subset of valid i (resampled every forward call)
y        = rho(z(lag))

Pooling a fixed-size subset (not every valid i) deconfounds "how far apart are the two compared
tokens" from "how many pairs got averaged" -- both otherwise move together as `lag` grows for a
fixed window length (see docs/model-layer.md, "Comparing training runs" / distance sweep).

Input:  [batch, obs_len, 1] integer token ids (e.g. token_level="quantize_diff")
Output: [batch, pred_len, output_dim]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PairLagModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 10,
        lag: int = 1,
        pool_size: int = 20,
        hidden: int = 32,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.lag = lag
        self.pool_size = pool_size
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, pred_len * output_dim))
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, obs_len, 1] integer token ids (float-castable). tgt is accepted for
             interface compatibility with model/trainers/supervised.py's train_epoch/eval_epoch
             (same convention as NBEATSModel) but never used -- this model has no decoder side.
        Returns: [batch, pred_len, output_dim]
        """
        x = src.squeeze(-1).float()  # [batch, obs_len]
        batch, n = x.shape
        if n - self.lag < self.pool_size:
            raise ValueError(
                f"obs_len={n} and lag={self.lag} leave only {n - self.lag} valid positions, "
                f"need >= pool_size={self.pool_size}"
            )
        positions = np.random.choice(n - self.lag, size=self.pool_size, replace=False)
        positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        xi = x[:, positions]
        xj = x[:, positions + self.lag]
        pairs = torch.stack([xi, xj], dim=-1)  # [batch, pool_size, 2]
        h = self.phi(pairs)
        pooled = h.mean(dim=1)  # [batch, hidden]
        out = self.rho(pooled)  # [batch, pred_len * output_dim]
        return out.view(batch, self.pred_len, self.output_dim)
