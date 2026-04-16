"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

Each block receives the residual backcast from the previous block, produces its own
backcast (subtracted from the residual) and a partial forecast (summed into the output).
No recurrence, no attention — pure stacked MLP with skip connections.

Input:  [batch, obs_len, input_dim]
Output: [batch, pred_len, output_dim]

Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting" (ICLR 2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _NBeatsBlock(nn.Module):
    """Single N-BEATS block: FC stack → backcast + forecast projections."""

    def __init__(
        self,
        input_size: int,
        hidden_units: int,
        theta_dim: int,
        pred_size: int,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, theta_dim),
        )
        self.backcast_proj = nn.Linear(theta_dim, input_size)
        self.forecast_proj = nn.Linear(theta_dim, pred_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.fc(x)
        return self.backcast_proj(theta), self.forecast_proj(theta)


class NBEATSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        obs_len: int = 60,
        pred_len: int = 10,
        hidden_units: int = 256,
        nb_blocks: int = 3,
        theta_dim: int = 64,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        input_size = obs_len * input_dim
        pred_size = pred_len * output_dim

        self.blocks = nn.ModuleList([
            _NBeatsBlock(input_size, hidden_units, theta_dim, pred_size)
            for _ in range(nb_blocks)
        ])
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, obs_len, input_dim]
        Returns: [batch, pred_len, output_dim]
        """
        x = src.reshape(src.size(0), -1)  # [batch, obs_len * input_dim]
        forecast = torch.zeros(src.size(0), self.pred_len * self.output_dim, device=src.device)

        for block in self.blocks:
            backcast, partial_forecast = block(x)
            x = x - backcast
            forecast = forecast + partial_forecast

        return forecast.view(src.size(0), self.pred_len, self.output_dim)
