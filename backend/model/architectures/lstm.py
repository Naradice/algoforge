"""
LSTM model — adapted from stocknet/stocknet/nets/lstm.py.

Input:  [batch, obs_len, input_dim]
Output: [batch, pred_len, output_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pred_len: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_dim * pred_len)
        self.output_dim = output_dim
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, obs_len, input_dim]
        Returns: [batch, pred_len, output_dim]
        """
        out, _ = self.lstm(src)
        last = out[:, -1, :]                         # [batch, hidden_dim]
        pred = self.head(last)                       # [batch, output_dim * pred_len]
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
