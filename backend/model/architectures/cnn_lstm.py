"""
CNN-LSTM hybrid model.

CNN extracts local spatial/pattern features from the input window,
then LSTM reads the resulting feature sequence to capture long-range dependencies.

Input:  [batch, obs_len, input_dim]
Output: [batch, pred_len, output_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 10,
        cnn_filters: int = 64,
        kernel_size: int = 3,
        cnn_layers: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        # 1-D CNN over the time axis: input [batch, input_dim, obs_len]
        cnn_blocks: list[nn.Module] = []
        in_ch = input_dim
        for _ in range(cnn_layers):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, cnn_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = cnn_filters
        self.cnn = nn.Sequential(*cnn_blocks)

        # LSTM reads [batch, obs_len, cnn_filters]
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, output_dim * pred_len)
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, obs_len, input_dim]
        Returns: [batch, pred_len, output_dim]
        """
        x = src.permute(0, 2, 1)   # [batch, input_dim, obs_len]
        x = self.cnn(x)             # [batch, cnn_filters, obs_len]
        x = x.permute(0, 2, 1)     # [batch, obs_len, cnn_filters]
        out, _ = self.lstm(x)       # [batch, obs_len, lstm_hidden]
        last = out[:, -1, :]        # [batch, lstm_hidden]
        pred = self.head(last)      # [batch, output_dim * pred_len]
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
