"""
Temporal Convolutional Network (TCN).

Stacks dilated causal convolutional blocks with residual connections.
Dilation doubles at each level (1, 2, 4, 8, …), giving an exponentially growing
receptive field without the vanishing-gradient issues of deep RNNs.

Input:  [batch, obs_len, input_dim]
Output: [batch, pred_len, output_dim]

Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" (2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TemporalBlock(nn.Module):
    """Dilated causal conv block with residual skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Left-pad so the output length equals the input length (causal)
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self._pad = pad

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future-leaking padding (right side)."""
        return x[:, :, : -self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._chomp(self.conv1(x))
        out = self.drop(self.relu(out))
        out = self._chomp(self.conv2(out))
        out = self.drop(self.relu(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 10,
        num_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        blocks: list[nn.Module] = []
        in_ch = input_dim
        for i in range(num_levels):
            blocks.append(_TemporalBlock(in_ch, num_channels, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = num_channels

        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(num_channels, output_dim * pred_len)
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, obs_len, input_dim]
        Returns: [batch, pred_len, output_dim]
        """
        x = src.permute(0, 2, 1)   # [batch, input_dim, obs_len]
        x = self.network(x)          # [batch, num_channels, obs_len]
        last = x[:, :, -1]           # [batch, num_channels]
        pred = self.head(last)        # [batch, output_dim * pred_len]
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
