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
        vocab_size: int | None = None,
        embedding_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.device = device

        # vocab_size (opt-in): src is a stream of integer token ids (see OHLCWindowDataset's
        # token_level) rather than continuous features -- embed instead of feeding directly into
        # the LSTM. tgt is unaffected; it's always continuous (see dataset.py's token_level docs).
        self.embed = nn.Embedding(vocab_size, embedding_dim or hidden_dim) if vocab_size else None
        lstm_input_size = (embedding_dim or hidden_dim) if vocab_size else input_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
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
        src: [batch, obs_len, input_dim] continuous, or [batch, obs_len, k] integer token ids if
             self.embed -- k=1 for most token_levels, k=1+n_digits for token_level="digits"
             (one sign token + one token per digit place, per underlying time step).
        Returns: [batch, pred_len, output_dim]
        """
        if self.embed is not None:
            src = self.embed(src.long())               # [batch, obs_len, k, embedding_dim]
            src = src.reshape(src.size(0), -1, src.size(-1))  # [batch, obs_len*k, embedding_dim]
        out, _ = self.lstm(src)
        last = out[:, -1, :]                         # [batch, hidden_dim]
        pred = self.head(last)                       # [batch, output_dim * pred_len]
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
