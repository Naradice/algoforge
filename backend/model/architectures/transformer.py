"""
Seq2Seq Transformer — adapted from stocknet/stocknet/nets/transformer.py.

Encoder input:  src  [batch, obs_len, d_model]
Decoder input:  tgt  [batch, pred_len, d_model]   (teacher-forced during training)
Output:              [batch, pred_len, output_dim]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.device = device

        self.src_proj = nn.Linear(input_dim, d_model)
        self.tgt_proj = nn.Linear(output_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, output_dim)
        self.to(device)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask_tgt: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        src: [batch, obs_len, input_dim]
        tgt: [batch, pred_len, output_dim]
        Returns: [batch, pred_len, output_dim]
        """
        src_emb = self.pos_enc(self.src_proj(src))
        tgt_emb = self.pos_enc(self.tgt_proj(tgt))

        if mask_tgt is None:
            tgt_len = tgt.size(1)
            mask_tgt = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=mask_tgt)
        return self.out_proj(out)
