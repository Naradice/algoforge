"""
Decoder-only Transformer (GPT-style causal decoder -- no encoder, no cross-attention) -- built
specifically as an ablation vehicle: `use_attention=False` swaps out self-attention for a
fixed-shape, causal *learned* linear mixing layer (CausalLinearMix below), while every other part
of the block (FFN, LayerNorm, residual, positional encoding, depth, width) stays bit-for-bit
identical to the `use_attention=True` variant. This isolates whether attention's specific
content-dependent (per-input, dynamic) weighting matters for a result -- e.g. a scaling-law
curve -- as opposed to "any learned cross-position mixing at all", which CausalLinearMix still
provides.

Input:  src [batch, seq_len, input_dim] (or [batch, seq_len_raw, k] integer tokens if embedding)
Output: [batch, pred_len, output_dim], read off the last sequence position's representation
        (same convention as LSTMModel -- see backend/model/architectures/lstm.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import PositionalEncoding


class CausalLinearMix(nn.Module):
    """Replaces self-attention with a fixed-shape, causal-masked *learned* linear mixing layer:
    position t's output is a learned linear combination of positions 0..t's representations. The
    mixing weights are static -- they don't depend on the input content, unlike attention's
    per-input dynamic weights -- but they ARE learned during training, not a fixed/uniform
    average. This isolates attention's content-dependent weighting specifically, rather than
    testing "no cross-position mixing at all" (a much more extreme, less informative ablation --
    see the module docstring)."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(seq_len, seq_len))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        w = self.weight * self.causal_mask
        return torch.einsum("ts,bsd->btd", w, x)


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, seq_len: int, use_attention: bool
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.mix: nn.Module = (
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            if use_attention
            else CausalLinearMix(seq_len)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None) -> torch.Tensor:
        if self.use_attention:
            mixed, _ = self.mix(x, x, x, attn_mask=causal_mask, need_weights=False)
        else:
            mixed = self.mix(x)
        x = self.norm1(x + self.dropout(mixed))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_attention: bool = True,
        device: str = "cpu",
        vocab_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        # vocab_size (opt-in): src is a stream of integer token ids (see OHLCWindowDataset's
        # token_level) -- embed directly to d_model, same convention as Seq2SeqTransformer.
        self.embed = nn.Embedding(vocab_size, d_model) if vocab_size else None
        self.src_proj = None if vocab_size else nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, nhead, dim_feedforward, dropout, seq_len, use_attention)
                for _ in range(num_layers)
            ]
        )
        if use_attention:
            self.register_buffer(
                "causal_mask", nn.Transformer.generate_square_subsequent_mask(seq_len), persistent=False
            )
        else:
            self.causal_mask = None

        self.head = nn.Linear(d_model, output_dim * pred_len)
        self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        src: [batch, seq_len, input_dim] continuous, or [batch, seq_len_raw, k] integer tokens if
             self.embed (k=1+n_digits for token_level="digits"; the caller must size seq_len to
             seq_len_raw*k accordingly -- see celery_worker.py's effective_config["seq_len"]).
        tgt: unused -- this model reads the whole src window and predicts the full pred_len
             horizon at once from the last position's representation, same convention as
             LSTMModel; it isn't autoregressive at inference the way a GPT-style decoder usually
             is at generation time.
        Returns: [batch, pred_len, output_dim]
        """
        if self.embed is not None:
            x = self.embed(src.long())  # [batch, seq_len_raw, k, d_model]
            x = x.reshape(x.size(0), -1, x.size(-1))  # [batch, seq_len, d_model]
        else:
            x = self.src_proj(src)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, self.causal_mask)
        last = x[:, -1, :]
        pred = self.head(last)
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
