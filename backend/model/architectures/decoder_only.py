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

import numpy as np
import torch
import torch.nn as nn

from .transformer import PositionalEncoding

_COVERAGE_MODES = ("contiguous", "uniform", "random")


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
        token_coverage_k: int | None = None,
        token_coverage_mode: str = "contiguous",
        device: str = "cpu",
        vocab_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        # token_coverage_k (opt-in): instead of attending over every one of the seq_len positions,
        # keep only k of them (always including seq_len-1, the most recent observed step, so no
        # strategy is handicapped by losing the single most informative position as a side effect
        # of sampling) -- see model/architectures/pair_lag.py's PairLagModel for the analogous
        # "random" convention (resampled every forward call). Requires use_attention=True:
        # CausalLinearMix's weight is a fixed [seq_len, seq_len] table of learned, absolute-
        # position-indexed entries with no coherent sparse-subset semantic, unlike
        # nn.MultiheadAttention which is shape-agnostic.
        if token_coverage_k is not None:
            if not use_attention:
                raise ValueError("token_coverage_k requires use_attention=True")
            if token_coverage_mode not in _COVERAGE_MODES:
                raise ValueError(f"token_coverage_mode must be one of {_COVERAGE_MODES}, got {token_coverage_mode!r}")
            if not (1 <= token_coverage_k <= seq_len):
                raise ValueError(f"token_coverage_k must be in [1, {seq_len}], got {token_coverage_k}")
        self.token_coverage_k = token_coverage_k
        self.token_coverage_mode = token_coverage_mode
        self.seq_len = seq_len

        # vocab_size (opt-in): src is a stream of integer token ids (see OHLCWindowDataset's
        # token_level) -- embed directly to d_model, same convention as Seq2SeqTransformer.
        self.embed = nn.Embedding(vocab_size, d_model) if vocab_size else None
        self.src_proj = None if vocab_size else nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        effective_len = token_coverage_k or seq_len
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, nhead, dim_feedforward, dropout, effective_len, use_attention)
                for _ in range(num_layers)
            ]
        )
        if use_attention:
            self.register_buffer(
                "causal_mask", nn.Transformer.generate_square_subsequent_mask(effective_len), persistent=False
            )
        else:
            self.causal_mask = None

        # Fixed index buffers for the deterministic modes -- computed once, reused every forward
        # call. "random" is resampled per call instead (see forward()), so it has no buffer here.
        if token_coverage_k is not None and token_coverage_k < seq_len:
            if token_coverage_mode == "contiguous":
                idx = np.arange(seq_len - token_coverage_k, seq_len)
            elif token_coverage_mode == "uniform":
                idx = np.unique(np.round(np.linspace(0, seq_len - 1, token_coverage_k)).astype(int))
                if len(idx) != token_coverage_k:
                    raise ValueError(
                        f"uniform token_coverage_k={token_coverage_k} at seq_len={seq_len} produced "
                        f"{len(idx)} unique positions after rounding -- pick a k that spaces out cleanly"
                    )
            else:
                idx = None  # random: computed fresh in forward()
            if idx is not None:
                self.register_buffer("_coverage_idx", torch.as_tensor(idx, dtype=torch.long), persistent=False)
            else:
                self._coverage_idx = None
        else:
            self._coverage_idx = None

        self.head = nn.Linear(d_model, output_dim * pred_len)
        self.to(device)

    def _select_positions(self, batch_device: torch.device) -> torch.Tensor:
        """Returns a 1D LongTensor of the token_coverage_k positions to keep, sorted ascending,
        always including seq_len-1."""
        if self.token_coverage_mode == "random":
            rest = np.random.choice(self.seq_len - 1, size=self.token_coverage_k - 1, replace=False)
            idx = np.sort(np.append(rest, self.seq_len - 1))
            return torch.as_tensor(idx, dtype=torch.long, device=batch_device)
        return self._coverage_idx.to(batch_device)

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

        if self.token_coverage_k is not None and self.token_coverage_k < self.seq_len:
            positions = self._select_positions(x.device)
            x = x[:, positions, :]
            x = x + self.pos_enc.pe[:, positions, :]
            x = self.pos_enc.dropout(x)
        else:
            x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, self.causal_mask)
        last = x[:, -1, :]
        pred = self.head(last)
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
