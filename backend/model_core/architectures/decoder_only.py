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
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, seq_len: int, use_attention: bool,
        layernorm_mode: str = "post", layerscale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.mix: nn.Module = (
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            if use_attention
            else CausalLinearMix(seq_len)
        )
        # layernorm_mode (opt-in, default "post" preserves existing behaviour exactly):
        #   "post" (original): norm(x + sublayer(x)) -- LayerNorm renormalizes the *residual
        #       stream itself* every time, independently per position. For a target that depends
        #       on the magnitude of within-window activity (e.g. realized volatility) this erases
        #       the signal at every single layer -- confirmed via a standalone diagnostic
        #       (attention_no_layernorm_diagnostic.py) where dropping LayerNorm entirely recovered
        #       R^2~0.31 on ddm_regime_delta_regression's held-out regimes, vs. R^2~0 (a collapsed
        #       constant prediction) with it, regardless of learning rate, attn_window, pooling
        #       mode, or output bias init.
        #   "pre" (GPT-2-style): x + sublayer(norm(x)) -- normalizes only the sublayer's *input*,
        #       never the running residual stream, which is free to accumulate magnitude
        #       information across depth. Standard modern convention, generally more stable to
        #       train than "post" -- the natural next step after "none" (dropping normalization
        #       altogether) showed a real signal but highly seed-unstable training (R^2 ranged
        #       0.35 to -0.03 across 3 seeds on this same task).
        #   "none": no normalization at all (nn.Identity) -- the first fix tried; kept as an
        #       option since it's still a valid point of comparison.
        if layernorm_mode not in ("post", "pre", "none"):
            raise ValueError(f"layernorm_mode must be 'post', 'pre', or 'none', got {layernorm_mode!r}")
        self.layernorm_mode = layernorm_mode
        self.norm1 = nn.LayerNorm(d_model) if layernorm_mode != "none" else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model) if layernorm_mode != "none" else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # layerscale_init (opt-in, default None = no LayerScale, ungated residual as before):
        # Pre-LN's residual stream is never renormalized, which is exactly what lets magnitude
        # information survive -- but with nothing capping it either, plain Pre-LN on this task
        # blew activations up across 4 layers into a range the (default-initialized) output head
        # can't handle (empirically: R^2 around -20, predictions wildly outside delta's true
        # [-1.5, 0] range). A learnable per-channel gate on each sublayer's contribution
        # (`x + gamma * sublayer(...)`, gamma initialized small) lets training start with a near-
        # identity residual stream -- controlled growth -- and only let a channel's contribution
        # grow if it's actually useful, rather than every sublayer's raw output being added at
        # full strength from step one.
        self.gamma1 = nn.Parameter(torch.full((d_model,), layerscale_init)) if layerscale_init is not None else None
        self.gamma2 = nn.Parameter(torch.full((d_model,), layerscale_init)) if layerscale_init is not None else None

    def _mix(self, x: torch.Tensor, causal_mask: torch.Tensor | None) -> torch.Tensor:
        if self.use_attention:
            mixed, _ = self.mix(x, x, x, attn_mask=causal_mask, need_weights=False)
            return mixed
        return self.mix(x)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None) -> torch.Tensor:
        if self.layernorm_mode == "pre":
            mixed = self.dropout(self._mix(self.norm1(x), causal_mask))
            x = x + (mixed * self.gamma1 if self.gamma1 is not None else mixed)
            ffn_out = self.dropout(self.ffn(self.norm2(x)))
            x = x + (ffn_out * self.gamma2 if self.gamma2 is not None else ffn_out)
        else:
            x = self.norm1(x + self.dropout(self._mix(x, causal_mask)))
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
        attn_window: int | None = None,
        pooling: str = "last",
        output_bias_init: float | None = None,
        layernorm_mode: str = "post",
        layerscale_init: float | None = None,
        device: str = "cpu",
        vocab_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        # pooling (opt-in, default "last" preserves existing behaviour exactly): for a target
        # that is a window-level, order-independent statistic (e.g. a generative parameter
        # constant across the whole window) rather than a genuinely sequential "predict what
        # comes next", reading only the final causally-masked position is a mismatched inductive
        # bias -- positional encoding actively injects order information the target doesn't
        # depend on, and causal masking prevents early positions from seeing the rest of the
        # window at all. pooling="mean" switches to a permutation-invariant readout: no
        # positional encoding, full (non-causal) attention so every position sees the whole
        # window, then mean-pool all positions' final hidden states before the head.
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")
        if pooling == "mean" and not use_attention:
            raise ValueError(
                "pooling='mean' requires use_attention=True -- CausalLinearMix bakes in a "
                "hardcoded causal structure internally that this readout mode is meant to remove"
            )
        self.pooling = pooling

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

        # attn_window (opt-in): unlike token_coverage_k (which drops entire positions from the
        # whole model), every position stays in the sequence here -- each query position i is
        # instead restricted to keys j with i-attn_window+1 <= j <= i, on top of the existing
        # causal constraint j <= i (attn_window >= seq_len recovers plain causal attention
        # exactly). Not combined with token_coverage_k in practice: a window over a coverage-
        # compressed index space wouldn't correspond to true temporal locality. Requires
        # use_attention=True for the same reason token_coverage_k does -- CausalLinearMix's
        # weight is a fixed table with no natural windowing extension.
        if attn_window is not None:
            if not use_attention:
                raise ValueError("attn_window requires use_attention=True")
            if not (1 <= attn_window <= seq_len):
                raise ValueError(f"attn_window must be in [1, {seq_len}], got {attn_window}")
        self.attn_window = attn_window

        # vocab_size (opt-in): src is a stream of integer token ids (see OHLCWindowDataset's
        # token_level) -- embed directly to d_model, same convention as Seq2SeqTransformer.
        self.embed = nn.Embedding(vocab_size, d_model) if vocab_size else None
        self.src_proj = None if vocab_size else nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        effective_len = token_coverage_k or seq_len
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, nhead, dim_feedforward, dropout, effective_len, use_attention, layernorm_mode, layerscale_init)
                for _ in range(num_layers)
            ]
        )
        if use_attention:
            if pooling == "mean":
                # Full (non-causal) attention -- every position sees the whole window. Overrides
                # attn_window: a window restriction is specifically what a symmetric, whole-window
                # readout needs removed, so combining the two isn't a meaningful configuration.
                self.causal_mask = None
            elif attn_window is not None:
                i_idx = torch.arange(effective_len).unsqueeze(1)
                j_idx = torch.arange(effective_len).unsqueeze(0)
                allowed = (j_idx <= i_idx) & (j_idx >= i_idx - attn_window + 1)
                mask = torch.zeros(effective_len, effective_len)
                mask.masked_fill_(~allowed, float("-inf"))
                self.register_buffer("causal_mask", mask, persistent=False)
            else:
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
        # output_bias_init (opt-in, default None preserves nn.Linear's usual small-random init):
        # when the loss-minimizing "ignore the input, output a constant" solution sits far from
        # the head's random init, early training spends most of its budget just discovering that
        # constant via the bias term before any input-dependent gradient signal (already weak
        # relative to it -- see ddm_regime_delta_regression's baseline R^2) gets a chance to act.
        # Starting the bias at the target's known train-mean skips that phase, so training begins
        # already at the "predict-the-mean" solution and any further loss reduction must come from
        # actually using the input.
        if output_bias_init is not None:
            with torch.no_grad():
                self.head.bias.fill_(output_bias_init)
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
            if self.pooling == "mean":
                x = self.pos_enc.dropout(x)
            else:
                x = x + self.pos_enc.pe[:, positions, :]
                x = self.pos_enc.dropout(x)
        elif self.pooling == "mean":
            x = self.pos_enc.dropout(x)
        else:
            x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, self.causal_mask)
        pooled = x.mean(dim=1) if self.pooling == "mean" else x[:, -1, :]
        pred = self.head(pooled)
        return pred.view(pred.size(0), self.pred_len, self.output_dim)
