"""Unit tests for model/architectures/decoder_only.py — the attention-ablation vehicle."""
from __future__ import annotations

import torch

from model.architectures.decoder_only import CausalLinearMix, DecoderOnlyTransformer


class TestCausalLinearMix:
    def test_output_shape(self):
        mix = CausalLinearMix(seq_len=10)
        x = torch.randn(4, 10, 16)
        out = mix(x)
        assert out.shape == (4, 10, 16)

    def test_causal_earlier_positions_unaffected_by_later_input_changes(self):
        torch.manual_seed(0)
        mix = CausalLinearMix(seq_len=8)
        x = torch.randn(2, 8, 4)
        out1 = mix(x)

        x2 = x.clone()
        x2[:, 5, :] = torch.randn(2, 4)  # perturb only position 5
        out2 = mix(x2)

        # positions 0..4 (strictly before the perturbed position) must be identical
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)
        # position 5 itself (and everything after, which can see position 5) must change
        assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-6)

    def test_weights_are_trainable_parameters(self):
        mix = CausalLinearMix(seq_len=6)
        assert mix.weight.requires_grad
        x = torch.randn(3, 6, 8, requires_grad=True)
        out = mix(x)
        out.sum().backward()
        assert mix.weight.grad is not None
        assert torch.any(mix.weight.grad != 0)


class TestDecoderOnlyTransformer:
    def _build(self, use_attention: bool, vocab_size: int | None = None, seq_len: int = 10):
        return DecoderOnlyTransformer(
            input_dim=1, output_dim=1, seq_len=seq_len, pred_len=3,
            d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
            dropout=0.0, use_attention=use_attention, device="cpu", vocab_size=vocab_size,
        )

    def test_continuous_input_output_shape(self):
        for use_attention in (True, False):
            model = self._build(use_attention)
            src = torch.randn(5, 10, 1)
            out = model(src)
            assert out.shape == (5, 3, 1)

    def test_tokenized_input_with_embedding(self):
        # k=4 token positions per step (mirrors token_level="digits": 1 sign + 3 digit tokens) --
        # seq_len passed to the constructor must already be obs_len_raw * k, matching how
        # celery_worker.py computes effective_config["seq_len"].
        obs_len_raw, k = 5, 4
        model = self._build(use_attention=True, vocab_size=12, seq_len=obs_len_raw * k)
        src = torch.randint(0, 12, (5, obs_len_raw, k))
        out = model(src)
        assert out.shape == (5, 3, 1)

    def test_use_attention_true_uses_multihead_attention_module(self):
        model = self._build(use_attention=True)
        assert all(hasattr(b.mix, "in_proj_weight") for b in model.blocks)

    def test_use_attention_false_uses_causal_linear_mix(self):
        model = self._build(use_attention=False)
        assert all(isinstance(b.mix, CausalLinearMix) for b in model.blocks)

    def test_gradients_flow_through_both_variants(self):
        for use_attention in (True, False):
            model = self._build(use_attention)
            src = torch.randn(4, 10, 1)
            out = model(src)
            out.sum().backward()
            # every parameter with requires_grad should have received a gradient
            for name, p in model.named_parameters():
                assert p.grad is not None, f"no gradient reached {name} (use_attention={use_attention})"

    def test_attention_and_no_attention_give_different_outputs(self):
        torch.manual_seed(0)
        model_a = self._build(use_attention=True)
        torch.manual_seed(0)
        model_b = self._build(use_attention=False)
        src = torch.randn(2, 10, 1)
        out_a = model_a(src)
        out_b = model_b(src)
        assert not torch.allclose(out_a, out_b)

    def test_tgt_argument_is_accepted_and_ignored(self):
        model = self._build(use_attention=True)
        src = torch.randn(2, 10, 1)
        tgt = torch.randn(2, 3, 1)
        out_with_tgt = model(src, tgt)
        out_without_tgt = model(src)
        assert torch.allclose(out_with_tgt, out_without_tgt)
