"""Unit tests for model/architectures/decoder_only.py — the attention-ablation vehicle."""
from __future__ import annotations

import numpy as np
import torch

from model_core.architectures.decoder_only import CausalLinearMix, DecoderOnlyTransformer


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


class TestTokenCoverage:
    def _build(self, token_coverage_k=None, token_coverage_mode="contiguous", use_attention=True, seq_len=10):
        return DecoderOnlyTransformer(
            input_dim=1, output_dim=1, seq_len=seq_len, pred_len=3,
            d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
            dropout=0.0, use_attention=use_attention, device="cpu",
            token_coverage_k=token_coverage_k, token_coverage_mode=token_coverage_mode,
        )

    def test_none_reproduces_exact_baseline(self):
        torch.manual_seed(0)
        model_a = DecoderOnlyTransformer(
            input_dim=1, output_dim=1, seq_len=10, pred_len=3,
            d_model=16, nhead=2, num_layers=2, dim_feedforward=32, dropout=0.0, device="cpu",
        )
        torch.manual_seed(0)
        model_b = self._build(token_coverage_k=None)
        src = torch.randn(4, 10, 1)
        assert torch.allclose(model_a(src), model_b(src))

    def test_output_shape_per_mode(self):
        for mode in ("contiguous", "uniform", "random"):
            for k in (4, 7, 10):
                model = self._build(token_coverage_k=k, token_coverage_mode=mode)
                src = torch.randn(5, 10, 1)
                out = model(src)
                assert out.shape == (5, 3, 1), f"mode={mode} k={k}"

    def test_last_position_always_used(self):
        # Perturbing the most recent step (seq_len-1) must change the output under every mode --
        # if it didn't, that mode silently dropped the anchor position. For "random", pin the
        # RNG seed before each forward call so both calls sample the identical position set --
        # otherwise resampling noise alone would make the outputs differ, masking the thing this
        # test is actually checking.
        torch.manual_seed(0)
        for mode in ("contiguous", "uniform", "random"):
            model = self._build(token_coverage_k=4, token_coverage_mode=mode)
            model.eval()
            src = torch.randn(2, 10, 1)
            np.random.seed(123)
            out1 = model(src)
            src2 = src.clone()
            src2[:, -1, :] = torch.randn(2, 1)
            np.random.seed(123)
            out2 = model(src2)
            assert not torch.allclose(out1, out2), f"mode={mode} did not use seq_len-1"

    def test_contiguous_and_uniform_are_deterministic_across_calls(self):
        for mode in ("contiguous", "uniform"):
            model = self._build(token_coverage_k=4, token_coverage_mode=mode)
            model.eval()
            src = torch.randn(3, 10, 1)
            out1 = model(src)
            out2 = model(src)
            assert torch.allclose(out1, out2)

    def test_random_mode_varies_across_calls(self):
        model = self._build(token_coverage_k=4, token_coverage_mode="random")
        model.eval()
        src = torch.randn(3, 10, 1)
        outs = [model(src) for _ in range(8)]
        assert not all(torch.allclose(outs[0], o) for o in outs[1:])

    def test_raises_with_use_attention_false(self):
        try:
            self._build(token_coverage_k=4, use_attention=False)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_raises_on_invalid_mode(self):
        try:
            self._build(token_coverage_k=4, token_coverage_mode="bogus")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_raises_on_out_of_range_k(self):
        for bad_k in (0, -1, 11):
            try:
                self._build(token_coverage_k=bad_k)
                assert False, f"expected ValueError for k={bad_k}"
            except ValueError:
                pass

    def test_gradients_flow_with_coverage(self):
        for mode in ("contiguous", "uniform", "random"):
            model = self._build(token_coverage_k=4, token_coverage_mode=mode)
            src = torch.randn(4, 10, 1)
            out = model(src)
            out.sum().backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"no gradient reached {name} (mode={mode})"


class TestAttentionWindow:
    def _build(self, attn_window=None, use_attention=True, seq_len=10):
        return DecoderOnlyTransformer(
            input_dim=1, output_dim=1, seq_len=seq_len, pred_len=3,
            d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
            dropout=0.0, use_attention=use_attention, device="cpu",
            attn_window=attn_window,
        )

    def test_none_reproduces_exact_baseline(self):
        torch.manual_seed(0)
        model_a = DecoderOnlyTransformer(
            input_dim=1, output_dim=1, seq_len=10, pred_len=3,
            d_model=16, nhead=2, num_layers=2, dim_feedforward=32, dropout=0.0, device="cpu",
        )
        torch.manual_seed(0)
        model_b = self._build(attn_window=None)
        src = torch.randn(4, 10, 1)
        assert torch.allclose(model_a(src), model_b(src))

    def test_full_window_reproduces_plain_causal(self):
        # attn_window == seq_len should behave identically to attn_window=None (both allow
        # every j <= i), even though the mask is built via a different code path.
        torch.manual_seed(0)
        model_a = self._build(attn_window=None)
        torch.manual_seed(0)
        model_b = self._build(attn_window=10)
        src = torch.randn(4, 10, 1)
        assert torch.allclose(model_a(src), model_b(src))

    def test_output_shape(self):
        for w in (1, 2, 5, 10):
            model = self._build(attn_window=w)
            src = torch.randn(5, 10, 1)
            out = model(src)
            assert out.shape == (5, 3, 1), f"window={w}"

    def test_query_unaffected_by_input_outside_its_window(self):
        # Query position i=7 with attn_window=3 can see keys {5,6,7} -- perturbing position 3
        # (outside that window) must leave the model's output completely unaffected, since the
        # model only ever reads from the last position (i=9 here) -- so use seq_len small enough
        # that the *last* position's own window excludes an early position.
        torch.manual_seed(0)
        model = self._build(attn_window=3, seq_len=10)
        model.eval()
        src = torch.randn(2, 10, 1)
        out1 = model(src)
        src2 = src.clone()
        src2[:, 0, :] = torch.randn(2, 1)  # position 0 is outside the last position's window {7,8,9}
        out2 = model(src2)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_query_affected_by_input_inside_its_window(self):
        torch.manual_seed(0)
        model = self._build(attn_window=3, seq_len=10)
        model.eval()
        src = torch.randn(2, 10, 1)
        out1 = model(src)
        src2 = src.clone()
        src2[:, 8, :] = torch.randn(2, 1)  # position 8 is inside the last position's window {7,8,9}
        out2 = model(src2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_raises_with_use_attention_false(self):
        try:
            self._build(attn_window=3, use_attention=False)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_raises_on_out_of_range_window(self):
        for bad_w in (0, -1, 11):
            try:
                self._build(attn_window=bad_w)
                assert False, f"expected ValueError for attn_window={bad_w}"
            except ValueError:
                pass

    def test_gradients_flow_with_window(self):
        for w in (1, 3, 10):
            model = self._build(attn_window=w)
            src = torch.randn(4, 10, 1)
            out = model(src)
            out.sum().backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"no gradient reached {name} (attn_window={w})"
