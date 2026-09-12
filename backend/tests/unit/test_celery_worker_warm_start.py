"""Unit tests for celery_worker._apply_warm_start_checkpoint -- the transfer-learning hook
behind the `warm_start_checkpoint` training hyperparam (pretrain-on-synthetic,
fine-tune-on-USDJPY experiments)."""
from __future__ import annotations

import torch

from celery_worker import _apply_warm_start_checkpoint
from model_core.architectures.decoder_only import DecoderOnlyTransformer


def _build_model(seed: int) -> DecoderOnlyTransformer:
    torch.manual_seed(seed)
    return DecoderOnlyTransformer(
        input_dim=1, output_dim=1, seq_len=10, pred_len=1,
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
        dropout=0.0, device="cpu",
    )


class TestApplyWarmStartCheckpoint:
    def test_loads_source_models_weights(self, tmp_path):
        source = _build_model(seed=1)
        target = _build_model(seed=2)
        # Different seeds -> different random init, so this assertion is meaningful evidence
        # the checkpoint load (not coincidence) produced the match below.
        assert not torch.allclose(
            next(source.parameters()), next(target.parameters())
        )

        ckpt_path = tmp_path / "best.pt"
        torch.save({"model_state": source.state_dict()}, ckpt_path)

        _apply_warm_start_checkpoint(target, "best.pt", tmp_path, "cpu")

        for p_source, p_target in zip(source.parameters(), target.parameters()):
            assert torch.equal(p_source, p_target)

    def test_resolves_checkpoint_path_relative_to_store(self, tmp_path):
        source = _build_model(seed=1)
        target = _build_model(seed=2)

        nested = tmp_path / "models" / "7" / "training_42"
        nested.mkdir(parents=True)
        torch.save({"model_state": source.state_dict()}, nested / "best.pt")

        _apply_warm_start_checkpoint(target, "models/7/training_42/best.pt", tmp_path, "cpu")

        for p_source, p_target in zip(source.parameters(), target.parameters()):
            assert torch.equal(p_source, p_target)

    def test_output_matches_after_warm_start_given_same_input(self, tmp_path):
        # End-to-end evidence beyond raw parameter equality: a warm-started model must actually
        # reproduce the source model's forward pass, not just have matching state_dict keys.
        source = _build_model(seed=1)
        target = _build_model(seed=2)
        torch.save({"model_state": source.state_dict()}, tmp_path / "best.pt")
        _apply_warm_start_checkpoint(target, "best.pt", tmp_path, "cpu")

        source.eval()
        target.eval()
        x = torch.randn(3, 10, 1)
        with torch.no_grad():
            assert torch.allclose(source(x), target(x))
