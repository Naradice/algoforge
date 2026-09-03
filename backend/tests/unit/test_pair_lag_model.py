"""Unit tests for model/architectures/pair_lag.py — PairLagModel."""
import torch

from model_core.architectures import build_model
from model_core.architectures.pair_lag import PairLagModel


class TestPairLagModel:
    def test_forward_shape_matches_pred_len_and_output_dim(self):
        model = PairLagModel(input_dim=1, output_dim=1, pred_len=12, lag=4, pool_size=10, hidden=16)
        src = torch.randint(0, 7, (8, 60, 1))
        out = model(src)
        assert out.shape == (8, 12, 1)

    def test_tgt_argument_is_accepted_and_ignored(self):
        model = PairLagModel(input_dim=1, output_dim=1, pred_len=12, lag=4, pool_size=10, hidden=16)
        src = torch.randint(0, 7, (4, 60, 1))
        tgt = torch.randn(4, 13, 1)
        out_with_tgt = model(src, tgt)
        assert out_with_tgt.shape == (4, 12, 1)

    def test_raises_when_pool_size_exceeds_valid_positions(self):
        model = PairLagModel(input_dim=1, output_dim=1, pred_len=12, lag=55, pool_size=20, hidden=16)
        src = torch.randint(0, 7, (2, 60, 1))
        try:
            model(src)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_build_model_dispatches_pair_lag(self):
        model = build_model("pair_lag", {"pred_len": 12, "lag": 8, "pool_size": 20, "hidden": 32})
        assert isinstance(model, PairLagModel)
        src = torch.randint(0, 7, (4, 60, 1))
        out = model(src)
        assert out.shape == (4, 12, 1)
