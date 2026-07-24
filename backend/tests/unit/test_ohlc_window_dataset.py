"""Unit tests for model/trainers/dataset.py — OHLCWindowDataset normalize/token_level."""
import os

import numpy as np
import pandas as pd
import pytest


def _make_sine_parquet(path, n=600, period=60, amplitude=0.5, base=100.0):
    t = np.arange(n)
    close = base + amplitude * np.sin(2 * np.pi * t / period)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.ones(n, dtype=int),
    }, index=idx)
    df.index.name = "datetime"
    df.to_parquet(path)
    return path


@pytest.fixture
def artifact_store(tmp_path):
    orig = os.environ.get("ARTIFACT_STORE_PATH")
    os.environ["ARTIFACT_STORE_PATH"] = str(tmp_path)
    yield tmp_path
    if orig is None:
        os.environ.pop("ARTIFACT_STORE_PATH", None)
    else:
        os.environ["ARTIFACT_STORE_PATH"] = orig


class TestNormalizeDiff:
    def test_diff_matches_raw_first_differences(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="diff", val_split=0.2)

        # Reconstruct the raw closes independently and compare the target array's own diff.
        raw = pd.read_parquet(artifact_store / "ds.parquet")["close"].values.astype(np.float32)
        expected_diff = np.diff(raw)
        # First train window's tgt should equal a slice of expected_diff at the corresponding offset.
        first_tgt = ds._train_tgt[0, :, 0]
        assert np.allclose(first_tgt, expected_diff[ds.obs_len - 1: ds.obs_len - 1 + len(first_tgt)], atol=1e-4)

    def test_diff_has_near_zero_mean_unlike_raw_level(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds_diff = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="diff")
        ds_none = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="none")

        assert abs(ds_diff._train_tgt.mean()) < abs(ds_none._train_tgt.mean())


class TestTokenLevel:
    def test_default_none_preserves_src_equals_tgt(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="zscore")

        assert ds.vocab_size is None
        # src/tgt windows overlap by 1 element by construction (teacher forcing) -- when they're
        # built from the same underlying array (token_level=None), that shared element must match
        # exactly. Shapes differ (obs_len vs pred_len+1), so this is the meaningful equality check.
        assert np.allclose(ds._train_src[:, -1, :], ds._train_tgt[:, 0, :])

    def test_diff_token_level_gives_continuous_differenced_src_with_continuous_tgt(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="zscore", token_level="diff")

        assert ds.vocab_size is None
        assert ds._train_src.dtype == np.float32
        assert ds._train_tgt.dtype == np.float32
        # Same number of windows (src and tgt are built from index-aligned, equal-length arrays),
        # but different per-window length (obs_len vs pred_len+1) and different values, since src
        # is differenced and tgt stays at zscore-normalized levels.
        assert ds._train_src.shape[0] == ds._train_tgt.shape[0]
        assert ds._train_src.shape[1] == ds.obs_len
        assert ds._train_tgt.shape[1] == ds.pred_len + 1

    def test_quantize_diff_produces_integer_tokens_within_vocab(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=1200)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="quantize_diff", n_bins=7,
        )

        assert ds.vocab_size == 7
        assert ds.token_bin_edges is not None
        assert len(ds.token_bin_edges) == 6  # n_bins - 1 internal edges
        assert ds._train_src.dtype == np.int64
        assert ds._train_src.min() >= 0
        assert ds._train_src.max() < 7
        # tgt stays continuous and unaffected by the vocabulary.
        assert ds._train_tgt.dtype == np.float32

    def test_quantize_diff_bins_are_roughly_balanced_via_quantile_edges(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=3000, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="quantize_diff", n_bins=7, val_split=0.2,
        )

        tokens = ds._train_src.reshape(-1)
        counts = np.bincount(tokens, minlength=7)
        # Quantile bins on the fitting region should be roughly equal-frequency; allow slack
        # since windowing overlaps and this is a small, deterministic synthetic series.
        assert counts.min() > 0
        assert counts.max() / max(counts.min(), 1) < 5

    def test_token_level_rejects_multiple_feature_cols(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        with pytest.raises(ValueError, match="single feature_col|one feature_col"):
            OHLCWindowDataset(
                "ds.parquet", obs_len=10, pred_len=5, feature_cols=["open", "close"],
                token_level="diff",
            )

    def test_unknown_token_level_raises(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        with pytest.raises(ValueError, match="Unknown token_level"):
            OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, token_level="bogus")
