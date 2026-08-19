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


def _make_sine_parquet_with_vol(path, n=600, period=60, amplitude=0.5, base=100.0):
    """Like _make_sine_parquet, but with an extra 'vol' column independent of close, for
    exercising tgt_feature_cols (predicting a different column than the model's input)."""
    t = np.arange(n)
    close = base + amplitude * np.sin(2 * np.pi * t / period)
    vol = 1.0 + 0.2 * np.cos(2 * np.pi * t / period)  # always positive, different phase/shape
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.ones(n, dtype=int), "vol": vol,
    }, index=idx)
    df.index.name = "datetime"
    df.to_parquet(path)
    return path


class TestCrossColumnTarget:
    def test_omitted_tgt_feature_cols_preserves_src_equals_tgt(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet_with_vol(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, normalize="diff")
        # Same overlap check as TestTokenLevel's default-None case: src/tgt windows share one
        # element by construction (teacher forcing) when built from the same underlying array.
        assert np.allclose(ds._train_src[:, -1, :], ds._train_tgt[:, 0, :])

    def test_tgt_feature_cols_predicts_a_different_column_than_src(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet_with_vol(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5,
            feature_cols=["close"], tgt_feature_cols=["vol"],
            src_normalize="diff", normalize="none",
        )
        # src is obs_len long, tgt is pred_len+1 long (unaffected by tgt_feature_cols).
        assert ds._train_src.shape[1:] == (10, 1)
        assert ds._train_tgt.shape[1:] == (6, 1)
        # With src and tgt now drawn from different columns, the teacher-forcing overlap element
        # should NOT match (would only coincide by chance) -- the clearest sign src != tgt here.
        assert not np.allclose(ds._train_src[:, -1, :], ds._train_tgt[:, 0, :])

        raw_vol = pd.read_parquet(artifact_store / "ds.parquet")["vol"].values.astype(np.float32)
        raw_close = pd.read_parquet(artifact_store / "ds.parquet")["close"].values.astype(np.float32)
        expected_src_flat = np.diff(raw_close)  # length n-1, src_normalize="diff"
        # tgt (normalize="none") has length n, one longer than src_flat -- the front-trim drops
        # tgt's first element so both end up tail-aligned at the same length.
        expected_tgt_flat = raw_vol[-len(expected_src_flat):]
        first_src = ds._train_src[0, :, 0]
        first_tgt = ds._train_tgt[0, :, 0]
        assert np.allclose(first_src, expected_src_flat[: len(first_src)], atol=1e-4)
        assert np.allclose(first_tgt, expected_tgt_flat[ds.obs_len - 1: ds.obs_len - 1 + len(first_tgt)], atol=1e-4)

    def test_token_level_combined_with_tgt_feature_cols_predicts_different_column(self, artifact_store):
        """The token-representation comparison (§07's next step) needs token_level (src
        representation) and tgt_feature_cols (cross-column target) to compose correctly --
        token_level's own tgt-length-alignment logic (line ~119 in dataset.py) was written before
        tgt_feature_cols existed, so this checks they don't silently misalign."""
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet_with_vol(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5,
            feature_cols=["close"], tgt_feature_cols=["vol"],
            token_level="diff", normalize="none",
        )
        raw_close = pd.read_parquet(artifact_store / "ds.parquet")["close"].values.astype(np.float32)
        raw_vol = pd.read_parquet(artifact_store / "ds.parquet")["vol"].values.astype(np.float32)
        expected_src_flat = np.diff(raw_close)          # token_level="diff": src = diff of close
        expected_tgt_flat = raw_vol[1:]                  # normalize="none", trimmed to align with the diff

        first_src = ds._train_src[0, :, 0]
        first_tgt = ds._train_tgt[0, :, 0]
        assert np.allclose(first_src, expected_src_flat[: len(first_src)], atol=1e-4)
        assert np.allclose(first_tgt, expected_tgt_flat[ds.obs_len - 1: ds.obs_len - 1 + len(first_tgt)], atol=1e-4)


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

    def test_quantize_diff_populates_token_stream_for_characteristics(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="quantize_diff", n_bins=7,
        )

        assert ds.token_stream is not None
        assert ds.token_stream.ndim == 1
        assert ds.token_stream.min() >= 0 and ds.token_stream.max() < 7

    def test_diff_and_none_leave_token_stream_unset(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        assert OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5).token_stream is None
        assert OHLCWindowDataset("ds.parquet", obs_len=10, pred_len=5, token_level="diff").token_stream is None


class TestClusterTokenLevel:
    def test_cluster_produces_integer_tokens_within_vocab_and_correct_length(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=1500, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="cluster", cluster_window=20, n_clusters=5,
        )

        assert ds.vocab_size == 5
        assert ds.token_stream is not None
        assert ds.token_stream.dtype == np.int64
        assert ds.token_stream.min() >= 0 and ds.token_stream.max() < 5
        # n=1500 -> diff length 1499 -> n_shapes = 1499 - 20 + 1 = 1480
        assert len(ds.token_stream) == 1480

    def test_cluster_centroids_have_expected_shape(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=1500, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="cluster", cluster_window=20, n_clusters=5,
        )

        assert ds.cluster_centroids is not None
        assert ds.cluster_centroids.shape == (5, 20)

    def test_cluster_tgt_stays_continuous_and_index_aligned_in_length(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=1500, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="cluster", cluster_window=20, n_clusters=5,
        )

        assert ds._train_tgt.dtype == np.float32
        assert ds._train_src.shape[0] == ds._train_tgt.shape[0]

    def test_cluster_rejects_series_shorter_than_cluster_window(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=30)
        with pytest.raises(ValueError, match="cluster_window"):
            OHLCWindowDataset(
                "ds.parquet", obs_len=5, pred_len=2, token_level="cluster", cluster_window=50,
            )

    def test_cluster_token_stream_recurs_at_the_signal_period(self, artifact_store):
        """On a periodic signal, clustering should recover something close to a phase
        partition of the cycle -- the same shape recurs every `period` steps, so the cluster
        id sequence should repeat at that lag far more often than chance (1/n_clusters)."""
        from model.trainers.dataset import OHLCWindowDataset

        period = 60
        _make_sine_parquet(artifact_store / "ds.parquet", n=3000, period=period)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="cluster", cluster_window=20, n_clusters=8,
        )

        stream = ds.token_stream
        match_rate = np.mean(stream[:-period] == stream[period:])
        assert match_rate > 3 / 8  # well above the 1/n_clusters chance rate


class TestDigitsTokenLevel:
    def test_digits_expand_one_step_into_sign_plus_n_digits_token_positions(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="digits", n_digits=3,
        )

        assert ds.vocab_size == 12
        assert ds._train_src.dtype == np.int64
        # obs_len=10 time steps, each expanding to (1 sign + 3 digits) = 4 token positions
        assert ds._train_src.shape[1:] == (10, 4)

    def test_digit_tokens_are_0_9_and_sign_tokens_are_10_or_11(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="digits", n_digits=3,
        )

        sign_col = ds.token_stream.reshape(-1, 4)[:, 0]
        digit_cols = ds.token_stream.reshape(-1, 4)[:, 1:]
        assert set(np.unique(sign_col)).issubset({10, 11})
        assert digit_cols.min() >= 0 and digit_cols.max() <= 9

    def test_digit_decoding_reconstructs_the_true_diff_within_scale_precision(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="digits", n_digits=3,
        )
        raw = pd_read_close(artifact_store / "ds.parquet")
        true_diff = np.diff(raw)

        toks = ds.token_stream.reshape(-1, 4)
        sign = np.where(toks[:, 0] == 11, -1.0, 1.0)
        magnitude = toks[:, 1] * 100 + toks[:, 2] * 10 + toks[:, 3]
        decoded = sign * magnitude / ds.digit_scale

        assert np.allclose(decoded, true_diff, atol=1.0 / ds.digit_scale + 1e-6)

    def test_digits_tgt_stays_continuous_with_the_standard_one_row_trim(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        ds_digits = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="digits", n_digits=3,
        )
        ds_diff = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore", token_level="diff",
        )

        assert ds_digits._train_tgt.dtype == np.float32
        # digits doesn't drop any time steps (unlike cluster's window trim) -- same tgt length as diff.
        assert ds_digits._train_tgt.shape[0] == ds_diff._train_tgt.shape[0]

    def test_digits_rejects_multiple_feature_cols(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=600)
        with pytest.raises(ValueError, match="one feature_col"):
            OHLCWindowDataset(
                "ds.parquet", obs_len=10, pred_len=5, feature_cols=["open", "close"],
                token_level="digits",
            )


class TestSaxTokenLevel:
    def test_sax_produces_integer_tokens_within_vocab(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=1500, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="sax", sax_paa_size=5, n_bins=7,
        )

        assert ds.vocab_size == 7
        assert ds.token_bin_edges is not None
        assert ds.token_stream.dtype == np.int64
        assert ds.token_stream.min() >= 0 and ds.token_stream.max() < 7
        # n=1500 -> diff length 1499 -> n_segments = 1499 // 5 = 299
        assert len(ds.token_stream) == 299

    def test_sax_bins_are_roughly_balanced_via_quantile_edges(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        # period=63 (not an exact multiple of sax_paa_size=5) avoids the degenerate case where a
        # purely periodic, noise-free signal PAA-averages down to only a handful of exactly
        # repeated values -- with period=60 (60/5=12 exactly), np.digitize's strict "<" boundary
        # against a quantile edge that lands exactly on one of only ~12 distinct repeated values
        # can leave a bin genuinely empty, which is correct digitize behaviour, not a bug.
        _make_sine_parquet(artifact_store / "ds.parquet", n=3000, period=63)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="sax", sax_paa_size=5, n_bins=7,
        )

        counts = np.bincount(ds.token_stream, minlength=7)
        assert counts.min() > 0
        assert counts.max() / max(counts.min(), 1) < 5

    def test_sax_tgt_realigns_to_the_last_step_of_each_paa_segment(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        p, obs_len, pred_len = 5, 10, 5
        _make_sine_parquet(artifact_store / "ds.parquet", n=1500, period=60)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=obs_len, pred_len=pred_len, normalize="zscore",
            token_level="sax", sax_paa_size=p, n_bins=7,
        )
        raw = pd_read_close(artifact_store / "ds.parquet")
        zscore = (raw - raw.mean()) / raw.std()

        # First training window's last tgt element (index obs_len-1+pred_len into the
        # SAX-realigned tgt array) should equal the zscored raw value at time (obs_len+pred_len)*p
        # -- tgt_data[k] = zscore[(k+1)*p] by construction (segment k covers up to time (k+1)*p).
        expected = zscore[(obs_len + pred_len) * p]
        assert ds._train_tgt[0, -1, 0] == pytest.approx(expected, abs=1e-4)

    def test_sax_rejects_series_shorter_than_paa_size(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        _make_sine_parquet(artifact_store / "ds.parquet", n=10)
        with pytest.raises(ValueError, match="sax_paa_size"):
            OHLCWindowDataset(
                "ds.parquet", obs_len=3, pred_len=1, token_level="sax", sax_paa_size=50,
            )

    def test_sax_token_stream_recurs_at_the_signal_period_in_paa_units(self, artifact_store):
        from model.trainers.dataset import OHLCWindowDataset

        period = 60
        p = 5
        _make_sine_parquet(artifact_store / "ds.parquet", n=3000, period=period)
        ds = OHLCWindowDataset(
            "ds.parquet", obs_len=10, pred_len=5, normalize="zscore",
            token_level="sax", sax_paa_size=p, n_bins=7,
        )

        stream = ds.token_stream
        lag = period // p  # segments per signal period
        match_rate = np.mean(stream[:-lag] == stream[lag:])
        assert match_rate > 1 / 7  # well above the 1/n_bins chance rate


def pd_read_close(path):
    return pd.read_parquet(path)["close"].values.astype(np.float32)


class TestComputeTokenCharacteristics:
    def test_uniform_random_tokens_have_near_max_entropy_and_low_mutual_information(self):
        from model.trainers.dataset import compute_token_characteristics

        rng = np.random.default_rng(0)
        vocab_size = 8
        tokens = rng.integers(0, vocab_size, size=20_000)

        result = compute_token_characteristics(tokens, vocab_size)

        assert result["token_entropy"]["normalized"] > 0.95  # near-uniform usage
        assert result["effective_vocab_size"] > vocab_size * 0.9
        assert abs(result["token_mutual_information"]) < 0.05  # iid -> ~0 bits shared with next token
        # 8 symbols is 3 bits/token of true entropy packed into 8-bit bytes, so the theoretical
        # floor is ~0.375; zlib doesn't hit the entropy bound exactly, but this should land well
        # above the near-zero ratio a truly compressible (e.g. constant) stream gets.
        assert result["lz_compression_ratio"] > 0.3

    def test_constant_token_stream_has_zero_entropy_and_compresses_well(self):
        from model.trainers.dataset import compute_token_characteristics

        tokens = np.zeros(5000, dtype=np.int64)
        result = compute_token_characteristics(tokens, vocab_size=8)

        assert result["token_entropy"]["bits"] == pytest.approx(0.0, abs=1e-9)
        assert result["effective_vocab_size"] == pytest.approx(1.0, abs=1e-6)
        assert result["lz_compression_ratio"] < 0.05  # trivially compressible

    def test_periodic_pattern_has_low_conditional_entropy_rate_at_its_own_period(self):
        from model.trainers.dataset import compute_token_characteristics

        # period-3 pattern repeated many times -- fully predictable given 2 tokens of context.
        pattern = np.array([0, 1, 2], dtype=np.int64)
        tokens = np.tile(pattern, 2000)

        result = compute_token_characteristics(tokens, vocab_size=3)

        rates = result["ngram_entropy"]["conditional_rates"]
        assert rates[max(rates)] < 0.05  # near-zero surprise once context captures the period
        assert result["token_mutual_information"] > 1.0  # strong adjacent-token structure

    def test_mi_curve_is_high_at_every_lag_for_a_periodic_pattern(self):
        """A period-3 tiled sequence is a deterministic function of position mod 3, so knowing
        the token at t fully determines the token at t+lag for *any* lag (not just lag=1) --
        this is the case §07's next step needs to distinguish from "only next-step structure"."""
        from model.trainers.dataset import compute_token_characteristics

        pattern = np.array([0, 1, 2], dtype=np.int64)
        tokens = np.tile(pattern, 2000)

        result = compute_token_characteristics(tokens, vocab_size=3, mi_lags=[1, 2, 4, 8])

        curve = result["token_mi_curve"]
        for lag in [1, 2, 4, 8]:
            assert curve[lag] > 1.5  # near log2(3) ~ 1.585 bits at every lag

    def test_mi_curve_is_near_zero_at_every_lag_for_iid_tokens(self):
        from model.trainers.dataset import compute_token_characteristics

        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 8, size=20_000)

        result = compute_token_characteristics(tokens, vocab_size=8, mi_lags=[1, 2, 4, 8, 16])

        curve = result["token_mi_curve"]
        for lag in [1, 2, 4, 8, 16]:
            assert abs(curve[lag]) < 0.05

    def test_token_target_mi_detects_a_predictive_relationship_only_at_the_true_horizon(self):
        """target[i] = tokens[i-1] by construction, so a token fully determines the target
        exactly one step ahead and carries no information about it at any other horizon (i.i.d.
        tokens otherwise) -- the clean synthetic case for "does token_target_mi actually find
        the horizon where the relationship lives, and only that one"."""
        from model.trainers.dataset import compute_token_characteristics

        rng = np.random.default_rng(2)
        vocab_size = 4
        tokens = rng.integers(0, vocab_size, size=5000)
        target = np.empty(5000)
        target[0] = 0.0
        target[1:] = tokens[:-1].astype(np.float64)

        result = compute_token_characteristics(
            tokens, vocab_size, target=target, target_horizons=[1, 2, 4], n_target_bins=vocab_size
        )

        tmi = result["token_target_mi"]
        assert tmi[1] > 1.5  # near log2(4) = 2 bits -- token fully determines target one step ahead
        assert abs(tmi[2]) < 0.3  # independent draws at any other horizon
        assert abs(tmi[4]) < 0.3

    def test_omitting_mi_lags_and_target_leaves_existing_keys_unaffected(self):
        """Both extensions are opt-in -- confirms the default call (no mi_lags, no target) is
        byte-for-byte the same result shape as before this feature existed."""
        from model.trainers.dataset import compute_token_characteristics

        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 8, size=2000)
        result = compute_token_characteristics(tokens, vocab_size=8)

        assert "token_mi_curve" not in result
        assert "token_target_mi" not in result

    def test_zipf_distributed_tokens_recover_a_positive_alpha_with_good_fit(self):
        from model.trainers.dataset import compute_token_characteristics

        rng = np.random.default_rng(1)
        vocab_size = 50
        # Construct an explicit Zipf-like frequency table (rank^-1) and sample from it.
        ranks = np.arange(1, vocab_size + 1)
        weights = 1.0 / ranks
        probs = weights / weights.sum()
        tokens = rng.choice(vocab_size, size=50_000, p=probs)

        result = compute_token_characteristics(tokens, vocab_size)

        zipf = result["token_zipf"]
        assert zipf["alpha"] is not None
        assert 0.5 < zipf["alpha"] < 2.0
        assert zipf["r2"] > 0.8

    def test_best_effort_does_not_raise_on_pathological_input(self):
        from model.trainers.dataset import compute_token_characteristics

        # A single repeated token: too few distinct tokens for a stable Zipf fit, but the call
        # must still succeed and report the degenerate case per-metric rather than raising.
        tokens = np.array([0, 0, 0], dtype=np.int64)
        result = compute_token_characteristics(tokens, vocab_size=1)

        assert "token_entropy" in result
        assert result["token_zipf"]["alpha"] is None
