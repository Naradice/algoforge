"""Unit tests for the synthetic_function collector."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from data.collectors.synthetic_function import collect, _generate_series


@pytest.fixture
def artifact_store(tmp_path):
    orig = os.environ.get("ARTIFACT_STORE_PATH")
    os.environ["ARTIFACT_STORE_PATH"] = str(tmp_path)
    # synthetic_function.py reads ARTIFACT_STORE_PATH at import time into a module-level
    # constant, so patch it directly rather than relying on re-import picking up the env var.
    import data.collectors.synthetic_function as mod
    orig_store = mod.ARTIFACT_STORE
    mod.ARTIFACT_STORE = tmp_path
    yield tmp_path
    mod.ARTIFACT_STORE = orig_store
    if orig is None:
        os.environ.pop("ARTIFACT_STORE_PATH", None)
    else:
        os.environ["ARTIFACT_STORE_PATH"] = orig


class TestGenerateSeries:
    def test_sine_is_periodic_with_given_period(self):
        period = 40
        s = _generate_series("sine", length=200, period=period, amplitude=1.0, freq_ratio=5)
        # Same phase every `period` steps → near-identical values (float roundoff only)
        assert np.allclose(s[:100], s[period : 100 + period], atol=1e-9)

    def test_sine_amplitude_bounds_the_range(self):
        s = _generate_series("sine", length=500, period=33, amplitude=2.5, freq_ratio=5)
        assert s.max() <= 2.5 + 1e-9
        assert s.min() >= -2.5 - 1e-9

    def test_sine_sum_combines_both_frequencies(self):
        base = _generate_series("sine", length=200, period=40, amplitude=1.0, freq_ratio=1)
        combined = _generate_series("sine_sum", length=200, period=40, amplitude=0.0, freq_ratio=5)
        # amplitude=0 on the second wave → sine_sum reduces to the plain base sine
        assert np.allclose(base, combined)

    def test_rejects_unknown_function(self):
        with pytest.raises(ValueError, match="Unknown synthetic function"):
            _generate_series("cosine", length=10, period=5, amplitude=1.0, freq_ratio=1)


class TestDelay:
    def test_deterministic_given_tau(self):
        s1 = _generate_series("delay", length=300, period=50, amplitude=1.0, freq_ratio=1, tau=17)
        s2 = _generate_series("delay", length=300, period=50, amplitude=1.0, freq_ratio=1, tau=17)
        assert np.array_equal(s1, s2)  # no RNG involved -- exact equality, not just allclose

    def test_different_tau_gives_a_different_trajectory(self):
        s17 = _generate_series("delay", length=300, period=50, amplitude=1.0, freq_ratio=1, tau=17)
        s30 = _generate_series("delay", length=300, period=50, amplitude=1.0, freq_ratio=1, tau=30)
        assert not np.allclose(s17, s30)

    def test_stays_bounded_and_non_constant(self):
        s = _generate_series("delay", length=2000, period=50, amplitude=1.0, freq_ratio=1, tau=17)
        assert np.all(np.isfinite(s))
        assert s.std() > 0.01  # genuinely moving, not settled to a fixed point at this tau


class TestTemporalXor:
    def test_values_are_binary(self):
        from data.collectors.synthetic_function import _temporal_xor

        x = _temporal_xor(length=500, seed=1)
        assert set(np.unique(x)).issubset({0.0, 1.0})

    def test_matches_direct_xor_definition(self):
        from data.collectors.synthetic_function import _temporal_xor

        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, size=502)
        expected = (bits[:500] ^ bits[1:501]).astype(np.float64)
        x = _temporal_xor(length=500, seed=1)
        assert np.array_equal(x, expected)

    def test_same_seed_is_reproducible(self):
        from data.collectors.synthetic_function import _temporal_xor

        assert np.array_equal(_temporal_xor(200, seed=5), _temporal_xor(200, seed=5))

    def test_different_seeds_differ(self):
        from data.collectors.synthetic_function import _temporal_xor

        assert not np.array_equal(_temporal_xor(200, seed=5), _temporal_xor(200, seed=6))


class TestLfsr:
    def test_values_are_binary(self):
        from data.collectors.synthetic_function import _lfsr

        x = _lfsr(length=500, bits=8, seed=1)
        assert set(np.unique(x)).issubset({0.0, 1.0})

    def test_period_is_exactly_2_pow_bits_minus_1(self):
        from data.collectors.synthetic_function import _lfsr

        for bits in (4, 5, 8):
            period = 2 ** bits - 1
            x = _lfsr(length=period * 2 + 5, bits=bits, seed=1)
            # the sequence must repeat with period exactly `period`, not sooner
            assert np.array_equal(x[:period], x[period : 2 * period])
            # not a shorter period than the maximal one (spot-check a few divisors don't already repeat)
            for shorter in (period // p for p in (2, 3) if period % p == 0):
                assert not np.array_equal(x[:shorter], x[shorter : 2 * shorter])

    def test_rejects_unsupported_bit_width(self):
        from data.collectors.synthetic_function import _lfsr

        with pytest.raises(ValueError, match="Unsupported lfsr_bits"):
            _lfsr(length=10, bits=6, seed=1)

    def test_same_seed_is_reproducible(self):
        from data.collectors.synthetic_function import _lfsr

        assert np.array_equal(_lfsr(200, 8, seed=5), _lfsr(200, 8, seed=5))


class TestCollect:
    def test_writes_parquet_with_expected_shape(self, artifact_store):
        result = collect(1, {"function": "sine", "period": "50", "length": "500", "timeframe": "M5"})

        assert result.row_count == 500
        full_path = artifact_store / result.artifact_path
        assert full_path.exists()

        df = pd.read_parquet(full_path)
        assert len(df) == 500
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
        # Flat OHLC candle — the series is a single point value, not a simulated market
        assert (df["open"] == df["close"]).all()
        assert (df["high"] == df["close"]).all()
        assert (df["low"] == df["close"]).all()

    def test_base_price_shifts_the_series(self, artifact_store):
        result = collect(1, {"function": "sine", "period": "50", "amplitude": "1", "base_price": "1000", "length": "200"})
        df = pd.read_parquet(artifact_store / result.artifact_path)
        assert df["close"].mean() == pytest.approx(1000, abs=1)

    def test_noise_zero_is_deterministic(self, artifact_store):
        cfg = {"function": "sine_sum", "period": "50", "amplitude": "0.5", "freq_ratio": "5", "length": "300", "noise": "0"}
        r1 = collect(1, cfg)
        df1 = pd.read_parquet(artifact_store / r1.artifact_path)
        r2 = collect(2, cfg)
        df2 = pd.read_parquet(artifact_store / r2.artifact_path)
        assert np.allclose(df1["close"].values, df2["close"].values)

    def test_noise_with_same_seed_is_reproducible(self, artifact_store):
        cfg = {"function": "sine", "period": "50", "noise": "0.1", "seed": "7", "length": "300"}
        r1 = collect(1, cfg)
        df1 = pd.read_parquet(artifact_store / r1.artifact_path)
        r2 = collect(2, cfg)
        df2 = pd.read_parquet(artifact_store / r2.artifact_path)
        assert np.allclose(df1["close"].values, df2["close"].values)

    def test_rejects_non_positive_period(self, artifact_store):
        with pytest.raises(ValueError, match="period"):
            collect(1, {"function": "sine", "period": "0"})

    def test_rejects_too_short_length(self, artifact_store):
        with pytest.raises(ValueError, match="length"):
            collect(1, {"function": "sine", "length": "1"})

    def test_collect_delay(self, artifact_store):
        result = collect(1, {"function": "delay", "tau": "17", "length": "500"})
        df = pd.read_parquet(artifact_store / result.artifact_path)
        assert len(df) == 500
        assert df["close"].std() > 0.01

    def test_collect_xor(self, artifact_store):
        result = collect(1, {"function": "xor", "amplitude": "0.5", "seed": "3", "length": "500"})
        df = pd.read_parquet(artifact_store / result.artifact_path)
        assert len(df) == 500
        # base_price +/- amplitude only -- a two-valued series (before any noise is added)
        assert set(np.round(df["close"].values - 100.0, 6)).issubset({-0.5, 0.5})

    def test_collect_lfsr(self, artifact_store):
        result = collect(1, {"function": "lfsr", "lfsr_bits": "8", "amplitude": "0.5", "seed": "3", "length": "500"})
        df = pd.read_parquet(artifact_store / result.artifact_path)
        assert len(df) == 500
        assert set(np.round(df["close"].values - 100.0, 6)).issubset({-0.5, 0.5})

    def test_collect_lfsr_rejects_unsupported_bit_width(self, artifact_store):
        with pytest.raises(ValueError, match="Unsupported lfsr_bits"):
            collect(1, {"function": "lfsr", "lfsr_bits": "6", "length": "100"})
