"""Unit tests for data/characteristics.py — load_df_for_dataset."""
import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


def _make_ohlc_parquet(path: Path) -> Path:
    idx = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": np.ones(20) * 100,
        "high": np.ones(20) * 101,
        "low": np.ones(20) * 99,
        "close": np.ones(20) * 100,
        "volume": np.ones(20, dtype=int),
    }, index=idx)
    df.index.name = "datetime"
    df.to_parquet(path)
    return path


def _make_ddm_dir(base: Path, n_batches: int = 3) -> Path:
    d = base / "ddm_ticks"
    d.mkdir()
    for i in range(n_batches):
        idx = pd.date_range("2024-01-01", periods=5, freq="1s", tz="UTC")
        pd.DataFrame({"price": np.ones(5) * (100 + i)}, index=idx).to_parquet(
            d / f"batch_{i:04d}.parquet"
        )
    return d


class TestLoadDfForDataset:
    def _with_store(self, store_path: str):
        orig = os.environ.get("ARTIFACT_STORE_PATH")
        os.environ["ARTIFACT_STORE_PATH"] = store_path
        return orig

    def _restore(self, orig):
        if orig is None:
            os.environ.pop("ARTIFACT_STORE_PATH", None)
        else:
            os.environ["ARTIFACT_STORE_PATH"] = orig

    def test_loads_flat_parquet_file(self):
        from data.characteristics import load_df_for_dataset
        with tempfile.TemporaryDirectory() as tmp:
            orig = self._with_store(tmp)
            try:
                _make_ohlc_parquet(Path(tmp) / "ohlc.parquet")
                df = load_df_for_dataset("ohlc.parquet")
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 20
                assert "close" in df.columns
            finally:
                self._restore(orig)

    def test_loads_ddm_batch_directory(self):
        from data.characteristics import load_df_for_dataset
        with tempfile.TemporaryDirectory() as tmp:
            orig = self._with_store(tmp)
            try:
                _make_ddm_dir(Path(tmp))
                df = load_df_for_dataset("ddm_ticks")
                assert isinstance(df, pd.DataFrame)
                assert "price" in df.columns
                assert len(df) == 15  # 3 batches × 5 rows
            finally:
                self._restore(orig)

    def test_empty_directory_raises_value_error(self):
        """A directory with no batch_*.parquet files (e.g. a web_report folder) raises
        ValueError — this is why web_report datasets must skip compute_characteristics."""
        from data.characteristics import load_df_for_dataset
        with tempfile.TemporaryDirectory() as tmp:
            orig = self._with_store(tmp)
            try:
                web_dir = Path(tmp) / "web_reports" / "mizuho-sc"
                web_dir.mkdir(parents=True)
                (web_dir / "20240101_digest.pdf").write_bytes(b"%PDF")
                with pytest.raises(ValueError, match="No batch parquet files"):
                    load_df_for_dataset("web_reports/mizuho-sc")
            finally:
                self._restore(orig)

    def test_missing_file_raises(self):
        from data.characteristics import load_df_for_dataset
        with tempfile.TemporaryDirectory() as tmp:
            orig = self._with_store(tmp)
            try:
                with pytest.raises(Exception):
                    load_df_for_dataset("does_not_exist.parquet")
            finally:
                self._restore(orig)


# ── Structure / complexity analyses ────────────────────────────────────────────

def _df_from_close(close: np.ndarray, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq, tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    df.index.name = "datetime"
    return df


class TestLongRangeDependence:
    def test_trending_series_has_higher_hurst_than_white_noise(self):
        from data.characteristics import compute_long_range_dependence
        rng = np.random.default_rng(0)
        n = 3000
        trending = _df_from_close(100 + np.cumsum(rng.standard_normal(n) * 0.1))
        white_noise_returns = rng.standard_normal(n) * 0.01
        white_noise = _df_from_close(100 * np.exp(np.cumsum(white_noise_returns)))

        h_trend = compute_long_range_dependence(trending)["hurst"]
        h_noise = compute_long_range_dependence(white_noise)["hurst"]

        assert h_trend > 0.5
        assert abs(h_noise - 0.5) < 0.15

    def test_output_keys_and_interpretation_label(self):
        from data.characteristics import compute_long_range_dependence
        rng = np.random.default_rng(1)
        df = _df_from_close(100 + np.cumsum(rng.standard_normal(500) * 0.1))
        result = compute_long_range_dependence(df)
        assert set(result.keys()) >= {
            "hurst", "interpretation", "memory_length", "acf_significance_band", "adf_statistic", "adf_pvalue",
        }
        assert result["interpretation"] in {"trending", "mean-reverting", "random walk", "undetermined"}

    def test_too_short_series_raises(self):
        from data.characteristics import compute_long_range_dependence
        df = _df_from_close(np.linspace(100, 101, 10))
        with pytest.raises(ValueError):
            compute_long_range_dependence(df)


class TestSpectralPeriodicity:
    def test_periodic_series_has_high_periodicity_strength_and_matching_period(self):
        from data.characteristics import compute_spectral_periodicity
        rng = np.random.default_rng(0)
        n = 3000
        period = 50
        t = np.arange(n)
        close = 100 + 5 * np.sin(2 * np.pi * t / period) + rng.standard_normal(n) * 0.1
        result = compute_spectral_periodicity(_df_from_close(close))

        assert result["periodicity_strength"] > 10
        assert abs(result["dominant_period"] - period) < period * 0.2

    def test_white_noise_has_low_periodicity_strength(self):
        from data.characteristics import compute_spectral_periodicity
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.01))
        result = compute_spectral_periodicity(_df_from_close(close))
        assert result["periodicity_strength"] < 10

    def test_band_energy_fractions_sum_to_one(self):
        from data.characteristics import compute_spectral_periodicity
        rng = np.random.default_rng(2)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(1000) * 0.01))
        result = compute_spectral_periodicity(_df_from_close(close))
        assert abs(sum(result["band_energy"].values()) - 1.0) < 1e-6


class TestMultiscaleWavelet:
    def test_runs_without_error_and_energy_fractions_are_sane(self):
        from data.characteristics import compute_multiscale_wavelet
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01))
        result = compute_multiscale_wavelet(_df_from_close(close))

        assert len(result["energy_fraction"]) == result["level"] + 1
        assert len(result["labels"]) == len(result["energy_fraction"])
        assert abs(sum(result["energy_fraction"]) - 1.0) < 1e-6
        assert 0.0 <= result["flatness_score"] <= 1.0


class TestComplexityNonlinearity:
    def test_deterministic_periodic_series_flagged_nonlinear(self):
        from data.characteristics import compute_complexity_nonlinearity
        rng = np.random.default_rng(0)
        n = 3000
        t = np.arange(n)
        close = 100 + 5 * np.sin(2 * np.pi * t / 50) + rng.standard_normal(n) * 0.1
        result = compute_complexity_nonlinearity(_df_from_close(close))

        assert result["nonlinear"] is True
        assert result["sample_entropy"] >= 0 or np.isnan(result["sample_entropy"])
        assert 0.0 <= result["permutation_entropy"] <= 1.0

    def test_random_walk_not_flagged_nonlinear(self):
        from data.characteristics import compute_complexity_nonlinearity
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.01))
        result = compute_complexity_nonlinearity(_df_from_close(close))
        assert result["nonlinear"] is False

    def test_long_series_is_downsampled_to_cap(self):
        from data.characteristics import compute_complexity_nonlinearity, MAX_ANALYSIS_N
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(MAX_ANALYSIS_N * 3) * 0.01))
        result = compute_complexity_nonlinearity(_df_from_close(close))
        assert result["downsampled"] is True
        assert result["n_used"] <= MAX_ANALYSIS_N


class TestRegimeChanges:
    def test_mean_shift_detected_near_known_index(self):
        from data.characteristics import compute_regime_changes
        rng = np.random.default_rng(0)
        shift_idx = 1500
        shift = np.concatenate([
            rng.standard_normal(shift_idx) * 0.1,
            rng.standard_normal(3000 - shift_idx) * 0.1 + 2,
        ])
        close = 100 + np.cumsum(shift)
        result = compute_regime_changes(_df_from_close(close))

        assert result["n_changepoints"] >= 1
        assert any(abs(cp - shift_idx) < 100 for cp in result["changepoints"])

    def test_no_regime_change_in_stationary_noise(self):
        from data.characteristics import compute_regime_changes
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.01))
        result = compute_regime_changes(_df_from_close(close))
        assert result["n_changepoints"] == 0

    def test_long_series_is_downsampled_to_cap(self):
        from data.characteristics import compute_regime_changes, MAX_ANALYSIS_N
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(MAX_ANALYSIS_N * 3) * 0.01))
        result = compute_regime_changes(_df_from_close(close))
        assert result["downsampled"] is True
        assert result["n_used"] <= MAX_ANALYSIS_N
