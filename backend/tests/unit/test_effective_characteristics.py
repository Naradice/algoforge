"""Unit tests for model/trainers/dataset.py — compute_effective_characteristics."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_ohlc_parquet(path: Path, n: int = 300, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
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


class TestComputeEffectiveCharacteristics:
    def test_computes_structure_metrics_on_close_with_no_preprocessing(self, artifact_store):
        from model.trainers.dataset import compute_effective_characteristics

        _make_ohlc_parquet(artifact_store / "ds.parquet", n=300)

        result = compute_effective_characteristics("ds.parquet", feature_cols=["close"], preprocessing=None)

        assert set(result.keys()) == {
            "long_range_dependence", "spectral_periodicity", "multiscale_wavelet",
            "complexity_nonlinearity", "regime_changes",
        }
        assert "error" not in result["long_range_dependence"]
        assert isinstance(result["long_range_dependence"]["hurst"], float)
        assert "error" not in result["regime_changes"]

    def test_uses_preprocessing_indicator_column_when_selected(self, artifact_store):
        from model.trainers.dataset import compute_effective_characteristics

        _make_ohlc_parquet(artifact_store / "ds.parquet", n=300)
        preprocessing = {"indicators": [{"type": "sma", "period": 5}]}

        result = compute_effective_characteristics(
            "ds.parquet", feature_cols=["sma_5"], preprocessing=preprocessing
        )

        # sma_5 is a smoothed version of close — still enough non-NaN points to compute on,
        # and shouldn't blow up just because it's an indicator column rather than raw close.
        assert "error" not in result["long_range_dependence"]

    def test_degrades_gracefully_per_analysis_on_short_series(self, artifact_store):
        from model.trainers.dataset import compute_effective_characteristics

        _make_ohlc_parquet(artifact_store / "short.parquet", n=10)

        result = compute_effective_characteristics("short.parquet", feature_cols=["close"], preprocessing=None)

        # All 5 registered analyses require >= 32 returns; each should fail independently
        # with an {"error": ...} entry rather than raising out of compute_effective_characteristics.
        assert set(result.keys()) == {
            "long_range_dependence", "spectral_periodicity", "multiscale_wavelet",
            "complexity_nonlinearity", "regime_changes",
        }
        for name, value in result.items():
            assert "error" in value, f"expected {name} to degrade to an error on a short series"
