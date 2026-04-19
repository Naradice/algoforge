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
