"""Unit tests for data/parquet_reader.py, focused on resample_ddm_ticks_streaming's
chunk-boundary correctness -- it must produce identical output to a plain
concat-everything-then-resample regardless of how the ticks happen to be split
across fragment files.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.parquet_reader import resample_ddm_ticks_streaming


def _write_fragments(out_dir: Path, prices: np.ndarray, timestamps: pd.DatetimeIndex, ticks_per_file: int) -> None:
    """Write ticks into out_dir/batch_NNNNNN.parquet fragments (legacy layout),
    mirroring how ddm_simulator.py writes real tick data."""
    n = len(prices)
    for i, start in enumerate(range(0, n, ticks_per_file)):
        end = min(start + ticks_per_file, n)
        df = pd.DataFrame({"price": prices[start:end]}, index=timestamps[start:end])
        df.index.name = "datetime"
        df.to_parquet(out_dir / f"batch_{i:06d}.parquet")


def _reference_ohlc(prices: np.ndarray, timestamps: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """The ground truth: load everything into one DataFrame, resample in one shot."""
    series = pd.Series(prices, index=timestamps.rename("datetime"))
    ohlc = series.resample(freq).ohlc()
    ohlc["volume"] = series.resample(freq).count()
    return ohlc.dropna()


def _make_irregular_ticks(n: int, seed: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Ticks at irregular sub-second intervals so minute-bucket boundaries fall at
    unpredictable points relative to any fixed chunk size -- the scenario that
    would expose an off-by-one in the carry-forward boundary logic."""
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    # Random gaps between ~0.05s and ~2s so ticks land at varying points within
    # each minute, including ties right at a minute boundary.
    gaps_s = rng.uniform(0.05, 2.0, size=n)
    offsets = pd.to_timedelta(np.cumsum(gaps_s), unit="s")
    start = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    timestamps = start + offsets
    return prices, timestamps


@pytest.mark.parametrize("ticks_per_file,chunk_files", [
    (500, 3),    # many small fragments per chunk -- boundary crossed often
    (500, 1),    # worst case: one fragment per chunk, boundary logic exercised every step
    (2000, 5),
])
def test_streaming_matches_reference_across_chunk_sizes(ticks_per_file, chunk_files):
    prices, timestamps = _make_irregular_ticks(n=15_000, seed=1)
    reference = _reference_ohlc(prices, timestamps, freq="1min")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_fragments(out_dir, prices, timestamps, ticks_per_file=ticks_per_file)
        result = resample_ddm_ticks_streaming(out_dir, freq="1min", chunk_files=chunk_files)

    pd.testing.assert_frame_equal(result, reference, check_dtype=False)


def test_streaming_handles_tick_exactly_on_bucket_boundary():
    """A tick landing exactly on a minute boundary must be attributed to the same
    bucket regardless of which chunk it happens to fall into."""
    start = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    timestamps = pd.DatetimeIndex([
        start + pd.Timedelta(seconds=58),
        start + pd.Timedelta(seconds=59, milliseconds=999),
        start + pd.Timedelta(minutes=1),          # exactly on the boundary
        start + pd.Timedelta(minutes=1, seconds=1),
        start + pd.Timedelta(minutes=2),
    ])
    prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4])
    reference = _reference_ohlc(prices, timestamps, freq="1min")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_fragments(out_dir, prices, timestamps, ticks_per_file=1)  # one tick per file
        result = resample_ddm_ticks_streaming(out_dir, freq="1min", chunk_files=1)

    pd.testing.assert_frame_equal(result, reference, check_dtype=False)


def test_streaming_single_bucket_whole_run():
    """All ticks fall in one bucket across many fragments -- must not finalize
    early just because a chunk boundary was crossed."""
    start = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    n = 50
    timestamps = start + pd.to_timedelta(np.arange(n) * 0.1, unit="s")  # all within 5s
    prices = 100.0 + np.arange(n) * 0.01
    reference = _reference_ohlc(prices, timestamps, freq="1min")
    assert len(reference) == 1

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_fragments(out_dir, prices, timestamps, ticks_per_file=5)
        result = resample_ddm_ticks_streaming(out_dir, freq="1min", chunk_files=2)

    pd.testing.assert_frame_equal(result, reference, check_dtype=False)


def test_streaming_raises_on_missing_directory():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            resample_ddm_ticks_streaming(missing)


def test_streaming_raises_on_empty_directory():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            resample_ddm_ticks_streaming(Path(tmp))
