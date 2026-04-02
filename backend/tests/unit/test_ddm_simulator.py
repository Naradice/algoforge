"""Unit tests for the DDM simulator collector."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ddm(num_agent: int = 50, seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    from data.collectors.ddm_simulator import DDMv3
    return DDMv3(
        num_agent=num_agent,
        max_volatility=0.02,
        min_volatility=0.01,
        trade_unit=0.001,
        initial_price=100.0,
        spread=1.0,
        wma=5,
    )


def _prices(stream_result):
    """Extract just prices from simulate_stream results."""
    return [p for p, _ in stream_result]


def _ticks(stream_result):
    """Extract just tick_times from simulate_stream results."""
    return [t for _, t in stream_result]


# ---------------------------------------------------------------------------
# simulate_stream: fixed-length
# ---------------------------------------------------------------------------

class TestSimulateStreamFixed:
    def test_yields_exactly_n_trades(self):
        ddm = _make_ddm()
        result = list(ddm.simulate_stream(n_trades=50))
        assert len(result) == 50

    def test_yields_price_tick_tuples(self):
        ddm = _make_ddm()
        result = list(ddm.simulate_stream(n_trades=5))
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_all_prices_are_finite(self):
        ddm = _make_ddm()
        prices = _prices(ddm.simulate_stream(n_trades=200))
        assert all(np.isfinite(p) for p in prices)

    def test_all_prices_are_positive(self):
        ddm = _make_ddm()
        prices = _prices(ddm.simulate_stream(n_trades=200))
        assert all(p > 0 for p in prices)

    def test_tick_times_are_monotonically_increasing(self):
        """tick_time must advance on every step — never go backwards."""
        ddm = _make_ddm()
        ticks = _ticks(ddm.simulate_stream(n_trades=100))
        assert all(ticks[i] < ticks[i + 1] for i in range(len(ticks) - 1))

    def test_simulate_returns_dataframe(self):
        ddm = _make_ddm()
        df = ddm.simulate(n_trades=30)
        assert len(df) == 30
        assert "price" in df.columns


# ---------------------------------------------------------------------------
# simulate_stream: total_seconds stop condition
# ---------------------------------------------------------------------------

class TestSimulateStreamTotalSeconds:
    def test_stops_when_total_seconds_reached(self):
        """All yielded tick_times must be < total_seconds."""
        ddm = _make_ddm(num_agent=50, seed=42)
        total = 0.5  # 0.5 simulated seconds
        results = list(ddm.simulate_stream(total_seconds=total))
        assert len(results) > 0
        ticks = _ticks(results)
        assert all(t <= total for t in ticks)

    def test_yields_at_least_one_trade(self):
        ddm = _make_ddm(num_agent=50, seed=42)
        results = list(ddm.simulate_stream(total_seconds=1.0))
        assert len(results) > 0


# ---------------------------------------------------------------------------
# simulate_stream: endless mode
# ---------------------------------------------------------------------------

class TestSimulateStreamEndless:
    def test_runs_beyond_fixed_count_when_none(self):
        ddm = _make_ddm()
        gen = ddm.simulate_stream(n_trades=None)
        results = [next(gen) for _ in range(500)]
        assert len(results) == 500


# ---------------------------------------------------------------------------
# price_history cap
# ---------------------------------------------------------------------------

class TestPriceHistoryCap:
    def test_history_stays_bounded(self):
        ddm = _make_ddm()
        cap = ddm.wma + 2
        for _ in ddm.simulate_stream(n_trades=200):
            assert len(ddm.price_history) <= cap


# ---------------------------------------------------------------------------
# _write_batch: timestamp handling
# ---------------------------------------------------------------------------

class TestWriteBatch:
    def _find_parquet_files(self, out_dir: Path) -> list[Path]:
        """Find all part-*.parquet files under the partitioned directory."""
        return sorted(out_dir.rglob("part-*.parquet"))

    def test_timestamps_match_tick_times(self):
        from data.collectors.ddm_simulator import _write_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            start_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
            prices = [100.0, 101.0, 102.0]
            tick_times = [10.0, 20.0, 30.0]  # 10 seconds between each trade

            _write_batch(out_dir, 0, prices, tick_times, start_ts)
            files = self._find_parquet_files(out_dir)
            assert len(files) == 1, f"Expected 1 parquet file, found {len(files)}"
            written = pd.read_parquet(files[0])
            written.index = pd.to_datetime(written.index, utc=True)

            assert len(written) == 3
            expected_ts0 = start_ts + pd.Timedelta(seconds=10.0)
            assert written.index[0] == expected_ts0
            diffs = written.index.to_series().diff().dropna()
            assert (diffs == pd.Timedelta(seconds=10.0)).all()

    def test_two_batches_have_non_overlapping_timestamps(self):
        from data.collectors.ddm_simulator import _write_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            start_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")

            _write_batch(out_dir, 0, [100.0], [5.0], start_ts)
            _write_batch(out_dir, 1, [101.0], [10.0], start_ts)

            files = sorted(self._find_parquet_files(out_dir))
            assert len(files) == 2, f"Expected 2 parquet files, found {len(files)}"
            ts0 = pd.to_datetime(pd.read_parquet(files[0]).index, utc=True)[0]
            ts1 = pd.to_datetime(pd.read_parquet(files[1]).index, utc=True)[0]
            assert ts1 > ts0

    def test_meta_json_written(self):
        from data.collectors.ddm_simulator import _write_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            start_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
            _write_batch(out_dir, 0, [100.0, 101.0], [1.0, 2.0], start_ts)

            meta = json.loads((out_dir / "_meta.json").read_text())
            assert "from_ts" in meta
            assert "to_ts" in meta

    def test_partition_directory_structure(self):
        """Files must land under year=/month=/day= subdirectories."""
        from data.collectors.ddm_simulator import _write_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            start_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
            _write_batch(out_dir, 0, [100.0], [5.0], start_ts)

            files = self._find_parquet_files(out_dir)
            assert len(files) == 1
            # Path must contain year=/month=/day= segments
            parts = files[0].parts
            assert any(p.startswith("year=") for p in parts)
            assert any(p.startswith("month=") for p in parts)
            assert any(p.startswith("day=") for p in parts)


# ---------------------------------------------------------------------------
# collect(): fixed-length end-to-end
# ---------------------------------------------------------------------------

class TestCollectFixed:
    def test_produces_correct_ohlc_row_count(self):
        """collect() with length=10, M1 should produce ~10 OHLC candles."""
        from data.collectors.ddm_simulator import collect

        with tempfile.TemporaryDirectory() as tmpdir:
            import data.collectors.ddm_simulator as mod
            original_store = mod.ARTIFACT_STORE
            mod.ARTIFACT_STORE = Path(tmpdir)
            try:
                config = {
                    "num_agent": 50,
                    "max_volatility": 0.02,
                    "min_volatility": 0.01,
                    "trade_unit": 0.001,
                    "initial_price": 100.0,
                    "spread": 1.0,
                    "wma": 5,
                    "length": 10,
                    "timeframe": "M1",
                    "seed": 42,
                }
                result = collect(datasource_id=1, config=config)
                assert abs(result.row_count - 10) <= 2, (
                    f"Expected ~10 OHLC rows, got {result.row_count}"
                )
                assert result.artifact_path.endswith("ddm_ticks")
                assert result.from_ts < result.to_ts
            finally:
                mod.ARTIFACT_STORE = original_store

    def test_artifact_directory_contains_parquet_files(self):
        from data.collectors.ddm_simulator import collect

        with tempfile.TemporaryDirectory() as tmpdir:
            import data.collectors.ddm_simulator as mod
            original_store = mod.ARTIFACT_STORE
            mod.ARTIFACT_STORE = Path(tmpdir)
            try:
                config = {"num_agent": 50, "length": 5, "timeframe": "M1", "seed": 1}
                collect(datasource_id=2, config=config)
                artifact_dir = Path(tmpdir) / "datasets" / "src_2" / "ddm_ticks"
                # New layout: part-*.parquet files under year=/month=/day/ subdirs
                assert len(list(artifact_dir.rglob("part-*.parquet"))) >= 1
            finally:
                mod.ARTIFACT_STORE = original_store

    def test_parquet_readable_as_tick_data(self):
        from data.collectors.ddm_simulator import collect

        with tempfile.TemporaryDirectory() as tmpdir:
            import data.collectors.ddm_simulator as mod
            original_store = mod.ARTIFACT_STORE
            mod.ARTIFACT_STORE = Path(tmpdir)
            try:
                config = {"num_agent": 50, "length": 5, "timeframe": "M1", "seed": 2}
                result = collect(datasource_id=3, config=config)
                from data.parquet_reader import load_ddm_ticks
                tick_df = load_ddm_ticks(Path(tmpdir) / result.artifact_path)
                assert "price" in tick_df.columns
                assert len(tick_df) > 0
                assert tick_df["price"].notna().all()
                assert np.isfinite(tick_df["price"].values).all()
            finally:
                mod.ARTIFACT_STORE = original_store

    def test_rerun_clears_stale_batches(self):
        """Running collect() twice must not accumulate old partition files."""
        from data.collectors.ddm_simulator import collect

        with tempfile.TemporaryDirectory() as tmpdir:
            import data.collectors.ddm_simulator as mod
            original_store = mod.ARTIFACT_STORE
            mod.ARTIFACT_STORE = Path(tmpdir)
            try:
                cfg_base = {"num_agent": 50, "length": 5, "timeframe": "M1"}

                collect(datasource_id=10, config={**cfg_base, "seed": 1})
                artifact_dir = Path(tmpdir) / "datasets" / "src_10" / "ddm_ticks"
                count_first = len(list(artifact_dir.rglob("part-*.parquet")))

                collect(datasource_id=10, config={**cfg_base, "seed": 2})
                count_second = len(list(artifact_dir.rglob("part-*.parquet")))

                assert count_second == count_first, (
                    f"Second run left {count_second} files, first left {count_first}. "
                    "Stale partitions were not cleaned."
                )
                from data.parquet_reader import load_ddm_ticks
                tick_df = load_ddm_ticks(artifact_dir)
                assert np.isfinite(tick_df["price"].values).all()
            finally:
                mod.ARTIFACT_STORE = original_store

    def test_ohlc_spans_multiple_candles(self):
        """Fixed simulation must produce multiple distinct OHLC candles."""
        from data.collectors.ddm_simulator import collect

        with tempfile.TemporaryDirectory() as tmpdir:
            import data.collectors.ddm_simulator as mod
            original_store = mod.ARTIFACT_STORE
            mod.ARTIFACT_STORE = Path(tmpdir)
            try:
                config = {"num_agent": 50, "length": 20, "timeframe": "M1", "seed": 42}
                result = collect(datasource_id=11, config=config)
                assert result.row_count > 1, (
                    f"Expected multiple OHLC candles, got {result.row_count}"
                )
            finally:
                mod.ARTIFACT_STORE = original_store
