"""
Shared helpers for reading DDM tick parquet artifacts.

Supports two on-disk layouts transparently:

  Legacy (batch files):
    artifacts/datasets/src_N/ddm_ticks/
      batch_000000.parquet
      batch_000001.parquet
      ...

  Current (date-partitioned Hive):
    artifacts/datasets/src_N/ddm_ticks/
      year=2000/month=01/day=03/part-000000.parquet
      year=2000/month=01/day=04/part-000000.parquet
      ...

Both layouts return a DataFrame with a datetime index and a "price" column,
sorted ascending. The caller is responsible for OHLC resampling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Maximum number of parquet fragment files to read at once (caps memory use).
# 100 files × ~10 000 ticks each ≈ 1 M ticks — enough for all analysis tasks.
_MAX_FILES = 100


def _is_partitioned(path: Path) -> bool:
    """Return True if path contains Hive year= subdirectories."""
    try:
        return any(p.is_dir() and p.name.startswith("year=") for p in path.iterdir())
    except (OSError, NotADirectoryError):
        return False


def load_ddm_ticks(path: Path, max_files: int = _MAX_FILES) -> pd.DataFrame:
    """Load DDM tick data from either layout, capped at max_files fragments.

    Sampling strategy: if there are more files than max_files, take an evenly
    spaced sample so the full time range is represented rather than only the
    start or end.

    Returns a DataFrame with a UTC-aware DatetimeIndex and a "price" column,
    sorted ascending by time.
    """
    if not path.exists():
        raise FileNotFoundError(f"DDM artifact directory not found: {path}")

    if _is_partitioned(path):
        return _load_partitioned(path, max_files)
    else:
        return _load_legacy(path, max_files)


def load_ddm_ticks_recent(path: Path, n_files: int = 20) -> pd.DataFrame:
    """Load the most-recent n_files fragments only (used for live preview).

    For a running simulation this returns the latest data without scanning the
    whole artifact directory.
    """
    if not path.exists():
        return pd.DataFrame(columns=["price"])

    if _is_partitioned(path):
        fragments = _sorted_fragments(path)
        recent = fragments[-n_files:] if len(fragments) > n_files else fragments
    else:
        files = sorted(path.glob("batch_*.parquet"))
        recent = files[-n_files:] if len(files) > n_files else files

    if not recent:
        return pd.DataFrame(columns=["price"])

    df = pd.concat([pd.read_parquet(f) for f in recent]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_ddm_ticks_windowed(
    path: Path,
    n_files: int = 20,
    to_ts: float | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load the most-recent n_files fragments, optionally capped at to_ts.

    Returns (df, has_more) where has_more=True means older fragments exist
    beyond the loaded window.
    """
    if not path.exists():
        return pd.DataFrame(columns=["price"]), False

    if _is_partitioned(path):
        all_fragments = _sorted_fragments(path)
    else:
        all_fragments = sorted(path.glob("batch_*.parquet"))

    if not all_fragments:
        return pd.DataFrame(columns=["price"]), False

    has_more = len(all_fragments) > n_files
    recent = all_fragments[-n_files:] if len(all_fragments) > n_files else all_fragments

    df = pd.concat([pd.read_parquet(f) for f in recent]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)

    if to_ts is not None:
        cutoff = pd.Timestamp(to_ts, unit="s", tz="UTC")
        df = df[df.index <= cutoff]

    return df, has_more


def resample_ddm_ticks_streaming(path: Path, freq: str = "1min", target_ticks_per_chunk: int = 2_000_000) -> pd.DataFrame:
    """Resample DDM tick data to OHLC candles without ever holding the full tick
    history in memory at once.

    load_ddm_ticks(path, max_files=<all fragments>) -- what collect() used for a
    finite run's one-time materialization -- concatenates every tick fragment into
    a single DataFrame before resampling. For a large finite run (hundreds of
    millions of ticks across tens of thousands of fragments) that OOMs; the crash
    consistently happened right after "DDM collect done" was logged, i.e. in this
    exact read-back step, not in the simulation loop itself.

    Chunks by accumulated tick count, not file count -- ddm_simulator.py scales its
    batch (fragment) size with the requested run length, so a fixed file-count
    chunk would mean a wildly different (and potentially still-OOMing) amount of
    data per chunk depending on how the source run was written. A chunk's last
    resample bucket may still receive ticks from the next chunk, so its raw ticks
    (not the partial candle) are carried forward and merged before that bucket is
    finalized -- fragments are chronologically sorted and non-overlapping, so this
    never revisits an already-finalized bucket.

    Returns a DataFrame with columns [open, high, low, close, volume], matching
    _ticks_to_ohlc's output.
    """
    if not path.exists():
        raise FileNotFoundError(f"DDM artifact directory not found: {path}")

    fragments = _sorted_fragments(path) if _is_partitioned(path) else sorted(path.glob("batch_*.parquet"))
    if not fragments:
        raise ValueError(f"No parquet files found in directory: {path}")

    ohlc_chunks: list[pd.DataFrame] = []
    carry: pd.DataFrame | None = None
    buf: list[pd.DataFrame] = []
    buf_len = 0

    def _flush(is_last_chunk: bool) -> None:
        nonlocal carry, buf, buf_len
        chunk = pd.concat(buf).sort_index()
        chunk.index = pd.to_datetime(chunk.index, utc=True)
        combined = pd.concat([carry, chunk]).sort_index() if carry is not None else chunk

        ohlc = combined["price"].resample(freq).ohlc()
        ohlc["volume"] = combined["price"].resample(freq).count()

        if is_last_chunk:
            ohlc_chunks.append(ohlc)
            carry = None
        elif len(ohlc) <= 1:
            # Whole chunk fell in a single bucket -- can't finalize it yet since
            # the next chunk might extend it; carry everything forward as-is.
            carry = combined
        else:
            last_bucket_start = ohlc.index[-1]
            ohlc_chunks.append(ohlc.iloc[:-1])
            carry = combined[combined.index >= last_bucket_start]
        buf = []
        buf_len = 0

    for i, f in enumerate(fragments):
        buf.append(pd.read_parquet(f))
        buf_len += len(buf[-1])
        is_last_fragment = i == len(fragments) - 1
        if buf_len >= target_ticks_per_chunk or is_last_fragment:
            _flush(is_last_chunk=is_last_fragment)

    result = pd.concat(ohlc_chunks) if ohlc_chunks else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return result.dropna()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_fragments(path: Path) -> list[Path]:
    """Return all part-*.parquet files under a Hive-partitioned directory,
    sorted by path (which is lexicographically equivalent to time order for
    zero-padded year/month/day/part naming)."""
    return sorted(path.rglob("part-*.parquet"))


def _load_partitioned(path: Path, max_files: int) -> pd.DataFrame:
    fragments = _sorted_fragments(path)
    if not fragments:
        raise ValueError(f"No parquet files found in partitioned directory: {path}")

    if len(fragments) > max_files:
        step = max(1, len(fragments) // max_files)
        fragments = fragments[::step][:max_files]

    df = pd.concat([pd.read_parquet(f) for f in fragments]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _load_legacy(path: Path, max_files: int) -> pd.DataFrame:
    files = sorted(path.glob("batch_*.parquet"))
    if not files:
        raise ValueError(f"No batch parquet files found in directory: {path}")

    if len(files) > max_files:
        step = max(1, len(files) // max_files)
        files = files[::step][:max_files]

    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df
