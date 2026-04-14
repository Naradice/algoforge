"""Shared utilities for data collectors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_into_parquet(artifact_path: Path, new_df: pd.DataFrame) -> int:
    """Append new_df to an existing parquet file, dedup by index, return total row count.

    - Existing rows are preserved; new rows with the same timestamp overwrite them
      (keep="last" so the freshest download wins on overlap).
    - The file is overwritten atomically via pandas parquet write.
    """
    existing = pd.read_parquet(artifact_path)
    merged = pd.concat([existing, new_df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(artifact_path)
    from data.artifact_store import upload as _upload
    _upload(artifact_path)
    return len(merged)
