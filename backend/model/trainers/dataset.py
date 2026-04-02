"""
Sliding-window dataset for OHLC parquet files.

Compatible with the stocknet trainer interface:
    ds[start : start + batch_size]  →  (src_tensor, tgt_tensor)
where tensors are [batch, seq_len, features].

Normalisation modes:
    "returns"  — log returns of each feature column (stationary, recommended)
    "minmax"   — min-max scale each column to [0, 1]
    "none"     — raw values
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class OHLCWindowDataset:
    # Max tick batch files to load from a DDM directory (100 × 10 000 = 1 M ticks)
    _MAX_TICK_FILES = 100
    # Max OHLC rows to keep after loading; keeps window arrays from exceeding ~1 GB RAM
    _MAX_OHLC_ROWS = 50_000

    def __init__(
        self,
        artifact_path: str,
        obs_len: int,
        pred_len: int,
        feature_cols: list[str] | None = None,
        normalize: str = "returns",
        val_split: float = 0.2,
        device: str = "cpu",
    ) -> None:
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        full_path = store / artifact_path

        if full_path.is_dir():
            # DDM tick directory: load a capped sample and resample to M1 OHLC
            from data.parquet_reader import load_ddm_ticks
            tick_df = load_ddm_ticks(full_path, max_files=self._MAX_TICK_FILES)
            ohlc = tick_df["price"].resample("1min").ohlc()
            ohlc.columns = ["open", "high", "low", "close"]
            ohlc["volume"] = tick_df["price"].resample("1min").count()
            df = ohlc.dropna()
        else:
            df = pd.read_parquet(full_path)

        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index().dropna()
        if len(df) > self._MAX_OHLC_ROWS:
            df = df.iloc[-self._MAX_OHLC_ROWS:]

        if feature_cols is None:
            feature_cols = ["close"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            feature_cols = [df.columns[-1]]

        data = df[feature_cols].values.astype(np.float32)

        # Normalise
        if normalize == "returns":
            data = np.log(data + 1e-8)
            data = np.diff(data, axis=0)
        elif normalize == "minmax":
            mn, mx = data.min(axis=0), data.max(axis=0)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            data = (data - mn) / rng

        # Store normalisation params for inverse transform at inference
        self._normalize = normalize
        self._norm_min = data.min(axis=0) if normalize == "minmax" else None
        self._norm_max = data.max(axis=0) if normalize == "minmax" else None

        # Build windows
        n = len(data)
        total_len = obs_len + pred_len + 1   # +1 for the teacher-forced tgt shift
        n_windows = n - total_len + 1

        split_idx = int(n_windows * (1 - val_split))
        self._train_src, self._train_tgt = self._make_windows(data[:split_idx + total_len - 1], obs_len, pred_len + 1)
        val_data = data[split_idx:]
        if len(val_data) >= total_len:
            self._val_src, self._val_tgt = self._make_windows(val_data, obs_len, pred_len + 1)
        else:
            self._val_src, self._val_tgt = self._train_src, self._train_tgt

        self._is_train = True
        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = len(feature_cols)

    @staticmethod
    def _make_windows(data: np.ndarray, obs_len: int, tgt_len: int):
        n = len(data)
        total = obs_len + tgt_len
        srcs, tgts = [], []
        for i in range(n - total + 1):
            srcs.append(data[i : i + obs_len])
            tgts.append(data[i + obs_len - 1 : i + obs_len - 1 + tgt_len])  # overlapping by 1 for teacher forcing
        return np.array(srcs, dtype=np.float32), np.array(tgts, dtype=np.float32)

    def train(self) -> None:
        self._is_train = True

    def eval(self) -> None:
        self._is_train = False

    def __len__(self) -> int:
        return len(self._train_src) if self._is_train else len(self._val_src)

    def __getitem__(self, key):
        src_arr = self._train_src if self._is_train else self._val_src
        tgt_arr = self._train_tgt if self._is_train else self._val_tgt

        if isinstance(key, slice):
            src = torch.tensor(src_arr[key], device=self.device)
            tgt = torch.tensor(tgt_arr[key], device=self.device)
        else:
            src = torch.tensor(src_arr[key], device=self.device)
            tgt = torch.tensor(tgt_arr[key], device=self.device)

        return src, tgt
