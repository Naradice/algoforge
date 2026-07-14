"""
Sliding-window dataset for OHLC parquet files.

Compatible with the stocknet trainer interface:
    ds[start : start + batch_size]  →  (src_tensor, tgt_tensor)
where tensors are [batch, seq_len, features].

Normalisation modes:
    "returns"  — log returns of each feature column (stationary, recommended)
    "minmax"   — min-max scale each column to [0, 1]
    "zscore"   — standardize to zero mean and unit variance
    "robust"   — scale by median and IQR (robust to outliers)
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
        preprocessing: dict | None = None,
    ) -> None:
        df, feature_cols = self._load_preprocessed_df(artifact_path, feature_cols, preprocessing)
        data = df[feature_cols].values.astype(np.float32)

        # Normalise
        if normalize == "returns":
            data = np.log(data + 1e-8)
            data = np.diff(data, axis=0)
        elif normalize == "minmax":
            mn, mx = data.min(axis=0), data.max(axis=0)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            data = (data - mn) / rng
        elif normalize == "zscore":
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            sigma = np.where(sigma == 0, 1.0, sigma)
            data = (data - mu) / sigma
        elif normalize == "robust":
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = np.where(q75 - q25 == 0, 1.0, q75 - q25)
            data = (data - median) / iqr

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

    @classmethod
    def _load_preprocessed_df(
        cls,
        artifact_path: str,
        feature_cols: list[str] | None,
        preprocessing: dict | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load + preprocess a dataset artifact, up to (but not including) normalization.

        Returns the DataFrame with its real DatetimeIndex intact — this is "what will be fed
        to the model" after indicators/clustering and the row cap, before feature_cols are
        pulled out as a plain ndarray and normalized. Shared by __init__ and
        compute_effective_characteristics below.
        """
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        full_path = store / artifact_path

        if full_path.is_dir():
            # DDM tick directory: load a capped sample and resample to M1 OHLC
            from data.parquet_reader import load_ddm_ticks
            tick_df = load_ddm_ticks(full_path, max_files=cls._MAX_TICK_FILES)
            ohlc = tick_df["price"].resample("1min").ohlc()
            ohlc.columns = ["open", "high", "low", "close"]
            ohlc["volume"] = tick_df["price"].resample("1min").count()
            df = ohlc.dropna()
        else:
            df = pd.read_parquet(full_path)

        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index().dropna()

        # Apply indicators and clustering before row cap so indicators have full history
        if preprocessing:
            from model.trainers.preprocessing import apply_preprocessing
            df = apply_preprocessing(df, preprocessing)

        if len(df) > cls._MAX_OHLC_ROWS:
            df = df.iloc[-cls._MAX_OHLC_ROWS:]

        if feature_cols is None:
            feature_cols = ["close"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            feature_cols = [df.columns[-1]]

        return df, feature_cols

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


# Structure/complexity analyses from data/characteristics.py's registry — see that module's
# docstring for what each one measures (long-range dependence, periodicity, multiscale
# wavelet structure, entropy/nonlinearity, regime changes).
_EFFECTIVE_CHARACTERISTIC_KEYS = [
    "long_range_dependence",
    "spectral_periodicity",
    "multiscale_wavelet",
    "complexity_nonlinearity",
    "regime_changes",
]


def compute_effective_characteristics(
    artifact_path: str,
    feature_cols: list[str] | None,
    preprocessing: dict | None,
) -> dict:
    """Structure characteristics of the data a training run will actually consume: after
    preprocessing (indicators/clustering) and the row cap, on the primary feature column,
    before normalization (whose output breaks the log-return math these analyses rely on).

    Best-effort per analysis — an error in one doesn't blank the rest. Callers should also
    wrap the call itself, since a completely unreadable/degenerate dataset can still raise
    before any per-analysis try/except is reached (e.g. during loading).
    """
    from data.characteristics import CHARACTERISTIC_REGISTRY

    df, resolved_feature_cols = OHLCWindowDataset._load_preprocessed_df(artifact_path, feature_cols, preprocessing)
    series_df = pd.DataFrame({"close": df[resolved_feature_cols[0]]})

    results: dict = {}
    for name in _EFFECTIVE_CHARACTERISTIC_KEYS:
        try:
            results[name] = CHARACTERISTIC_REGISTRY[name](series_df)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
