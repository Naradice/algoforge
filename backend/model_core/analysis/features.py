"""Inputs of the strong volatility-memory baseline, plus clock features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model_core.analysis.targets import EPS, rms

SCALES = (1, 2, 5, 10, 20, 40, 60)


def har_features(X: np.ndarray, scales=SCALES) -> np.ndarray:
    """log RMS over the last s returns for each scale (multi-scale HAR in RMS form)."""
    return np.log(np.column_stack([rms(X[:, -s:]) for s in scales if s <= X.shape[1]]) + EPS)


def vol_memory_features(X: np.ndarray, scales=SCALES) -> np.ndarray:
    """Multi-scale log RMS, every log|r| in the window, signed sums over 5 / 20 / all returns."""
    return np.c_[har_features(X, scales), np.log(np.abs(X) + 1e-6),
                 X[:, -5:].sum(1), X[:, -20:].sum(1), X.sum(1)]


def time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """Time of day (4 harmonics) and weekday one-hot (Mon-Fri)."""
    mod = (ts.hour * 60 + ts.minute).to_numpy()
    tod = np.column_stack([f(2 * np.pi * k * mod / 1440) for k in (1, 2, 3, 4) for f in (np.sin, np.cos)])
    return np.c_[tod, np.eye(7)[ts.dayofweek.to_numpy()][:, :5]]


def linear_extras(X: np.ndarray, ts: pd.DatetimeIndex | None, scales=SCALES) -> np.ndarray:
    """Terms a linear model needs to compete fairly: squares and pairwise products of the
    multi-scale log RMS, and (with timestamps) hour-of-day dummies x RMS(5/20/60) --
    session-dependent persistence."""
    h = har_features(X, scales)
    k = h.shape[1]
    parts = [h ** 2, np.column_stack([h[:, i] * h[:, j] for i in range(k) for j in range(i + 1, k)])]
    if ts is not None:
        hour = np.eye(24)[ts.hour.to_numpy()]
        used = [s for s in scales if s <= X.shape[1]]
        cols = [used.index(s) for s in (5, 20, 60) if s in used]
        parts += [hour, np.column_stack([hour[:, a] * h[:, c] for a in range(24) for c in cols])]
    return np.column_stack(parts)
