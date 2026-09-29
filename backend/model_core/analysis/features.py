"""Inputs of the strong volatility-memory baseline, plus clock features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model_core.analysis.targets import EPS, efficiency_ratio, rms

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


def trend_features(X: np.ndarray, scales=(5, 10, 20, 40, 60)) -> np.ndarray:
    """Efficiency ratio |net move| / path length over the last s returns for each scale (0 if no move):
    the past trend-vs-range state, which trend targets are defined relative to."""
    return np.column_stack([np.nan_to_num(efficiency_ratio(X[:, -s:]), nan=0.0)
                            for s in scales if s <= X.shape[1]])


EXOG_SUM_SCALES = (1, 2, 5, 10, 20, 60)
EXOG_RMS_SCALES = (5, 20, 60)


def exog_features(w, sum_scales=EXOG_SUM_SCALES, rms_scales=EXOG_RMS_SCALES) -> np.ndarray:
    """Per other instrument: signed return sums and log RMS over the last s returns ending at each
    window's r_t (never later). Read off prefix sums, so it costs O(n * k) for any window length."""
    if w.exog_r is None:
        return np.empty((len(w), 0))
    r = w.exog_r
    c1 = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    c2 = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r ** 2, axis=0)])
    end = w.t + 1
    cols = [c1[end] - c1[end - s] for s in sum_scales if s <= w.X.shape[1]]
    cols += [np.log(np.sqrt(np.maximum(c2[end] - c2[end - s], 0) / s) + EPS) for s in rms_scales if s <= w.X.shape[1]]
    return np.concatenate(cols, axis=1)
