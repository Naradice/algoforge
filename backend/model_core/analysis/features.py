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


def market_features(w) -> np.ndarray:
    """For a panel: per anchor day, the cross-instrument mean of the 5/20/60-return signed sums and of
    the 20-return log RMS, plus the cross-sectional std of the 20-return sum. Every instrument's
    window ends on that day, so only past bars are used."""
    obs = w.X.shape[1]
    own = np.column_stack([w.X[:, -s:].sum(1) for s in (5, 20, 60) if s <= obs] +
                          [np.log(rms(w.X[:, -min(20, obs):]) + EPS)])
    df = pd.DataFrame(own)
    df["day"] = w.day
    g = df.groupby("day")
    mean = g.transform("mean").to_numpy()
    disp = g[1 if own.shape[1] > 2 else 0].transform("std").fillna(0.0).to_numpy()
    return np.c_[mean, disp]


AUX_SCALES = (1, 5, 20, 60)


def _window_means(v: np.ndarray, t: np.ndarray, scales) -> list[np.ndarray]:
    c = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    return [(c[t + 1] - c[t + 1 - s]) / s for s in scales]


def aux_features(w, scales=AUX_SCALES) -> np.ndarray:
    """Per aux series (e.g. log volume): its mean over the last s bars ending at the anchor bar."""
    if w.aux_v is None:
        return np.empty((len(w), 0))
    return np.concatenate(_window_means(w.aux_v, w.t, [s for s in scales if s <= w.X.shape[1]]), axis=1)


def future_aux_mean(w, h: int, col: int = 0) -> np.ndarray:
    """Target: mean of aux column `col` over the next h bars (after the anchor)."""
    v = w.aux_v[:, [col]]
    c = np.vstack([np.zeros((1, 1)), np.cumsum(v, axis=0)])
    return ((c[w.t + 1 + h] - c[w.t + 1]) / h)[:, 0]


def past_aux_mean(w, h: int, col: int = 0) -> np.ndarray:
    return _window_means(w.aux_v[:, [col]], w.t, [h])[0][:, 0]


def aux_hour_interactions(w) -> np.ndarray:
    """Hour-of-day dummies x each aux series' 20-bar mean: session-dependent persistence of e.g.
    volume, the aux analogue of linear_extras' hour x RMS terms. Without them a linear baseline
    cannot express 'volume persists differently at 8:00 than at 22:00' and trees win on that alone."""
    if w.aux_v is None:
        return np.empty((len(w), 0))
    hour = np.eye(24)[w.anchor_ts.hour.to_numpy()]
    m = _window_means(w.aux_v, w.t, [min(20, w.X.shape[1])])[0]
    return np.column_stack([hour[:, a] * m[:, c] for c in range(m.shape[1]) for a in range(24)])
