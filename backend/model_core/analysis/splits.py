"""Blocked train/test split by calendar day with a purge at every boundary."""
from __future__ import annotations

import numpy as np


def blocked_split(day: np.ndarray, test_frac: float = 0.2, purge: int = 80, seed: int = 0):
    """Whole days go to train or test at random; windows within `purge` positions of a train/test
    boundary are dropped, so no train window shares returns with a test window (with overlapping
    windows and a future target, neighbours share most of their target). `day` must be in window
    order; purge should be at least obs + horizon. Returns (train_idx, test_idx)."""
    days = np.unique(day)
    rng = np.random.default_rng(seed)
    test_days = rng.choice(days, size=max(1, int(len(days) * test_frac)), replace=False)
    is_test = np.isin(day, test_days)
    change = np.flatnonzero(is_test[1:] != is_test[:-1])
    near = np.zeros(len(day) + 1, dtype=np.int64)
    np.add.at(near, np.maximum(change - purge + 1, 0), 1)
    np.add.at(near, np.minimum(change + purge + 1, len(day)), -1)
    near = np.cumsum(near)[: len(day)] > 0
    return np.flatnonzero(~is_test & ~near), np.flatnonzero(is_test & ~near)
