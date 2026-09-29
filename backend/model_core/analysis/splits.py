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


def chronological_split(day: np.ndarray, test_frac: float = 0.2, purge_days: int = 80):
    """The last `test_frac` of distinct days is the test set; the `purge_days` distinct days before it
    are dropped, so no train window's inputs or target reach into the test period. Works for a
    panel (windows of many instruments, any order): every instrument is cut at the same date.
    The honest split for long horizons, where a random-day split would purge most of the data.
    Returns (train_idx, test_idx)."""
    days = np.unique(day)
    n_test = max(1, int(len(days) * test_frac))
    if len(days) <= n_test + purge_days:
        raise ValueError(f"{len(days)} distinct days is too few for {n_test} test days plus a {purge_days}-day purge")
    test_start, cutoff = days[-n_test], days[-(n_test + purge_days)]
    return np.flatnonzero(day < cutoff), np.flatnonzero(day >= test_start)


def holdout_split(day: np.ndarray, test_start, test_end, purge_days: int = 80):
    """Days in [test_start, test_end) are the test set (`day`: int64 ns day stamps, as ReturnWindows.day); `purge_days` distinct days on each side of
    it are dropped and everything else trains (blocked cross-validation: train may lie after the
    test block). Moving the block over the sample checks that a chronological result is not one
    period's luck. `day` holds int64 ns day stamps; bounds are anything pd.Timestamp accepts."""
    import pandas as pd
    lo, hi = pd.Timestamp(test_start).value, pd.Timestamp(test_end).value
    days = np.unique(day)
    inside = np.flatnonzero((days >= lo) & (days < hi))
    if len(inside) == 0:
        raise ValueError(f"no days in the test period {test_start} .. {test_end}")
    before = days[max(inside[0] - purge_days, 0)]
    after = days[min(inside[-1] + purge_days, len(days) - 1)]
    is_test = (day >= lo) & (day < hi)
    return np.flatnonzero(~is_test & ((day < before) | (day > after))), np.flatnonzero(is_test)
