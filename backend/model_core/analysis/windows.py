"""Gap-aware windows of log returns: `obs` inputs before an anchor, `horizon` returns after it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from finance_client.fprocess.fprocess.validation import contiguous_segment_ids


@dataclass
class ReturnWindows:
    X: np.ndarray                 # [n, obs]     log returns r_{t-obs+1} .. r_t (the model's input)
    F: np.ndarray                 # [n, horizon] log returns r_{t+1} .. r_{t+horizon} (never an input)
    anchor_ts: pd.DatetimeIndex   # timestamp of the close that ends the input window
    day: np.ndarray               # calendar day (int64) of anchor_ts, the unit for blocked splits

    def __len__(self) -> int:
        return len(self.X)


def build_return_windows(close: pd.Series, obs: int = 60, horizon: int = 20,
                         expected_delta: pd.Timedelta | None = None) -> ReturnWindows:
    """All windows whose `obs` input returns and `horizon` future returns lie inside one gap-free
    run. r_k = log(close_{k+1} / close_k) is only defined when rows k and k+1 are in the same
    contiguous segment (finance_client.fprocess validation.contiguous_segment_ids); a window is
    kept only if every one of its obs + horizon returns is defined. `close` needs a DatetimeIndex."""
    close = close.astype(np.float64)
    seg = contiguous_segment_ids(close.index, expected_delta)
    r = np.diff(np.log(close.to_numpy()))
    if len(r) < obs + horizon:
        empty = pd.DatetimeIndex([])
        return ReturnWindows(X=np.empty((0, obs)), F=np.empty((0, horizon)), anchor_ts=empty,
                             day=np.empty(0, dtype=np.int64))
    bad = np.concatenate([[0], np.cumsum(seg[1:] != seg[:-1])])   # prefix count of gap-spanning returns
    t = np.arange(obs - 1, len(r) - horizon)
    t = t[(bad[t + horizon + 1] - bad[t - obs + 1]) == 0]
    win = np.lib.stride_tricks.sliding_window_view(r, obs)
    fut = np.lib.stride_tricks.sliding_window_view(r, horizon)
    anchor = close.index[t + 1]
    return ReturnWindows(X=win[t - obs + 1], F=fut[t + 1], anchor_ts=anchor,
                         day=np.asarray(anchor.normalize().asi8))
