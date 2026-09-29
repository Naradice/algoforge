"""Gap-aware windows of log returns: `obs` inputs before an anchor, `horizon` returns after it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from finance_client.fprocess.fprocess.validation import contiguous_segment_ids


@dataclass
class ReturnWindows:
    X: np.ndarray                 # [n, obs]     log returns r_{t-obs+1} .. r_t (the model's input)
    F: np.ndarray                 # [n, horizon] log returns r_{t+1} .. r_{t+horizon} (never an input)
    anchor_ts: pd.DatetimeIndex   # timestamp of the close that ends the input window
    day: np.ndarray               # calendar day (int64) of anchor_ts, the unit for blocked splits
    t: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # index of r_t per window
    # Log returns of other instruments on the same rows, [len(r), k] -- kept whole (not windowed:
    # [n, obs, k] would be GBs); features are read off it by `t` (features.exog_features).
    exog_r: np.ndarray | None = None
    exog_names: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.X)

    def take(self, idx: np.ndarray) -> "ReturnWindows":
        return ReturnWindows(X=self.X[idx], F=self.F[idx], anchor_ts=self.anchor_ts[idx], day=self.day[idx],
                             t=self.t[idx] if len(self.t) else self.t, exog_r=self.exog_r,
                             exog_names=self.exog_names)


def build_return_windows(close: pd.Series, obs: int = 60, horizon: int = 20,
                         expected_delta: pd.Timedelta | None = None,
                         exog: pd.DataFrame | None = None) -> ReturnWindows:
    """All windows whose `obs` input returns and `horizon` future returns lie inside one gap-free
    run. r_k = log(close_{k+1} / close_k) is only defined when rows k and k+1 are in the same
    contiguous segment (finance_client.fprocess validation.contiguous_segment_ids); a window is
    kept only if every one of its obs + horizon returns is defined. `close` needs a DatetimeIndex.

    `exog`: close prices of other instruments on exactly `close`'s index (no NaN -- align them
    first, e.g. align_closes). Their returns on the input rows (<= t) become `exog_r`; nothing after
    the anchor is read from them."""
    close = close.astype(np.float64)
    names: tuple[str, ...] = ()
    exog_r = None
    if exog is not None:
        if not exog.index.equals(close.index) or exog.isna().any().any():
            raise ValueError("exog must be on exactly close's index with no missing values")
        names = tuple(str(c) for c in exog.columns)
        exog_r = np.diff(np.log(exog.to_numpy(np.float64)), axis=0)
    seg = contiguous_segment_ids(close.index, expected_delta)
    r = np.diff(np.log(close.to_numpy()))
    if len(r) < obs + horizon:
        empty = pd.DatetimeIndex([])
        return ReturnWindows(X=np.empty((0, obs)), F=np.empty((0, horizon)), anchor_ts=empty,
                             day=np.empty(0, dtype=np.int64), exog_r=exog_r, exog_names=names)
    bad = np.concatenate([[0], np.cumsum(seg[1:] != seg[:-1])])   # prefix count of gap-spanning returns
    t = np.arange(obs - 1, len(r) - horizon)
    t = t[(bad[t + horizon + 1] - bad[t - obs + 1]) == 0]
    win = np.lib.stride_tricks.sliding_window_view(r, obs)
    fut = np.lib.stride_tricks.sliding_window_view(r, horizon)
    anchor = close.index[t + 1]
    return ReturnWindows(X=win[t - obs + 1], F=fut[t + 1], anchor_ts=anchor,
                         day=np.asarray(anchor.normalize().asi8), t=t, exog_r=exog_r, exog_names=names)


def align_closes(close: pd.Series, others: dict[str, pd.Series], ffill_limit: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Put other instruments' closes on `close`'s timestamps. A missing bar is filled from that
    instrument's previous close for at most `ffill_limit` rows (a bar with no quote in that
    minute -- the last price is what a trader saw); rows still missing any instrument are dropped,
    which the window builder then treats as gaps. Only past values are carried forward."""
    df = pd.DataFrame({k: v.astype(np.float64) for k, v in others.items()}).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    df = df.reindex(df.index.union(close.index))
    if ffill_limit:
        df = df.ffill(limit=ffill_limit)
    df = df.reindex(close.index)
    keep = df.notna().all(axis=1).to_numpy()
    return close[keep], df[keep]
