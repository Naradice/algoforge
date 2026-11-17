"""F1 (Study Manager #889, plan v0.3): common data, window, scoring and leak-test utilities.

Unit of analysis: the 60-candle block (= 1 "window", as in all earlier DDM evaluations).
Block j = rows [60j, 60j+59] of the derived USDJPY M1 series; its origin row is t_j = 60j + 59
(the last candle of the block). Information available at the origin: closes up to and including row t_j.
Target: R_h(j) = log(C[t_j + 60h] / C[t_j]), h in {1, 4, 12, 24} windows.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 60
HORIZONS = (1, 4, 12, 24)
MAX_H = max(HORIZONS)
ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"

PERIODS = {   # origin and target end must both lie inside the period (inclusive dates)
    "estimation": ("2016-08-31", "2020-08-31"),
    "validation": ("2020-09-01", "2021-08-31"),
    "final": ("2021-09-01", "2022-08-31"),
}
GAP_TOL_SECONDS = 300      # a step longer than 5 minutes is a market gap (weekend, holiday, outage); missing quiet minutes are not
N_QUANTILES = 20
M_SAMPLES = 1000


def artifact_path() -> Path:
    store = Path(os.getenv("ARTIFACT_STORE_PATH", str(Path(__file__).resolve().parents[2] / "artifacts"))).resolve()
    return store / ARTIFACT


class Series:
    """Close series with block bookkeeping. All derived arrays are built from `close` and `index` only."""

    def __init__(self, close: np.ndarray, index: pd.DatetimeIndex):
        self.close = np.asarray(close, float)
        self.index = index
        n = len(self.close)
        self.nblocks = n // BLOCK
        step_ok = np.diff(index.values.astype("datetime64[s]").astype("int64")) <= GAP_TOL_SECONDS
        self.bad_cum = np.concatenate([[0], np.cumsum(~step_ok)])    # bad_cum[k] = # bad steps among rows < k (step k-1 -> k)
        self.t = np.arange(self.nblocks) * BLOCK + BLOCK - 1         # origin row of every block
        self.t_time = index[self.t]
        self.logc = np.log(self.close)

    def contiguous(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where rows a..b (inclusive, vectors) form an unbroken 1-minute sequence."""
        ok = (a >= 0) & (b < len(self.close))
        out = np.zeros(len(a), bool)
        aa, bb = a[ok], b[ok]
        out[ok] = (self.bad_cum[bb] - self.bad_cum[aa]) == 0
        return out

    # --- per-origin information available at t (only rows <= t) ---
    def atr_window(self) -> np.ndarray:
        """Close-only ATR over the 60 closes of the observation block: mean |diff| of 59 steps (no row after t)."""
        d = np.abs(np.diff(self.close))
        out = np.full(self.nblocks, np.nan)
        for j in range(self.nblocks):
            a = j * BLOCK
            out[j] = d[a:a + BLOCK - 1].mean()
        return out

    def block_returns(self) -> np.ndarray:
        """r_j = log(C[t_j]/C[t_{j-1}]) when rows t_{j-1}..t_j are contiguous, else NaN. Uses rows <= t_j only."""
        r = np.full(self.nblocks, np.nan)
        j = np.arange(1, self.nblocks)
        ok = self.contiguous(self.t[j - 1], self.t[j])
        r[j[ok]] = self.logc[self.t[j[ok]]] - self.logc[self.t[j[ok] - 1]]
        return r

    def block_rv(self) -> np.ndarray:
        """Realised variance of block j from the 60 M1 returns ending at t_j (needs rows t_{j-1}..t_j contiguous)."""
        r1 = np.diff(self.logc)
        sq = r1 ** 2
        cs = np.concatenate([[0.0], np.cumsum(sq)])
        rv = np.full(self.nblocks, np.nan)
        j = np.arange(1, self.nblocks)
        ok = self.contiguous(self.t[j - 1], self.t[j])
        # M1 return k covers rows k -> k+1; block j uses returns t_{j-1} .. t_j - 1
        a = self.t[j - 1]
        b = self.t[j]
        rv[j[ok]] = (cs[b[ok]] - cs[a[ok]])
        return rv

    def target(self, h: int) -> np.ndarray:
        """R_h at each origin, NaN where rows t_j..t_j+60h are not contiguous or run past the data."""
        a = self.t
        b = self.t + BLOCK * h
        ok = self.contiguous(a, b)
        out = np.full(self.nblocks, np.nan)
        out[ok] = self.logc[b[ok]] - self.logc[a[ok]]
        return out

    def atr_valid(self) -> np.ndarray:
        a = self.t - (BLOCK - 1)
        return self.contiguous(a, self.t)

    def period_mask(self, name: str, h: int | None = None) -> np.ndarray:
        """Origins whose origin time and (if h given) target-end time lie inside the period."""
        lo, hi = (pd.Timestamp(PERIODS[name][0]), pd.Timestamp(PERIODS[name][1]) + pd.Timedelta(days=1))
        m = (self.t_time >= lo) & (self.t_time < hi)
        if h is not None:
            end_row = np.minimum(self.t + BLOCK * h, len(self.close) - 1)
            end_t = self.index[end_row]
            m &= (end_t < hi)
        return np.asarray(m)


def load_series() -> Series:
    df = pd.read_parquet(artifact_path(), columns=["close"])
    return Series(df["close"].values, df.index)


# ----------------------------------------------------------------------------- scoring
def fair_crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair (finite-sample unbiased) sample CRPS for each row: mean|x-y| - sum_{i<j}|xi-xj| / (M(M-1)).
    samples: [n, M], y: [n]."""
    x = np.sort(samples, axis=1)
    m = x.shape[1]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    k = np.arange(1, m + 1)
    w = (2 * k - m - 1).astype(float)
    pair = (x * w[None, :]).sum(axis=1) / (m * (m - 1))      # = (1/(2M(M-1))) * sum_{i,j}|xi-xj|
    return term1 - pair


def plain_crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard (biased) sample CRPS: mean|x-y| - sum_{i,j}|xi-xj|/(2 M^2)."""
    x = np.sort(samples, axis=1)
    m = x.shape[1]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    k = np.arange(1, m + 1)
    w = (2 * k - m - 1).astype(float)
    pair = (x * w[None, :]).sum(axis=1) / (m * m)
    return term1 - pair


COVER_LEVELS = (0.5, 0.8, 0.9, 0.95)
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)


def score_samples(samples: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-origin scores for a sample-based predictive distribution. All arrays have length n."""
    n, m = samples.shape
    x = np.sort(samples, axis=1)
    out = {"crps": fair_crps(x, y), "crps_plain": plain_crps(x, y)}
    lt = (x < y[:, None]).sum(axis=1)
    eq = (x == y[:, None]).sum(axis=1)
    u = rng.random(n)
    out["pit"] = (lt + u * eq) / m                                   # randomised PIT (ties)
    for c in COVER_LEVELS:
        lo = np.quantile(x, (1 - c) / 2, axis=1)
        hi = np.quantile(x, 1 - (1 - c) / 2, axis=1)
        out[f"hit{int(c * 100)}"] = ((y >= lo) & (y <= hi)).astype(float)
    for q in TAIL_LEVELS:
        qq = np.quantile(x, q, axis=1)
        d = y - qq
        out[f"ql{q}"] = np.where(d >= 0, q * d, (q - 1) * d)           # pinball loss
    return out


def block_bootstrap_weeks(week_id: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Index sets from resampling whole calendar-week blocks (shared by all horizons)."""
    weeks = np.unique(week_id)
    by_week = {w: np.flatnonzero(week_id == w) for w in weeks}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(weeks, size=len(weeks))
        out.append(np.concatenate([by_week[w] for w in pick]))
    return out
