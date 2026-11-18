"""F1 (#889) P1 independent verification library (plan v0.3, docs/research-f1-p1-verification-plan.md).

Everything here is written from the specification (docs/research-f1-statistical-baselines.md), NOT from the production code:
this module imports nothing from f1_common / f1_analyze / f1_run / f1_models. Plain loops and brute force are used where cheap.
Constants are typed from the specification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

# ---- constants from the specification --------------------------------------------------------------------------------
BLOCK = 60
HORIZONS = (1, 4, 12, 24)
GAP_SECONDS = 300                       # a step longer than 5 minutes is a market gap
M = 1000
PERIODS = {"estimation": ("2016-08-31", "2020-08-31"), "validation": ("2020-09-01", "2021-08-31"),
           "final": ("2021-09-01", "2022-08-31")}
SELECT_POOL = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T"]
TIE_ORDER = ["B1", "B2", "B3", "B4", "B7N", "B5", "B6", "B7T", "B0"]       # spec: B1<B2<B3<B4<B7-N<B5<B6<B7-t<B0
TIE_TOL = 0.005
MARGIN = 0.02
COVER = (50, 80, 90, 95)
TAILS = (0.01, 0.05, 0.95, 0.99)
STAT_MODELS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T", "B8", "B4s"]
DDM_MODELS = ["DDM1", "DDM2", "DDM3"]
ALL_MODELS = STAT_MODELS + DDM_MODELS


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---- raw data and origin sets (V1) ------------------------------------------------------------------------------------
class Raw:
    """Raw M1 closes with an independent bookkeeping of gaps (searchsorted over the gap positions)."""

    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path, columns=["close"])
        self.index = df.index
        self.close = np.asarray(df["close"].values, dtype=float)
        self.n = len(self.close)
        ns = self.index.values.astype("datetime64[ns]").astype("int64")
        step = np.diff(ns)
        self.gaps = np.flatnonzero(step > GAP_SECONDS * 10 ** 9)         # gap g: between row g and g+1
        self.nb = self.n // BLOCK
        self.t = np.arange(self.nb) * BLOCK + (BLOCK - 1)                # origin row of each block
        self.t_time = self.index[self.t]

    def contiguous(self, a, b):
        """rows a..b (inclusive) contain no gap step; vectorised, False when out of range."""
        a = np.asarray(a)
        b = np.asarray(b)
        ok = (a >= 0) & (b < self.n)
        out = np.zeros(a.shape, bool)
        aa, bb = a[ok], b[ok]
        out[ok] = (np.searchsorted(self.gaps, bb, side="left") - np.searchsorted(self.gaps, aa, side="left")) == 0
        return out

    def atr(self):
        """mean absolute close-to-close change over the 59 steps inside each block (close-only)."""
        c = self.close[: self.nb * BLOCK].reshape(self.nb, BLOCK)
        return np.abs(c[:, 1:] - c[:, :-1]).mean(axis=1)

    def origin_ok(self):
        """ATR window contiguous, previous-block return defined, realised variance of the block positive."""
        t = self.t
        atr_ok = self.contiguous(t - (BLOCK - 1), t)
        j = np.arange(self.nb)
        prev_ok = np.zeros(self.nb, bool)
        prev_ok[1:] = self.contiguous(t[:-1], t[1:])
        lr = np.diff(np.log(self.close))                                 # M1 log returns, lr[k] = row k -> k+1
        rv_pos = np.zeros(self.nb, bool)
        for jj in range(1, self.nb):                                     # returns t_{j-1} .. t_j - 1 (60 returns)
            a = t[jj - 1]
            seg = lr[a:a + BLOCK]
            rv_pos[jj] = bool(np.sum(seg * seg) > 0.0)
        return atr_ok & prev_ok & rv_pos

    def target(self, h):
        """R_h at each origin; NaN when the rows are not contiguous or run past the data."""
        b = self.t + BLOCK * h
        ok = self.contiguous(self.t, b)
        out = np.full(self.nb, np.nan)
        out[ok] = np.log(self.close[b[ok]] / self.close[self.t[ok]])
        return out

    def period_ok(self, name, h):
        lo = pd.Timestamp(PERIODS[name][0])
        hi = pd.Timestamp(PERIODS[name][1]) + pd.Timedelta(days=1)
        m = np.asarray((self.t_time >= lo) & (self.t_time < hi))
        end_row = np.minimum(self.t + BLOCK * h, self.n - 1)
        m &= np.asarray(self.index[end_row] < hi)
        return m

    def origin_sets(self):
        """{(period, h): sorted array of block numbers} and the independent targets."""
        ok = self.origin_ok()
        tg = {h: self.target(h) for h in HORIZONS}
        sets = {}
        for per in ("validation", "final"):
            for h in HORIZONS:
                sets[(per, h)] = np.flatnonzero(ok & self.period_ok(per, h) & np.isfinite(tg[h]))
        return sets, tg, ok


def edges_from_estimation(raw: Raw, n_buckets=20):
    """19 interior quantiles (linear interpolation) of the block ATR over estimation-period origins whose ATR window is valid."""
    atr = raw.atr()
    t = raw.t
    lo = pd.Timestamp(PERIODS["estimation"][0])
    hi = pd.Timestamp(PERIODS["estimation"][1]) + pd.Timedelta(days=1)
    m = np.asarray((raw.t_time >= lo) & (raw.t_time < hi)) & raw.contiguous(t - (BLOCK - 1), t)
    x = np.sort(atr[m])
    return np.array([linear_quantile(x, k / n_buckets) for k in range(1, n_buckets)])


# ---- scoring definitions -----------------------------------------------------------------------------------------------
def linear_quantile(xs, q):
    """Quantile of an ascending sample by linear interpolation between order statistics at position q*(n-1)."""
    n = len(xs)
    pos = q * (n - 1)
    k = int(math.floor(pos))
    f = pos - k
    if k >= n - 1:
        return float(xs[n - 1])
    return float(xs[k] + f * (xs[k + 1] - xs[k]))


def pair_term(xs):
    """(1 / (2 M (M-1))) * sum_{i,j} |x_i - x_j| by brute force over blocks of rows (exact definition, O(M^2))."""
    x = np.asarray(xs, float)
    m = len(x)
    tot = 0.0
    step = 100
    for a in range(0, m, step):
        tot += float(np.abs(x[a:a + step, None] - x[None, :]).sum())
    return tot / (2.0 * m * (m - 1))


def fair_crps_brute(xs, y):
    x = np.asarray(xs, float)
    return float(np.abs(x - y).mean()) - pair_term(x)


def point_scores(xs_sorted, y, pt=None):
    """hit{50,80,90,95}, pinball{q}, pit_lo, pit_w for one sorted sample and realisation (independent definitions)."""
    x = xs_sorted
    out = {}
    for c in COVER:
        a = (1 - c / 100.0) / 2
        lo, hi = linear_quantile(x, a), linear_quantile(x, 1 - a)
        out[f"hit{c}"] = 1.0 if (y >= lo and y <= hi) else 0.0
    for q in TAILS:
        qq = linear_quantile(x, q)
        d = y - qq
        out[f"ql{q}"] = q * d if d >= 0 else (q - 1) * d
    left = int(np.searchsorted(x, y, side="left"))
    right = int(np.searchsorted(x, y, side="right"))
    out["pit_lo"] = left / len(x)
    out["pit_w"] = (right - left) / len(x)
    return out


# ---- comparison helper -------------------------------------------------------------------------------------------------
def close_enough(a, b, rtol, atol):
    return abs(a - b) <= rtol * abs(b) + atol


def max_excess(a, b, rtol, atol):
    """max over elements of |a-b| / (rtol*|b| + atol) (<= 1 means within tolerance) and the worst absolute difference."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = np.abs(a - b)
    tol = rtol * np.abs(b) + atol
    return float(np.max(d / tol)) if d.size else 0.0, float(d.max()) if d.size else 0.0


# ---- inference (V4, V6, V7) --------------------------------------------------------------------------------------------
def nw_dm(d, lag):
    """Diebold-Mariano t statistic with a Bartlett-kernel (Newey-West) long-run variance of the loss differential d."""
    d = np.asarray(d, float)
    n = len(d)
    dm = d - d.mean()
    g = [float(np.dot(dm[k:], dm[:n - k]) / n) for k in range(lag + 1)]
    var = g[0] + 2.0 * sum((1.0 - k / (lag + 1.0)) * g[k] for k in range(1, lag + 1))
    t = float(d.mean() / math.sqrt(var / n))
    return t, var


def log10_p_two_sided(t):
    return float(stats.norm.logsf(abs(t)) / math.log(10.0) + math.log10(2.0))


def week_ids(times):
    iso = times.isocalendar()
    return (iso.year.values.astype(int) * 100 + iso.week.values.astype(int))


def week_sums(values_by_h, weeks_by_h, all_weeks):
    """per-horizon per-week sums aligned to the common week universe."""
    out = {}
    for h in HORIZONS:
        s = {}
        for w, v in zip(weeks_by_h[h], values_by_h[h]):
            s[w] = s.get(w, 0.0) + float(v)
        out[h] = np.array([s.get(w, 0.0) for w in all_weeks])
    return out


def bootstrap(sd, ss, point, n_boot, rng_factory):
    """Week-block bootstrap of Delta_h; rng_factory() -> numpy Generator. Returns se_h, critical value c."""
    W = len(next(iter(sd.values())))
    rng = rng_factory()
    draws = np.empty((n_boot, len(HORIZONS)))
    for b in range(n_boot):
        pick = rng.integers(0, W, W)
        for i, h in enumerate(HORIZONS):
            draws[b, i] = sd[h][pick].sum() / ss[h][pick].sum() - 1.0
    se = draws.std(axis=0)
    pt = np.array([point[h] for h in HORIZONS])
    crit = float(np.quantile(np.max(np.abs(draws - pt) / se, axis=1), 0.95))
    return se, crit


def classify(lo, hi):
    if hi < -MARGIN:
        return "DDM better"
    if lo > MARGIN:
        return "statistical model better"
    if lo >= -MARGIN and hi <= MARGIN:
        return "equivalent"
    return "inconclusive"


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def file_hash_manifest(path):
    return sha256_file(path)


def listdir_hashes(d):
    return {f: sha256_file(os.path.join(d, f)) for f in sorted(os.listdir(d))}


# ---- execution record ---------------------------------------------------------------------------------------------------
def code_state(require_clean=True):
    """HEAD of the repository and whether backend/research/p1_verification has uncommitted changes. Aborts when dirty."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = subprocess.check_output(["git", "-C", here, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", "backend/research/p1_verification"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p1_verification has uncommitted changes -> commit first so that the executed commit is recorded: " + dirty)
    return {"head": head, "dirty_p1_files": dirty}
