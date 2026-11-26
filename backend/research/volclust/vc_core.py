"""Pure functions: volclust estimators, window construction (the rule of stage B of the previous study), time-of-day profiles with leave-week-out
estimation, the two uncertainty layers, and the three-way classification. No file I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import vc_env as E


# ---------------------------------------------------------------- estimators ---------------------------------------------------------------
def acf_rows(X, lag):
    """sample ACF at one lag of every row (statsmodels acf, adjusted=False): sum (x_t-m)(x_{t+lag}-m) / sum (x_t-m)^2."""
    x = X - X.mean(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", x[:, :-lag], x[:, lag:]) / np.einsum("ij,ij->i", x, x)


def volclust(R, lag=E.LAG):
    return acf_rows(np.abs(R), lag)


def estimator(R, name):
    if name.startswith("lag"):
        return volclust(R, int(name[3:]))
    if name == "mean41_60":
        return np.mean([volclust(R, k) for k in range(41, 61)], axis=0)
    if name == "rank":
        return acf_rows(rankdata(np.abs(R), axis=1), E.LAG)
    if name == "squared":
        return acf_rows(R ** 2, E.LAG)
    raise KeyError(name)


ESTIMATORS = ("lag1", "lag5", "lag10", "lag20", "lag50", "lag100", "mean41_60", "rank", "squared")


# ---------------------------------------------------------------- time basis ----------------------------------------------------------------
def convert_hist(df):
    """HistData (naive New York local time) -> the basis of the main series (New York + 7 h). A calendar week (Sunday..Saturday) whose first Sunday row is stamped
    16:00-16:29 is shifted by 8 h as a whole, every other week by 7 h (plan r1 2.2; time stamps only). Returns (frame, info)."""
    idx = df.index
    week = idx.normalize() - pd.to_timedelta((idx.dayofweek.values + 1) % 7, unit="D")
    sunday = idx.dayofweek == 6
    first = pd.Series(idx[sunday], index=week[sunday]).groupby(level=0).min()
    early = first.index[(first.dt.hour.values == 16) & (first.dt.minute.values < E.OPEN_MINUTES)]
    off = np.where(week.isin(early), 8, 7)
    out = df.copy()
    out.index = idx + pd.to_timedelta(off, unit="h")
    out = out.sort_index(kind="stable")
    dup = out.index.duplicated(keep="first")
    return out[~dup], {"weeks_shifted_8h": [str(d.date()) for d in early], "duplicates_dropped_after_conversion": int(dup.sum())}


# ---------------------------------------------------------------- series and windows ------------------------------------------------------
def prepare(df):
    """df: DatetimeIndex (naive, basis of the main series = New York + 7 h) with column close -> arrays (seconds, close, trading week id)."""
    ts = df.index.values.astype("datetime64[s]").astype("int64")
    iso = (pd.DatetimeIndex(df.index) + pd.Timedelta(hours=E.WEEK_SHIFT_HOURS)).isocalendar()
    wk = iso["year"].values.astype(np.int64) * 100 + iso["week"].values.astype(np.int64)
    return {"ts": ts, "close": df["close"].values.astype(float), "week": wk}


def _bounds(period):
    a = np.datetime64(period[0], "s").astype("int64")
    b = (np.datetime64(period[1], "s") + np.timedelta64(1, "D")).astype("int64")
    return a, b


def build_windows(S, L, period):
    """non-overlapping windows of L returns: rows i..i+L within one trading week with no step > 300 s; stride L from each segment start."""
    ts, close, wk = S["ts"], S["close"], S["week"]
    brk = np.flatnonzero((np.diff(ts) > E.GAP_SECONDS) | (np.diff(wk) != 0)) + 1
    a = np.concatenate([[0], brk])
    b = np.concatenate([brk, [len(close)]])
    p0, p1 = _bounds(period)
    rows = []
    for s, e in zip(a, b):
        i = s
        while i + L < e:
            if ts[i] >= p0 and ts[i + L] < p1:
                rows.append(i)
            i += L
    rows = np.array(rows, dtype=np.int64)
    return windows_at(S, rows, L)


def windows_at(S, rows, L):
    idx = rows[:, None] + np.arange(L + 1)[None, :] if len(rows) else np.zeros((0, L + 1), np.int64)
    C = S["close"][idx]
    R = np.diff(np.log(C), axis=1)
    zero = (R == 0.0).mean(axis=1)
    excluded = (zero > E.ZERO_FRAC_MAX) | ~np.isfinite(R).all(axis=1)
    return {"row": rows, "R": R, "ret_ts": S["ts"][idx[:, 1:]], "week": S["week"][rows], "zero_frac": zero, "excluded": excluded, "start_ts": S["ts"][rows]}


def keep(W):
    m = ~W["excluded"]
    return {k: v[m] for k, v in W.items()}


def pool_returns(S, period):
    """all 1-step returns inside one trading week with step <= 300 s whose two rows lie in the period (profile estimation sample)."""
    ts, close, wk = S["ts"], S["close"], S["week"]
    p0, p1 = _bounds(period)
    ok = (np.diff(ts) <= E.GAP_SECONDS) & (np.diff(wk) == 0) & (ts[:-1] >= p0) & (ts[1:] < p1)
    r = np.diff(np.log(close))
    ok &= np.isfinite(r)
    return {"r": r[ok], "ts": ts[1:][ok], "week": wk[1:][ok]}


# ---------------------------------------------------------------- time-of-day profiles --------------------------------------------------
def bins_of(ts_main, scheme):
    """ts_main: seconds of naive time stamps on the basis of the main series (New York + 7 h). 'ny15' / 'ny_wd30' use New York local time; 'est15' (sensitivity)
    converts New York -> UTC -> a fixed UTC-5 clock, which ignores daylight saving time."""
    ts_main = np.asarray(ts_main, np.int64)
    ny = ts_main.ravel() - 7 * 3600
    if scheme == "ny15":
        return ((ny % 86400) // 900).reshape(ts_main.shape), 96
    if scheme == "ny_wd30":
        dow = (ny // 86400 + 3) % 7                      # 1970-01-01 was a Thursday; Monday = 0
        return (dow * 48 + (ny % 86400) // 1800).reshape(ts_main.shape), 336
    if scheme == "est15":
        local = pd.to_datetime(ny, unit="s")
        utc = local.tz_localize("America/New_York", ambiguous=np.ones(len(local), bool), nonexistent="shift_forward").tz_convert("UTC")
        sec = utc.tz_localize(None).values.astype("datetime64[s]").astype("int64") - 5 * 3600
        return ((sec % 86400) // 900).reshape(ts_main.shape), 96
    raise KeyError(scheme)


class Season:
    """per-week sums of |r| by time-of-day bin; profile for week w excludes every copy of week w from the (resampled) weeks."""

    def __init__(self, pool, win, scheme="ny15"):
        self.units = np.unique(np.concatenate([pool["week"], win["week"]]))
        pb, nb = bins_of(pool["ts"], scheme)
        self.wb, _ = bins_of(win["ret_ts"], scheme)
        self.nb = nb
        W = len(self.units)
        key = np.searchsorted(self.units, pool["week"]) * nb + pb
        self.S = np.bincount(key, weights=np.abs(pool["r"]), minlength=W * nb).reshape(W, nb)
        self.N = np.bincount(key, minlength=W * nb).reshape(W, nb).astype(float)
        self.win_unit = np.searchsorted(self.units, win["week"])

    def profiles(self, m):
        """[units, bins]; bins with fewer than MIN_BIN_OBS observations are merged into the preceding bin (leading ones into the following)."""
        W, nb = self.S.shape
        s = (m @ self.S)[None, :] - m[:, None] * self.S
        n = (m @ self.N)[None, :] - m[:, None] * self.N
        gid = np.cumsum(n >= E.MIN_BIN_OBS - 0.5, axis=1) - 1
        np.maximum(gid, 0, out=gid)
        key = (np.arange(W)[:, None] * nb + gid).ravel()
        gs = np.bincount(key, weights=s.ravel(), minlength=W * nb)
        gn = np.bincount(key, weights=n.ravel(), minlength=W * nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (gs[key] / gn[key]).reshape(W, nb)

    def adjusted(self, R, m, sel=None):
        """returns divided by the profile of their week and bin (m = multiplicity of every unit); sel = window subset."""
        P = self.profiles(m)
        u = self.win_unit if sel is None else self.win_unit[sel]
        wb = self.wb if sel is None else self.wb[sel]
        return (R if sel is None else R[sel]) / P[u[:, None], wb]


# ---------------------------------------------------------------- uncertainty layers ------------------------------------------------------
def multiplicities(n_units, rng):
    return np.bincount(rng.integers(0, n_units, n_units), minlength=n_units).astype(float)


def real_layer(values, unit, n_units, B, rng, season=None, R=None, median=False):
    """week-block bootstrap of the window mean (and optionally median). With season/R the time-of-day profile is re-estimated from the resampled weeks in every
    replicate and the adjusted volclust recomputed; the raw values of the same replicate are returned too (paired)."""
    out = {"raw": np.empty(B), "adj": np.empty(B) if season is not None else None, "raw_median": np.empty(B) if median else None, "adj_median": np.empty(B) if (median and season is not None) else None}
    for b in range(B):
        m = multiplicities(n_units, rng)
        w = m[unit]
        sel = w > 0
        out["raw"][b] = np.dot(w[sel], values[sel]) / w[sel].sum()
        if median:
            out["raw_median"][b] = np.median(np.repeat(values[sel], w[sel].astype(int)))
        if season is not None:
            va = volclust(season.adjusted(R, m, sel))
            out["adj"][b] = np.dot(w[sel], va) / w[sel].sum()
            if median:
                out["adj_median"][b] = np.median(np.repeat(va, w[sel].astype(int)))
    return out


def pad_runs(runs):
    """list (per point) of 1-d arrays of run values -> ([P, 5] padded, counts)."""
    n = np.array([len(r) for r in runs])
    X = np.zeros((len(runs), n.max()))
    for i, r in enumerate(runs):
        X[i, : len(r)] = r
    return X, n


def point_means(X, n):
    return X.sum(axis=1) / n


def ddm_max_boot(X, n, B, rng, chunk=200):
    """run-level bootstrap within every point; the maximum over all points is recomputed in every replicate."""
    P, K = X.shape
    out = np.empty(B)
    valid = np.arange(K)[None, :] < n[:, None]
    rows = np.arange(P)[None, :, None]
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        idx = (rng.random((b, P, K)) * n[None, :, None]).astype(np.int64)
        v = X[rows, idx] * valid[None]
        out[s:s + b] = (v.sum(axis=2) / n[None, :]).max(axis=1)
    return out


def ddm_max_t(X, n, B, rng):
    """reference only: parametric t4 draws of the point means."""
    mean = point_means(X, n)
    se = np.array([X[i, : n[i]].std(ddof=1) / np.sqrt(n[i]) for i in range(len(n))])
    return (mean[None, :] + se[None, :] * rng.standard_t(4, (B, len(n)))).max(axis=1)


def interval(x):
    return [float(np.percentile(x, 5)), float(np.percentile(x, 95))]


def classify(intervals, delta, small_enabled):
    """intervals: the four 90% intervals of Delta (real/DDM layer x SA/SB)."""
    if all(lo > delta for lo, _ in intervals):
        return "差あり"
    if small_enabled and all(hi < delta for _, hi in intervals):
        return "差は小さい"
    return "判定保留"


def compare(mu_real, real_boot, ddm, B, rng, small_enabled, t_ref=False):
    """ddm: {'SA': (X, n), 'SB': (X, n)}. Delta = mu_real - max of the point means; classification at DELTA and the sensitivity thresholds."""
    out = {"mu_real": float(mu_real), "by_set": {}}
    ivs = []
    for s in ("SA", "SB"):
        X, n = ddm[s]
        pm = point_means(X, n)
        mx = float(pm.max())
        a = interval(real_boot - mx)
        mb = ddm_max_boot(X, n, B, rng)
        d = interval(mu_real - mb)
        ivs += [a, d]
        out["by_set"][s] = {"mu_max": mx, "argmax_index": int(pm.argmax()), "n_points": int(len(n)), "Delta": float(mu_real - mx), "real_layer_90": a, "ddm_layer_90": d,
                            "ddm_max_boot_median": float(np.median(mb))}
        if t_ref:
            out["by_set"][s]["ddm_layer_t_reference_90"] = interval(mu_real - ddm_max_t(X, n, B, rng))
    out["classification"] = {str(dl): classify(ivs, dl, small_enabled) for dl in (E.DELTA,) + E.DELTAS_SENS}
    return out


def combine(base, extended):
    """primary rule of V1: if adding the partially failed points changes the category, the category is 'held'."""
    return {k: (base["classification"][k] if base["classification"][k] == extended["classification"][k] else "判定保留") for k in base["classification"]}
