"""F1 (#889 v0.3): predictive-distribution samplers for all models. See docs/research-f1-statistical-baselines.md.

Every sampler is a pure function of (learned state, context built from the series, origin block j, seed) and may only read
rows <= t_j of the context. The leak test rebuilds the context from a perturbed series and requires identical output.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats

import f1_common as fc

warnings.filterwarnings("ignore")

M = fc.M_SAMPLES
SCALE = 1000.0                      # block returns are fitted in units of 1e-3 (std ~ 1)
HIDX = {h: i for i, h in enumerate(fc.HORIZONS)}
MODEL_IDS = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6, "B7N": 7, "B7T": 8, "B8": 9, "B4s": 10}
ARCH_SPECS = {
    "B3": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "B4": dict(vol="GARCH", p=1, q=1, dist="t"),
    "B5": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "B6": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "B4s": dict(vol="GARCH", p=1, q=1, dist="t"),
}
NU_MIN = 2.1


class Context:
    """All series-derived arrays; built only from (close, index)."""

    def __init__(self, close, index, edges):
        self.s = fc.Series(close, index)
        s = self.s
        self.atr = s.atr_window()
        self.r = s.block_returns()
        self.rv = s.block_rv()
        self.targets = {h: s.target(h) for h in fc.HORIZONS}
        self.atr_ok = s.atr_valid()
        self.tt = s.t_time
        self.edges = np.asarray(edges)
        self.valid_r = np.flatnonzero(np.isfinite(self.r))
        self.y_all = self.r[self.valid_r] * SCALE
        self.hour = np.asarray(self.tt.hour)
        self.bucket = np.digitize(self.atr, self.edges)
        self.lrv = np.log(self.rv)
        self.origin_ok = self.atr_ok & np.isfinite(self.r) & np.isfinite(self.rv)

    def end_time(self, h):
        end_row = np.minimum(self.s.t + fc.BLOCK * h, len(self.s.close) - 1)
        return self.s.index[end_row]


# ----------------------------------------------------------------------------- distributions
def std_t(rng, nu, size):
    nu = max(nu, NU_MIN)
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


def fit_std_t_nu(x):
    """MLE of nu for a standardised (unit-variance) Student-t on already-standardised data."""
    x = np.asarray(x, float)

    def nll(nu):
        c = np.sqrt(nu / (nu - 2.0))
        return -np.sum(stats.t.logpdf(x * c, nu) + np.log(c))
    res = optimize.minimize_scalar(nll, bounds=(NU_MIN, 100.0), method="bounded")
    return float(res.x)


# ----------------------------------------------------------------------------- learned state (data strictly before month start)
class State:
    pass


def month_starts(period):
    lo, hi = pd.Timestamp(fc.PERIODS[period][0]), pd.Timestamp(fc.PERIODS[period][1])
    return list(pd.date_range(lo.replace(day=1), hi, freq="MS"))


def build_state(ctx: Context, month_start: pd.Timestamp, prev: "State | None") -> "State":
    st = State()
    st.month_start = month_start
    st.edges = ctx.edges
    ok_atr = ctx.atr_ok
    st.R, st.Rb = {}, {}
    st.v, st.nu = {}, {}
    for h in fc.HORIZONS:
        done = ok_atr & np.isfinite(ctx.targets[h]) & np.asarray(ctx.end_time(h) < month_start)
        R = ctx.targets[h][done]
        st.R[h] = R
        st.Rb[h] = ctx.bucket[done]
        st.v[h] = float(np.mean(R ** 2))
        st.nu[h] = fit_std_t_nu(R / np.sqrt(st.v[h]))
    # GARCH family on block returns before the month start
    past = ctx.valid_r[np.asarray(ctx.tt[ctx.valid_r] < month_start)]
    y = ctx.r[past] * SCALE
    st.arch = {}
    st.arch_fail = []
    # hour-of-day factors (sensitivity model)
    hrs = ctx.hour[past]
    fac = np.ones(24)
    for hh in range(24):
        sel = hrs == hh
        if sel.sum() >= 50:
            fac[hh] = y[sel].std()
    fac = fac / np.sqrt(np.mean(fac ** 2))
    st.hour_factor = fac
    for name, spec in ARCH_SPECS.items():
        yy = y / fac[hrs] if name == "B4s" else y
        try:
            res = arch_model(yy, mean="Zero", rescale=False, **spec).fit(disp="off", show_warning=False)
            if res.convergence_flag != 0 or not np.all(np.isfinite(res.params.values)):
                raise RuntimeError("no convergence")
            st.arch[name] = res.params.copy()
        except Exception:
            st.arch_fail.append(name)
            st.arch[name] = prev.arch[name] if prev is not None else None
    # HAR-RV
    feats, tgt = har_features(ctx)
    jj = np.flatnonzero(np.isfinite(tgt) & np.all(np.isfinite(feats), axis=1) & np.asarray(ctx.tt < month_start))
    # target block j+1 must also lie before the month start
    jj = jj[np.asarray(ctx.tt[np.minimum(jj + 1, len(ctx.tt) - 1)] < month_start)]
    X = np.column_stack([np.ones(len(jj)), feats[jj]])
    beta, *_ = np.linalg.lstsq(X, tgt[jj], rcond=None)
    resid = tgt[jj] - X @ beta
    st.har_beta, st.har_resid = beta, resid - resid.mean()
    pj = past[np.isfinite(ctx.rv[past]) & (ctx.rv[past] > 0)]
    z = ctx.r[pj] / np.sqrt(ctx.rv[pj])
    st.z_std = float(z.std())
    zeta = (z - z.mean()) / z.std()
    st.har_nu = fit_std_t_nu(zeta)
    return st


def har_features(ctx: Context):
    """Features at block j (rows <= t_j) and the next-block target log RV_{j+1}."""
    rv = ctx.rv
    n = len(rv)
    f = np.full((n, 3), np.nan)
    s24 = pd.Series(rv).rolling(24, min_periods=1).mean().values
    s120 = pd.Series(rv).rolling(120, min_periods=1).mean().values
    f[:, 0] = np.log(rv)
    f[:, 1] = np.log(s24)
    f[:, 2] = np.log(s120)
    tgt = np.full(n, np.nan)
    tgt[:-1] = np.log(rv[1:])
    return f, tgt


# ----------------------------------------------------------------------------- samplers: return [4, M] cumulative returns
def rng_for(model: str, j: int):
    return np.random.default_rng([MODEL_IDS[model], int(j)])


def sample_B0(st, ctx, j):
    g = rng_for("B0", j)
    return np.stack([g.choice(st.R[h], size=M) for h in fc.HORIZONS])


def sample_B1(st, ctx, j):
    g = rng_for("B1", j)
    return np.stack([g.standard_normal(M) * np.sqrt(st.v[h]) for h in fc.HORIZONS])


def sample_B2(st, ctx, j):
    g = rng_for("B2", j)
    return np.stack([std_t(g, st.nu[h], M) * np.sqrt(st.v[h]) for h in fc.HORIZONS])


def sample_B8(st, ctx, j):
    g = rng_for("B8", j)
    b = ctx.bucket[j]
    out = []
    for h in fc.HORIZONS:
        pool = st.R[h][st.Rb[h] == b]
        if len(pool) < 30:
            pool = st.R[h]
        out.append(g.choice(pool, size=M))
    return np.stack(out)


def sample_arch(name):
    spec = ARCH_SPECS[name]
    is_t = spec["dist"] == "t"

    def f(st, ctx, j):
        params = st.arch[name]
        if params is None:
            return None
        k = int(np.searchsorted(ctx.valid_r, j, side="right"))
        y = ctx.y_all[:k]
        fac = None
        if name == "B4s":
            fac = st.hour_factor
            y = y / fac[ctx.hour[ctx.valid_r[:k]]]
        g = rng_for(name, j)
        nu = float(params["nu"]) if is_t else None
        rng = (lambda size: std_t(g, nu, size)) if is_t else (lambda size: g.standard_normal(size))
        res = arch_model(y, mean="Zero", rescale=False, **spec).fix(params)
        fcst = res.forecast(horizon=fc.MAX_H, start=len(y) - 1, method="simulation", simulations=M, rng=rng, reindex=False)
        sim = fcst.simulations.values[0]                     # [M, 24] per-step returns (scaled)
        if fac is not None:
            hrs = (ctx.hour[j] + np.arange(1, fc.MAX_H + 1)) % 24
            sim = sim * fac[hrs][None, :]
        cum = sim.cumsum(axis=1) / SCALE
        return np.stack([cum[:, h - 1] for h in fc.HORIZONS])
    return f


def sample_har(innov: str):
    def f(st, ctx, j):
        g = rng_for("B7N" if innov == "N" else "B7T", j)
        lo = max(0, j - 119)
        hist = ctx.rv[lo:j + 1].copy()
        if len(hist) < 120:
            hist = np.concatenate([np.full(120 - len(hist), np.nan), hist])
        paths = np.tile(hist, (M, 1))                       # [M, 120]
        b = st.har_beta
        cum = np.zeros(M)
        out = np.zeros((fc.MAX_H, M))
        for s_ in range(fc.MAX_H):
            last = paths[:, -1]
            x1 = np.log(last)
            x2 = np.log(np.nanmean(paths[:, -24:], axis=1))
            x3 = np.log(np.nanmean(paths, axis=1))
            e = g.choice(st.har_resid, size=M)               # (a) log-RV residual law (empirical)
            lrv = b[0] + b[1] * x1 + b[2] * x2 + b[3] * x3 + e
            rv_new = np.exp(lrv)
            zeta = g.standard_normal(M) if innov == "N" else std_t(g, st.har_nu, M)   # (b) return innovation law
            cum = cum + st.z_std * np.sqrt(rv_new) * zeta
            out[s_] = cum
            paths = np.concatenate([paths[:, 1:], rv_new[:, None]], axis=1)
        return np.stack([out[h - 1] for h in fc.HORIZONS])
    return f


SAMPLERS = {
    "B0": sample_B0, "B1": sample_B1, "B2": sample_B2, "B8": sample_B8,
    "B3": sample_arch("B3"), "B4": sample_arch("B4"), "B5": sample_arch("B5"), "B6": sample_arch("B6"), "B4s": sample_arch("B4s"),
    "B7N": sample_har("N"), "B7T": sample_har("T"),
}
