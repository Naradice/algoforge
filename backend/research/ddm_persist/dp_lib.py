"""DP1 (Study Manager #1151): time-window trend feedback x self-modulated clock -- shared definitions.

Plan: docs/research-ddm-volatility-persistence-plan.md v0.1 (approved 2026-11-28, with the two definitions in
its section 11). The production simulator is NOT modified; the time-window model is a subclass here.

Implementation choice not spelled out in the plan, fixed before any run: the shuffled clock permutes the
post-burn-in intervals only (as in T1), so ordered and shuffled clocks have the same multiset of intervals in
the evaluated period.

Every constant is in CFG; jobs receive the configuration explicitly so that the tests can run a miniature one.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
BACKEND = os.path.dirname(RESEARCH)
REPO = os.path.dirname(BACKEND)
for _p in (BACKEND, RESEARCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.collectors.ddm_simulator import DDMv3  # noqa: E402
from tdm_fig3_audit import model1_intervals  # noqa: E402
from tdm_fig3_t0c import simulate_ar  # noqa: E402

CFG = {
    # model (plan section 4)
    "model": {"num_agent": 300, "max_volatility": 0.02, "min_volatility": 0.01, "wma": 5, "spread": 1.0,
              "initial_price": 100.0, "dealer_sensitive_min": -3.5, "dealer_sensitive_max": -1.5},
    "eta": 15.0, "burn": 5_000, "k": 100_000, "bar": 20,
    # clocks
    "clocks": {"K-short": [16, 0.1], "K-long": [50, 0.03]},
    "target_mean": 3.0, "alpha": 1.25,
    "pool_seeds": [9001, 9002, 9003], "pool_trades": 20_000, "pool_n": 100,
    "cal_seeds": list(range(8001, 8021)), "cal_k": 400_000, "cal_burn": 500,
    "diag_chains": 4_000, "diag_seed": 6001, "diag_tol": 0.05, "half_tol": 0.25, "op_min": 0.05,
    # seeds (plan section 5)
    "main_seeds": list(range(201, 221)), "pilot_seeds": list(range(191, 196)),
    "equiv_seeds": [1, 2, 3], "equiv_trades": 20_000,
    "clock_offset": 1_000, "shuffle_offset": 50_000, "barperm_offset": 60_000, "pos_offset": 70_000,
    # judgement (plan section 6)
    "lags_p": list(range(2, 11)), "lags_sec": [1, 5, 20, 60, 240],
    "delta": 0.03, "min_pairs": 18, "n_boot": 10_000, "boot_seed": 20261128,
    "level_main": 0.975, "level_check": 0.95,
    # checks and stop rules (plan section 7-8)
    "pilot_max_fail": 1, "neg_band": 0.01, "pos_min": 0.05, "pos_tau": 200.0, "pos_sd": 0.5,
    "price_lo": 50.0, "price_hi": 200.0, "run_limit_s": 600.0, "cpu_cap_s": 3 * 3600.0,
    "har_train_n": 14,
    "workers": 10,
    "fake": False,      # tests only: replace the experimental simulations by a random walk
}
CONDITIONS = {"B0": (None, None), "S-ord": ("K-short", "ord"), "S-shuf": ("K-short", "shuf"),
              "L-ord": ("K-long", "ord"), "L-shuf": ("K-long", "shuf")}
COMPARISONS = {"K-short": ("S-ord", "S-shuf"), "K-long": ("L-ord", "L-shuf")}


# ------------------------------------------------------------------------------------------- model
class DDMWindow(DDMv3):
    """DDMv3 whose trend term is the equal-weight mean of the price changes of the trades with
    t_k - t_j < eta (the latest trade k always included); 0 while t_k < eta. Trade k happens at
    t_k = t_{k-1} + intervals[k-1]. The clock never depends on the price (one-way coupling)."""

    def __init__(self, intervals, eta, **kwargs):
        super().__init__(**kwargs)
        n = len(intervals)
        self._iv = np.asarray(intervals, float)
        self._eta = float(eta)
        self._t = 0.0
        self._k = 0
        self._start = 1
        self._times = np.zeros(n + 1)
        self._diffs = np.zeros(n + 1)
        self._last_price = self.market_price
        self._mean = 0.0

    def _wma_diff(self) -> float:
        return self._mean

    def _common_step(self):
        price, tick = super()._common_step()
        if price is not None:
            k = self._k + 1
            self._t += self._iv[k - 1]
            self._times[k] = self._t
            self._diffs[k] = price - self._last_price
            self._last_price = price
            self._k = k
            while self._t - self._times[self._start] >= self._eta:
                self._start += 1
            if self._t >= self._eta:
                w = np.ascontiguousarray(self._diffs[self._start:k + 1][::-1])   # newest first, as DDMv3
                self._mean = float(np.dot(np.ones(len(w)), w) / len(w))
            else:
                self._mean = 0.0
        return price, tick


def collect_prices(model, n_trades, cfg):
    """Run `model` for n_trades; raise on the failure conditions of the plan."""
    out = np.empty(n_trades)
    t0 = time.perf_counter()
    got = 0
    for p, _ in model.simulate_stream(n_trades=n_trades):
        if not np.isfinite(p) or p < cfg["price_lo"] or p > cfg["price_hi"]:
            raise RuntimeError(f"price out of range at trade {got + 1}: {p}")
        out[got] = p
        got += 1
        if got % 1000 == 0 and time.perf_counter() - t0 > cfg["run_limit_s"]:
            raise RuntimeError("run time limit exceeded")
    if got != n_trades:
        raise RuntimeError(f"only {got} of {n_trades} trades")
    return out


def simulate_condition(cfg, cond, seed, intervals):
    random.seed(seed)
    np.random.seed(seed)
    n = cfg["burn"] + cfg["k"]
    if cfg["fake"]:
        rng = np.random.default_rng(seed * 10 + list(CONDITIONS).index(cond))
        return cfg["model"]["initial_price"] + cfg["model"]["spread"] + np.cumsum(rng.normal(0.0, 0.002, n))
    if cond == "B0":
        model = DDMv3(**cfg["model"])
    else:
        model = DDMWindow(intervals, cfg["eta"], **cfg["model"])
    return collect_prices(model, n, cfg)


def equivalence(cfg, seed, n_trades):
    """S0 check 1: unit intervals and eta = 5 must reproduce DDMv3 with wma = [1,1,1,1,1] bit for bit."""
    kw = dict(cfg["model"], wma=[1, 1, 1, 1, 1])
    random.seed(seed)
    np.random.seed(seed)
    a = collect_prices(DDMv3(**kw), n_trades, cfg)
    random.seed(seed)
    np.random.seed(seed)
    b = collect_prices(DDMWindow(np.ones(n_trades), 5.0, **kw), n_trades, cfg)
    return {"seed": seed, "n": n_trades, "equal": bool(np.array_equal(a, b)),
            "price_changes": int(np.count_nonzero(np.diff(a)))}


# ------------------------------------------------------------------------------------------- clock
def pool_part(args):
    cfg, seed = args
    t0 = time.process_time()
    _, tp, _ = model1_intervals(cfg["pool_trades"], cfg["pool_n"], 1.0, 0.01, 0.01, "signed", seed)
    return seed, tp, time.process_time() - t0


def make_logf(parts, cfg):
    tp = np.concatenate(parts)
    tp = tp[tp > 0]
    x = tp ** cfg["alpha"]
    lf = np.log(x / x.mean())
    return lf - lf.mean()


def calibrate(lf, n, lam, cfg):
    """Same procedure as T1: kappa = E[T]/m0 over the calibration seeds, m0 = target / kappa."""
    kap = []
    for s in cfg["cal_seeds"]:
        rng = np.random.default_rng(s)
        t = simulate_ar(rng.choice(lf, size=cfg["cal_k"]), n, lam, m0=1.0)
        kap.append(t[cfg["cal_burn"]:].mean())
    kappa = float(np.mean(kap))
    return {"n": n, "lam": lam, "kappa": kappa, "m0": cfg["target_mean"] / kappa}


def transient_diag(lf, n, lam, m0, cfg):
    rng = np.random.default_rng(cfg["diag_seed"])
    burn = cfg["burn"]
    out = {}
    for disp in (+2.0, -2.0):
        vals = np.empty(cfg["diag_chains"])
        for i in range(cfg["diag_chains"]):
            t = simulate_ar(rng.choice(lf, size=burn), n, lam, m0=m0, init_log=np.log(m0) + disp)
            vals[i] = np.log(t[burn - 50:burn]).mean() - np.log(m0)
        out[f"bias_disp{disp:+.0f}"] = float(vals.mean())
        out[f"se_disp{disp:+.0f}"] = float(vals.std() / np.sqrt(len(vals)))
    out["pass"] = bool(all(abs(v) < cfg["diag_tol"] for k, v in out.items() if k.startswith("bias")))
    return out


def make_clock(lf, cal, seed, order, cfg):
    burn, k = cfg["burn"], cfg["k"]
    rng = np.random.default_rng(seed + cfg["clock_offset"])
    t = simulate_ar(rng.choice(lf, size=burn + k), cal["n"], cal["lam"], m0=cal["m0"])
    if order == "shuf":
        t = t.copy()
        t[burn:] = np.random.default_rng(cfg["shuffle_offset"] + seed).permutation(t[burn:])
    return t


def half_difference(t, cfg):
    lg = np.log(t[cfg["burn"]:])
    h = len(lg) // 2
    return float(lg[h:].mean() - lg[:h].mean())


# ------------------------------------------------------------------------------------------- metrics
def acf(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-lag], x[lag:]) / d) if d > 0 else float("nan")


def p_stat(x, cfg):
    return float(np.mean([acf(x, k) for k in cfg["lags_p"]]))


def bar_returns(prices, cfg):
    """Log returns of the 20-trade bars after the burn-in; the first return starts at the last burn-in price."""
    b, bar = cfg["burn"], cfg["bar"]
    n = cfg["k"] // bar
    closes = prices[b - 1 + bar * np.arange(n + 1)]
    return np.diff(np.log(closes))


def bar_durations(t, cfg):
    b, bar = cfg["burn"], cfg["bar"]
    n = cfg["k"] // bar
    return t[b:b + n * bar].reshape(n, bar).sum(axis=1)


def p_of_run(prices, cfg):
    return p_stat(np.abs(bar_returns(prices, cfg)), cfg)


def permuted_p(prices, seed, cfg):
    r = bar_returns(prices, cfg)
    r = np.random.default_rng(cfg["barperm_offset"] + seed).permutation(r)
    return p_stat(np.abs(r), cfg)


def positive_control(prices, seed, cfg):
    """Per-trade log returns times exp(h_k), h a stationary AR(1) (known persistence; instrument check only)."""
    p0 = cfg["model"]["initial_price"] + cfg["model"]["spread"]
    dr = np.diff(np.log(np.concatenate([[p0], prices])))
    rng = np.random.default_rng(cfg["pos_offset"] + seed)
    phi = np.exp(-1.0 / cfg["pos_tau"])
    eps = rng.normal(0.0, cfg["pos_sd"] * np.sqrt(1 - phi * phi), len(dr))
    eps[0] = rng.normal(0.0, cfg["pos_sd"])
    h = np.empty(len(dr))
    prev = 0.0
    for i, e in enumerate(eps):
        prev = e if i == 0 else phi * prev + e
        h[i] = prev
    return np.exp(np.log(p0) + np.cumsum(dr * np.exp(h)))


def boot_ci(values, level, cfg):
    """Percentile bootstrap of the mean with the price seed as the unit."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(cfg["boot_seed"])
    means = v[rng.integers(0, len(v), size=(cfg["n_boot"], len(v)))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return {"n": int(len(v)), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "level": level}


def judge(lo, hi, delta):
    if lo > delta:
        return "伝播を支持"
    if hi < -delta:
        return "逆向き"
    if lo > -delta and hi < delta:
        return "実質的な伝播なし"
    return "判定保留"


# ------------------------------------------------------------------------------------------- secondary
def pooled_acf_32(series, lag):
    """ACF with one mean over all runs (the definition of Study Manager #32)."""
    allv = np.concatenate(series)
    mu = allv.mean()
    num = sum(float(((s - mu)[:-lag] * (s - mu)[lag:]).sum()) for s in series if len(s) > lag)
    pairs = sum(max(len(s) - lag, 0) for s in series)
    den = float(((allv - mu) ** 2).sum())
    return (num / max(pairs, 1)) / (den / len(allv))


def har_windows(r):
    sw = np.lib.stride_tricks.sliding_window_view
    n = len(r)
    anchors = np.arange(59, n - 20)
    x = sw(r, 60)[anchors - 59]
    y = sw(r, 20)[anchors + 1].std(axis=1, ddof=1)
    f = np.c_[x[:, -5:].std(axis=1, ddof=1), x[:, -20:].std(axis=1, ddof=1), x.std(axis=1, ddof=1)]
    keep = (y > 0) & (f > 0).all(axis=1)
    f, y = np.log(f[keep]), np.log(y[keep])
    return f - f.mean(axis=0), y - y.mean()          # within-run demeaning


def har_r2(returns_by_seed, cfg):
    seeds = sorted(returns_by_seed)
    cut = cfg["main_seeds"][cfg["har_train_n"] - 1]
    train = [har_windows(returns_by_seed[s]) for s in seeds if s <= cut]
    test = [har_windows(returns_by_seed[s]) for s in seeds if s > cut]
    if not train or not test:
        return None
    ftr, ytr = np.concatenate([a for a, _ in train]), np.concatenate([b for _, b in train])
    fte, yte = np.concatenate([a for a, _ in test]), np.concatenate([b for _, b in test])
    coef, *_ = np.linalg.lstsq(ftr, ytr, rcond=None)
    return {"r2": float(1 - np.mean((yte - fte @ coef) ** 2) / np.var(yte)), "coef": [float(c) for c in coef],
            "n_test": int(len(yte))}


def fixed_time_returns(prices, t, cfg, width=60.0):
    """60 s bars on the continuous grid (empty bars forward filled), as in T1."""
    times = np.cumsum(t)
    b = cfg["burn"]
    start = np.ceil(times[b - 1] / width) * width
    end = np.floor(times[-1] / width) * width
    nb = int(round((end - start) / width))
    i0, i1 = np.searchsorted(times, [start, end], "left")
    tt, p = times[i0:i1], prices[i0:i1]
    before = prices[i0 - 1]
    bins = ((tt - start) // width).astype(np.int64)
    last = np.full(nb, -1)
    np.maximum.at(last, bins, np.arange(len(p)))      # last trade of each bar
    idx = np.maximum.accumulate(last)
    close = np.where(idx >= 0, p[np.maximum(idx, 0)], before)
    return np.diff(np.log(np.concatenate([[before], close])))


def window_counts(t, cfg):
    """Number of trades inside the time window after each post-burn-in trade."""
    times = np.cumsum(t)
    first = np.searchsorted(times, times - cfg["eta"], "right")     # t_k - t_j < eta  <=>  t_j > t_k - eta
    return (np.arange(len(times)) - first + 1)[cfg["burn"]:]


def secondary_run(prices, t, cfg):
    r = bar_returns(prices, cfg)
    a = np.abs(r)
    out = {"p": p_stat(a, cfg), "abs_acf": {str(k): acf(a, k) for k in cfg["lags_sec"]},
           "ret_acf1": acf(r, 1), "kurtosis": float(pd.Series(r).kurt())}
    if t is not None:
        dur = bar_durations(t, cfg)
        out["spearman_duration_absr"] = float(sps.spearmanr(dur, a).statistic)
        out["duration_p"] = p_stat(dur, cfg)
        ft = np.abs(fixed_time_returns(prices, t, cfg))
        out["fixed60_p"] = p_stat(ft, cfg)
        out["fixed60_abs_acf"] = {str(k): acf(ft, k) for k in cfg["lags_sec"] if len(ft) > k}
        n = window_counts(t, cfg)
        out["window_n"] = {"mean": float(n.mean()), "share_1": float(np.mean(n == 1)),
                           "q05_25_50_75_95": [float(v) for v in np.quantile(n, [0.05, 0.25, 0.5, 0.75, 0.95])]}
    return out


def mean_tree(items):
    """Mean over runs of a list of identically shaped nested dicts / lists of numbers."""
    first = items[0]
    if isinstance(first, dict):
        return {k: mean_tree([d[k] for d in items]) for k in first}
    if isinstance(first, list):
        return [mean_tree([d[i] for d in items]) for i in range(len(first))]
    return float(np.mean(items))


# ------------------------------------------------------------------------------------------- jobs
def run_job(args):
    """One simulation run. Pilot runs (save_dir None) keep nothing but success/failure and time."""
    cfg, cond, seed, intervals, save_dir = args
    c0, w0 = time.process_time(), time.perf_counter()
    out = {"cond": cond, "seed": seed, "ok": True, "error": None}
    try:
        prices = simulate_condition(cfg, cond, seed, intervals)
        if save_dir is not None:
            np.savez(os.path.join(save_dir, f"{cond}_{seed}.npz"), prices=prices)
    except Exception as e:  # failure of the run as defined in the plan
        out.update(ok=False, error=repr(e))
    out.update(cpu=time.process_time() - c0, wall=time.perf_counter() - w0)
    return out
