"""F1 follow-up P2 (calibration diagnostics) library — plan v0.5 (docs/research-f1-p2-calibration-diagnostics-plan.md, commit c18ffc4).

Pure functions (no file I/O) so the same code runs on real data and on the synthetic-data gate (plan section 6).
Conventions: y = log C1 - log C0 (plan section 4); quantiles by linear interpolation (numpy 'linear' = P1 definition);
the week-block bootstrap resamples week ids with replacement (rng.integers(0, W, W)); p = (1 + #{|theta*-theta| >= |theta-theta0|}) / (B+1).
"""
from __future__ import annotations

import math

import numpy as np

COVER = (50, 80, 90, 95)
QS = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
TOL = 1e-12


# ---- predictive quantities from sorted ensembles ------------------------------------------------------------------------
def ens_table(sorted_ens):
    """sorted_ens [B, M] (sorted along axis 1) -> dict of per-bucket quantiles: med, lo{c}, hi{c}, q{q}."""
    t = {"med": np.quantile(sorted_ens, 0.5, axis=1)}
    for c in COVER:
        a = (1 - c / 100.0) / 2
        t[f"lo{c}"] = np.quantile(sorted_ens, a, axis=1)
        t[f"hi{c}"] = np.quantile(sorted_ens, 1 - a, axis=1)
    for q in QS:
        t[f"q{q}"] = np.quantile(sorted_ens, q, axis=1)
    return t


def origin_stats(sorted_ens, bucket, y):
    """per-origin diagnostics given sorted ensembles [B, M], bucket id per origin, realised y.
    Returns dict of arrays: med, lo{c}, hi{c}, q{q}, hit{c}, pit_lo, pit_w."""
    tab = ens_table(sorted_ens)
    out = {k: v[bucket] for k, v in tab.items()}
    M = sorted_ens.shape[1]
    pit_lo = np.empty(len(y))
    pit_r = np.empty(len(y))
    for b in np.unique(bucket):
        m = bucket == b
        left = np.searchsorted(sorted_ens[b], y[m], side="left")
        right = np.searchsorted(sorted_ens[b], y[m], side="right")
        pit_lo[m] = left / M
        pit_r[m] = (right - left) / M              # integer difference first (float subtraction of fractions is not exact)
    out["pit_lo"] = pit_lo
    out["pit_w"] = pit_r
    for c in COVER:
        out[f"hit{c}"] = ((y >= out[f"lo{c}"]) & (y <= out[f"hi{c}"])).astype(float)
    return out


# ---- week-block bootstrap ------------------------------------------------------------------------------------------------
class WeekBoot:
    """Week-block bootstrap on per-origin arrays. Origins are grouped by week id; a replicate is a draw of W weeks with replacement.
    Statistics are computed through per-week sufficient statistics (sums and counts), so a replicate costs O(W)."""

    def __init__(self, week, B, seed):
        self.uw, self.inv = np.unique(week, return_inverse=True)
        self.W = len(self.uw)
        self.B = B
        rng = np.random.default_rng(seed)
        self.draw = rng.integers(0, self.W, size=(B, self.W))
        # multiplicity of each week in each replicate: [B, W]
        self.mult = (self.draw[:, :, None] == np.arange(self.W)[None, None, :]).sum(axis=1).astype(float)

    def wsum(self, v):
        return np.bincount(self.inv, weights=np.asarray(v, float), minlength=self.W)

    def wcount(self, mask=None):
        return np.bincount(self.inv, weights=None if mask is None else np.asarray(mask, float), minlength=self.W)

    def ratio(self, num_v, den_v):
        """bootstrap replicates of sum(num)/sum(den) over resampled weeks; NaN where the denominator is 0."""
        n = self.wsum(num_v)
        d = self.wsum(den_v)
        N = self.mult @ n
        D = self.mult @ d
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(D > 0, N / D, np.nan)
        return n.sum() / d.sum() if d.sum() > 0 else np.nan, r, D


def boot_test(theta_hat, theta0, theta_star):
    """centred bootstrap test (plan 5.2). theta_star: replicate array (NaN = undefined replicate, excluded).
    Returns dict: theta, se, t, p, excl_frac, status ('ok'|'判定不能'), ci (percentile 95%)."""
    ts = np.asarray(theta_star, float)
    ok = np.isfinite(ts)
    excl = 1.0 - ok.mean()
    res = {"theta": float(theta_hat), "theta0": float(theta0), "excl_frac": float(excl), "B_valid": int(ok.sum())}
    if not np.isfinite(theta_hat) or excl > 0.05 or ok.sum() < 2:
        res.update(status="判定不能", se=None, t=None, p=None, ci=None)
        return res
    ts = ts[ok]
    se = float(ts.std(ddof=1))
    if not se > 0:
        res.update(status="判定不能", se=0.0, t=None, p=None, ci=[float(np.quantile(ts, .025)), float(np.quantile(ts, .975))])
        return res
    dev = abs(theta_hat - theta0)
    cnt = int((np.abs(ts - theta_hat) >= dev - TOL).sum())
    p = (1 + cnt) / (len(ts) + 1)
    res.update(status="ok", se=se, t=float((theta_hat - theta0) / se), p=float(p),
               ci=[float(np.quantile(ts, .025)), float(np.quantile(ts, .975))])
    return res


def holm(pvals):
    """Holm step-down over a list (None = 判定不能: counted in m, adjusted p = None)."""
    m = len(pvals)
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [None] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i]))
        adj[i] = run
    return adj


# ---- D3 k_c ------------------------------------------------------------------------------------------------------------------
def u_values(y, med, lo, hi):
    """normalised exceedance (plan D3). Returns u, side (+1 up, -1 down, 0 equal), valid mask (denominator > 0)."""
    up = y > med
    dn = y < med
    side = np.where(up, 1, np.where(dn, -1, 0))
    den = np.where(up, hi - med, np.where(dn, med - lo, 1.0))
    num = np.where(up, y - med, np.where(dn, med - y, 0.0))
    valid = (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(valid, num / den, np.nan)
    u = np.where(side == 0, 0.0, u)
    return u, side, valid


def k_c(y, med, lo, hi, c):
    """pooled, up and down k_c (c-quantile of u; side-specific conditional on the side) + exclusion counts.
    status 'unestimable' if exclusions (denominator <= 0) exceed 0.1% of the observations."""
    u, side, valid = u_values(y, med, lo, hi)
    n = len(y)
    n_excl = int((~valid).sum())
    out = {"n": n, "n_excl": n_excl, "n_equal": int((side == 0).sum()), "n_up": int(((side == 1) & valid).sum()),
           "n_dn": int(((side == -1) & valid).sum())}
    if n_excl > 0.001 * n:
        out.update(status="推定不能", k=None, k_up=None, k_dn=None)
        return out
    g = valid
    out["k"] = float(np.quantile(u[g], c / 100.0)) if g.any() else None
    for nm, s in (("k_up", 1), ("k_dn", -1)):
        m = g & (side == s)
        out[nm] = float(np.quantile(u[m], c / 100.0)) if m.sum() > 0 else None
    out["status"] = "ok"
    return out


# ---- D1 -----------------------------------------------------------------------------------------------------------------------
def pit_summary(pit):
    pit = np.asarray(pit, float)
    ks = np.sort(pit)
    n = len(ks)
    i = np.arange(1, n + 1)
    ksd = float(max((i / n - ks).max(), (ks - (i - 1) / n).max()))
    h, _ = np.histogram(pit, bins=20, range=(0, 1))
    return {"mean": float(pit.mean()), "var": float(pit.var()), "tail5_low": float((pit < 0.05).mean()),
            "tail5_high": float((pit > 0.95).mean()), "ks": ksd, "hist": (h / n).tolist()}


def randomized_pit(pit_lo, pit_w, seed, reps=100):
    rng = np.random.default_rng(seed)
    return pit_lo[None, :] + rng.random((reps, len(pit_lo))) * pit_w[None, :]


# ---- helpers ---------------------------------------------------------------------------------------------------------------------
def clopper_pearson(k, n, alpha=0.05):
    from scipy import stats
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


# ---- test families FA-FE on one (period, h) cell --------------------------------------------------------------------------
def pit_moments(pit_lo, pit_w):
    """E and E^2 of the randomised PIT pit_lo + U*pit_w in expectation over the tie randomisation (deterministic)."""
    e1 = pit_lo + pit_w / 2
    e2 = pit_lo ** 2 + pit_lo * pit_w + pit_w ** 2 / 3
    return e1, e2


def family_tests(o, y, wb):
    """o: origin_stats dict; wb: WeekBoot. Returns {name: boot_test result} for FA(c), FB, FC, FD(q), FE."""
    n = len(y)
    one = np.ones(n)
    res = {}
    for c in COVER:
        th, st, _ = wb.ratio(o[f"hit{c}"], one)
        res[f"FA{c}"] = boot_test(th, c / 100.0, st)
    low = (y < o["lo90"]).astype(float)
    upx = (y > o["hi90"]).astype(float)
    d = low - upx
    th, st, _ = wb.ratio(d, one)
    res["FB"] = boot_test(th, 0.0, st)
    th, st, _ = wb.ratio((y > o["med"]).astype(float), one)
    res["FC"] = boot_test(th, 0.5, st)
    for q in QS:
        th, st, _ = wb.ratio((y < o[f"q{q}"]).astype(float), one)
        res[f"FD{q}"] = boot_test(th, q, st)
    e1, e2 = pit_moments(o["pit_lo"], o["pit_w"])
    m1, s1, _ = wb.ratio(e1, one)
    m2, s2, _ = wb.ratio(e2, one)
    # replicate variance = E2* - E1*^2
    _, r1, _ = wb.ratio(e1, one)
    _, r2, _ = wb.ratio(e2, one)
    res["FE"] = boot_test(m2 - m1 ** 2, 1.0 / 12.0, r2 - r1 ** 2)
    return res


# ---- D4 max-t --------------------------------------------------------------------------------------------------------------------
def maxt_test(group, hit, wb, min_cell=100, max_missing=0.05):
    """plan D4. group: integer/str label per origin. Returns dict with the fixed target set, T, p, invalid rate, status."""
    labels = sorted(set(group.tolist()))
    gi = {g: np.asarray(group == g, float) for g in labels}
    n_g = {g: int(gi[g].sum()) for g in labels}
    gp = hit.mean()
    out = {"groups": {}, "status": "ok"}
    stats = {}
    for g in labels:
        th_g, st_g, D = wb.ratio(hit * gi[g], gi[g])
        miss = float((D <= 0).mean())
        stats[g] = (th_g, st_g, miss)
        out["groups"][str(g)] = {"n": n_g[g], "coverage": float(th_g), "missing_frac": miss}
    _, pool_star, _ = wb.ratio(hit, np.ones(len(hit)))
    target = []
    for g in labels:
        th_g, st_g, miss = stats[g]
        dstar = st_g - pool_star
        se = float(np.nanstd(dstar, ddof=1)) if np.isfinite(dstar).sum() > 1 else 0.0
        out["groups"][str(g)]["se"] = se
        out["groups"][str(g)]["diff_from_pool"] = float(th_g - gp)
        if n_g[g] < min_cell:
            out["groups"][str(g)]["note"] = "n<min_cell: descriptive only"
        elif miss > max_missing:
            out["groups"][str(g)]["note"] = "absent in >5% of bootstrap samples: descriptive only"
        elif not se > 0:
            out["groups"][str(g)]["note"] = "se=0: 判定不能"
        else:
            target.append(g)
    out["target_groups"] = [str(g) for g in target]
    if len(target) < 2:
        out["status"] = "判定不能"
        out["reason"] = "fewer than 2 target groups"
        return out
    Tobs = max(abs(stats[g][0] - gp) / out["groups"][str(g)]["se"] for g in target)
    star = np.stack([np.abs((stats[g][1] - pool_star) - (stats[g][0] - gp)) / out["groups"][str(g)]["se"] for g in target])
    valid = np.all(np.isfinite(star), axis=0)
    inv = 1.0 - valid.mean()
    out["invalid_rate"] = float(inv)
    out["T"] = float(Tobs)
    if inv > max_missing:
        out["status"] = "判定不能"
        out["reason"] = "invalid bootstrap rate > 5%"
        return out
    Ts = star[:, valid].max(axis=0)
    out["B_valid"] = int(valid.sum())
    out["p"] = float((1 + (Ts >= Tobs - TOL).sum()) / (valid.sum() + 1))
    return out
