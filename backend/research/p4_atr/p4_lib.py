"""P4 library (plan v0.2 @ 9d9825d): predictive objects with per-cell samples, fair CRPS for unequal sample sizes, per-origin metrics,
exact decomposition (Shapley), conditioning features (origin-time information only), week-block bootstrap helpers."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p4_env as E  # noqa: E402,F401  (sets sys.path for P1-P3 modules)
import p2_lib as L  # noqa: E402
import p3_eval as V  # noqa: E402

PERIODS = V.PERIODS
HZ = V.HZ
GRID_M = 2000          # quantile grid used for coverage / PIT quantities of the empirical baselines
BOOT_B = 2000
SEEDS = (1, 2, 3)
WIN = 720              # 12 candles of 60 M1 rows
MIN_CELL = 100


# ---- fair CRPS for per-cell samples of different sizes -------------------------------------------------------------------
def fair_crps_list(samples, cell, y):
    """samples: list of sorted 1-D arrays (one per cell); cell: cell index per origin. Same estimator as p3_eval.fair_crps."""
    out = np.empty(len(y))
    for c in np.unique(cell):
        x = samples[c]
        M = len(x)
        m = cell == c
        yy = y[m]
        S = np.concatenate([[0.0], np.cumsum(x)])
        kk = np.arange(1, M + 1)
        pair = ((2 * kk - M - 1) * x).sum() / (M * (M - 1))
        k = np.searchsorted(x, yy, side="right")
        out[m] = (yy * (2 * k - M) + S[M] - 2 * S[k]) / M - pair
    return out


def fair_crps_brute(x, y):
    x = np.asarray(x, float)
    M = len(x)
    d = np.abs(x[:, None] - x[None, :]).sum()
    return np.array([np.abs(x - yy).mean() - d / (2 * M * (M - 1)) for yy in np.atleast_1d(y)])


def grid_from_samples(x, m=GRID_M):
    return np.quantile(x, (np.arange(m) + 0.5) / m)


class Pred:
    """A predictive distribution that depends only on a cell index; each cell is an equally weighted sample."""

    def __init__(self, samples, grid=None):
        self.samples = samples
        self.grid = grid if grid is not None else np.stack([grid_from_samples(s) for s in samples])

    @classmethod
    def from_ens(cls, ens):                      # ens [cells, M] sorted: the ensemble is its own grid
        return cls([ens[b] for b in range(ens.shape[0])], grid=ens)

    def per_origin(self, cell, y):
        o = L.origin_stats(self.grid, cell, y)
        o["crps"] = fair_crps_list(self.samples, cell, y)
        return o


def metrics(o, y, mask=None):
    if mask is not None:
        o = {k: v[mask] for k, v in o.items()}
        y = y[mask]
    n = len(y)
    if n == 0:
        return {"n": 0}
    m = {"n": int(n), "crps": float(o["crps"].mean())}
    for c in (50, 80, 90, 95):
        m[f"cov{c}"] = float(o[f"hit{c}"].mean())
    for c in (50, 90):
        m[f"k{c}"] = L.k_c(y, o["med"], o[f"lo{c}"], o[f"hi{c}"], c)["k"]
    m["P_up"] = float((y > o["med"]).mean())
    e1, e2 = L.pit_moments(o["pit_lo"], o["pit_w"])
    m["pit_var"] = float(e2.mean() - e1.mean() ** 2)
    m["width90"] = float((o["hi90"] - o["lo90"]).mean())
    return m


def group_metrics(o, y, bucket, mask=None):
    g = bucket // 4
    base = np.ones(len(y), bool) if mask is None else mask
    return {str(a): metrics(o, y, base & (g == a)) for a in range(5)}


def avg_metrics(ms):
    """average numeric metrics over seeds (None entries are skipped; n is taken from the first)."""
    out = {}
    for k in ms[0]:
        vals = [m[k] for m in ms if m.get(k) is not None]
        out[k] = ms[0][k] if k == "n" else (float(np.mean(vals)) if vals else None)
    return out


# ---- week-block bootstrap helpers ---------------------------------------------------------------------------------------------
_WB = {}


def wboot(week, key, B=BOOT_B):
    k = (key, len(week), B)
    if k not in _WB:
        _WB[k] = L.WeekBoot(week, B, seed=[20261501] + [ord(ch) for ch in key])      # deterministic (no Python hash())
    return _WB[k]


def rel_diff(x, ref, week, key):
    """relative change of mean(x)/mean(ref) - 1 with a descriptive week-block bootstrap 95% interval."""
    wb = wboot(week, key)
    th, st, _ = wb.ratio(x, ref)
    ok = np.isfinite(st)
    return {"est": float(th - 1.0), "ci": [float(np.quantile(st[ok], .025) - 1), float(np.quantile(st[ok], .975) - 1)],
            "n": int(len(x)), "weeks": int(wb.W)}


# ---- decomposition (A1) ---------------------------------------------------------------------------------------------------------
def shapley(wv, cv, wf, cf):
    wv, cv, wf, cf = (np.asarray(a, float) for a in (wv, cv, wf, cf))
    total = (wf * cf).sum() - (wv * cv).sum()
    comp1, within1 = ((wf - wv) * cv).sum(), (wf * (cf - cv)).sum()          # reference = earlier period's group values
    comp2, within2 = ((wf - wv) * cf).sum(), (wv * (cf - cv)).sum()          # reference = later period's group values
    return {"total": total, "composition_order1": comp1, "within_order1": within1, "composition_order2": comp2, "within_order2": within2,
            "composition_shapley": (comp1 + comp2) / 2, "within_shapley": (within1 + within2) / 2}


def group_stats(arr, grp, week, key):
    """group means c_g, group shares w_g and bootstrap replicates of both (arr per origin, grp in 0..4)."""
    wb = wboot(week, key)
    cs, ws, cr, wr = [], [], [], []
    Ds = []
    for g in range(5):
        m = (grp == g).astype(float)
        th, r, D = wb.ratio(arr * m, m)
        cs.append(th)
        cr.append(r)
        Ds.append(D)
    Ds = np.array(Ds)                       # [5, B]
    tot = Ds.sum(axis=0)
    n = np.array([(grp == g).sum() for g in range(5)], float)
    return {"c": np.array(cs), "w": n / n.sum(), "c_rep": np.array(cr), "w_rep": Ds / tot}


def decomposition(arr_v, grp_v, week_v, arr_f, grp_f, week_f, key):
    sv = group_stats(arr_v, grp_v, week_v, key + "v")
    sf = group_stats(arr_f, grp_f, week_f, key + "f")
    pt = shapley(sv["w"], sv["c"], sf["w"], sf["c"])
    B = sv["c_rep"].shape[1]
    reps = {k: np.empty(B) for k in pt}
    for b in range(B):
        r = shapley(sv["w_rep"][:, b], sv["c_rep"][:, b], sf["w_rep"][:, b], sf["c_rep"][:, b])
        for k in pt:
            reps[k][b] = r[k]
    out = {}
    for k in pt:
        ok = np.isfinite(reps[k])
        out[k] = {"est": float(pt[k]), "ci": [float(np.quantile(reps[k][ok], .025)), float(np.quantile(reps[k][ok], .975))]}
    out["groups"] = {"w_v": sv["w"].tolist(), "c_v": sv["c"].tolist(), "w_f": sf["w"].tolist(), "c_f": sf["c"].tolist()}
    return out


# ---- features at the origin (information up to and including the origin row only) ------------------------------------------------------
def build_features(raw):
    """vr = mean |dclose| over the last WIN rows divided by the block ATR; ret = log C_t - log C_{t-WIN}; NaN where the window has a gap."""
    t = raw.t
    c = raw.close
    atr = raw.atr()
    ok = raw.contiguous(t - WIN, t)
    d = np.abs(np.diff(c))
    cs = np.concatenate([[0.0], np.cumsum(d)])
    vr = np.full(raw.nb, np.nan)
    ret = np.full(raw.nb, np.nan)
    i = np.flatnonzero(ok)
    vr[i] = ((cs[t[i]] - cs[t[i] - WIN]) / WIN) / atr[i]
    ret[i] = np.log(c[t[i]]) - np.log(c[t[i] - WIN])
    return {"vr": vr, "ret": ret}


def feat_one(close, t):
    """same quantities from a single origin row using only close[:t+1]."""
    seg = close[t - WIN:t + 1]
    atr = np.abs(np.diff(close[t - 59:t + 1])).mean()
    return float((np.abs(np.diff(seg)).sum() / WIN) / atr), float(np.log(close[t]) - np.log(close[t - WIN]))


def session_of(hour):
    return np.digitize(hour, [8, 16])


def terciles_by_bucket(x_est, b_est):
    cuts = np.full((20, 2), np.nan)
    for b in range(20):
        v = x_est[(b_est == b) & np.isfinite(x_est)]
        cuts[b] = np.quantile(v, [1 / 3, 2 / 3])
    return cuts


def level_of(x, b, cuts):
    lv = (x > cuts[b, 0]).astype(int) + (x > cuts[b, 1]).astype(int)
    return np.where(np.isfinite(x), lv, -1)
