"""P5-A library (plan v0.2 @ b995b5a): quantile-function comparison (A), week-weighted bootstrap quantiles, transforms T1-T4 (C),
candidate recovery (D). Diagnostics only; nothing here is a recalibration proposal."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5_env as E  # noqa: E402,F401
import p4_lib as P  # noqa: E402
import p4_core as C4  # noqa: E402

V, L = P.V, P.L
HZ = P.HZ
PERIODS = P.PERIODS
SEEDS = P.SEEDS
QS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
QA = np.array(QS)
I25, I50, I75 = QS.index(0.25), QS.index(0.50), QS.index(0.75)
EPS_REL = 1e-9                       # ratio undefined when |Qe(p) - Qe(.5)| <= EPS_REL * IQR_E1a
SGRID = np.round(np.arange(0.50, 2.0 + 1e-9, 0.01), 2)
GRID_M = 1000                        # quantile-grid size of the PIT-recalibrated (T0g, T3, T4) distributions
BOOT_B = 2000
N_CAND = 8
APPROX_TOL = 1e-3                    # gate 2c tolerance (relative CRPS error of grid vs sample representation)


# ---- A: quantile comparison --------------------------------------------------------------------------------------------------------------
def compare(qd, qe):
    """qd, qe: [..., K] quantiles at QS. Returns r (scale ratio |Qd(p)-Qd(.5)|/|Qe(p)-Qe(.5)|), shape (tail relative to the quartile of the
    same side), loc ((med_d - med_e)/IQR_e). Undefined entries are NaN (never replaced by 0 or 1)."""
    qd = np.asarray(qd, float)
    qe = np.asarray(qe, float)
    iqr = qe[..., I75] - qe[..., I25]
    side = np.where(QA > 0.5, 1.0, -1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        loc = np.where(iqr > 0, (qd[..., I50] - qe[..., I50]) / iqr, np.nan)
        dd = qd - qd[..., I50:I50 + 1]
        de = qe - qe[..., I50:I50 + 1]
        ok = (np.abs(de) > EPS_REL * np.maximum(iqr, 0)[..., None]) & (de * side > 0) & (iqr[..., None] > 0)
        r = np.where(ok, np.abs(dd) / np.where(ok, np.abs(de), 1.0), np.nan)
        r[..., I50] = np.nan
        shape = np.full_like(r, np.nan)
        up = QA > 0.75
        lo = QA < 0.25
        shape[..., up] = r[..., up] / r[..., I75:I75 + 1]
        shape[..., lo] = r[..., lo] / r[..., I25:I25 + 1]
    return {"loc": loc, "r": r, "shape": shape}


def to_json(x):
    """NaN -> None recursively (JSON-safe; undefined values are kept as null)."""
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_json(x.tolist())
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def wquantile(vals, wts, ps):
    """quantiles of a weighted sample (mid-point interpolation of the weighted ECDF); used for mixtures over buckets."""
    o = np.argsort(vals, kind="stable")
    v, w = vals[o], wts[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(ps, c, v)


class WBoot:
    """Week-block bootstrap of quantiles: week multiplicities are shared across all arrays of one universe (so replicates are coherent across
    horizons / buckets). The replicate quantile equals numpy 'linear' on the expanded sample. Descriptive intervals only."""

    def __init__(self, universe, B, tag):
        self.u = np.unique(universe)
        self.W = len(self.u)
        rng = np.random.default_rng([20261601] + [ord(ch) for ch in tag])
        draw = rng.integers(0, self.W, size=(B, self.W))
        self.mult = np.stack([np.bincount(d, minlength=self.W) for d in draw]).astype(np.int32)
        self.B = B

    def quantiles(self, y, weeks, ps):
        o = np.argsort(y, kind="stable")
        ys = y[o]
        wi = np.searchsorted(self.u, weeks[o])
        cum = np.cumsum(self.mult[:, wi], axis=1)
        N = cum[:, -1]
        ps = np.asarray(ps, float)
        out = np.full((self.B, len(ps)), np.nan)
        for b in range(self.B):
            if N[b] < 2:
                continue
            h = ps * (N[b] - 1)
            k0 = np.floor(h).astype(np.int64)
            k1 = np.minimum(k0 + 1, N[b] - 1)
            fr = h - k0
            i0 = np.searchsorted(cum[b], k0, side="right")
            i1 = np.searchsorted(cum[b], k1, side="right")
            out[b] = ys[i0] * (1 - fr) + ys[i1] * fr
        return out


def ci(rep, lo=2.5, hi=97.5):
    """descriptive interval over replicates ignoring undefined (NaN); also the undefined fraction."""
    ok = np.isfinite(rep)
    if ok.sum() < 20:
        return {"ci": None, "undefined_frac": float(1 - ok.mean())}
    return {"ci": [float(np.percentile(rep[ok], lo)), float(np.percentile(rep[ok], hi))], "undefined_frac": float(1 - ok.mean())}


# ---- C: transforms ---------------------------------------------------------------------------------------------------------------------
def scale_about_median(ens, s):
    """ens [..., M] sorted; scale about the ensemble median (s > 0 keeps the order)."""
    if s == 1.0:
        return ens.copy()                    # exact: the original ensemble is used as is
    med = np.median(ens, axis=-1, keepdims=True)
    return med + s * (ens - med)


def pit_mid(ens_row, y):
    left = np.searchsorted(ens_row, y, side="left")
    right = np.searchsorted(ens_row, y, side="right")
    return (left + right) / (2.0 * len(ens_row))


def grid_p(m=GRID_M):
    return (np.arange(m) + 0.5) / m


def recal_grid(ens_row, ginv=None):
    """quantile-grid representation of F' = G(F): x_i = Q_ens(Ginv(p_i)); ginv=None is the identity (T0g). ens_row sorted [M]."""
    p = grid_p()
    u = p if ginv is None else np.clip(ginv(p), 0.0, 1.0)
    return np.quantile(ens_row, u)


def fit_scales(data, ens_by_seed, h):
    """T1 (common) and T2 (per bucket) scale factors from ESTIMATION origins only: minimise the mean CRPS over SGRID (seed-averaged).
    Uses only data.cells[('estimation', h)]. Returns (s1, s2[20], crps_curves)."""
    c = data.cells[("estimation", h)]
    y, b = c["y"], c["b"]
    tot = np.zeros(len(SGRID))
    per_b = np.zeros((20, len(SGRID)))
    nb = np.bincount(b, minlength=20)
    for si, s in enumerate(SGRID):
        acc = np.zeros(len(y))
        for ens in ens_by_seed.values():
            sc = scale_about_median(ens[:, c["hi"], :], s)
            acc += P.fair_crps_list([sc[k] for k in range(20)], b, y)
        acc /= len(ens_by_seed)
        tot[si] = acc.mean()
        per_b[:, si] = np.bincount(b, weights=acc, minlength=20) / nb
    return float(SGRID[int(np.argmin(tot))]), SGRID[np.argmin(per_b, axis=1)].astype(float), {"pooled": tot, "per_bucket": per_b}


def fit_pit(data, ens_by_seed, h):
    """PIT samples of ESTIMATION origins (mid-PIT): pooled over buckets and seeds (T3) and per bucket (T4). Only estimation data are used."""
    c = data.cells[("estimation", h)]
    y, b = c["y"], c["b"]
    per_bucket = [[] for _ in range(20)]
    for ens in ens_by_seed.values():
        e = ens[:, c["hi"], :]
        for k in range(20):
            m = b == k
            per_bucket[k].append(pit_mid(e[k], y[m]))
    per_bucket = [np.sort(np.concatenate(v)) for v in per_bucket]
    pooled = np.sort(np.concatenate(per_bucket))
    return pooled, per_bucket


def make_ginv(pit_sorted):
    return lambda p: np.quantile(pit_sorted, p)


def fit_all(data, ens_by_seed):
    """all estimation-only fits (gate 4 runs this with a data object that raises on any non-estimation access)."""
    out = {"s1": {}, "s2": {}, "pit_pooled": {}, "pit_bucket": {}}
    for h in HZ:
        s1, s2, _ = fit_scales(data, ens_by_seed, h)
        pooled, pb = fit_pit(data, ens_by_seed, h)
        out["s1"][h], out["s2"][h], out["pit_pooled"][h], out["pit_bucket"][h] = s1, s2, pooled, pb
    return out


def apply_transform(kind, ens_by_seed, fits):
    """returns {seed: [20, 4, M'] sorted arrays}. kind in T0, T0g, T1, T2, T3, T4."""
    res = {}
    for s, ens in ens_by_seed.items():
        if kind == "T0":
            res[s] = ens
            continue
        out = np.empty((20, 4, GRID_M if kind in ("T0g", "T3", "T4") else ens.shape[2]))
        for hi, h in enumerate(HZ):
            for k in range(20):
                row = ens[k, hi, :]
                if kind == "T1":
                    out[k, hi] = scale_about_median(row, fits["s1"][h])
                elif kind == "T2":
                    out[k, hi] = scale_about_median(row, fits["s2"][h][k])
                elif kind == "T0g":
                    out[k, hi] = recal_grid(row, None)
                elif kind == "T3":
                    out[k, hi] = recal_grid(row, make_ginv(fits["pit_pooled"][h]))
                elif kind == "T4":
                    out[k, hi] = recal_grid(row, make_ginv(fits["pit_bucket"][h][k]))
                else:
                    raise ValueError(kind)
        res[s] = out
    return res


class EnsEval:
    """Per-origin diagnostics for a set of per-seed ensembles (any representation); seed-wise nonlinear metrics averaged as in P4."""

    def __init__(self, data, ens_by_seed):
        self.data = data
        self.o = {}
        for s, ens in ens_by_seed.items():
            for per in PERIODS:
                for h in HZ:
                    c = data.cells[(per, h)]
                    self.o.setdefault((per, h), []).append(P.Pred.from_ens(ens[:, c["hi"], :]).per_origin(c["b"], c["y"]))
        self.crps = {k: np.mean([o["crps"] for o in v], axis=0) for k, v in self.o.items()}

    def metrics(self, per, h, mask=None):
        c = self.data.cells[(per, h)]
        return P.avg_metrics([P.metrics(o, c["y"], mask) for o in self.o[(per, h)]])

    def group_metrics(self, per, h, mask=None):
        c = self.data.cells[(per, h)]
        gm = [P.group_metrics(o, c["y"], c["b"], mask) for o in self.o[(per, h)]]
        return {g: P.avg_metrics([x[g] for x in gm]) for g in gm[0]}


# ---- D: candidates ------------------------------------------------------------------------------------------------------------------------
def load_raw_ens(cond, seed):
    """unsorted ensemble [20, 4, 1000] in attempt order and the attempt id of each path [20, 1000]."""
    z = np.load(os.path.join(E.ENS_DIR, f"{cond}_seed{seed}.npz"), allow_pickle=True)
    return z["ens"], z["attempt_ids"]


def load_mapping():
    import json
    return json.load(open(os.path.join(E.RUN_DIR, "ddm_mapping.json")))


def cand_samples(raw, att, b, hi, cand, seeds):
    """paths of one candidate in bucket b / horizon index hi, per seed, in attempt order. cand = attempt id modulo N_CAND (F1 generation rule)."""
    return [raw[s][b, hi][(att[s][b] % N_CAND) == cand] for s in seeds]


def pooled(arrs):
    return np.sort(np.concatenate(arrs))


def pair_samples(la, lb):
    """equal weights: per seed both candidates are truncated to the same number of paths (the first ones in attempt order)."""
    parts = []
    for a, b in zip(la, lb):
        n = min(len(a), len(b))
        parts += [a[:n], b[:n]]
    return np.sort(np.concatenate(parts))


def _crps_one(sample, y):
    return float(P.fair_crps_list([sample], np.zeros(len(y), int), y).mean())


def select_subsets(data, raw, att, seeds, h):
    """fixed rule (plan section 4D): per bucket, using ESTIMATION origins only, pick the single candidate (8 options) and the candidate pair
    (28 options, 50:50) with the smallest mean CRPS (ties -> first in lexicographic order). Returns per-bucket dicts."""
    c = data.cells[("estimation", h)]
    hi = HZ.index(h)
    out = []
    pairs = [(i, j) for i in range(N_CAND) for j in range(i + 1, N_CAND)]
    for b in range(20):
        ye = c["y"][c["b"] == b]
        cs = [cand_samples(raw, att, b, hi, k, seeds) for k in range(N_CAND)]
        single = np.array([_crps_one(pooled(x), ye) for x in cs])
        pair = np.array([_crps_one(pair_samples(cs[i], cs[j]), ye) for i, j in pairs])
        mix = _crps_one(pooled([raw[s][b, hi] for s in seeds]), ye)
        out.append({"single": int(np.argmin(single)), "single_crps": single, "pair": pairs[int(np.argmin(pair))], "pair_crps": pair, "mix8_crps": mix,
                    "n_paths": [int(sum(len(a) for a in x)) for x in cs]})
    return out
