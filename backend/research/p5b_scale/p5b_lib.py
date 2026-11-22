"""P5-B library (plan v0.2 @ 5277e17): mapping windows, atlas lookup, selection replication, real-side variance ratios, DDM-side per-candidate
statistics, exact decomposition. Stored data only; no simulation."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5b_env as E  # noqa: E402,F401
import p5_lib as K  # noqa: E402  (P5-A library: WBoot, candidate helpers; also pulls the P4/P3/P2/P1 paths)
import p4_lib as P  # noqa: E402

V = P.V
HZ = (1, 4, 12, 24)
WIN = 1000                 # mapping window: 1001 closes = 1000 log returns, rows [60 j, 60 j + 1000]
BLOCK = 60
N_CAND = 8
SHORTLIST = 30
W_ACF1, W_VC = 0.25, 0.25
B_BOOT = 2000
CONDS = ("C0s", "C3")


# ---- mapping windows (same definition as f1_ddm.build_mapping) -------------------------------------------------------------------------
def load_mapping():
    return json.load(open(os.path.join(E.RUN_DIR, "ddm_mapping.json")))


def load_atlas():
    return pd.read_csv(E.ATLAS)


class Windows:
    """per-window variance of the 1000 log returns of rows [a, a+1000] via cumulative sums (ddof=0, as numpy ndarray.std in the mapping)."""

    def __init__(self, close):
        self.close = close
        lr = np.diff(np.log(close))
        self.S1 = np.concatenate([[0.0], np.cumsum(lr)])
        self.S2 = np.concatenate([[0.0], np.cumsum(lr * lr)])
        self.n = len(close)

    def var(self, a):
        a = np.asarray(a)
        s1 = self.S1[a + WIN] - self.S1[a]
        s2 = self.S2[a + WIN] - self.S2[a]
        return (s2 - s1 * s1 / WIN) / WIN

    def var_direct(self, a):
        return np.array([np.diff(np.log(self.close[i:i + WIN + 1])).var() for i in np.atleast_1d(a)])

    def std_direct(self, a):
        return np.array([np.diff(np.log(self.close[i:i + WIN + 1])).std() for i in np.atleast_1d(a)])


def valid_windows(raw, blocks, period_end):
    """windows [60 j, 60 j + 1000] that are gap-free and end before `period_end` (exclusive, a Timestamp)."""
    a = np.asarray(blocks) * BLOCK
    b = a + WIN
    ok = raw.contiguous(a, b) & (b < raw.n)
    ok &= np.asarray(raw.index[np.minimum(b, raw.n - 1)] < period_end)
    return ok


# ---- atlas and selection replication --------------------------------------------------------------------------------------------------
def atlas_row(atlas, delta, wma):
    m = np.isclose(atlas["delta"].values, delta) & (atlas["wma"].values == wma)
    return atlas[m]


def sigma2_atlas(row):
    """seed-averaged variance recovered from the stored mean of per-seed std and the std across seeds (np.nanstd, ddof=0)."""
    return float(row["std"] ** 2 + row["std_std_across_seeds"] ** 2)


def replicate_selection(real_stats, atlas):
    """replicates ddm_ensemble_selection.select_ensemble (std-first shortlist of 30, then acf_1 / volclust_50 z-distance, top 8)."""
    valid = atlas.dropna(subset=["std", "acf_1", "volclust_50"]).copy()
    valid["std_distance"] = (np.log(valid["std"]) - np.log(real_stats["std"])).abs()
    sl = valid.nsmallest(SHORTLIST, "std_distance").copy()
    d2 = np.zeros(len(sl))
    for name, w in (("acf_1", W_ACF1), ("volclust_50", W_VC)):
        mu, sig = sl[name].mean(), sl[name].std()
        sig = sig if sig > 0 else 1.0
        d2 += w * ((sl[name] - mu) / sig - (real_stats[name] - mu) / sig) ** 2
    sl["distance"] = np.sqrt(d2)
    sl = sl.sort_values("distance")
    return sl.head(N_CAND), sl, valid


# ---- variance bootstrap on week blocks ------------------------------------------------------------------------------------------------
def var_rep(wb, y, weeks):
    """replicates of the population variance of y over week-resampled data; wb = K.WBoot (shared multiplicities)."""
    idx = np.searchsorted(wb.u, weeks)
    n = np.bincount(idx, minlength=wb.W).astype(float)
    s1 = np.bincount(idx, weights=y, minlength=wb.W)
    s2 = np.bincount(idx, weights=y * y, minlength=wb.W)
    N = wb.mult @ n
    S1 = wb.mult @ s1
    S2 = wb.mult @ s2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(N > 1, S2 / N - (S1 / N) ** 2, np.nan)


def pct(rep, lo=2.5, hi=97.5):
    ok = np.isfinite(rep)
    return [float(np.percentile(rep[ok], lo)), float(np.percentile(rep[ok], hi))] if ok.sum() >= 20 else None


# ---- DDM-side statistics from the stored ensembles --------------------------------------------------------------------------------------
def load_cond(cond):
    raw, att = {}, {}
    for s in K.SEEDS:
        raw[s], att[s] = K.load_raw_ens(cond, s)
    return raw, att


def cand_stats(raw, att, b, h):
    """per-candidate and mixture statistics of the 60h-step cumulative returns of bucket b (3 seeds pooled)."""
    hi = HZ.index(h)
    per = [np.concatenate(K.cand_samples(raw, att, b, hi, c, K.SEEDS)) for c in range(N_CAND)]
    mix = np.concatenate([raw[s][b, hi] for s in K.SEEDS])
    seed_sd = [float(np.std(raw[s][b, hi])) for s in K.SEEDS]
    q = lambda x: np.quantile(x, [0.05, 0.95])  # noqa: E731
    return {"n_c": [len(x) for x in per], "mean_c": [float(x.mean()) for x in per], "var_c": [float(x.var()) for x in per], "sd_c": [float(x.std()) for x in per],
            "w90_c": [float(np.subtract(*q(x)[::-1])) for x in per], "mix_mean": float(mix.mean()), "mix_var": float(mix.var()), "mix_sd": float(mix.std()),
            "mix_w90": float(np.subtract(*q(mix)[::-1])), "seed_sd": seed_sd}


# ---- decomposition (plan section 3D) ------------------------------------------------------------------------------------------------------
def decompose(sd_c0s, sd_c3, var_y, v_real, s2_mix, h):
    """log(sd_C0s / sd_E1a) = M1 + M3 + M2 with
       M1 = log(sigma_atlas,mix / sqrt(V_real,M1)),  M3 = 1/2 log(VR_C3 / VR_real),  M2 = log(sd_C0s / sd_C3),
       VR_real = Var(y_h) / (60h V_real,M1),  VR_C3 = sd_C3^2 / (60h sigma^2_atlas,mix)   (definitions; sd_E1a = sqrt(Var(y_h)))."""
    n = BLOCK * h
    vr_real = var_y / (n * v_real)
    vr_c3 = sd_c3 ** 2 / (n * s2_mix)
    m1 = np.log(np.sqrt(s2_mix) / np.sqrt(v_real))
    m3 = 0.5 * np.log(vr_c3 / vr_real)
    m2 = np.log(sd_c0s / sd_c3)
    total = np.log(sd_c0s / np.sqrt(var_y))
    return {"M1": m1, "M3": m3, "M2": m2, "total": total, "VR_real": vr_real, "VR_C3": vr_c3, "residual": m1 + m3 + m2 - total}
