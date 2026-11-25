"""Matching, noise floor and residual statistics (plan v0.6 sections 7.2, 7.3). Pure functions on arrays; no I/O."""
from __future__ import annotations

import numpy as np

MIN_REF = 3
K_CROSSFIT = 200
K_INNER = 50
CROSSFIT_SEED = 20261125


def rules_from_reference(MA, EA):
    """SD_j of the matching components, caliper c_j = 0.5 SD_j, scale sigma_j of the evaluation statistics (window-to-window SD over the reference windows, ddof=1)."""
    sd = MA.std(axis=0, ddof=1)
    return {"SD": sd, "c": 0.5 * sd, "sigma": EA.std(axis=0, ddof=1)}


def box_mask(Mx, My, c):
    """[len(Mx), len(My)] True where all components differ by at most c."""
    d = np.abs(Mx[:, None, :] - My[None, :, :])
    return (d <= c[None, None, :]).all(axis=2)


def wpercentile(x, w, q):
    """percentile of x with integer multiplicities w (repeat then np.percentile: identical to the unweighted version when w = 1)."""
    return np.percentile(np.repeat(x, w.astype(int)), q)


def dn_real_real(EB, EA, mask, wA, sigma):
    """d^rr for each B window: (E_B - weighted mean of E over the reference A windows) / sigma; valid where the reference weight is >= MIN_REF."""
    cnt = mask @ wA
    ok = cnt >= MIN_REF
    num = (mask * wA[None, :]) @ EA
    ref = np.where(ok[:, None], num / np.maximum(cnt, 1)[:, None], np.nan)
    return (EB - ref) / sigma, ok


def tau_from(d, ok, wB, q=95.0):
    """tau_j = 95th percentile of |d^rr_j| over valid B windows (with multiplicities wB)."""
    tau = np.empty(d.shape[1])
    for j in range(d.shape[1]):
        m = ok & (wB > 0)
        tau[j] = wpercentile(np.abs(d[m, j]), wB[m], q)
    return tau


def crossfit_f0(d, ok, wB, weeks_B, k, rng):
    """f_0j: random 50/50 splits of the B weeks; tau on one half, exceedance fraction on the other half; mean over k splits (k is a stabilisation count, not a sample size)."""
    uw = np.unique(weeks_B)
    out = np.zeros((k, d.shape[1]))
    for s in range(k):
        perm = rng.permutation(len(uw))
        h1 = np.isin(weeks_B, uw[perm[: len(uw) // 2]])
        h2 = ~h1
        for j in range(d.shape[1]):
            m1 = ok & (wB > 0) & h1
            m2 = ok & (wB > 0) & h2
            if m1.sum() < 5 or m2.sum() < 5:
                out[s, j] = np.nan
                continue
            tau = wpercentile(np.abs(d[m1, j]), wB[m1], 95.0)
            out[s, j] = np.sum(wB[m2] * (np.abs(d[m2, j]) > tau)) / np.sum(wB[m2])
    return np.nanmean(out, axis=0), out


def ddm_reference(reach, Etheta):
    """E_DDM(w) = equal-weight mean of E_theta over the matched points; windows without a matched point are NaN."""
    n = reach.sum(axis=1)
    ed = (reach.astype(float) @ Etheta) / np.maximum(n, 1)[:, None]
    ed[n == 0] = np.nan
    return ed, n


def residual_stats(EDDM, EA, wA, sigma, tau):
    """d_j(w), D_j (weighted mean), median, f_j (weighted exceedance fraction) over the reachable windows."""
    ok = np.isfinite(EDDM).all(axis=1) & (wA > 0)
    d = (EDDM[ok] - EA[ok]) / sigma
    w = wA[ok]
    D = (w[:, None] * d).sum(axis=0) / w.sum()
    f = (w[:, None] * (np.abs(d) > tau)).sum(axis=0) / w.sum()
    return d, D, f, int(ok.sum())
