"""V0 step 3 (plan 6): operating characteristics of the three-way rule on pseudo data. Truth per point from all 10 runs (SA and SB are not treated differently),
(a) the 10-run means, (b) all points equal to the maximum of (a); two independent pseudo sets are generated; the analysis is the same as the main one (max over all
136 points, recomputed in the bootstrap; four intervals). Output: results/v0c.json"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

import vc_env as E
import vc_core as C
import vc_data as D


def clopper(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def calibrate(m_truth, dev, values, week, n_units, rng, n_cal, b_cal, shifts=(0.0, 0.01, -0.01)):
    """m_truth [P], dev [P, 10] centred and inflated run deviations; values/week: real windows used as the shape of the pseudo real data."""
    uw, inv = np.unique(week, return_inverse=True)
    ssum = np.bincount(inv, weights=values)
    scnt = np.bincount(inv).astype(float)
    shift0 = float(m_truth.max() + E.DELTA - values.mean())
    counts = {s: {"diff": 0, "small": 0} for s in shifts}
    P = len(m_truth)
    n = np.full(P, 5)
    for _ in range(n_cal):
        pick = rng.integers(0, len(uw), n_units)
        c = scnt[pick]
        s = ssum[pick] + shift0 * c
        mu = s.sum() / c.sum()
        mm = rng.integers(0, n_units, (b_cal, n_units))
        boot = s[mm].sum(axis=1) / c[mm].sum(axis=1)
        ivs = []
        for _set in range(2):
            X = m_truth[:, None] + np.take_along_axis(dev, rng.integers(0, dev.shape[1], (P, 5)), axis=1)
            mx = X.mean(axis=1).max()
            ivs.append(C.interval(boot - mx))
            ivs.append(C.interval(mu - C.ddm_max_boot(X, n, b_cal, rng)))
        for sh in shifts:
            moved = [[lo + sh, hi + sh] for lo, hi in ivs]
            counts[sh]["diff"] += all(lo > E.DELTA for lo, _ in moved)
            counts[sh]["small"] += all(hi < E.DELTA for _, hi in moved)
    return counts


def main():
    code = E.begin("vc_v0c")
    v0a = E.load("v0a.json")
    base, _, ok = D.ddm_points()
    vals = D.ddm_run_values(base, ok)
    ten = np.array([np.concatenate([vals["SA"][i], vals["SB"][i]]) for i in range(len(base))])      # [P, 10]
    mean10 = ten.mean(axis=1)
    dev = (ten - mean10[:, None]) * np.sqrt(10 / 9)
    T = D.b_windows()
    RA = T[(T["period"] == "estimation") & (~T["excluded"]) & (T["parity"] == 1)]
    n_units = v0a["independent_counts"]["n_weeks_with_valid_windows"]
    rng = np.random.default_rng(E.SEEDS["cal"])
    out = {"_code": code, "n_points": len(base), "n_pseudo": E.N_CAL, "inner_B": E.B_CAL, "pseudo_weeks": n_units, "delta": E.DELTA, "settings": {}}
    for name, truth in (("a_point_means", mean10), ("b_all_equal_to_max", np.full(len(base), mean10.max()))):
        cnt = calibrate(truth, dev, RA["volclust_50"].values, RA["week"].values, n_units, rng, E.N_CAL, E.B_CAL)
        res = {}
        for sh, c in cnt.items():
            res[f"true_Delta=delta{sh:+.3f}"] = {k: {"rate": c[k] / E.N_CAL, "count": int(c[k]), "ci95": clopper(c[k], E.N_CAL)} for k in ("diff", "small")}
        out["settings"][name] = {"truth_max": float(truth.max()), "truth_sd": float(truth.std()), "results": res}
    at = [out["settings"][s]["results"]["true_Delta=delta+0.000"] for s in out["settings"]]
    out["false_diff_rate_max"] = max(a["diff"]["rate"] for a in at)
    out["false_small_rate_max"] = max(a["small"]["rate"] for a in at)
    out["stop_false_diff"] = bool(out["false_diff_rate_max"] > 0.10)
    out["small_category_enabled"] = bool(out["false_small_rate_max"] <= 0.10)
    E.save("v0c.json", out)
    print({k: out[k] for k in ("false_diff_rate_max", "false_small_rate_max", "stop_false_diff", "small_category_enabled")}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
