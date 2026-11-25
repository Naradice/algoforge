"""Stage C-min (plan v0.6 sections 7.2, 7.3, 10): reachability of the R-A real windows from the A1 DDM points, binding components, and (only if the plan's reachability conditions
hold) conditional residuals with the two-layer bootstrap. Requires results/frozen_rules.json committed before it runs. Output: results/c_min.json"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_match as MT  # noqa: E402
from dr_a1_stats import IDX  # noqa: E402
from dr_b_rules import split_est  # noqa: E402

B_SEED, B_REAL, B_REAL_SENS = 2000, 2000, 1000
RNG_SEED = 20261126


def load_ddm(fr, exclude_first, R, PT):
    """point arrays for the points with all 10 seeds successful: SA/SB M means and SEs, SB run-level E arrays."""
    Mn, En = fr["M"], fr["E"]
    P = PT[PT["all10"]].reset_index(drop=True)
    d = {"pid": P["pid"].values, "delta": P["delta"].values, "wma": P["wma"].values}
    for s in ("SA", "SB"):
        d[f"M_{s}"] = P[[f"{s}_{m}" for m in Mn]].values
        d[f"SEM_{s}"] = P[[f"{s}_{m}_se" for m in Mn]].values
    d["E_SB"] = P[[f"SB_{e}" for e in En]].values
    run = []
    for pid in d["pid"]:
        rr = []
        for k in range(E.SEEDS_PER_SET):
            w = R[("SB", pid, k)]["win"][E.TPC_MAIN]
            if exclude_first:
                w = w[1:]
            rr.append(w[:, [IDX[e] for e in En]].mean(axis=0))
        run.append(rr)
    d["E_SB_runs"] = np.array(run)                                  # [P, 5 runs, nE]
    d["SE_E_SB"] = d["E_SB_runs"].std(axis=1, ddof=1) / np.sqrt(E.SEEDS_PER_SET)
    return d


def reach_block(WA_M, M_pts, SE_pts, c, sd):
    """box reachability with the matching eligibility (SE <= c/2 for all components)."""
    elig = (SE_pts <= c / 2).all(axis=1)
    mask_full = np.zeros((len(WA_M), len(M_pts)), bool)
    mask_full[:, elig] = MT.box_mask(WA_M, M_pts[elig], c)
    return mask_full, elig


def descriptive_reach(WA, Mn, d, fr):
    c, sd = np.array(fr["c"]), np.array(fr["SD"])
    MA = WA[list(Mn)].values
    mask, elig = reach_block(MA, d["M_SA"], d["SEM_SA"], c, sd)
    n_w = mask.sum(axis=1)
    reach = n_w > 0
    out = {"n_points_all10": int(len(elig)), "n_points_matching_eligible": int(elig.sum()), "fraction_points_unresolved_for_matching": float(1 - elig.mean()),
           "n_RA_windows": int(len(WA)), "n_reachable_windows": int(reach.sum()), "fraction_reachable_box": float(reach.mean()),
           "n_weeks_reachable": int(WA.loc[reach, "week"].nunique()), "n_weeks_RA": int(WA["week"].nunique())}
    out["by_atr_group"] = {str(g): {"n": int((WA["atr_group"] == g).sum()), "fraction_reachable": float(reach[WA["atr_group"].values == g].mean()) if (WA["atr_group"] == g).any() else None} for g in range(4)}
    hr = {}
    for h, g in WA.groupby("hour"):
        hr[str(h)] = {"n": int(len(g)), "fraction_reachable": float(reach[g.index.values].mean())}
    out["by_hour"] = hr
    # binding component: remove one component's caliper at a time
    bind = {}
    for j, m in enumerate(Mn):
        cj = c.copy()
        cj[j] = np.inf
        mk, _ = reach_block(MA, d["M_SA"], d["SEM_SA"], cj, sd)
        bind[m] = float((mk.sum(axis=1) > 0).mean())
    out["fraction_reachable_if_caliper_removed_for"] = bind
    # Euclidean (stricter) sensitivity
    z = (MA[:, None, :] - d["M_SA"][None, elig, :]) / sd
    eu = np.linalg.norm(z, axis=2)
    out["fraction_reachable_euclid_0.5"] = float((eu <= 0.5).any(axis=1).mean())
    # nearest eligible point in Chebyshev distance (units of SD): how far the windows are from the DDM region
    ch = np.abs(z).max(axis=2)
    near = ch.min(axis=1)
    arg = np.abs(z[np.arange(len(MA)), ch.argmin(axis=1), :]).argmax(axis=1)
    out["nearest_chebyshev_distance_SD_units"] = {"min": float(near.min()), "p10": float(np.percentile(near, 10)), "median": float(np.median(near)), "p90": float(np.percentile(near, 90)),
                                                  "max": float(near.max()), "fraction_le_0.5": float((near <= 0.5).mean())}
    out["binding_component_at_nearest_point_counts"] = {m: int((arg == j).sum()) for j, m in enumerate(Mn)}
    # marginal coverage: real window M inside the DDM (eligible points) range
    cov = {}
    for j, m in enumerate(Mn):
        lo, hi = d["M_SA"][elig, j].min(), d["M_SA"][elig, j].max()
        cov[m] = {"ddm_min": float(lo), "ddm_max": float(hi), "real_min": float(MA[:, j].min()), "real_median": float(np.median(MA[:, j])), "real_max": float(MA[:, j].max()),
                  "fraction_real_inside_ddm_range": float(((MA[:, j] >= lo) & (MA[:, j] <= hi)).mean())}
    out["marginal_coverage"] = cov
    # Jaccard of the reachable point sets, SA-based versus SB-based M (A1 gate iii)
    mask_b, elig_b = reach_block(MA, d["M_SB"], d["SEM_SB"], c, sd)
    sa_set, sb_set = set(np.flatnonzero(mask.any(axis=0))), set(np.flatnonzero(mask_b.any(axis=0)))
    union = sa_set | sb_set
    out["jaccard_reachable_points_SA_vs_SB"] = {"n_SA": len(sa_set), "n_SB": len(sb_set), "jaccard": (len(sa_set & sb_set) / len(union)) if union else None}
    return out, mask, elig


def seed_layer(mask, d, EA, sigma, tau, f0, rng, thr_list):
    """point-level seed bootstrap (percentile and parametric t, df=4) of D_j, f_j, g_j with tau and f0 fixed; analytic linear interval of D_j."""
    ok_pts = np.flatnonzero(mask.any(axis=0))
    Rm = mask[:, ok_pts]
    runs = d["E_SB_runs"][ok_pts]
    Eth = d["E_SB"][ok_pts]
    SEth = d["SE_E_SB"][ok_pts]
    wA = np.ones(len(EA))
    ed, n = MT.ddm_reference(Rm, Eth)
    dd, D, f, nr = MT.residual_stats(ed, EA, wA, sigma, tau)
    res = {"D": D, "f": f, "g": f - f0, "n_reach": nr}

    def boot(kind):
        Ds, fs = [], []
        for _ in range(B_SEED):
            if kind == "pct":
                idx = rng.integers(0, E.SEEDS_PER_SET, (len(ok_pts), E.SEEDS_PER_SET))
                Es = runs[np.arange(len(ok_pts))[:, None], idx].mean(axis=1)
            else:
                Es = Eth + SEth * rng.standard_t(4, Eth.shape)
            e2, _ = MT.ddm_reference(Rm, Es)
            _, D_, f_, _ = MT.residual_stats(e2, EA, wA, sigma, tau)
            Ds.append(D_)
            fs.append(f_)
        return np.array(Ds), np.array(fs)
    out = {}
    for kind in ("pct", "t"):
        Ds, fs = boot(kind)
        gs = fs - f0
        out[kind] = {"D_p5": np.percentile(Ds, 5, axis=0), "D_p95": np.percentile(Ds, 95, axis=0), "f_p5": np.percentile(fs, 5, axis=0), "f_p95": np.percentile(fs, 95, axis=0),
                     "g_p5": np.percentile(gs, 5, axis=0), "g_p95": np.percentile(gs, 95, axis=0)}
    # analytic interval for D_j
    okw = n > 0
    a = np.zeros((len(ok_pts), len(sigma)))
    for pi in range(len(ok_pts)):
        ws = np.flatnonzero(Rm[:, pi] & okw)
        a[pi] = (1.0 / n[ws]).sum() / (okw.sum())
    se_D = np.sqrt(((a / sigma) ** 2 * SEth ** 2).sum(axis=0))
    out["analytic_D"] = {"se": se_D, "lo": D - 1.645 * se_D, "hi": D + 1.645 * se_D}
    # reuse description
    use = Rm[okw].sum(axis=0)
    out["reuse"] = {"n_distinct_points_used": int((use > 0).sum()), "share_top1": float(np.sort(use)[::-1][:1].sum() / max(1, Rm[okw].sum())), "share_top5": float(np.sort(use)[::-1][:5].sum() / max(1, Rm[okw].sum())),
                    "mean_seed_variance_over_sigma2": (np.nanmean(np.where(Rm[okw][:, :, None], (SEth ** 2)[None], np.nan) / sigma ** 2, axis=(0, 1))).tolist() if True else None}
    return res, out, dd


def real_layer(mask, d, WA, WB, fr, rng, f0, tau, sensitivity=False):
    """week-block bootstrap of R-A and R-B (B_REAL): tau_j re-estimated each replicate (and f0j re-estimated by cross-fit, K'=50); conditional (tau, f0 fixed) and joint (E_theta also resampled) variants."""
    Mn, En = fr["M"], fr["E"]
    c0, sigma0 = np.array(fr["c"]), np.array(fr["sigma"])
    EA, EB = WA[list(En)].values, WB[list(En)].values
    MA, MB = WA[list(Mn)].values, WB[list(Mn)].values
    mask_BA = MT.box_mask(MB, MA, c0)
    ok_pts = np.flatnonzero(mask.any(axis=0))
    Rm = mask[:, ok_pts]
    Eth = d["E_SB"][ok_pts]
    uwA, invA = np.unique(WA["week"].values, return_inverse=True)
    uwB, invB = np.unique(WB["week"].values, return_inverse=True)
    out = {"fixed": {"D": [], "f_cond": [], "f_rep": [], "g_rep": [], "tau": [], "f0": []}}
    nrep = B_REAL
    for _ in range(nrep):
        mA = np.bincount(rng.integers(0, len(uwA), len(uwA)), minlength=len(uwA))[invA].astype(float)
        mB = np.bincount(rng.integers(0, len(uwB), len(uwB)), minlength=len(uwB))[invB].astype(float)
        dd, ok = MT.dn_real_real(EB, EA, mask_BA, mA, sigma0)
        tau_r = MT.tau_from(dd, ok, mB)
        f0_r, _ = MT.crossfit_f0(dd, ok, mB, WB["week"].values, MT.K_INNER, rng)
        ed, n = MT.ddm_reference(Rm, Eth)
        _, D_, f_cond, _ = MT.residual_stats(ed, EA, mA, sigma0, tau)
        _, _, f_rep, _ = MT.residual_stats(ed, EA, mA, sigma0, tau_r)
        out["fixed"]["D"].append(D_)
        out["fixed"]["f_cond"].append(f_cond)
        out["fixed"]["f_rep"].append(f_rep)
        out["fixed"]["g_rep"].append(f_rep - f0_r)
        out["fixed"]["tau"].append(tau_r)
        out["fixed"]["f0"].append(f0_r)
    return {k: np.array(v) for k, v in out["fixed"].items()}


def summarize(x):
    return {"p5": np.percentile(x, 5, axis=0), "p95": np.percentile(x, 95, axis=0), "median": np.median(x, axis=0)}


def main():
    code = E.code_state()
    fr_all = json.load(open(os.path.join(E.RESULTS, "frozen_rules.json"), encoding="utf-8"))
    fr = fr_all["primary"]
    dec = json.load(open(os.path.join(E.RESULTS, "a1_decisions.json"), encoding="utf-8"))
    PT = pd.read_csv(os.path.join(E.RESULTS, "a1_points.csv"))
    R = pickle.load(open(os.path.join(E.OUT, "a1_stats.pkl"), "rb"))["runs"]
    W = pd.read_csv(os.path.join(E.RESULTS, "b_windows_L1000_bar1.csv"))
    WA, WB = split_est(W)
    d = load_ddm(fr, dec["exclude_first_window"], R, PT)
    reach, mask, elig = descriptive_reach(WA, fr["M"], d, fr)
    out = {"_code": code, "frozen_rules": fr, "reachability": reach}
    cond = reach["fraction_reachable_box"] >= 0.05 and reach["n_reachable_windows"] >= 50 and reach["n_weeks_reachable"] >= 20
    out["residual_analysis_run"] = bool(cond)
    out["residual_analysis_condition"] = "plan section 10: reachable windows >= 5% of R-A, >= 50 windows, >= 20 distinct weeks; and noise floor computable (>= 100 R-B windows with >= 3 reference windows)"
    if cond and fr["noise_floor_computable"]:
        rng = np.random.default_rng(RNG_SEED)
        EA = WA[list(fr["E"])].values
        sigma, tau, f0 = np.array(fr["sigma"]), np.array(fr["tau"]), np.array(fr["f0"])
        res, sl, dd = seed_layer(mask, d, EA, sigma, tau, f0, rng, (0.0, 0.05, 0.10))
        rl = real_layer(mask, d, WA, WB, fr, rng, f0, tau)
        out["residual"] = {"point": res, "seed_layer": sl, "real_layer": {k: summarize(v) for k, v in rl.items()}}
        out["residual"]["D_median_over_windows"] = np.median(dd, axis=0)
    elif cond:
        out["residual_skipped_reason"] = "noise floor not computable"
    else:
        out["residual_skipped_reason"] = "reachability conditions not met (plan section 10): report the reachability and the binding components only"
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(out, open(os.path.join(E.RESULTS, "c_min.json"), "w", encoding="utf-8"), indent=1, default=conv)
    print(json.dumps({k: reach[k] for k in ("n_points_all10", "n_points_matching_eligible", "n_reachable_windows", "fraction_reachable_box", "fraction_reachable_euclid_0.5")}), flush=True)
    print("residual analysis run:", cond, flush=True)


if __name__ == "__main__":
    main()
