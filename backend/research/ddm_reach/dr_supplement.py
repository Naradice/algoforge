"""Supplements to C-min / D. Two clearly separated blocks:
  planned  - descriptive items of plan v0.6 that stages C-min/D did not output (aggregation profile real vs DDM, H1s std ratio, time-of-day record) and the M-space figure;
  post_hoc - analyses added AFTER the reachability result was seen (why the SA/SB reachable-point sets differ, reach without the matching eligibility). They change no decision.
Output: results/supplement.json, results/fig_m_space.png"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_match as MT  # noqa: E402
from dr_a1_stats import IDX  # noqa: E402
from dr_b_rules import split_est  # noqa: E402
from dr_c_min import reach_block  # noqa: E402

BLUE, ORANGE, INK, INK2, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def q(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))} if len(x) else {"n": 0}


def main():
    code = E.code_state()
    fr = json.load(open(os.path.join(E.RESULTS, "frozen_rules.json"), encoding="utf-8"))["primary"]
    PT = pd.read_csv(os.path.join(E.RESULTS, "a1_points.csv"))
    P = PT[PT["all10"]].reset_index(drop=True)
    R = pickle.load(open(os.path.join(E.OUT, "a1_stats.pkl"), "rb"))["runs"]
    W = pd.read_csv(os.path.join(E.RESULTS, "b_windows_L1000_bar1.csv"))
    WA, _ = split_est(W)
    M = fr["M"]
    c, sd = np.array(fr["c"]), np.array(fr["SD"])
    MA = WA[M].values
    st = {}
    for s in ("SA", "SB"):
        Mp, SE = P[[f"{s}_{m}" for m in M]].values, P[[f"{s}_{m}_se" for m in M]].values
        mask, elig = reach_block(MA, Mp, SE, c, sd)
        st[s] = {"M": Mp, "SE": SE, "mask": mask, "elig": elig, "raw": MT.box_mask(MA, Mp, c), "reach": mask.any(axis=1)}
    reach = st["SA"]["reach"]

    planned = {}
    # aggregation profile, per window: DDM (SA runs of all-10 points, TPC 20) and real R-A windows
    ddm = {1: [], 2: [], 4: []}
    for pid in P["pid"]:
        for k in range(E.SEEDS_PER_SET):
            r = R[("SA", pid, k)]
            ddm[1].append(r["win"][E.TPC_MAIN][:, [IDX["acf_1"], IDX["kurtosis"]]])
            for a in (2, 4):
                ddm[a].append(r["agg"][a])
    prof = {}
    for a in (1, 2, 4):
        d = np.vstack(ddm[a])
        cols = ["acf_1", "kurtosis"] if a == 1 else [f"agg{a}_acf_1", f"agg{a}_kurtosis"]
        x = WA[cols].values
        prof[str(a)] = {"ddm_window_median": {"acf_1": float(np.median(d[:, 0])), "kurtosis": float(np.median(d[:, 1]))},
                        "real_RA_window_median": {"acf_1": float(np.median(x[:, 0])), "kurtosis": float(np.median(x[:, 1]))}, "n_ddm_windows": int(len(d)), "n_real_windows": int(len(x))}
    planned["aggregation_profile"] = prof
    # H1s: std ratio (real / matched DDM mean std, SB seeds) over reachable windows
    stdb = P["SB_std"].values
    m = st["SA"]["mask"][reach]
    matched_std = (m @ stdb) / m.sum(axis=1)
    lr = np.log10(WA.loc[reach, "std"].values / matched_std)
    planned["H1s_std_ratio_reachable"] = {"ratio_real_over_ddm": {k: (float(10 ** v) if k != "n" else v) for k, v in q(lr).items()}, "p90_over_p10": float(10 ** (np.percentile(lr, 90) - np.percentile(lr, 10))),
                                          "spearman_real_std_vs_ddm_std": float(pd.Series(WA.loc[reach, "std"].values).corr(pd.Series(matched_std), method="spearman"))}
    # time of day: window counts and M medians by start hour (hours with at least 50 windows)
    planned["by_start_hour_RA"] = {str(h): {"n": int(len(g)), **{k: float(g[k].median()) for k in M}, "fraction_reachable": float(reach[g.index.values].mean())} for h, g in WA.groupby("hour") if len(g) >= 50}
    planned["M_of_windows"] = {"all_RA": {k: q(WA[k]) for k in M}, "reachable": {k: q(WA.loc[reach, k]) for k in M}}

    post = {}
    for s in ("SA", "SB"):
        x = st[s]
        post[s] = {"n_eligible_points": int(x["elig"].sum()), "eligibility_pass_by_component": dict(zip(M, map(float, (x["SE"] <= c / 2).mean(axis=0)))), "n_reachable_windows": int(x["reach"].sum()),
                   "n_reachable_windows_ignoring_eligibility": int(x["raw"].any(axis=1).sum()), "fraction_ignoring_eligibility": float(x["raw"].any(axis=1).mean()),
                   "n_points_reaching_windows_ignoring_eligibility": int(x["raw"].any(axis=0).sum())}
    ea, eb = st["SA"]["elig"], st["SB"]["elig"]
    a, b = st["SA"]["reach"], st["SB"]["reach"]
    ra, rb = st["SA"]["raw"].any(axis=1), st["SB"]["raw"].any(axis=1)
    sa_pts, sb_pts = st["SA"]["raw"].any(axis=0), st["SB"]["raw"].any(axis=0)
    post["overlap"] = {"eligible_points_both": int((ea & eb).sum()), "windows_both": int((a & b).sum()), "windows_jaccard": float((a & b).sum() / (a | b).sum()),
                       "windows_jaccard_ignoring_eligibility": float((ra & rb).sum() / (ra | rb).sum()), "point_jaccard_ignoring_eligibility": float((sa_pts & sb_pts).sum() / (sa_pts | sb_pts).sum())}
    used = np.flatnonzero((ea & st["SA"]["mask"].any(axis=0)) | (eb & st["SB"]["mask"].any(axis=0)))
    post["reaching_eligible_points"] = [{"pid": int(P.pid[i]), "delta": float(P.delta[i]), "wma": int(P.wma[i]),
                                         **{s: {"eligible": bool(st[s]["elig"][i]), "n_windows_if_eligible_ignored": int(st[s]["raw"][:, i].sum()), "n_windows_counted": int(st[s]["mask"][:, i].sum()),
                                                "M": dict(zip(M, map(float, st[s]["M"][i]))), "SE_over_caliper": dict(zip(M, map(float, st[s]["SE"][i] / c)))} for s in ("SA", "SB")}} for i in used]
    # windows beyond the DDM range of one component by more than its caliper: unreachable from any all-10-success point, whatever the eligibility
    beyond = {}
    for s in ("SA", "SB"):
        lo, hi = st[s]["M"].min(axis=0) - c, st[s]["M"].max(axis=0) + c
        out_j = (MA < lo) | (MA > hi)
        beyond[s] = {**{m_: float(out_j[:, j].mean()) for j, m_ in enumerate(M)}, "any_component": float(out_j.any(axis=1).mean()), "volclust_50_upper_bound": float(hi[M.index("volclust_50")])}
    post["fraction_windows_beyond_ddm_range_plus_caliper"] = beyond
    post["windows_per_counted_point_SA"] = {int(P.pid[i]): int(st["SA"]["mask"][:, i].sum()) for i in np.flatnonzero(st["SA"]["mask"].any(axis=0))}
    k = M.index("kurtosis")
    mk = st["SA"]["mask"][reach]
    post["kurtosis_of_reachable"] = {"real_windows": q(WA.loc[reach, "kurtosis"]), "matched_ddm_mean": q((mk @ st["SA"]["M"][:, k]) / mk.sum(axis=1)), "caliper_kurtosis": float(c[k]),
                                     "real_RA_median_kurtosis": float(WA["kurtosis"].median())}
    json.dump({"_code": code, "planned": planned, "post_hoc": post}, open(os.path.join(E.RESULTS, "supplement.json"), "w", encoding="utf-8"), indent=1)

    # figure: pairwise projections of the matching space
    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "axes.unicode_minus": False, "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    pairs = [("acf_1", "volclust_50"), ("acf_1", "kurtosis"), ("volclust_50", "kurtosis")]
    for ax, (xn, yn) in zip(axes, pairs):
        i, j = M.index(xn), M.index(yn)
        ax.set_facecolor(SURFACE)
        ax.scatter(MA[~reach, i], MA[~reach, j], s=9, c=BLUE, alpha=0.35, linewidths=0, label=f"実市場 R-A の窓: 到達不可({int((~reach).sum())})")
        ax.scatter(MA[reach, i], MA[reach, j], s=26, c=BLUE, edgecolors=INK, linewidths=0.9, label=f"実市場 R-A の窓: 到達可能({int(reach.sum())})", zorder=3)
        X = st["SA"]["M"]
        ax.scatter(X[~ea, i], X[~ea, j], s=30, marker="D", facecolors="none", edgecolors=ORANGE, linewidths=1.0, label=f"DDM の点(SA): マッチング不能({int((~ea).sum())})", zorder=4)
        ax.scatter(X[ea, i], X[ea, j], s=30, marker="D", c=ORANGE, edgecolors=SURFACE, linewidths=0.8, label=f"DDM の点(SA): マッチング可能({int(ea.sum())})", zorder=5)
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        if yn == "kurtosis":
            ax.set_yscale("log")
        ax.grid(True, color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("マッチング統計量の空間: 実市場の窓と DDM の格子点(推定期間 R-A、TPC=20、L=1,000)。2 次元の射影で重なっても、3 成分すべての条件を満たすとは限らない", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(os.path.join(E.RESULTS, "fig_m_space.png"), dpi=130, facecolor=SURFACE)
    print(json.dumps({"planned": planned, "post_hoc": post}, ensure_ascii=False)[:5000], flush=True)


if __name__ == "__main__":
    main()
