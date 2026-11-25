"""Stage A1 analysis (plan v0.6 sections 4.3, 7.2, 10): N_grid, DDM-precision eligibility, tail-measure fallback rule, initial-transient rule, SA/SB agreement, TPC sensitivity.
Uses no real-market data. Output: results/a1_analysis.json, results/a1_points.csv, results/a1_decisions.json (the frozen DDM-side decisions)."""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
from dr_a1_stats import IDX  # noqa: E402

COMPS_ALL = ("acf_1", "kurtosis", "Q", "volclust_50")
E_ALL = ("acf_5", "acf_20", "vr20", "vr60", "skew")


def run_means(R, set_name, pid, tpc, names, drop_first):
    """list over runs of the window-mean (rows = runs) of the statistics `names`."""
    rows = []
    for k in range(E.SEEDS_PER_SET):
        r = R.get((set_name, pid, k))
        if r is None:
            continue
        w = r["win"][tpc]
        if drop_first:
            w = w[1:]
        rows.append(w[:, [IDX[n] for n in names]].mean(axis=0))
    return np.array(rows)


def point_mean_se(rm):
    n = len(rm)
    if n < 2:
        return np.full(rm.shape[1], np.nan), np.full(rm.shape[1], np.nan), n
    return rm.mean(axis=0), rm.std(axis=0, ddof=1) / np.sqrt(n), n


def main():
    code = E.code_state()
    st = pickle.load(open(os.path.join(E.OUT, "a1_stats.pkl"), "rb"))
    R = st["runs"]
    metas = [json.loads(line) for line in open(os.path.join(E.A1_DIR, "meta.jsonl"), encoding="utf-8")]
    ok = {(m["set"], m["pid"], m["k"]): m["success"] for m in metas}
    grid = E.grid()
    pts = []
    for pid, d, w in grid:
        n_ok = {s: sum(ok.get((s, pid, k), False) for k in range(E.SEEDS_PER_SET)) for s in ("SA", "SB")}
        pts.append({"pid": pid, "delta": d, "wma": w, "n_ok_SA": n_ok["SA"], "n_ok_SB": n_ok["SB"], "all10": n_ok["SA"] == 5 and n_ok["SB"] == 5})
    P = pd.DataFrame(pts)
    n_grid = int(P["all10"].sum())
    out = {"_code": code, "n_points": len(P), "n_runs_success": int(sum(ok.values())), "n_runs": len(ok), "N_grid": n_grid,
           "success_by_wma": {int(w): float(g["all10"].mean()) for w, g in P.groupby("wma")}, "success_by_delta": {float(d): float(g["all10"].mean()) for d, g in P.groupby("delta")},
           "run_success_rate": float(np.mean(list(ok.values())))}
    if n_grid < 30:
        out["stop"] = "N_grid < 30: plan section 10 stop condition"
        json.dump(out, open(os.path.join(E.RESULTS, "a1_analysis.json"), "w", encoding="utf-8"), indent=1)
        print("STOP: N_grid < 30")
        sys.exit(1)
    out["flag_review_N_grid_lt_60"] = bool(n_grid < 60)
    A = P[P["all10"]]
    # ---- SD_grid and DDM-precision eligibility (all windows, SA; plan 4.3) ------------------------------------------------------------------
    mean_sa, se_sa = {}, {}
    for pid in A["pid"]:
        m, s, _ = point_mean_se(run_means(R, "SA", pid, E.TPC_MAIN, COMPS_ALL, False))
        mean_sa[pid], se_sa[pid] = m, s
    MA = np.array([mean_sa[p] for p in A["pid"]])
    SEA = np.array([se_sa[p] for p in A["pid"]])
    sd_grid = MA.std(axis=0, ddof=1)
    out["SD_grid"] = dict(zip(COMPS_ALL, map(float, sd_grid)))
    elig = SEA <= 0.25 * sd_grid
    out["eligibility_fraction_by_component"] = dict(zip(COMPS_ALL, map(float, elig.mean(axis=0))))
    frac_fail_kurt = float(1 - elig[:, 1].mean())
    tail = "Q" if frac_fail_kurt > 0.5 else "kurtosis"
    out["fallback_rule"] = {"fraction_of_N_grid_points_failing_kurtosis_precision": frac_fail_kurt, "primary_tail_measure": tail}
    comps = ("acf_1", tail, "volclust_50")
    ci = [COMPS_ALL.index(c) for c in comps]
    elig_pts = elig[:, ci].all(axis=1)
    out["n_DDM_precision_eligible_all_windows"] = int(elig_pts.sum())
    elig_pid = list(A["pid"][elig_pts])
    # ---- initial-transient rule (paired within-run differences, SA, eligible points) ----------------------------------------------------------
    def transient(pids):
        res = {c: [] for c in comps}
        for pid in pids:
            d = []
            for k in range(E.SEEDS_PER_SET):
                w = R[("SA", pid, k)]["win"][E.TPC_MAIN][:, [IDX[c] for c in comps]]
                d.append(w[0] - w[1:].mean(axis=0))
            d = np.array(d)
            m, s = d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(len(d))
            for j, c in enumerate(comps):
                res[c].append(abs(m[j]) > 3 * s[j])
        return {c: (float(np.mean(v)) if v else None) for c, v in res.items()}
    tr = transient(elig_pid)
    out["transient"] = {"fraction_points_abs_mean_delta_gt_3SE_eligible": tr, "n_eligible": len(elig_pid), "fraction_all_N_grid_points_for_information": transient(list(A["pid"]))}
    vals = [v for v in tr.values() if v is not None]
    exclude_first = bool(len(vals) > 0 and max(vals) >= 0.25)
    out["transient"]["exclude_first_window"] = exclude_first
    # ---- final point statistics (after the transient decision) ----------------------------------------------------------------------------------
    names_all = comps + E_ALL + ("std", "Q" if tail == "kurtosis" else "kurtosis")      # both tail measures are stored (PB evaluates "kurtosis and the robust tail measure")
    rows = []
    for pid, d, w in grid:
        r = {"pid": pid, "delta": d, "wma": w}
        for s in ("SA", "SB"):
            if (A["pid"] == pid).any():
                m, se, n = point_mean_se(run_means(R, s, pid, E.TPC_MAIN, names_all, exclude_first))
                for j, nm in enumerate(names_all):
                    r[f"{s}_{nm}"], r[f"{s}_{nm}_se"] = float(m[j]), float(se[j])
        r["all10"] = bool((A["pid"] == pid).any())
        rows.append(r)
    PT = pd.DataFrame(rows)
    PT.to_csv(os.path.join(E.RESULTS, "a1_points.csv"), index=False)
    # DDM-precision eligibility, final windows (SD_grid fixed from the all-window SA means above)
    fin_el = []
    for _, r in PT.iterrows():
        fin_el.append(bool(r["all10"] and all(r[f"SA_{c}_se"] <= 0.25 * sd_grid[COMPS_ALL.index(c)] for c in comps)))
    PT["ddm_precision_eligible"] = fin_el
    PT.to_csv(os.path.join(E.RESULTS, "a1_points.csv"), index=False)
    out["n_DDM_precision_eligible_final"] = int(PT["ddm_precision_eligible"].sum())
    # ---- SA/SB agreement ------------------------------------------------------------------------------------------------------------------------
    sub = PT[PT["all10"]]
    agree = np.ones(len(sub), bool)
    comp_agree = {}
    for c in comps:
        z = np.abs(sub[f"SA_{c}"] - sub[f"SB_{c}"]) <= 2 * np.sqrt(sub[f"SA_{c}_se"] ** 2 + sub[f"SB_{c}_se"] ** 2)
        comp_agree[c] = float(z.mean())
        agree &= z.values
    out["SA_SB_agreement"] = {"fraction_all_components": float(agree.mean()), "by_component": comp_agree, "criterion": "|SA-SB| <= 2*sqrt(SE_SA^2+SE_SB^2) per component, points with all 10 seeds successful", "gate_ge_0.80": bool(agree.mean() >= 0.8)}
    # ---- ranges, correlations (SA point means, final windows) ---------------------------------------------------------------------------------
    out["M_ranges_SA"] = {c: {"min": float(sub[f"SA_{c}"].min()), "median": float(sub[f"SA_{c}"].median()), "max": float(sub[f"SA_{c}"].max())} for c in names_all}
    cor = sub[[f"SA_{c}" for c in names_all]].corr(method="spearman")
    out["spearman_SA"] = {a: {b: float(cor.loc[f"SA_{a}", f"SA_{b}"]) for b in names_all} for a in names_all}
    # ---- TPC sensitivity (H2', exploratory) and aggregation profile -------------------------------------------------------------------------------
    tp = {}
    for tpc in E.TPCS:
        arr = []
        for pid in sub["pid"]:
            m, _, _ = point_mean_se(run_means(R, "SA", pid, tpc, comps, exclude_first and tpc == E.TPC_MAIN and False))
            arr.append(m)
        arr = np.array(arr)
        tp[tpc] = arr
    out["tpc_sensitivity_SA"] = {str(t): {c: {"median": float(np.nanmedian(tp[t][:, j])), "min": float(np.nanmin(tp[t][:, j])), "max": float(np.nanmax(tp[t][:, j]))} for j, c in enumerate(comps)} for t in E.TPCS}
    out["tpc_sensitivity_note"] = "all available windows (no first-window exclusion) for every TPC, SA seeds"
    agg = {}
    for a in (2, 4):
        v = []
        for pid in sub["pid"]:
            rr = np.array([R[("SA", pid, k)]["agg"][a].mean(axis=0) for k in range(E.SEEDS_PER_SET)]).mean(axis=0)
            v.append(rr)
        v = np.array(v)
        agg[str(a)] = {"acf_1_median": float(np.median(v[:, 0])), "kurtosis_median": float(np.median(v[:, 1]))}
    out["aggregation_profile_SA_median"] = agg
    out["decisions"] = {"primary_tail_measure": tail, "M_components": list(comps), "exclude_first_window": exclude_first, "SD_grid": out["SD_grid"], "N_grid": n_grid}
    json.dump(out, open(os.path.join(E.RESULTS, "a1_analysis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(out["decisions"], open(os.path.join(E.RESULTS, "a1_decisions.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["decisions"]), flush=True)


if __name__ == "__main__":
    main()
