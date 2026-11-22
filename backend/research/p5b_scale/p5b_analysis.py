"""P5-B first-stage analyses A-E (plan v0.2 @ 5277e17). Stored data only; descriptive; no p-values. Output: OUT/analysis.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5b_env as E  # noqa: E402
import p5b_lib as L5  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402
import p1_lib as P1  # noqa: E402

HZ = L5.HZ
PERS = ("estimation", "validation", "final")
ACF_KEYS = ("acf_1", "acf_5", "acf_20", "kurtosis", "volclust_50", "skewness")


def period_end(per):
    return pd.Timestamp(P1.PERIODS[per][1]) + pd.Timedelta(days=1)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(stats.spearmanr(x[ok], y[ok])[0]) if ok.sum() >= 4 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0 else None


def slope(y, x=None):
    y = np.asarray(y, float)
    x = np.arange(len(y)) if x is None else np.asarray(x, float)
    ok = np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def main():
    code = E.code_state()
    g = json.load(open(os.path.join(E.OUT, "gates.json"), encoding="utf-8"))
    assert g["all_pass"], "gates must pass first"
    data = P.V.Data()
    raw = data.raw
    mp = L5.load_mapping()
    atlas = L5.load_atlas()
    win = L5.Windows(raw.close)
    atr = raw.atr()
    est_end = period_end("estimation")
    wbs = {per: K.WBoot(np.concatenate([data.cells[(per, h)]["meta"]["week"] for h in HZ]), L5.B_BOOT, f"p5b{per}") for per in PERS}
    conds = {c: L5.load_cond(c) for c in L5.CONDS}
    out = {"_code": code, "A": {}, "B": {}, "C": {}, "D": {}, "E": {}, "notes": {}}

    # per-bucket real-side quantities (estimation) -------------------------------------------------------------------------------------------
    j1 = data.cells[("estimation", 1)]["j"]
    b1 = data.cells[("estimation", 1)]["b"]
    okw = L5.valid_windows(raw, j1, est_end)
    var_all = np.full(len(j1), np.nan)
    var_all[okw] = win.var(j1[okw] * L5.BLOCK)
    v_primary, v_all, real_std, sig2c, sig2mix, atr_med = {}, {}, {}, {}, {}, {}
    for b in range(20):
        mb = mp["buckets"][b]
        bl = np.array(mb["used_blocks"])
        vp = win.var(bl * L5.BLOCK)
        v_primary[b] = float(vp.mean())
        v_all[b] = float(np.nanmean(var_all[b1 == b]))
        real_std[b] = float(mb["real_stats"]["std"])
        rows = [L5.atlas_row(atlas, dl, int(wm)).iloc[0] for dl, wm in mb["candidates"]]
        sig2c[b] = [L5.sigma2_atlas(r) for r in rows]
        sig2mix[b] = float(np.mean(sig2c[b]))
        atr_med[b] = float(np.median(atr[j1[b1 == b]]))
        top, sl, valid = L5.replicate_selection(pd.Series(mb["real_stats"]), atlas)
        rs = mb["real_stats"]
        lr = np.log(valid["std"].values / rs["std"])
        cand = []
        for (dl, wm), r in zip(mb["candidates"], rows):
            cand.append({"delta": dl, "wma": wm, "std": float(r["std"]), "std_across_seeds": float(r["std_std_across_seeds"]), "sigma2": L5.sigma2_atlas(r),
                         "log_std_ratio_vs_real_std": float(np.log(r["std"] / rs["std"])), "log_ratio_vs_sqrtV": float(np.log(np.sqrt(L5.sigma2_atlas(r)) / np.sqrt(v_primary[b]))),
                         "acf_1": float(r["acf_1"]), "acf_5": float(r["acf_5"]), "acf_20": float(r["acf_20"]), "kurtosis": float(r["kurtosis"]), "volclust_50": float(r["volclust_50"]),
                         "skewness": float(r["skewness"]), "success_rate": float(r["success_rate"]), "n_success": int(r["n_success"])})
        ok_o = np.isfinite(var_all) & (b1 == b)
        out["A"][str(b)] = {
            "real_stats": rs, "n_windows": mb["n_windows_used"], "best_std_ratio": mb["best_std_ratio"],
            "V_real_M1_primary": v_primary[b], "V_real_M1_all_origins": v_all[b], "sqrtV_primary_over_real_std": float(np.sqrt(v_primary[b]) / real_std[b]),
            "sqrtV_all_over_real_std": float(np.sqrt(v_all[b]) / real_std[b]), "sqrtV_all_over_sqrtV_primary": float(np.sqrt(v_all[b] / v_primary[b])),
            "window_var_se_primary": float(win.var(np.array(mb["used_blocks"]) * L5.BLOCK).std(ddof=1) / np.sqrt(len(mb["used_blocks"]))),
            "candidates": cand, "mean_log_std_ratio_vs_real_std": float(np.mean([c["log_std_ratio_vs_real_std"] for c in cand])),
            "sigma_atlas_mix": float(np.sqrt(sig2mix[b])), "log_sigma_mix_over_sqrtV": float(np.log(np.sqrt(sig2mix[b]) / np.sqrt(v_primary[b]))),
            "log_sigma_mix_over_real_std": float(np.log(np.sqrt(sig2mix[b]) / real_std[b])),
            "atlas_coverage": {"n_valid_points": int(len(valid)), "atlas_std_min": float(valid["std"].min()), "atlas_std_max": float(valid["std"].max()),
                               "real_std_below_atlas_min": bool(real_std[b] < valid["std"].min()), "real_std_above_atlas_max": bool(real_std[b] > valid["std"].max()),
                               "n_within_10pct": int((np.abs(lr) <= np.log(1.10)).sum()), "n_within_25pct": int((np.abs(lr) <= np.log(1.25)).sum()),
                               "shortlist_std_min": float(sl["std"].min()), "shortlist_std_max": float(sl["std"].max()),
                               "shortlist_log_ratio_range": [float(np.log(sl["std"].min() / rs["std"])), float(np.log(sl["std"].max() / rs["std"]))]},
            "M5": {"atr_median": atr_med[b], "sqrtV_over_atr_median": float(np.sqrt(v_primary[b]) / atr_med[b]), "real_std_over_atr_median": real_std[b] / atr_med[b],
                   "within_bucket_spearman_atr_vs_window_std": spearman(atr[j1[ok_o]], np.sqrt(var_all[ok_o]))}}
    bidx = np.arange(20)
    out["A"]["across_buckets"] = {
        "loglog_slope_sqrtV_vs_atr_median": float(np.polyfit(np.log([atr_med[b] for b in range(20)]), np.log([np.sqrt(v_primary[b]) for b in range(20)]), 1)[0]),
        "spearman_bucket_vs_log_sigma_mix_over_sqrtV_0_16": spearman(bidx[:17], [out["A"][str(b)]["log_sigma_mix_over_sqrtV"] for b in range(17)]),
        "spearman_bucket_vs_mean_log_std_ratio_0_16": spearman(bidx[:17], [out["A"][str(b)]["mean_log_std_ratio_vs_real_std"] for b in range(17)])}

    # B: real-side variance ratios ----------------------------------------------------------------------------------------------------------
    wb_e = wbs["estimation"]
    for b in range(20):
        rec = {}
        for h in HZ:
            c = data.cells[("estimation", h)]
            m = c["b"] == b
            y, wk = c["y"][m], c["meta"]["week"][m]
            vy = float(y.var())
            rep = L5.var_rep(wb_e, y, wk)
            n = L5.BLOCK * h
            rec[f"h{h}"] = {"var_y": vy, "sd_y": float(np.sqrt(vy)), "VR_real": vy / (n * v_primary[b]), "VR_real_ci": [x / (n * v_primary[b]) for x in L5.pct(rep)] if L5.pct(rep) else None,
                            "VR_real_all_origins_windows": vy / (n * v_all[b]), "n": int(m.sum())}
        out["B"][str(b)] = rec
    out["B"]["groups"] = {}
    for per in PERS:
        pe = period_end(per)
        cj = data.cells[(per, 1)]
        okp = L5.valid_windows(raw, cj["j"], pe)
        vwin = np.full(len(cj["j"]), np.nan)
        vwin[okp] = win.var(cj["j"][okp] * L5.BLOCK)
        wb = wbs[per]
        for gi in range(5):
            gm = (cj["b"] // 4) == gi
            vg = float(np.nanmean(vwin[gm]))
            rec = {"V_real_M1_group_windows": vg, "n_windows": int((gm & okp).sum()), "n_origins": int(gm.sum())}
            for h in HZ:
                c = data.cells[(per, h)]
                m = (c["b"] // 4) == gi
                y, wk = c["y"][m], c["meta"]["week"][m]
                vy = float(y.var())
                rep = L5.var_rep(wb, y, wk)
                n = L5.BLOCK * h
                ci = L5.pct(rep)
                rec[f"h{h}"] = {"var_y": vy, "VR_real": vy / (n * vg), "VR_real_ci": [x / (n * vg) for x in ci] if ci else None, "n": int(m.sum()), "weeks": int(len(np.unique(wk)))}
            out["B"]["groups"][f"{per}|{gi}"] = rec

    # C and D: DDM-side statistics, shock-condition difference, decomposition, candidate coverage (E) -------------------------------------------------
    for b in range(20):
        recC, recD, recE = {}, {}, {}
        for h in HZ:
            c = data.cells[("estimation", h)]
            m = c["b"] == b
            y, wk = c["y"][m], c["meta"]["week"][m]
            vy = float(y.var())
            sd_e = float(np.sqrt(vy))
            w_e = float(np.subtract(*np.quantile(y, [0.95, 0.05])))
            s0 = L5.cand_stats(conds["C0s"][0], conds["C0s"][1], b, h)
            s3 = L5.cand_stats(conds["C3"][0], conds["C3"][1], b, h)
            n = L5.BLOCK * h
            vr_c = {cond: [vc / (n * s2) for vc, s2 in zip(st["var_c"], sig2c[b])] for cond, st in (("C0s", s0), ("C3", s3))}
            seed_ratio = [a / bb for a, bb in zip(s0["seed_sd"], s3["seed_sd"])]
            recC[f"h{h}"] = {"sd_E1a": sd_e, "w90_E1a": w_e, "C0s": s0, "C3": s3,
                             "sd_ratio_C0s_over_C3": s0["mix_sd"] / s3["mix_sd"], "sd_ratio_seed_range": [min(seed_ratio), max(seed_ratio)],
                             "w90_ratio_C0s_over_C3": s0["mix_w90"] / s3["mix_w90"],
                             "sd_ratio_C0s_over_E1a": s0["mix_sd"] / sd_e, "sd_ratio_C3_over_E1a": s3["mix_sd"] / sd_e,
                             "w90_ratio_C0s_over_E1a": s0["mix_w90"] / w_e, "w90_ratio_C3_over_E1a": s3["mix_w90"] / w_e,
                             "VR_candidate": vr_c, "VR_mix_C3": s3["mix_sd"] ** 2 / (n * sig2mix[b]), "VR_mix_C0s_with_atlas_sigma": s0["mix_sd"] ** 2 / (n * sig2mix[b]),
                             "mean_check": {cond: {"max_abs_mean_over_sd_candidates": float(np.max(np.abs(np.array(st["mean_c"]) / np.array(st["sd_c"])))),
                                                   "sd_of_candidate_means_over_mix_sd": float(np.std(st["mean_c"]) / st["mix_sd"])} for cond, st in (("C0s", s0), ("C3", s3))}}
            d = L5.decompose(s0["mix_sd"], s3["mix_sd"], vy, v_primary[b], sig2mix[b], h)
            rep = L5.var_rep(wb_e, y, wk)
            tot_rep = np.log(s0["mix_sd"]) - 0.5 * np.log(rep)
            recD[f"h{h}"] = {**{k: float(v) for k, v in d.items()}, "total_ci": L5.pct(tot_rep), "M3_ci": L5.pct(tot_rep - d["M1"] - d["M2"]),
                             "A_config_sensitivity": {str(f): {"M1": float(d["M1"] + np.log(f)), "M3": float(d["M3"] - np.log(f))} for f in (0.8, 0.9, 1.1, 1.25)},
                             "M1_uncertainty_components": {"window_var_se_rel": out["A"][str(b)]["window_var_se_primary"] / v_primary[b] / 2,
                                                           "atlas_sigma_se_rel": float(np.mean([r["std_across"] for r in [{"std_across": cd["std_across_seeds"] / cd["std"]} for cd in out["A"][str(b)]["candidates"]]]) / np.sqrt(5))},
                             "w90": {"log_C0s_over_E1a": float(np.log(s0["mix_w90"] / w_e)), "log_C3_over_E1a": float(np.log(s3["mix_w90"] / w_e)), "log_C0s_over_C3": float(np.log(s0["mix_w90"] / s3["mix_w90"]))}}
            wrep = wb_e.quantiles(y, wk, [0.05, 0.95])
            wd = wrep[:, 1] - wrep[:, 0]
            recD[f"h{h}"]["w90"]["log_C0s_over_E1a_ci"] = L5.pct(np.log(s0["mix_w90"]) - np.log(wd))
            recD[f"h{h}"]["w90"]["log_C3_over_E1a_ci"] = L5.pct(np.log(s3["mix_w90"]) - np.log(wd))
            for cond, st in (("C0s", s0), ("C3", s3)):
                ratios = [x / w_e for x in st["w90_c"]]
                recE.setdefault(cond, {})[f"h{h}"] = {"cand_ratio_to_E1a": ratios, "min": min(ratios), "max": max(ratios), "n_wider_than_E1a": int(sum(r > 1 for r in ratios)),
                                                       "mix_ratio": st["mix_w90"] / w_e, "E1a_inside_candidate_range": bool(min(ratios) <= 1 <= max(ratios))}
        out["C"][str(b)], out["D"][str(b)] = recC, recD
        out["E"][str(b)] = recE
    # across-bucket summaries -----------------------------------------------------------------------------------------------------------------
    out["D"]["across_buckets"] = {}
    for h in HZ:
        terms = {k: [out["D"][str(b)][f"h{h}"][k] for b in range(20)] for k in ("M1", "M3", "M2", "total")}
        sc = {cond: [out["C"][str(b)][f"h{h}"][f"sd_ratio_{cond}_over_E1a"] for b in range(20)] for cond in ("C0s", "C3")}
        wr = {cond: [out["C"][str(b)][f"h{h}"][f"w90_ratio_{cond}_over_E1a"] for b in range(20)] for cond in ("C0s", "C3")}
        out["D"]["across_buckets"][f"h{h}"] = {
            "ols_slope_vs_bucket_0_16": {k: slope(v[:17]) for k, v in terms.items()},
            "slope_share_of_total_0_16": {k: slope(terms[k][:17]) / slope(terms["total"][:17]) for k in ("M1", "M3", "M2")},
            "spearman_vs_bucket_0_16": {k: spearman(bidx[:17], v[:17]) for k, v in terms.items()},
            "spearman_vs_bucket_0_19": {k: spearman(bidx, v) for k, v in terms.items()},
            "sd_ratio_to_E1a_spearman_0_16": {cond: spearman(bidx[:17], v[:17]) for cond, v in sc.items()},
            "w90_ratio_to_E1a_spearman_0_16": {cond: spearman(bidx[:17], v[:17]) for cond, v in wr.items()},
            "shock_condition_sd_ratio_spearman_0_16": spearman(bidx[:17], [out["C"][str(b)][f"h{h}"]["sd_ratio_C0s_over_C3"] for b in range(17)]),
            "shock_condition_sd_ratio_min_max": [min(out["C"][str(b)][f"h{h}"]["sd_ratio_C0s_over_C3"] for b in range(20)), max(out["C"][str(b)][f"h{h}"]["sd_ratio_C0s_over_C3"] for b in range(20))],
            "VR_real_spearman_vs_bucket_0_16": spearman(bidx[:17], [out["B"][str(b)][f"h{h}"]["VR_real"] for b in range(17)]),
            "VR_C3_spearman_vs_bucket_0_16": spearman(bidx[:17], [out["C"][str(b)][f"h{h}"]["VR_mix_C3"] for b in range(17)])}
    out["notes"] = {"A_config": "sigma_atlas (atlas std, no shock, 1000 candles) is used as the proxy of the ensemble-configuration M1 std; sigma^2_atlas,mix is the within-candidate "
                                "(equal-weight) mixture proxy without between-candidate mean variance",
                    "windows": "V_real,M1 primary = mean variance of the 300 mapping windows (rows [60j, 60j+1000]); variant = all estimation origins with valid windows",
                    "h24": "VR_real for h12/h24 normalises by the 1000-row forward window variance (not the same time span)"}
    json.dump(K.to_json(out), open(os.path.join(E.OUT, "analysis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
