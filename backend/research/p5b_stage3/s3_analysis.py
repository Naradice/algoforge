"""P5-B stage 3 analyses V0, A3, B3, C3 (plan v0.3 @ 496ea6d). Stored data only; descriptive; no p-values. Output: results/analysis3.json and OUT/pathstats.npz"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_env as E  # noqa: E402
import s3_lib as L  # noqa: E402
import p5_lib as K  # noqa: E402

KEYS = ("std", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50")
THRESH = {"n_obs_within": 28, "spearman": 0.90, "median_bias": 0.10}


def sha(path):
    return L.S2.sha256_file(path)


def pstat(closes):
    s = L.S2.path_stats(closes)
    return [s["atlas"][k] for k in KEYS] + [s["vr_win"]["60"], s["var1000"], s["kurt_full"], *s["pos_std"]]


def path_table(cond, q, ex):
    f = os.path.join(E.OUT, f"ps_{cond}_q{q}.npy")
    if os.path.exists(f):
        return np.load(f)
    closes = np.load(os.path.join(E.S2_OUT, f"closes_{cond}_q{q}.npy"))
    arr = np.array(list(ex.map(pstat, list(closes), chunksize=16)))
    np.save(f, arr)
    return arr


COLS = {k: i for i, k in enumerate(KEYS + ("vr60", "var1000", "kurt_full", "pos0", "pos1", "pos2", "pos3"))}


def band_in(x, band):
    return (x >= band[0]) & (x <= band[1])


def main():
    code = E.code_state()
    os.makedirs(E.OUT, exist_ok=True)
    bands = json.load(open(E.BAND_FILE, encoding="utf-8"))
    mp = E.load_mapping()
    atlas = pd.read_csv(E.ATLAS)
    valid = L.valid_points(atlas)
    out = {"_code": code, "gates": {}, "hashes": {"atlas_csv": sha(E.ATLAS), "mapping_json": sha(os.path.join(E.RUN_DIR, "ddm_mapping.json")), "bands_json": sha(E.BAND_FILE)}}
    # ---- gate 1: input identity, stage-2 values --------------------------------------------------------------------------------------------
    sha_ok = True
    for cond in E.CONDS:
        man = json.load(open(os.path.join(E.S2_RESULTS, f"regen_{cond}.json"), encoding="utf-8"))["sha256"]
        for q in E.BUCKETS4:
            nm = f"closes_{cond}_q{q}.npy"
            sha_ok &= sha(os.path.join(E.S2_OUT, nm)) == man[nm]
    out["gates"]["1a_stage2_closes_sha256"] = bool(sha_ok)
    assert sha_ok
    a2 = json.load(open(os.path.join(E.S2_RESULTS, "analysis2.json"), encoding="utf-8"))
    # ---- per-path statistics (stage-2 definitions) -------------------------------------------------------------------------------------------
    T = {}
    with ProcessPoolExecutor(max_workers=E.WORKERS) as ex:
        for cond in E.CONDS:
            for q in E.BUCKETS4:
                T[(cond, q)] = path_table(cond, q, ex)
    worst = 0.0
    for q in E.BUCKETS4:
        for c in range(E.N_CAND):
            m = (np.arange(1000) % 8) == c
            ref = a2["cells"][str(q)]["candidates"][c]["C3"]
            t = T[("C3", q)][m]
            worst = max(worst, abs(t[:, COLS["acf_1"]].mean() - ref["atlas_stats_mean"]["acf_1"]), abs(t[:, COLS["vr60"]].mean() - ref["vr_win_mean"]["60"]),
                        abs(t[:, COLS["var1000"]].mean() / ref["V_ens_M1"] - 1))
    out["gates"]["1b_stage2_values_reproduced_max_abs_diff"] = float(worst)
    assert worst < 1e-9, worst
    # ---- gate 2: selection reproduction (all 20 buckets) -------------------------------------------------------------------------------------
    sel_ok, nvalid = True, int(len(valid))
    sel = {}
    for b in mp["buckets"]:
        q = b["q"]
        top, sl, v = L.L5.replicate_selection(pd.Series(b["real_stats"]), atlas)
        keys = [L.key_of(r.delta, r.wma) for r in top.itertuples()]
        sel_ok &= keys == [L.key_of(d, w) for d, w in b["candidates"]]
        sel[q] = (top, sl)
    out["gates"]["2_selection_reproduced_20_buckets"] = bool(sel_ok)
    out["gates"]["2_n_valid_atlas_points"] = nvalid
    assert sel_ok
    # ---- V0 -----------------------------------------------------------------------------------------------------------------------------------
    v0 = {"observations": {}, "supplement": {}}
    cond_obs = {}
    for cond in E.CONDS:
        obs = []
        for q in E.BUCKETS4:
            t = T[(cond, q)]
            cands = mp["buckets"][q]["candidates"]
            for c in range(E.N_CAND):
                tt = t[(np.arange(1000) % 8) == c]
                a1, a5, a20 = (tt[:, COLS[k]].mean() for k in ("acf_1", "acf_5", "acf_20"))
                px = float(L.vr_proxy(a1, a5, a20))
                vw = float(tt[:, COLS["vr60"]].mean())
                per_path = L.vr_proxy(tt[:, COLS["acf_1"]], tt[:, COLS["acf_5"]], tt[:, COLS["acf_20"]]) / tt[:, COLS["vr60"]]
                obs.append({"q": q, "delta": cands[c][0], "wma": cands[c][1], "vrproxy": px, "vrwin": vw, "e": float(np.log(px / vw)), "median_per_path_log_ratio": float(np.median(np.log(per_path)))})
        bk = []
        for q in E.BUCKETS4:
            t = T[(cond, q)]
            px = float(L.vr_proxy(*(t[:, COLS[k]].mean() for k in ("acf_1", "acf_5", "acf_20"))))
            vw = float(t[:, COLS["vr60"]].mean())
            bk.append({"q": q, "vrproxy": px, "vrwin": vw, "e": float(np.log(px / vw))})
        cond_obs[cond] = (obs, bk)

    def crit(obs, bk):
        e = np.array([o["e"] for o in obs])
        rho = float(sps.spearmanr([o["vrproxy"] for o in obs], [o["vrwin"] for o in obs])[0])
        return {"n_obs": len(obs), "n_within_25pct": int(np.sum(np.abs(e) <= E.LOG125)), "n_proxy_nonpositive_e_undefined_counted_outside": int(np.sum(~np.isfinite(e))), "spearman": rho, "median_e": float(np.median(e)),
                "bucket_e": {str(b["q"]): b["e"] for b in bk}, "buckets_within": int(np.sum([abs(b["e"]) <= E.LOG125 for b in bk]))}
    real_bk = []
    for q in E.BUCKETS4:
        r = bands["buckets"][str(q)]
        real_bk.append({"q": q, "vrproxy": r["vrproxy_from_mean_acf"], "vrwin": r["vr_win60_mean"], "e": float(np.log(r["vrproxy_from_mean_acf"] / r["vr_win60_mean"]))})
    c3 = crit(*cond_obs["C3"])
    pass_flags = {"a_28_of_32_within_25pct": c3["n_within_25pct"] >= THRESH["n_obs_within"], "b_spearman_ge_0.90": c3["spearman"] >= THRESH["spearman"],
                  "c_median_e_within_0.10": abs(c3["median_e"]) <= THRESH["median_bias"], "d_ddm_4_buckets_within_25pct": c3["buckets_within"] == 4,
                  "e_real_4_buckets_within_25pct": all(abs(b["e"]) <= E.LOG125 for b in real_bk)}
    v0_pass = all(pass_flags.values())
    # supplement: unique (delta, wma) pooled over buckets (paths of the same candidate from different buckets pooled)
    uniq = {}
    for cond in ("C3",):
        for q in E.BUCKETS4:
            t = T[(cond, q)]
            for c, (dl, wm) in enumerate(mp["buckets"][q]["candidates"]):
                uniq.setdefault(L.key_of(dl, wm), []).append(t[(np.arange(1000) % 8) == c])
    ux = []
    for k, lst in uniq.items():
        tt = np.vstack(lst)
        px = float(L.vr_proxy(*(tt[:, COLS[n]].mean() for n in ("acf_1", "acf_5", "acf_20"))))
        vw = float(tt[:, COLS["vr60"]].mean())
        ux.append((k, px, vw, float(np.log(px / vw))))
    ue = np.array([u[3] for u in ux])
    v0["supplement"] = {"n_observations": 32, "n_unique_delta_wma": len(ux), "n_duplicate_observations": 32 - len(ux),
                        "unique_spearman": float(sps.spearmanr([u[1] for u in ux], [u[2] for u in ux])[0]), "unique_n_within_25pct": int(np.sum(np.abs(ue) <= E.LOG125)),
                        "unique_fraction_within_25pct": float(np.mean(np.abs(ue) <= E.LOG125))}
    v0.update({"criteria": THRESH, "C3": c3, "pass_flags": pass_flags, "pass": bool(v0_pass), "real_buckets": real_bk, "C3_observations": cond_obs["C3"][0], "C3_buckets": cond_obs["C3"][1],
               "C0s_reference": {"summary": crit(*cond_obs["C0s"]), "observations": cond_obs["C0s"][0]}})
    out["V0"] = v0
    print("V0 pass:", v0_pass, pass_flags, flush=True)
    # ---- A3 / B3 (all 20 buckets) ---------------------------------------------------------------------------------------------------------------
    A, B = {}, {}
    atlas_proxy = pd.Series(L.vr_proxy(valid["acf_1"].values, valid["acf_5"].values, valid["acf_20"].values), index=valid.index)
    for b in mp["buckets"]:
        q = b["q"]
        bd = bands["buckets"][str(q)]
        bb = bd["bands"]
        real = pd.Series(b["real_stats"])
        sd_in, pc_in = band_in(valid["std"], bb["std_sd"]), band_in(valid["std"], bb["std_pct25"])
        a1_in = band_in(valid["acf_1"], bb["acf_1_sd"])
        rec = {"n_valid": nvalid, "n_std_sd": int(sd_in.sum()), "n_std_pct25": int(pc_in.sum()), "n_acf1_band_only": int(a1_in.sum()), "n_std_sd_and_acf1": int((sd_in & a1_in).sum()),
               "n_std_pct25_and_acf1": int((pc_in & a1_in).sum()), "min_acf1_dist_all_valid": float((valid["acf_1"] - real["acf_1"]).abs().min()),
               "min_acf1_dist_in_std_pct25": float((valid.loc[pc_in, "acf_1"] - real["acf_1"]).abs().min()) if pc_in.any() else None,
               "acf1_band": bb["acf_1_sd"], "real_acf_1": real["acf_1"], "acf1_window_sd": bd["window_sd"]["acf_1"]}
        if v0_pass:
            vr_in = band_in(atlas_proxy, bb["vrproxy_sd"])
            rec.update({"n_vrproxy_band_only": int(vr_in.sum()), "n_std_sd_acf1_vr": int((sd_in & a1_in & vr_in).sum()), "n_std_pct25_acf1_vr": int((pc_in & a1_in & vr_in).sum()),
                        "n_std_pct25_vr": int((pc_in & vr_in).sum())})
        A[q] = rec
        # B3: selection
        top, sl = sel[q]
        sl = sl.copy()
        sl["acf1_dist"] = (sl["acf_1"] - real["acf_1"]).abs()
        top = top.copy()
        top["acf1_dist"] = (top["acf_1"] - real["acf_1"]).abs()
        top["std_ratio"] = top["std"] / real["std"]
        rank_in_sl = [int((sl["acf1_dist"] < d).sum() + 1) for d in top["acf1_dist"]]
        pcv = valid[pc_in].copy()
        pcv["acf1_dist"] = (pcv["acf_1"] - real["acf_1"]).abs()
        rank_in_std = [int((pcv["acf1_dist"] < d).sum() + 1) for d in top["acf1_dist"]]
        cf, cfsl = L.cf1_selection(real, atlas)
        cf = cf.copy()
        cf["std_ratio"] = cf["std"] / real["std"]
        cf["acf1_dist"] = (cf["acf_1"] - real["acf_1"]).abs()
        cur_keys = {L.key_of(r.delta, r.wma) for r in top.itertuples()}
        cf_keys = {L.key_of(r.delta, r.wma) for r in cf.itertuples()}
        rb = {"selected": [{"delta": float(r.delta), "wma": int(r.wma), "std_ratio": float(r.std_ratio), "acf_1": float(r.acf_1), "acf_5": float(r.acf_5), "acf1_dist": float(r.acf1_dist)}
                           for r in top.itertuples()],
              "real": {"std": float(real["std"]), "acf_1": float(real["acf_1"]), "acf_5": float(real["acf_5"])},
              "sel_acf1_minus_real_mean": float((top["acf_1"] - real["acf_1"]).mean()), "sel_acf5_minus_real_mean": float((top["acf_5"] - real["acf_5"]).mean()),
              "rank_acf1_in_shortlist30": rank_in_sl, "n_std_pct25_points": int(pc_in.sum()), "rank_acf1_in_std_pct25": rank_in_std,
              "shortlist_acf1_band_count": int(band_in(sl["acf_1"], bb["acf_1_sd"]).sum()), "selected_acf1_band_count": int(band_in(top["acf_1"], bb["acf_1_sd"]).sum()),
              "shortlist_acf1_dist": {"min": float(sl["acf1_dist"].min()), "median": float(sl["acf1_dist"].median()), "max": float(sl["acf1_dist"].max())},
              "valid_acf1_dist": {"min": float((valid["acf_1"] - real["acf_1"]).abs().min()), "median": float((valid["acf_1"] - real["acf_1"]).abs().median()),
                                  "max": float((valid["acf_1"] - real["acf_1"]).abs().max())},
              "shortlist_std_ratio_range": [float((sl["std"] / real["std"]).min()), float((sl["std"] / real["std"]).max())],
              "CF1": {"selected": [{"delta": float(r.delta), "wma": int(r.wma), "std_ratio": float(r.std_ratio), "acf_1": float(r.acf_1), "acf_5": float(r.acf_5), "acf1_dist": float(r.acf1_dist)}
                                   for r in cf.itertuples()], "overlap_with_current": int(len(cur_keys & cf_keys)),
                      "median_std_ratio": float(cf["std_ratio"].median()), "max_abs_log_std_ratio": float(np.abs(np.log(cf["std_ratio"])).max()), "mean_acf1_dist": float(cf["acf1_dist"].mean())},
              "current": {"median_std_ratio": float(top["std_ratio"].median()), "max_abs_log_std_ratio": float(np.abs(np.log(top["std_ratio"])).max()), "mean_acf1_dist": float(top["acf1_dist"].mean())}}
        if v0_pass:
            px_top = atlas_proxy.loc[top.index].values
            px_cf = atlas_proxy.loc[cf.index].values
            rv = bd["vrproxy_from_mean_acf"]
            pc_vr = atlas_proxy[pc_in]
            rb["vrproxy"] = {"real": rv, "selected_mean": float(px_top.mean()), "selected_values": px_top.tolist(), "CF1_mean": float(px_cf.mean()),
                             "rank_in_shortlist30": [int((np.abs(atlas_proxy.loc[sl.index] - rv) < abs(x - rv)).sum() + 1) for x in px_top],
                             "rank_in_std_pct25": [int((np.abs(pc_vr - rv) < abs(x - rv)).sum() + 1) for x in px_top],
                             "min_abs_diff_in_std_pct25": float((pc_vr - rv).abs().min()) if len(pc_vr) else None}
        B[q] = rb
    out["A3"], out["B3"] = A, B
    # ---- C3 ----------------------------------------------------------------------------------------------------------------------------------------
    kb = pd.cut(valid["kurtosis"], [-np.inf, 4.5, 8, np.inf], labels=["<=4.5", "4.5-8", ">8"])
    rr = valid["std_std_across_seeds"] / valid["std"]
    cls = {}
    for lab in ("<=4.5", "4.5-8", ">8"):
        m = kb == lab
        x = rr[m]
        cls[lab] = {"n": int(m.sum()), "ratio_median": float(x.median()) if len(x) else None, "ratio_p90": float(x.quantile(0.9)) if len(x) else None, "ratio_max": float(x.max()) if len(x) else None}
    C = {"kurtosis_classes": cls, "n_valid": nvalid, "n_valid_kurtosis_missing": int(valid["kurtosis"].isna().sum()), "spearman_kurt_vs_relseedsd_all_valid": float(sps.spearmanr(valid["kurtosis"], rr, nan_policy="omit")[0]), "conds": {}}
    for cond in E.CONDS:
        rows = []
        for q in E.BUCKETS4:
            t = T[(cond, q)]
            for c, (dl, wm) in enumerate(mp["buckets"][q]["candidates"]):
                m = (np.arange(1000) % 8) == c
                tt = t[m]
                ar = atlas[(np.isclose(atlas["delta"], dl)) & (atlas["wma"] == wm)].iloc[0]
                sd = tt[:, COLS["std"]]
                p5, p95 = np.percentile(sd, [5, 95])
                sig2 = float(ar["std"] ** 2 + ar["std_std_across_seeds"] ** 2)
                v = float(tt[:, COLS["var1000"]].mean())
                ps = tt[:, COLS["pos0"]:COLS["pos3"] + 1].mean(axis=0)
                rows.append({"q": q, "delta": float(dl), "wma": int(wm), "n_paths": int(m.sum()), "atlas_std": float(ar["std"]), "atlas_std_std": float(ar["std_std_across_seeds"]),
                             "atlas_kurtosis": float(ar["kurtosis"]), "atlas_acf_1": float(ar["acf_1"]), "path_kurt_mean": float(tt[:, COLS["kurtosis"]].mean()), "path_std_mean": float(sd.mean()),
                             "path_std_sd": float(sd.std(ddof=1)), "path_std_p5": float(p5), "path_std_p95": float(p95), "path_std_p95_minus_p5": float(p95 - p5),
                             "seed_sd_over_path_sd": float(ar["std_std_across_seeds"] / sd.std(ddof=1)), "seed_sd_over_p95p5_range": float(ar["std_std_across_seeds"] / (p95 - p5)),
                             "log_V_ens_over_sigma2_atlas": float(np.log(v / sig2)), "rel_seed_sd": float(ar["std_std_across_seeds"] / ar["std"]), "pos_std_blocks": ps.tolist(),
                             "pos_block_max_over_min_first_three": float(ps[:3].max() / ps[:3].min())})
        d = pd.DataFrame(rows)
        C["conds"][cond] = {"rows": rows, "spearman_logratio_vs_atlas_kurtosis": float(sps.spearmanr(d["log_V_ens_over_sigma2_atlas"], d["atlas_kurtosis"])[0]),
                            "spearman_logratio_vs_rel_seed_sd": float(sps.spearmanr(d["log_V_ens_over_sigma2_atlas"], d["rel_seed_sd"])[0]),
                            "spearman_logratio_vs_atlas_acf1": float(sps.spearmanr(d["log_V_ens_over_sigma2_atlas"], d["atlas_acf_1"])[0]),
                            "n_heavy_kurt_gt8": int((d["atlas_kurtosis"] > 8).sum())}
    out["C3"] = C
    json.dump(K.to_json(out), open(os.path.join(E.RESULTS, "analysis3.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
