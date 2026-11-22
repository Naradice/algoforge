"""P5-B stage 2 analyses A2-E2 and gates 5-6 (plan v0.2 @ 6d6af4c). Requires the regenerated closes of C3 and C0s (gate 2/3 passed).
Descriptive only; no p-values. Output: OUT/analysis2.json"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2_env as E  # noqa: E402
import s2_lib as S2  # noqa: E402
import p5b_lib as L5  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402

HZ = (1, 4, 12, 24)
ATLAS_KEYS = ("std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50")
OFFSETS = (0, 120, 360, 720)
SUM_K = (60, 240)


def pstats(arr):
    return S2.path_stats(arr)


def stack(stats_list):
    """list of path_stats dicts -> dict of arrays."""
    out = {"atlas": {k: np.array([s["atlas"][k] for s in stats_list]) for k in ATLAS_KEYS}, "var1000": np.array([s["var1000"] for s in stats_list]),
           "mean1000": np.array([s["mean1000"] for s in stats_list]), "acf": np.array([s["acf"] for s in stats_list]),
           "vr_win": {k: np.array([s["vr_win"][k] for s in stats_list]) for k in stats_list[0]["vr_win"]}}
    if "var_full" in stats_list[0]:
        out["var_full"] = np.array([s["var_full"] for s in stats_list])
        out["var_60h"] = {h: np.array([s["var_60h"][h] for s in stats_list]) for h in stats_list[0]["var_60h"]}
        out["pos_std"] = np.array([s["pos_std"] for s in stats_list])
        out["kurt_full"] = np.array([s["kurt_full"] for s in stats_list])
    return out


def implied_vr(acf_mean, k):
    j = np.arange(1, k)
    return float(1 + 2 * np.sum((1 - j / k) * acf_mean[:k - 1]))


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def main():
    code = E.code_state()
    for cond in E.CONDS:
        g = json.load(open(os.path.join(E.OUT, f"regen_{cond}.json"), encoding="utf-8"))
        assert g["pass"], f"{cond} regeneration gate has not passed"
    pre = json.load(open(os.path.join(E.OUT, "pre_gates.json"), encoding="utf-8"))
    assert pre["all_pass"]
    s1 = json.load(open(os.path.join(E.STAGE1_RESULTS, "analysis.json"), encoding="utf-8"))
    mp = S2.load_mapping()
    atlas = L5.load_atlas()
    data = P.V.Data()
    raw = data.raw
    # ---- gate 6: storage reload ---------------------------------------------------------------------------------------------------------
    sha_ok = True
    for cond in E.CONDS:
        man = json.load(open(os.path.join(E.OUT, f"regen_{cond}.json"), encoding="utf-8"))["sha256"]
        for q in E.BUCKETS:
            sha_ok &= S2.sha256_file(S2.closes_path(cond, q)) == man[os.path.basename(S2.closes_path(cond, q))]
    out = {"_code": code, "gates": {"6_storage_sha256": bool(sha_ok)}, "cells": {}, "notes": {}}
    assert sha_ok
    # ---- load, per-path statistics (atlas definitions) -------------------------------------------------------------------------------------
    D = {}
    with ProcessPoolExecutor(max_workers=E.WORKERS) as ex:
        for cond in E.CONDS:
            for q in E.BUCKETS:
                closes = np.load(S2.closes_path(cond, q))
                st = stack(list(ex.map(pstats, list(closes), chunksize=16)))
                lc = np.log(closes)
                D[(cond, q)] = {"closes": closes, "r": np.diff(lc, axis=1), "lc": lc - lc[:, :1], "st": st}
                print("stats", cond, q, flush=True)
        real = {}
        for q in E.BUCKETS:
            bl = np.array(mp["buckets"][q]["used_blocks"])
            real[q] = stack(list(ex.map(pstats, [raw.close[b * 60:b * 60 + 1001] for b in bl], chunksize=16)))
    ens = {cond: np.load(os.path.join(E.ENS_DIR, f"{cond}_seed1.npz"), allow_pickle=True)["ens"] for cond in E.CONDS}
    att_idx = np.arange(E.N_PATHS) % E.N_CAND
    worst_id = 0.0
    for q in E.BUCKETS:
        rec = {"candidates": [], "bucket": {}, "D2": {}, "real": {}}
        cands = mp["buckets"][q]["candidates"]
        rows = [L5.atlas_row(atlas, dl, int(wm)).iloc[0] for dl, wm in cands]
        sig2c = [L5.sigma2_atlas(r) for r in rows]
        sig2mix = float(np.mean(sig2c))
        v_real = s1["A"][str(q)]["V_real_M1_primary"]
        d3, d0 = D[("C3", q)], D[("C0s", q)]
        # ---- real-side windows ---------------------------------------------------------------------------------------------------------------
        rl = real[q]
        racf = rl["acf"].mean(axis=0)
        rec["real"] = {"V_real_M1": v_real, "vr_win_mean": {k: float(v.mean()) for k, v in rl["vr_win"].items()}, "vr_win_se": {k: se(v) for k, v in rl["vr_win"].items()},
                       "acf_mean": racf.tolist(), "implied_vr": {str(k): implied_vr(racf, k) for k in S2.KS}, "atlas_stats_mean": {k: float(v.mean()) for k, v in rl["atlas"].items()}}
        # ---- per-candidate statistics (A2, B2, C2-a, E2) ---------------------------------------------------------------------------------------
        for c in range(E.N_CAND):
            m = att_idx == c
            entry = {"delta": cands[c][0], "wma": cands[c][1], "n": int(m.sum()), "sigma2_atlas": sig2c[c], "atlas": {k: float(rows[c][k]) for k in ATLAS_KEYS},
                     "atlas_std_across_seeds": {k: float(rows[c][f"{k}_std_across_seeds"]) for k in ATLAS_KEYS}}
            for cond, dd in (("C3", d3), ("C0s", d0)):
                st = dd["st"]
                v = st["var1000"][m]
                acfm = st["acf"][m].mean(axis=0)
                e = {"V_ens_M1": float(v.mean()), "V_ens_M1_se": se(v), "V_ens_60h": {h: float(st["var_60h"][h][m].mean()) for h in st["var_60h"]}, "V_ens_full": float(st["var_full"][m].mean()),
                     "atlas_stats_mean": {k: float(st["atlas"][k][m].mean()) for k in ATLAS_KEYS}, "atlas_stats_se": {k: se(st["atlas"][k][m]) for k in ATLAS_KEYS},
                     "vr_win_mean": {k: float(x[m].mean()) for k, x in st["vr_win"].items()}, "vr_win_se": {k: se(x[m]) for k, x in st["vr_win"].items()},
                     "acf_mean": acfm.tolist(), "implied_vr": {str(k): implied_vr(acfm, k) for k in S2.KS}, "kurt_full_mean": float(st["kurt_full"][m].mean()),
                     "pos_std_mean": st["pos_std"][m].mean(axis=0).tolist(), "mean1000_sd_over_expected": float(np.std(st["mean1000"][m]) / (np.sqrt(v.mean()) / np.sqrt(1000)))}
                cum = ens[cond][q][:, m]
                e["VR_ens"] = {str(h): float(cum[i].var() / (60 * h * v.mean())) for i, h in enumerate(HZ)}
                e["VR_ens_within_var_variants"] = {str(h): {"first_60h_window": float(cum[i].var() / (60 * h * st["var_60h"][str(h)][m].mean())),
                                                            "full_1440": float(cum[i].var() / (60 * h * st["var_full"][m].mean()))} for i, h in enumerate(HZ)}
                entry[cond] = e
            sa = float(rows[c]["std_std_across_seeds"])
            entry["A2_log_ratio_var_vs_atlas"] = float(np.log(entry["C3"]["V_ens_M1"] / sig2c[c]))
            entry["A2_se_log_ratio"] = float(np.sqrt((entry["C3"]["V_ens_M1_se"] / entry["C3"]["V_ens_M1"]) ** 2 + (2 * rows[c]["std"] * sa / np.sqrt(5) / sig2c[c]) ** 2))
            rec["candidates"].append(entry)
        # ---- bucket mixture (A2, C2-b, D2) ----------------------------------------------------------------------------------------------------
        b = {}
        for cond, dd in (("C3", d3), ("C0s", d0)):
            st = dd["st"]
            r1 = dd["r"][:, :1000]
            v_pool = float(np.var(r1))
            v_within = float(st["var1000"].mean())
            cum = ens[cond][q]
            sd_h = {h: float(np.std(cum[i])) for i, h in enumerate(HZ)}
            b[cond] = {"V_mix_pooled": v_pool, "V_mix_within_mean": v_within, "between_path_mean_var": float(np.var(st["mean1000"])), "V_pooled_over_within": v_pool / v_within,
                       "sd_h": sd_h, "VR_ens_mix_pooled": {str(h): sd_h[h] ** 2 / (60 * h * v_pool) for h in HZ}, "VR_ens_mix_within": {str(h): sd_h[h] ** 2 / (60 * h * v_within) for h in HZ},
                       "vr_win_mean": {k: float(x.mean()) for k, x in st["vr_win"].items()}, "acf_mean": st["acf"].mean(axis=0).tolist(),
                       "implied_vr": {str(k): implied_vr(st["acf"].mean(axis=0), k) for k in S2.KS}, "atlas_stats_mean": {k: float(st["atlas"][k].mean()) for k in ATLAS_KEYS},
                       "pos_std_mean": st["pos_std"].mean(axis=0).tolist(), "kurt_full_mean": float(st["kurt_full"].mean())}
            # cross-path variance of sums of k returns starting at different offsets (initial-state / transient diagnostic)
            offs = {}
            for k in SUM_K:
                offs[str(k)] = {str(t0): float(np.var(dd["lc"][:, t0 + k] - dd["lc"][:, t0]) / (k * v_pool)) for t0 in OFFSETS if t0 + k <= 1440}
            b[cond]["cross_path_vr_by_start_offset"] = offs
        b["sigma2_atlas_mix_proxy"] = sig2mix
        b["A2_log_ratio_mix_var_vs_atlas"] = {"pooled": float(np.log(b["C3"]["V_mix_pooled"] / sig2mix)), "within_mean": float(np.log(b["C3"]["V_mix_within_mean"] / sig2mix))}
        rec["bucket"] = b
        # ---- D2: decomposition re-evaluated with measured M1 variances (cross-path VR) -----------------------------------------------------------
        for h in HZ:
            s1d = s1["D"][str(q)][f"h{h}"]
            sd0, sd3 = s1["C"][str(q)][f"h{h}"]["C0s"]["mix_sd"], s1["C"][str(q)][f"h{h}"]["C3"]["mix_sd"]
            var_y = s1["B"][str(q)][f"h{h}"]["var_y"]
            vr_real = s1["B"][str(q)][f"h{h}"]["VR_real"]
            n = 60 * h
            row = {"stage1": {k: s1d[k] for k in ("M1", "M3", "M2", "total")}}
            for name, key in (("pooled", "V_mix_pooled"), ("within_mean", "V_mix_within_mean")):
                v3, v0 = b["C3"][key], b["C0s"][key]
                m1 = 0.5 * np.log(v3 / v_real)
                vr3 = sd3 ** 2 / (n * v3)
                m3 = 0.5 * np.log(vr3 / vr_real)
                m2a = 0.5 * np.log(v0 / v3)
                vr0 = sd0 ** 2 / (n * v0)
                m2b = 0.5 * np.log(vr0 / vr3)
                res = {"M1": m1, "M3": m3, "M2": np.log(sd0 / sd3), "M2_M1level": m2a, "M2_aggregation": m2b, "VR_C3_cross_path": vr3, "VR_C0s_cross_path": vr0, "VR_real": vr_real,
                       "dM1_vs_stage1": m1 - s1d["M1"], "dM3_vs_stage1": m3 - s1d["M3"]}
                worst_id = max(worst_id, abs(m1 + m3 - (s1d["M1"] + s1d["M3"])), abs(m2a + m2b - s1d["M2"]), abs(np.log(sd0 / sd3) - s1d["M2"]))
                row[name] = {k: float(v) for k, v in res.items()}
            rec["D2"][f"h{h}"] = row
        out["cells"][str(q)] = rec
    out["gates"]["5_identity_max_abs_residual"] = float(worst_id)
    assert worst_id < 1e-9, worst_id
    out["notes"] = {"V_mix_pooled": "variance of all first-1000 M1 returns pooled over the 1000 paths of the bucket (includes between-path mean differences)",
                    "V_mix_within_mean": "mean over paths of the within-path variance of the first 1000 M1 returns",
                    "VR_win": "within-path overlapping variance ratio; VR_ens: cross-path variance of the 60h-step cumulative return / (60h x M1 variance)"}
    json.dump(K.to_json(out), open(os.path.join(E.OUT, "analysis2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("ANALYSIS COMPLETE; identity residual", worst_id, flush=True)


if __name__ == "__main__":
    main()
