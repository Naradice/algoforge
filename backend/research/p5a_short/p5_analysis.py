"""P5-A analyses A, B, C, D (and optional E) (plan v0.2 @ b995b5a). Descriptive only; no p-value judgements; diagnostics, not recalibration.
Output: OUT/analysis_AB.json, analysis_C.json, analysis_D.json, analysis_E_<variant>.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5_env as E  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402
import p4_core as C4  # noqa: E402

HZ, PERIODS, SEEDS, QS, QA = K.HZ, K.PERIODS, K.SEEDS, K.QS, K.QA
I05, I95 = QS.index(0.05), QS.index(0.95)
HPAIRS = ((1, 4), (4, 12), (12, 24))


def jdump(obj, name):
    json.dump(K.to_json(obj), open(os.path.join(E.OUT, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def ci_cols(rep):
    """per-column descriptive 95% interval over replicates (NaN ignored) and undefined fraction. rep [B, K]"""
    lo = np.full(rep.shape[1], np.nan)
    hi = np.full(rep.shape[1], np.nan)
    und = np.zeros(rep.shape[1])
    for j in range(rep.shape[1]):
        ok = np.isfinite(rep[:, j])
        und[j] = 1 - ok.mean()
        if ok.sum() >= 20:
            lo[j], hi[j] = np.percentile(rep[ok, j], [2.5, 97.5])
    return {"lo": lo, "hi": hi, "undefined_frac": und}


def comp_with_ci(qd, qe_point, qe_rep):
    pt = K.compare(qd, qe_point)
    rp = K.compare(qd, qe_rep)
    return {"point": pt, "ci_r": ci_cols(rp["r"]), "ci_shape": ci_cols(rp["shape"]),
            "ci_loc": K.ci(rp["loc"]), "n_undefined_r": int(np.isnan(pt["r"][np.arange(len(QS)) != K.I50]).sum())}


def width_slopes(w):
    """w: dict h -> width (scalar or array); returns {pair: slope}"""
    return {f"h{a}->h{b}": (np.log(w[b]) - np.log(w[a])) / (np.log(b) - np.log(a)) for a, b in HPAIRS}


def mixture_quantiles(samples_by_bucket, nb, buckets):
    """quantiles of the mixture over `buckets` with weights = number of origins in each bucket (point estimate)."""
    vals, wts = [], []
    for k in buckets:
        if nb[k] == 0:
            continue
        s = samples_by_bucket[k]
        vals.append(s)
        wts.append(np.full(len(s), nb[k] / len(s)))
    if not vals:
        return np.full(len(QS), np.nan)
    return K.wquantile(np.concatenate(vals), np.concatenate(wts), QA)


# ---- A and B ----------------------------------------------------------------------------------------------------------------------------
def analysis_AB(data, ens_s, tag):
    out = {"buckets": {}, "groups": {}}
    wbs = {per: K.WBoot(np.concatenate([data.cells[(per, h)]["meta"]["week"] for h in HZ]), K.BOOT_B, f"{tag}{per}") for per in PERIODS}
    pooled = {(k, h): np.sort(np.concatenate([ens_s[s][k, HZ.index(h)] for s in SEEDS])) for k in range(20) for h in HZ}
    qd_pool = {kh: np.quantile(v, QA) for kh, v in pooled.items()}
    qd_seed = {(k, h, s): np.quantile(ens_s[s][k, HZ.index(h)], QA) for k in range(20) for h in HZ for s in SEEDS}
    # E1a samples per (period-independent) bucket: estimation origins
    e1a = {}
    wb_est = wbs["estimation"]
    rep_e = {}
    for h in HZ:
        c = data.cells[("estimation", h)]
        for k in range(20):
            m = c["b"] == k
            e1a[(k, h)] = np.sort(c["y"][m])
            rep_e[(k, h)] = wb_est.quantiles(c["y"][m], c["meta"]["week"][m], QA)
    # ---- buckets (estimation) ----
    for k in range(20):
        rec = {}
        for h in HZ:
            qe = np.quantile(e1a[(k, h)], QA)
            r = comp_with_ci(qd_pool[(k, h)], qe, rep_e[(k, h)])
            seed_r = np.array([K.compare(qd_seed[(k, h, s)], qe)["r"] for s in SEEDS])
            seed_sh = np.array([K.compare(qd_seed[(k, h, s)], qe)["shape"] for s in SEEDS])
            rec[f"h{h}"] = {"q_ddm": qd_pool[(k, h)], "q_e1a": qe, "n_e1a": int(len(e1a[(k, h)])), "loc": r["point"]["loc"], "loc_ci": r["ci_loc"], "r": r["point"]["r"],
                            "r_ci": r["ci_r"], "shape": r["point"]["shape"], "shape_ci": r["ci_shape"], "r_seed_min": np.nanmin(seed_r, axis=0) if np.isfinite(seed_r).any() else None,
                            "r_seed_max": np.nanmax(seed_r, axis=0) if np.isfinite(seed_r).any() else None, "shape_seed_min": np.nanmin(seed_sh, axis=0),
                            "shape_seed_max": np.nanmax(seed_sh, axis=0), "n_undefined_r": r["n_undefined_r"]}
        # B: slopes of log width90 vs log h (estimation): DDM, E1a (with bootstrap), per seed
        wd = {h: qd_pool[(k, h)][I95] - qd_pool[(k, h)][I05] for h in HZ}
        we = {h: np.quantile(e1a[(k, h)], QA)[I95] - np.quantile(e1a[(k, h)], QA)[I05] for h in HZ}
        wer = {h: rep_e[(k, h)][:, I95] - rep_e[(k, h)][:, I05] for h in HZ}
        sl_d, sl_e, sl_er = width_slopes(wd), width_slopes(we), width_slopes(wer)
        rec["slopes"] = {pair: {"ddm": sl_d[pair], "e1a": sl_e[pair], "e1a_ci": K.ci(sl_er[pair]), "diff_ddm_minus_e1a": sl_d[pair] - sl_e[pair],
                                "diff_ci": K.ci(sl_d[pair] - sl_er[pair]),
                                "ddm_seed": [width_slopes({h: qd_seed[(k, h, s)][I95] - qd_seed[(k, h, s)][I05] for h in HZ})[pair] for s in SEEDS]} for pair in sl_d}
        rec["width90"] = {"ddm": wd, "e1a": we}
        out["buckets"][str(k)] = rec
    # ---- ATR groups, per period ----
    for per in PERIODS:
        wb = wbs[per]
        gq = {}
        for h in HZ:
            c = data.cells[(per, h)]
            nbk = np.bincount(c["b"], minlength=20)
            for g in range(5):
                bk = range(4 * g, 4 * g + 4)
                m = (c["b"] // 4) == g
                yg, wg = c["y"][m], c["meta"]["week"][m]
                q_ddm = mixture_quantiles({k: pooled[(k, h)] for k in range(20)}, nbk, bk)
                q_e1a = mixture_quantiles({k: e1a[(k, h)] for k in range(20)}, nbk, bk)
                q_real = np.quantile(yg, QA)
                rep_real = wb.quantiles(yg, wg, QA)
                gq[(g, h)] = {"ddm": q_ddm, "e1a": q_e1a, "real": q_real, "rep_real": rep_real, "n": int(m.sum()), "weeks": int(len(np.unique(wg)))}
        for g in range(5):
            rec = {}
            for h in HZ:
                x = gq[(g, h)]
                vs_e = K.compare(x["ddm"], x["e1a"])
                vs_r = comp_with_ci(x["ddm"], x["real"], x["rep_real"])
                rec[f"h{h}"] = {"n": x["n"], "weeks": x["weeks"], "q_ddm": x["ddm"], "q_e1a": x["e1a"], "q_real": x["real"],
                                "ddm_vs_e1a": {"loc": vs_e["loc"], "r": vs_e["r"], "shape": vs_e["shape"]},
                                "ddm_vs_real": {"loc": vs_r["point"]["loc"], "loc_ci": vs_r["ci_loc"], "r": vs_r["point"]["r"], "r_ci": vs_r["ci_r"],
                                                "shape": vs_r["point"]["shape"], "shape_ci": vs_r["ci_shape"], "n_undefined_r": vs_r["n_undefined_r"]}}
            wd = {h: gq[(g, h)]["ddm"][I95] - gq[(g, h)]["ddm"][I05] for h in HZ}
            we = {h: gq[(g, h)]["e1a"][I95] - gq[(g, h)]["e1a"][I05] for h in HZ}
            wr = {h: gq[(g, h)]["real"][I95] - gq[(g, h)]["real"][I05] for h in HZ}
            wrr = {h: gq[(g, h)]["rep_real"][:, I95] - gq[(g, h)]["rep_real"][:, I05] for h in HZ}
            sd, se, sr, srr = width_slopes(wd), width_slopes(we), width_slopes(wr), width_slopes(wrr)
            rec["slopes"] = {pair: {"ddm": sd[pair], "e1a": se[pair], "real": sr[pair], "real_ci": K.ci(srr[pair]), "diff_ddm_minus_real": sd[pair] - sr[pair],
                                    "diff_ci": K.ci(sd[pair] - srr[pair])} for pair in sd}
            rec["width90"] = {"ddm": wd, "e1a": we, "real": wr}
            out["groups"][f"{per}|{g}"] = rec
    return out


# ---- C ------------------------------------------------------------------------------------------------------------------------------------
def analysis_C(data, ens_s, tag):
    fits = K.fit_all(data, ens_s)
    out = {"fits": {"s1": fits["s1"], "s2": {h: fits["s2"][h] for h in HZ},
                    "pit_quantiles": {h: np.quantile(fits["pit_pooled"][h], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) for h in HZ},
                    "sgrid": [float(K.SGRID[0]), float(K.SGRID[-1])]}, "cells": {}}
    kinds = ["T0", "T0g", "T1", "T2", "T3", "T4"]
    ev = {k: K.EnsEval(data, K.apply_transform(k, ens_s, fits)) for k in kinds}
    e1a = {}
    for h in HZ:
        y_fit, b_fit = C4.fit_set(data, h, "B-1")
        pred = P.Pred(C4.fit_E1("E1a", y_fit, b_fit))
        for per in PERIODS:
            c = data.cells[(per, h)]
            e1a[(per, h)] = pred.per_origin(c["b"], c["y"])
    for per in PERIODS:
        for h in HZ:
            c = data.cells[(per, h)]
            wk = c["meta"]["week"]
            g = c["b"] // 4
            oe = e1a[(per, h)]
            crps_e = oe["crps"]
            crps0 = ev["T0"].crps[(per, h)]
            row = {}
            for kind in kinds + ["E1a"]:
                if kind == "E1a":
                    rec = {"pooled": P.metrics(oe, c["y"]), "by_group": P.group_metrics(oe, c["y"], c["b"])}
                    cr = crps_e
                else:
                    rec = {"pooled": ev[kind].metrics(per, h), "by_group": ev[kind].group_metrics(per, h)}
                    cr = ev[kind].crps[(per, h)]
                if kind not in ("T0", "E1a"):
                    rec["crps_rel_vs_T0"] = P.rel_diff(cr, crps0, wk, f"P5C{tag}{kind}{per}{h}a")
                if kind != "E1a":
                    rec["crps_rel_vs_E1a"] = P.rel_diff(cr, crps_e, wk, f"P5C{tag}{kind}{per}{h}b")
                gap = (crps0.mean() - crps_e.mean()) / crps0.mean()
                rec["gap_T0_E1a_rel"] = float(gap)
                if kind not in ("T0", "E1a"):
                    rec["gap_closed_pooled"] = float((crps0.mean() - cr.mean()) / (crps0.mean() - crps_e.mean())) if abs(gap) >= 0.01 else None
                    gg = {}
                    for a in range(5):
                        m = g == a
                        d0, de = crps0[m].mean(), crps_e[m].mean()
                        gg[str(a)] = float((d0 - cr[m].mean()) / (d0 - de)) if abs((d0 - de) / d0) >= 0.01 else None
                    rec["gap_closed_by_group"] = gg
                    rec["crps_rel_vs_T0_by_group"] = {str(a): P.rel_diff(cr[g == a], crps0[g == a], wk[g == a], f"P5C{tag}{kind}{per}{h}g{a}") for a in range(5)}
                row[kind] = rec
            row["in_sample"] = per == "estimation"
            out["cells"][f"{per}|h{h}"] = row
    approx = {f"{per}|h{h}": abs(ev["T0g"].crps[(per, h)].mean() - ev["T0"].crps[(per, h)].mean()) / ev["T0"].crps[(per, h)].mean() for per in PERIODS for h in HZ}
    out["approx_error_T0g_vs_T0"] = {"tolerance": K.APPROX_TOL, "max": max(approx.values()), "n_over": int(sum(v > K.APPROX_TOL for v in approx.values())), "cells": approx}
    return out


# ---- D ---------------------------------------------------------------------------------------------------------------------------------------
def analysis_D(data, tag):
    raw = {s: K.load_raw_ens("C0s", s)[0] for s in SEEDS}
    att = {s: K.load_raw_ens("C0s", s)[1] for s in SEEDS}
    mp = K.load_mapping()
    out = {"candidates": {}, "selection": {}, "stability": {}, "eval": {}, "s5": {}}
    sel = {h: K.select_subsets(data, raw, att, SEEDS, h) for h in HZ}
    sel_seed = {h: {s: K.select_subsets(data, raw, att, (s,), h) for s in SEEDS} for h in HZ}
    # descriptive: candidate-wise width90 vs E1a
    for h in HZ:
        hi = HZ.index(h)
        c = data.cells[("estimation", h)]
        rows = []
        for b in range(20):
            qe = np.quantile(c["y"][c["b"] == b], QA)
            w_e = qe[I95] - qe[I05]
            wc = []
            for k in range(K.N_CAND):
                x = K.pooled(K.cand_samples(raw, att, b, hi, k, SEEDS))
                q = np.quantile(x, QA)
                wc.append(float(q[I95] - q[I05]))
            xm = K.pooled([raw[s][b, hi] for s in SEEDS])
            qm = np.quantile(xm, QA)
            rows.append({"bucket": b, "width90_e1a": float(w_e), "width90_mix8": float(qm[I95] - qm[I05]), "width90_candidates": wc,
                         "cand_ratio_to_e1a": [x / w_e for x in wc], "mix8_ratio_to_e1a": float((qm[I95] - qm[I05]) / w_e),
                         "n_candidates_wider_than_e1a": int(sum(x > w_e for x in wc))})
        out["candidates"][f"h{h}"] = rows
        out["selection"][f"h{h}"] = [{"bucket": b, "single": sel[h][b]["single"], "pair": list(sel[h][b]["pair"]), "mix8_crps": sel[h][b]["mix8_crps"],
                                      "single_crps": float(sel[h][b]["single_crps"].min()), "pair_crps": float(sel[h][b]["pair_crps"].min()),
                                      "single_crps_all": sel[h][b]["single_crps"], "n_paths": sel[h][b]["n_paths"],
                                      "est_improvement_single": float(sel[h][b]["single_crps"].min() / sel[h][b]["mix8_crps"] - 1),
                                      "est_improvement_pair": float(sel[h][b]["pair_crps"].min() / sel[h][b]["mix8_crps"] - 1)} for b in range(20)]
        # stability across seeds
        s1 = [[sel_seed[h][s][b]["single"] for b in range(20)] for s in SEEDS]
        p1 = [[tuple(sel_seed[h][s][b]["pair"]) for b in range(20)] for s in SEEDS]
        pooled_s = [sel[h][b]["single"] for b in range(20)]
        pooled_p = [tuple(sel[h][b]["pair"]) for b in range(20)]
        out["stability"][f"h{h}"] = {
            "single_seed_pairs_equal_frac": float(np.mean([s1[i][b] == s1[j][b] for i in range(3) for j in range(i + 1, 3) for b in range(20)])),
            "single_seed_vs_pooled_equal_frac": float(np.mean([s1[i][b] == pooled_s[b] for i in range(3) for b in range(20)])),
            "pair_seed_pairs_equal_frac": float(np.mean([p1[i][b] == p1[j][b] for i in range(3) for j in range(i + 1, 3) for b in range(20)])),
            "pair_seed_vs_pooled_equal_frac": float(np.mean([p1[i][b] == pooled_p[b] for i in range(3) for b in range(20)])),
            "pair_seed_vs_pooled_share_a_candidate_frac": float(np.mean([len(set(p1[i][b]) & set(pooled_p[b])) >= 1 for i in range(3) for b in range(20)])),
            "chance_single": 1 / K.N_CAND, "chance_pair_exact": 1 / 28}
    # evaluation of the fixed selections on estimation (in-sample), validation, final
    for h in HZ:
        hi = HZ.index(h)
        mix = [K.pooled([raw[s][b, hi] for s in SEEDS]) for b in range(20)]
        single = [K.pooled(K.cand_samples(raw, att, b, hi, sel[h][b]["single"], SEEDS)) for b in range(20)]
        pair = [K.pair_samples(K.cand_samples(raw, att, b, hi, sel[h][b]["pair"][0], SEEDS), K.cand_samples(raw, att, b, hi, sel[h][b]["pair"][1], SEEDS)) for b in range(20)]
        y_fit, b_fit = C4.fit_set(data, h, "B-1")
        e1 = C4.fit_E1("E1a", y_fit, b_fit)
        preds = {"mix8": P.Pred(mix), "single": P.Pred(single), "pair": P.Pred(pair), "E1a": P.Pred(e1)}
        for per in PERIODS:
            c = data.cells[(per, h)]
            wk = c["meta"]["week"]
            g = c["b"] // 4
            o = {k: v.per_origin(c["b"], c["y"]) for k, v in preds.items()}
            rec = {}
            for kind in ("mix8", "single", "pair", "E1a"):
                rec[kind] = {"pooled": P.metrics(o[kind], c["y"]), "by_group": P.group_metrics(o[kind], c["y"], c["b"])}
                if kind != "mix8":
                    rec[kind]["crps_rel_vs_mix8"] = P.rel_diff(o[kind]["crps"], o["mix8"]["crps"], wk, f"P5D{tag}{kind}{per}{h}a")
                    rec[kind]["crps_rel_vs_mix8_by_group"] = {str(a): P.rel_diff(o[kind]["crps"][g == a], o["mix8"]["crps"][g == a], wk[g == a], f"P5D{tag}{kind}{per}{h}g{a}") for a in range(5)}
                if kind in ("single", "pair"):
                    rec[kind]["crps_rel_vs_E1a"] = P.rel_diff(o[kind]["crps"], o["E1a"]["crps"], wk, f"P5D{tag}{kind}{per}{h}b")
            rec["in_sample"] = per == "estimation"
            out["eval"][f"{per}|h{h}"] = rec
    # S5: bucket-wise width shortfall vs mapping descriptors (descriptive; Spearman rho only, no test)
    for h in (1, 4):
        c = data.cells[("estimation", h)]
        hi = HZ.index(h)
        ratio = []
        for b in range(20):
            qe = np.quantile(c["y"][c["b"] == b], QA)
            xm = K.pooled([raw[s][b, hi] for s in SEEDS])
            qm = np.quantile(xm, QA)
            ratio.append((qm[I95] - qm[I05]) / (qe[I95] - qe[I05]))
        ratio = np.array(ratio)
        desc = {"best_std_ratio": [b["best_std_ratio"] for b in mp["buckets"]], "n_windows_available": [b["n_windows_available"] for b in mp["buckets"]],
                "out_of_range": [float(b["out_of_range"]) for b in mp["buckets"]], "real_std": [b["real_stats"]["std"] for b in mp["buckets"]],
                "real_kurtosis": [b["real_stats"]["kurtosis"] for b in mp["buckets"]], "real_acf_1": [b["real_stats"]["acf_1"] for b in mp["buckets"]],
                "real_volclust_50": [b["real_stats"]["volclust_50"] for b in mp["buckets"]], "bucket_index": list(range(20))}
        out["s5"][f"h{h}"] = {"width90_ratio_ddm_over_e1a": ratio, "descriptors": desc,
                              "spearman_rho": {k: (None if np.ptp(v) == 0 else float(stats.spearmanr(v, ratio)[0])) for k, v in desc.items()}, "n": 20,
                              "constant_descriptors": [k for k, v in desc.items() if np.ptp(v) == 0]}
    return out


def main(variants=("C0s",)):
    code = E.code_state()
    g = json.load(open(os.path.join(E.OUT, "gates.json"), encoding="utf-8"))
    assert g["all_pass"], "gates must pass first"
    data = P.V.Data()
    for v in variants:
        ens_s = {s: P.V.load_ens(v, s) for s in SEEDS}
        tag = v.replace("_", "")[:5]
        if v == "C0s":
            print("A,B", flush=True)
            jdump({"_code": code, **analysis_AB(data, ens_s, tag)}, "analysis_AB.json")
            print("C", flush=True)
            jdump({"_code": code, **analysis_C(data, ens_s, tag)}, "analysis_C.json")
            print("D", flush=True)
            jdump({"_code": code, **analysis_D(data, tag)}, "analysis_D.json")
        else:
            print("E", v, flush=True)
            jdump({"_code": code, "A_B": analysis_AB(data, ens_s, tag), "C": analysis_C(data, ens_s, tag)}, f"analysis_E_{v}.json")
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("C0s",))
