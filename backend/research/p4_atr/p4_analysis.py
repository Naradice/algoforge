"""P4 analyses A, B, C (plan v0.2 @ 9d9825d). Descriptive only; no p-value judgements. Output: OUT/analysis_{A,B,C}[_variant].json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p4_env as E  # noqa: E402
import p4_lib as P  # noqa: E402
import p4_core as C  # noqa: E402

PER_PAIRS = (("validation", "final"), ("estimation", "validation"), ("estimation", "final"))
YQ = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_BUCKET_N, MIN_BUCKET_WEEKS = 100, 10


def jdump(obj, name):
    json.dump(obj, open(os.path.join(E.OUT, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


# ---- A: composition vs within-group decomposition and within-group tables ------------------------------------------------------------------
def analysis_A(data, ddm, tag):
    out = {"decomposition": {}, "group_tables": {}, "bucket_tables": {}}
    for h in P.HZ:
        for name, arrs in (("cov90", ddm.hit90), ("crps", ddm.crps)):
            for pv, pf in PER_PAIRS:
                cv, cf = data.cells[(pv, h)], data.cells[(pf, h)]
                out["decomposition"][f"{name}|h{h}|{pv}->{pf}"] = P.decomposition(
                    arrs[(pv, h)], cv["b"] // 4, cv["meta"]["week"], arrs[(pf, h)], cf["b"] // 4, cf["meta"]["week"], f"A{tag}{name}h{h}{pv[:3]}{pf[:3]}")
        for per in P.PERIODS:
            c = data.cells[(per, h)]
            g = c["b"] // 4
            gm = ddm.group_metrics(per, h)
            wk = c["meta"]["week"]
            tab = {}
            for a in range(5):
                m = g == a
                ymed = {f"q{q}": float(np.quantile(c["y"][m], q)) for q in YQ}
                tab[str(a)] = {**gm[str(a)], "weeks": int(len(np.unique(wk[m]))), "y_quantiles": ymed}
            out["group_tables"][f"{per}|h{h}"] = tab
            bt = {}
            for b in range(20):
                m = c["b"] == b
                nw = int(len(np.unique(wk[m])))
                ok = bool(m.sum() >= MIN_BUCKET_N and nw >= MIN_BUCKET_WEEKS)
                bt[str(b)] = {"n": int(m.sum()), "weeks": nw, "evaluated": ok}
                if ok:
                    bt[str(b)]["cov90"] = float(ddm.hit90[(per, h)][m].mean())
                    bt[str(b)]["crps"] = float(ddm.crps[(per, h)][m].mean())
            out["bucket_tables"][f"{per}|h{h}"] = bt
        # y-quantile ratios within group across periods (realised side), descriptive
    out["y_quantile_ratios"] = {}
    for h in P.HZ:
        for pa, pb in (("validation", "estimation"), ("final", "validation"), ("final", "estimation")):
            r = {}
            for a in range(5):
                qa = out["group_tables"][f"{pa}|h{h}"][str(a)]["y_quantiles"]
                qb = out["group_tables"][f"{pb}|h{h}"][str(a)]["y_quantiles"]
                r[str(a)] = {"q95/q05 width ratio": float((qa["q0.95"] - qa["q0.05"]) / (qb["q0.95"] - qb["q0.05"])),
                             "q75/q25 width ratio": float((qa["q0.75"] - qa["q0.25"]) / (qb["q0.75"] - qb["q0.25"])),
                             "median shift": float(qa["q0.5"] - qb["q0.5"])}
            out["y_quantile_ratios"][f"h{h}|{pa} / {pb}"] = r
    return out


# ---- B: empirical baselines ---------------------------------------------------------------------------------------------------------------
def analysis_B(data, ddm, tag):
    out = {}
    stat = {}
    import json as _j
    sel = _j.load(open(os.path.join(P.V.E.RUN_DIR, "selection.json")))["selected"]
    for per in ("validation", "final"):
        sc = np.load(os.path.join(P.V.E.RUN_DIR, f"scores_{per}.npz"))
        for h in P.HZ:
            key = f"{sel[str(h)]}|h{h}"
            stat[(per, h)] = {"model": sel[str(h)], "crps": sc[f"{key}|crps"], "j": sc[f"{key}|j"]}
    for mode, evals in (("B-1", ("estimation", "validation", "final")), ("B-2", ("final",))):
        for kind in ("E1a", "E1b", "E1c"):
            for h in P.HZ:
                y_fit, b_fit = C.fit_set(data, h, mode)
                pred = P.Pred(C.fit_E1(kind, y_fit, b_fit))
                for per in evals:
                    c = data.cells[(per, h)]
                    o = pred.per_origin(c["b"], c["y"])
                    wk = c["meta"]["week"]
                    key = f"{mode}|{kind}|{per}|h{h}"
                    rec = {"baseline": P.metrics(o, c["y"]), "ddm": ddm.metrics(per, h),
                           "baseline_by_group": P.group_metrics(o, c["y"], c["b"]), "ddm_by_group": ddm.group_metrics(per, h),
                           "insample": per == "estimation"}
                    rec["crps_rel_vs_ddm"] = P.rel_diff(o["crps"], ddm.crps[(per, h)], wk, f"B{tag}{key}")
                    g = c["b"] // 4
                    rec["crps_rel_vs_ddm_by_group"] = {str(a): P.rel_diff(o["crps"][g == a], ddm.crps[(per, h)][g == a], wk[g == a], f"B{tag}{key}g{a}") for a in range(5)}
                    if (per, h) in stat:
                        s = stat[(per, h)]
                        assert np.array_equal(s["j"], c["j"])
                        rec["stat_model"] = {"model": s["model"], "crps": float(s["crps"].mean()),
                                             "crps_by_group": {str(a): float(s["crps"][g == a].mean()) for a in range(5)}}
                    out[key] = rec
    return out


# ---- C: additional conditioning information (one variable at a time) -------------------------------------------------------------------------
def analysis_C(data, ddm, tag):
    F = P.build_features(data.raw)
    out = {"cuts": {}, "cells": {}}
    ce = {h: data.cells[("estimation", h)] for h in P.HZ}
    j1 = ce[1]["j"]
    cuts = {v: P.terciles_by_bucket(F[v][j1], ce[1]["b"]) for v in ("vr", "ret")}
    out["cuts"] = {v: cuts[v].tolist() for v in cuts}
    out["note"] = ("terciles are computed within each ATR bucket from the estimation-period horizon-1 origins with a defined feature; "
                   "the session variable has the 3 fixed UTC classes (00-08, 08-16, 16-24)")

    def level(var, per, h):
        c = data.cells[(per, h)]
        if var == "session":
            return P.session_of(c["meta"]["hour"])
        return P.level_of(F[var][c["j"]], c["b"], cuts[var])

    for var in ("session", "vr", "ret"):
        for h in P.HZ:
            ey = ce[h]["y"]
            eb = ce[h]["b"]
            elv = level(var, "estimation", h)
            fit_ok = elv >= 0
            cell_fit = eb * 3 + elv
            samples_c = C.fit_cond_cells(ey[fit_ok], cell_fit[fit_ok], 60)
            n_fit = np.array([0 if s is None else len(s) for s in samples_c])
            n_fit_all = np.bincount(cell_fit[fit_ok], minlength=60)
            base_samples = C.fit_E1("E1a", ey, eb)
            for per in ("validation", "final"):
                c = data.cells[(per, h)]
                lv = level(var, per, h)
                cell = c["b"] * 3 + np.where(lv >= 0, lv, 0)
                n_eval = np.bincount(cell[lv >= 0], minlength=60)
                usable = (lv >= 0) & np.array([samples_c[x] is not None for x in cell]) & (n_eval[cell] >= P.MIN_CELL)
                # conditional predictive: cells without a fit sample fall back to E1a but are excluded from the comparison
                sc = [samples_c[x] if samples_c[x] is not None else base_samples[x // 3] for x in range(60)]
                pc = P.Pred(sc)
                oc = pc.per_origin(cell, c["y"])
                oa = P.Pred(base_samples).per_origin(c["b"], c["y"])
                wk = c["meta"]["week"]
                key = f"{var}|{per}|h{h}"
                m = usable
                g = c["b"] // 4
                rec = {"n_eval_total": int(len(m)), "n_usable": int(m.sum()), "excluded_fraction": float(1 - m.mean()),
                       "weeks_usable": int(len(np.unique(wk[m]))),
                       "fit_cell_n": {"min": int(n_fit_all.min()), "median": int(np.median(n_fit_all)), "max": int(n_fit_all.max())},
                       "eval_cell_n_among_usable": {"min": int(n_eval[np.unique(cell[m])].min()), "median": int(np.median(n_eval[np.unique(cell[m])]))},
                       "cond": P.metrics(oc, c["y"], m), "E1a": P.metrics(oa, c["y"], m),
                       "ddm": ddm.metrics(per, h, m),
                       "cond_by_group": P.group_metrics(oc, c["y"], c["b"], m), "E1a_by_group": P.group_metrics(oa, c["y"], c["b"], m)}
                rec["crps_rel_cond_vs_E1a"] = P.rel_diff(oc["crps"][m], oa["crps"][m], wk[m], f"C{tag}{key}")
                rec["crps_rel_cond_vs_E1a_by_group"] = {
                    str(a): (P.rel_diff(oc["crps"][m & (g == a)], oa["crps"][m & (g == a)], wk[m & (g == a)], f"C{tag}{key}g{a}") if (m & (g == a)).sum() > 0 else None) for a in range(5)}
                rec["crps_rel_ddm_vs_E1a"] = P.rel_diff(ddm.crps[(per, h)][m], oa["crps"][m], wk[m], f"C{tag}{key}d")
                out["cells"][key] = rec
    return out


def main(variants=("C0s",)):
    code = E.code_state()
    g = json.load(open(os.path.join(E.OUT, "gates.json"), encoding="utf-8"))
    assert g["all_pass"], "gates must pass first"
    data = P.V.Data()
    for v in variants:
        ddm = C.DdmEval(data, v)
        tag = v[:3]
        suffix = "" if v == "C0s" else f"_{v}"
        print("A", v, flush=True)
        jdump({"_code": code, **analysis_A(data, ddm, tag)}, f"analysis_A{suffix}.json")
        print("B", v, flush=True)
        jdump({"_code": code, "cells": analysis_B(data, ddm, tag)}, f"analysis_B{suffix}.json")
        if v == "C0s":
            print("C", v, flush=True)
            jdump({"_code": code, **analysis_C(data, ddm, tag)}, f"analysis_C{suffix}.json")
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("C0s",))
