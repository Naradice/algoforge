"""P4 pre-analysis gates (plan v0.2 section 5). Stops (exit 1) when any gate fails. Output: OUT/gates.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p4_env as E  # noqa: E402
import p4_lib as P  # noqa: E402
import p4_core as C  # noqa: E402
import p1_lib as P1  # noqa: E402

RES = {}


def gate(name, ok, detail=None):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    os.makedirs(E.OUT, exist_ok=True)
    code = E.code_state()
    data = P.V.Data()
    raw = data.raw
    # 1. input identity ---------------------------------------------------------------------------------------------------------
    import p2_env as E2
    sha = P1.sha256_file(E2.PARQUET)
    gate("1a raw parquet sha256 equals the recorded F1/P3 hash", sha == E.PARQUET_SHA256, {"sha256": sha})
    p3 = json.load(open(os.path.join(E.P3_RESULTS, "p3_report.json"), encoding="utf-8"))["metrics"]
    ddm = C.DdmEval(data, "C0s")
    worst, nbad = 0.0, 0
    for s_i, s in enumerate(P.SEEDS):
        for per in ("validation", "final"):
            for h in P.HZ:
                c = data.cells[(per, h)]
                m = P.metrics(ddm.o[(per, h)][s_i], c["y"])
                ref = p3[f"C0s|{s}"][f"{per}|h{h}"]
                d = max(abs(m["crps"] - ref["crps"]), abs(m["cov90"] - ref["cov90"]))
                worst = max(worst, d)
                nbad += int(m["n"] != ref["n"] or d > 1e-12)
    gate("1b DDM C0s metrics reproduce the stored P3 values (all seeds/cells, tol 1e-12; same n)", nbad == 0, {"max_abs_diff": worst, "n_bad": nbad})
    ens = P.V.load_ens("C0s", 1)
    c = data.cells[("final", 12)]
    a = P.fair_crps_list([ens[b, c["hi"], :] for b in range(20)], c["b"], c["y"])
    b_ = P.V.fair_crps(ens[:, c["hi"], :], c["b"], c["y"])
    gate("1c fair_crps_list equals p3_eval.fair_crps on the DDM ensemble", float(np.abs(a - b_).max()) < 1e-12, {"max_abs_diff": float(np.abs(a - b_).max())})
    # 2. decomposition identity ------------------------------------------------------------------------------------------------------
    worst_id, worst_tot = 0.0, 0.0
    for h in P.HZ:
        for name, arrs in (("cov90", ddm.hit90), ("crps", ddm.crps)):
            for pv, pf in (("estimation", "validation"), ("validation", "final"), ("estimation", "final")):
                av, af = arrs[(pv, h)], arrs[(pf, h)]
                gv, gf = data.cells[(pv, h)]["b"] // 4, data.cells[(pf, h)]["b"] // 4
                wv = np.bincount(gv, minlength=5) / len(gv)
                wf = np.bincount(gf, minlength=5) / len(gf)
                cv = np.array([av[gv == g].mean() for g in range(5)])
                cf = np.array([af[gf == g].mean() for g in range(5)])
                r = P.shapley(wv, cv, wf, cf)
                direct = af.mean() - av.mean()
                worst_tot = max(worst_tot, abs(r["total"] - direct))
                worst_id = max(worst_id, abs(r["composition_order1"] + r["within_order1"] - r["total"]), abs(r["composition_order2"] + r["within_order2"] - r["total"]),
                               abs(r["composition_shapley"] + r["within_shapley"] - r["total"]))
    gate("2 decomposition identity (cov90 and mean CRPS only; both orders and Shapley; total equals the direct difference)", worst_id < 1e-12 and worst_tot < 1e-12,
         {"max_identity_error": worst_id, "max_total_error": worst_tot})
    # 3. baselines --------------------------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    worst = 0.0
    for M in (5, 50, 300):
        x = np.sort(rng.standard_normal(M))
        yy = rng.standard_normal(20)
        worst = max(worst, float(np.abs(P.fair_crps_list([x], np.zeros(20, int), yy) - P.fair_crps_brute(x, yy)).max()))
    gate("3a fair CRPS (unequal M) equals brute force on random arrays", worst < 1e-12, {"max_abs_diff": worst})
    det = {}
    ok3 = True
    for h in P.HZ:
        y, b = C.fit_set(data, h, "B-1")
        for kind in ("E1a", "E1b", "E1c"):
            S = C.fit_E1(kind, y, b)
            mono = all((np.diff(s) >= 0).all() for s in S)
            pr = P.Pred(S)
            gm = (np.diff(pr.grid, axis=1) >= 0).all()
            o = pr.per_origin(b, y)
            cov = {c_: [float(o[f"hit{c_}"][b == k].mean()) for k in range(20)] for c_ in (50, 90)}
            det[f"{kind}|h{h}"] = {"samples_sorted": bool(mono), "grid_sorted": bool(gm),
                                   "insample_cov90_min_max": [min(cov[90]), max(cov[90])], "insample_cov50_min_max": [min(cov[50]), max(cov[50])]}
            ok3 &= bool(mono and gm)
            if kind == "E1a":                                   # rank/coverage consistency of the empirical quantiles
                ok3 &= all(abs(v - 0.9) <= 0.01 for v in cov[90]) and all(abs(v - 0.5) <= 0.01 for v in cov[50])
        S = C.fit_E1("E1a", y, b)
        k = 7
        idx = np.flatnonzero(b == k)[:30]
        brute = np.array([P.fair_crps_brute(S[k], y[i])[0] for i in idx])
        fast = P.fair_crps_list(S, b[idx], y[idx])
        det[f"E1a_crps_brute|h{h}"] = float(np.abs(brute - fast).max())
        ok3 &= float(np.abs(brute - fast).max()) < 1e-12
        S2 = C.fit_E1("E1b", y, b)
        sub = np.sort(rng.choice(S2[k], 1500, replace=False))
        idx = np.flatnonzero(b == k)[:10]
        brute = np.array([P.fair_crps_brute(sub, y[i])[0] for i in idx])
        fast = P.fair_crps_list([sub], np.zeros(len(idx), int), y[idx])
        det[f"E1b_crps_brute_subsample|h{h}"] = float(np.abs(brute - fast).max())
        ok3 &= float(np.abs(brute - fast).max()) < 1e-12
    gate("3b E1a: empirical quantile ranks / in-sample coverage consistent (|cov-nominal|<=0.01 per bucket, 50% and 90%); E1b/E1c: sorted samples, "
         "monotone quantile grid, CRPS equals brute force (E1b/E1c in-sample coverage recorded, not a pass condition)", ok3, det)
    # 4. purge ----------------------------------------------------------------------------------------------------------------------------
    pdet, ok4 = {}, True
    for h in P.HZ:
        for mode, fit_pers, ev_pers in (("B-1", ("estimation",), ("validation", "final")), ("B-2", ("estimation", "validation"), ("final",))):
            end = max(raw.index[raw.t[data.cells[(p, h)]["j"]] + P1.BLOCK * h].max() for p in fit_pers)
            last_origin = max(raw.t_time[data.cells[(p, h)]["j"]].max() for p in fit_pers)
            for ev in ev_pers:
                start = pd.Timestamp(P1.PERIODS[ev][0])
                first_origin = raw.t_time[data.cells[(ev, h)]["j"]].min()
                disjoint = not set(np.concatenate([data.cells[(p, h)]["j"] for p in fit_pers])) & set(data.cells[(ev, h)]["j"])
                ok = bool(end < start and last_origin < first_origin and disjoint)
                pdet[f"{mode}|h{h}|eval {ev}"] = {"max_fit_window_end": str(end), "eval_start": str(start), "ok": ok}
                ok4 &= ok
    gate("4 purge: every fit origin's forward window ends before the evaluation period starts; fit/eval origins disjoint and ordered (B-1 and B-2)", ok4, pdet)
    # 5. no future information in the conditioning variables -----------------------------------------------------------------------------
    F = P.build_features(raw)
    ok_idx = np.flatnonzero(np.isfinite(F["vr"]))
    pick = rng.choice(ok_idx, 300, replace=False)
    close = raw.close
    err_vec, same_future = 0.0, True
    for j in pick:
        t = int(raw.t[j])
        vr1, ret1 = P.feat_one(close, t)
        err_vec = max(err_vec, abs(vr1 - F["vr"][j]) / abs(vr1), abs(ret1 - F["ret"][j]) / (abs(ret1) + 1e-12) if abs(ret1) > 1e-9 else abs(ret1 - F["ret"][j]))
        cm = close.copy()
        nf = min(6000, len(close) - t - 1)
        cm[t + 1:t + 1 + nf] = rng.uniform(close.min(), close.max(), nf)
        same_future &= P.feat_one(cm, t) == (vr1, ret1)
    gate("5 conditioning variables use origin-time information only (vectorised == single-origin; altering rows after the origin leaves them unchanged; "
         "session from the origin hour)", bool(same_future and err_vec < 1e-8), {"max_rel_err_vectorised_vs_single": float(err_vec), "unchanged_when_future_replaced": bool(same_future),
                                                                              "n_origins_tested": 300, "n_origins_with_features": int(len(ok_idx))})
    json.dump({"_code": code, "all_pass": all(v["pass"] for v in RES.values()), "gates": RES}, open(os.path.join(E.OUT, "gates.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print("ALL PASS" if all(v["pass"] for v in RES.values()) else "GATE FAILURE", flush=True)
    sys.exit(0 if all(v["pass"] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
