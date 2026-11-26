"""R2 = V1 (plan 6, plan r1 5): the raw difference. Primary judgement on the independent period (only if R1 passed); the other 8 pairs (converted time basis)
descriptive. The raw values of the main series are not recomputed (see results/v1.json of the original run). Output: results_r1/v1.json, v1_sensitivity.json"""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D
import dr_env as DE


def real_block(W, B, rng, ddm_base, ddm_ext, small, t_ref=False, median=False):
    v = C.volclust(W["R"])
    units, unit = np.unique(W["week"], return_inverse=True)
    bt = C.real_layer(v, unit, len(units), B, rng, median=median)
    base = C.compare(v.mean(), bt["raw"], ddm_base, B, rng, small, t_ref)
    ext = C.compare(v.mean(), bt["raw"], ddm_ext, B, rng, small, t_ref)
    out = {"n_windows": int(len(v)), "n_weeks": int(len(units)), "mean": float(v.mean()), "median": float(np.median(v)),
           "quantiles": {q: float(np.quantile(v, q / 100)) for q in (10, 25, 75, 90)}, "base": base, "extended": ext, "combined_classification": C.combine(base, ext)}
    if median:
        med = float(np.median(v))
        mb = C.compare(med, bt["raw_median"], ddm_base, B, rng, small)
        out["median_sensitivity"] = mb
    return out, v


def window_level(v_real, base, ok, vals_pm):
    """(a) all windows of all base points, (b) windows of the argmax point (selected on the same data), per set."""
    out = {}
    R = D.a1()["runs"]
    from dr_a1_stats import IDX
    for s in ("SA", "SB"):
        allw = np.concatenate([R[(s, p, k)]["win"][DE.TPC_MAIN][:, IDX["volclust_50"]] for p in base for k in sorted(ok[(p, s)])])
        i = int(np.argmax(vals_pm[s]))
        arg = np.concatenate([R[(s, base[i], k)]["win"][DE.TPC_MAIN][:, IDX["volclust_50"]] for k in sorted(ok[(base[i], s)])])
        res = {}
        for name, w in (("all_points", allw), ("argmax_point", arg)):
            sw = np.sort(w)
            lo, hi = np.searchsorted(sw, v_real, "left"), np.searchsorted(sw, v_real, "right")
            res[name] = {"n_ddm_windows": int(len(w)), "ddm_mean": float(w.mean()), "ddm_sd": float(w.std(ddof=1)), "ddm_q95": float(np.quantile(w, 0.95)),
                         "fraction_real_above_ddm_q95": float(np.mean(v_real > np.quantile(w, 0.95))), "p_real_greater_than_ddm": float(np.mean((lo + hi) / 2 / len(w)))}
        res["argmax_pid"] = int(base[i])
        out[s] = res
    return out


def ddm_estimators(base, ok, tpc, L, names):
    """{estimator: {set: [point means]}} recomputed from the stored trade series."""
    out = {n: {"SA": [], "SB": []} for n in names}
    for s in ("SA", "SB"):
        for p in base:
            per = {n: [] for n in names}
            for k in sorted(ok[(p, s)]):
                R = D.ddm_returns(s, p, k, tpc, L)
                for n in names:
                    per[n].append(C.estimator(R, n).mean())
            for n in names:
                out[n][s].append(np.mean(per[n]))
    return out


def main():
    code = E.begin("vc_v1")
    r0, r1 = E.load("r0.json"), E.load("r1.json")
    hold = bool(r1["hold_independent_judgement"])
    small = E.load_previous("v0c.json")["small_category_enabled"]        # V0 is not rerun (plan r1 5)
    base, ext, ok = D.ddm_points()
    vb, ve = D.ddm_run_values(base, ok), D.ddm_run_values(ext, ok)
    dbase, dext = D.padded(vb), D.padded(ve)
    rng = np.random.default_rng(E.SEEDS["real"])
    out = {"_code": code, "independent_held": bool(hold), "small_category_enabled": bool(small), "n_points_base": len(base), "n_points_extended": len(ext), "datasets": {}}
    out["extended_points"] = [{"pid": int(p), "n_SA": len(ok[(p, "SA")]), "n_SB": len(ok[(p, "SB")])} for p in ext if p not in base]
    pm = {s: C.point_means(*dbase[s]) for s in ("SA", "SB")}
    out["ddm_point_means"] = {s: {"max": float(pm[s].max()), "argmax_pid": int(base[int(pm[s].argmax())]), "median": float(np.median(pm[s])), "p95": float(np.percentile(pm[s], 95))} for s in pm}
    real = None
    # independent (primary)
    if not hold:
        hist, info = D.load_hist("USDJPY", E.INDEP[1])
        S = C.prepare(hist)
        W = C.keep(C.build_windows(S, E.L_MAIN, E.INDEP))
        if len(W["row"]) >= 300:
            blk, v = real_block(W, E.B, rng, dbase, dext, small, t_ref=True, median=True)
            yr = (W["start_ts"].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
            blk["by_year_mean"] = {str(y): {"n": int((yr == y).sum()), "mean": float(v[yr == y].mean())} for y in np.unique(yr)}
            out["datasets"]["independent_USDJPY"] = {"role": "primary", **blk}
            real = (S, W)
            out["window_level_independent"] = window_level(v, base, ok, pm)
        else:
            out["datasets"]["independent_USDJPY"] = {"role": "primary", "not_judged": "fewer than 300 valid windows", "n_windows": int(len(W["row"]))}
        out["independent_source"] = info
    else:
        out["datasets"]["independent_USDJPY"] = {"role": "primary", "not_judged": "held: R0 or R1 failed (values not computed)"}
    E.save("v1.json", out)
    # secondary: other pairs (they depend on R0 only)
    for sym in E.PAIRS:
        if not (r0["usdjpy_ok"] and r0["pairs"][sym]["passes"]):
            out["datasets"][sym] = {"role": "secondary", "not_computed": "time-basis check of R0 failed"}
            continue
        hist, _ = D.load_hist(sym, E.SECOND[1])
        W = C.keep(C.build_windows(C.prepare(hist), E.L_MAIN, E.SECOND))
        if len(W["row"]) >= 300:
            blk, _ = real_block(W, E.B, rng, dbase, dext, small)
            out["datasets"][sym] = {"role": "secondary", **blk}
        else:
            out["datasets"][sym] = {"role": "secondary", "not_judged": "fewer than 300 valid windows", "n_windows": int(len(W["row"]))}
    E.save("v1.json", out)
    E.save("v1.json", out)
    # estimator sensitivity (descriptive; independent period only)
    if real is not None:
        S, W = real
        sens = {"_code": code, "estimators": {}, "L4000": {}, "ddm_tpc": {}}
        de = ddm_estimators(base, ok, DE.TPC_MAIN, E.L_MAIN, C.ESTIMATORS)
        sens["check_recomputed_lag50_vs_stored_max_abs_diff"] = max(float(np.max(np.abs(np.array(de["lag50"][s]) - pm[s]))) for s in ("SA", "SB"))
        for n in C.ESTIMATORS:
            row = {"ddm_max": {s: float(np.max(de[n][s])) for s in ("SA", "SB")}}
            v = C.estimator(W["R"], n)
            row["independent_USDJPY"] = {"mean": float(v.mean()), "Delta": {s: float(v.mean() - row["ddm_max"][s]) for s in ("SA", "SB")}}
            sens["estimators"][n] = row
        d4 = ddm_estimators(base, ok, DE.TPC_MAIN, E.L_SENS, ("lag50",))["lag50"]
        row = {"ddm_max": {s: float(np.max(d4[s])) for s in ("SA", "SB")}}
        v = C.volclust(C.keep(C.build_windows(S, E.L_SENS, E.INDEP))["R"])
        row["independent_USDJPY"] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "Delta": {s: float(v.mean() - row["ddm_max"][s]) for s in ("SA", "SB")} if len(v) else None}
        sens["L4000"] = row
        for tpc in (10, 40):
            vals = D.padded(D.ddm_run_values(base, ok, tpc))
            sens["ddm_tpc"][str(tpc)] = {s: float(C.point_means(*vals[s]).max()) for s in ("SA", "SB")}
        E.save("v1_sensitivity.json", sens)
    d = out["datasets"]["independent_USDJPY"]
    print({"independent": d.get("combined_classification", d.get("not_judged")), "mean": d.get("mean")}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
