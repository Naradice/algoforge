"""V1 (plan 6): reproducibility of the raw difference. Primary judgement on the independent period (unless held by V0 step 2); everything else descriptive.
Output: results/v1.json"""
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
    v0b, v0c = E.load("v0b.json"), E.load("v0c.json")
    hold = v0b["source_effect"]["hold_independent_judgement"]
    small = v0c["small_category_enabled"]
    base, ext, ok = D.ddm_points()
    vb, ve = D.ddm_run_values(base, ok), D.ddm_run_values(ext, ok)
    dbase, dext = D.padded(vb), D.padded(ve)
    rng = np.random.default_rng(E.SEEDS["real"])
    out = {"_code": code, "independent_held": bool(hold), "small_category_enabled": bool(small), "n_points_base": len(base), "n_points_extended": len(ext), "datasets": {}}
    out["extended_points"] = [{"pid": int(p), "n_SA": len(ok[(p, "SA")]), "n_SB": len(ok[(p, "SB")])} for p in ext if p not in base]
    pm = {s: C.point_means(*dbase[s]) for s in ("SA", "SB")}
    out["ddm_point_means"] = {s: {"max": float(pm[s].max()), "argmax_pid": int(base[int(pm[s].argmax())]), "median": float(np.median(pm[s])), "p95": float(np.percentile(pm[s], 95))} for s in pm}
    real = {}
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
            real["independent_USDJPY"] = (S, W, v)
            out["window_level_independent"] = window_level(v, base, ok, pm)
        else:
            out["datasets"]["independent_USDJPY"] = {"role": "primary", "not_judged": "fewer than 300 valid windows", "n_windows": int(len(W["row"]))}
        out["independent_source"] = info
    else:
        out["datasets"]["independent_USDJPY"] = {"role": "primary", "not_judged": "held: source effect flag of V0 step 2 (values not computed)"}
    # descriptive: estimation halves and validation of the main series
    T = D.b_windows()
    Sm = C.prepare(D.load_main())
    for name, mask in (("R-A", (T["period"] == "estimation") & (T["parity"] == 1)), ("R-B", (T["period"] == "estimation") & (T["parity"] == 0)), ("validation", T["period"] == "validation")):
        rows = T.loc[mask & (~T["excluded"]), "row"].values.astype(np.int64)
        W = C.windows_at(Sm, rows, E.L_MAIN)
        blk, v = real_block(W, E.B, rng, dbase, dext, small)
        out["datasets"][name] = {"role": "descriptive", **blk}
        if name == "R-A":
            real["R-A"] = (Sm, W, v)
            out["window_level_RA"] = window_level(v, base, ok, pm)
    sub = T[(~T["excluded"]) & T["period"].isin(["estimation", "validation"])].copy()
    sub["year"] = sub["start"].str[:4]
    out["main_series_breakdown"] = {key: {str(k): {"n": int(len(g)), "mean": float(g["volclust_50"].mean()), "median": float(g["volclust_50"].median())} for k, g in sub.groupby(key)} for key in ("year", "atr_group", "hour")}
    # secondary: other pairs
    for sym in E.PAIRS:
        hist, _ = D.load_hist(sym, E.SECOND[1])
        W = C.keep(C.build_windows(C.prepare(hist), E.L_MAIN, E.SECOND))
        if len(W["row"]) >= 300:
            blk, _ = real_block(W, E.B, rng, dbase, dext, small)
            out["datasets"][sym] = {"role": "secondary", **blk}
        else:
            out["datasets"][sym] = {"role": "secondary", "not_judged": "fewer than 300 valid windows", "n_windows": int(len(W["row"]))}
    E.save("v1.json", out)
    # estimator sensitivity (descriptive)
    sens = {"_code": code, "estimators": {}, "L4000": {}, "ddm_tpc": {}}
    de = ddm_estimators(base, ok, DE.TPC_MAIN, E.L_MAIN, C.ESTIMATORS)
    chk = max(float(np.max(np.abs(np.array(de["lag50"][s]) - pm[s]))) for s in ("SA", "SB"))
    sens["check_recomputed_lag50_vs_stored_max_abs_diff"] = chk
    for n in C.ESTIMATORS:
        row = {"ddm_max": {s: float(np.max(de[n][s])) for s in ("SA", "SB")}}
        for key, (_, W, _) in real.items():
            v = C.estimator(W["R"], n)
            row[key] = {"mean": float(v.mean()), "Delta": {s: float(v.mean() - row["ddm_max"][s]) for s in ("SA", "SB")}}
        sens["estimators"][n] = row
    d4 = ddm_estimators(base, ok, DE.TPC_MAIN, E.L_SENS, ("lag50",))["lag50"]
    row = {"ddm_max": {s: float(np.max(d4[s])) for s in ("SA", "SB")}}
    for key, (S, _, _) in real.items():
        period = E.INDEP if key != "R-A" else E.ESTIM
        W4 = C.keep(C.build_windows(S, E.L_SENS, period))
        if key == "R-A":
            odd = (W4["week"] % 100) % 2 == 1
            W4 = {k: v[odd] for k, v in W4.items()}
        v = C.volclust(W4["R"])
        row[key] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "Delta": {s: float(v.mean() - row["ddm_max"][s]) for s in ("SA", "SB")} if len(v) else None}
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
