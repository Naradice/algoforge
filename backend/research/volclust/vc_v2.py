"""V2 (plan 6): intraday seasonality. Primary adjustment: America/New_York 15-minute profile of |r| (fixed EST -> UTC -> New York), estimated from the other weeks of the
same period and source, re-estimated inside every week-block bootstrap replicate. Judgement on the independent period; R-A and validation descriptive. Output: results/v2.json"""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D
import dr_env as DE


def adjusted_block(S, W, period, B, rng, dbase, dext, small, parity=None, t_ref=False):
    """W: valid windows (possibly one parity); the profile uses every pool return of the period."""
    pool = C.pool_returns(S, period)
    se = C.Season(pool, W, "ny15")
    ones = np.ones(len(se.units))
    raw = C.volclust(W["R"])
    adj = C.volclust(se.adjusted(W["R"], ones))
    bt = C.real_layer(raw, se.win_unit, len(se.units), B, rng, season=se, R=W["R"])
    base = C.compare(adj.mean(), bt["adj"], dbase, B, rng, small, t_ref)
    ext = C.compare(adj.mean(), bt["adj"], dext, B, rng, small, t_ref)
    frac = {}
    for s in ("SA", "SB"):
        mx = base["by_set"][s]["mu_max"]
        d_raw, d_adj = bt["raw"] - mx, bt["adj"] - mx
        point = 1 - (adj.mean() - mx) / (raw.mean() - mx)
        frac[s] = {"point": float(point), "real_layer_90": C.interval(1 - d_adj / d_raw), "replicates_with_nonpositive_raw_Delta": int((d_raw <= 0).sum())}
    P = se.profiles(ones)
    prof = {"n_units": int(len(se.units)), "min_bin_count": float(se.N.sum(axis=0).min()), "bins_below_min_total": int((se.N.sum(axis=0) < E.MIN_BIN_OBS).sum()),
            "median_profile_max_over_min": float(np.nanmax(np.median(P, axis=0)) / np.nanmin(np.median(P, axis=0))), "n_nonfinite_adjusted_windows": int((~np.isfinite(adj)).sum())}
    out = {"n_windows": int(len(raw)), "raw_mean": float(raw.mean()), "adjusted_mean": float(adj.mean()), "adjusted_median": float(np.median(adj)), "base": base, "extended": ext,
           "combined_classification": C.combine(base, ext), "fraction_explained": frac, "profile": prof}
    return out, adj, se


def variants(S, W, period, maxima):
    """descriptive sensitivities: point estimates of the mean and Delta against the base maxima."""
    pool = C.pool_returns(S, period)
    R = W["R"]
    res = {}

    def put(name, v, extra=None):
        v = v[np.isfinite(v)]
        res[name] = {"n": int(len(v)), "mean": float(v.mean()), "Delta": {s: float(v.mean() - maxima[s]) for s in maxima}, **(extra or {})}
    for scheme in ("est15", "ny_wd30"):
        se = C.Season(pool, W, scheme)
        put(f"adjusted_{scheme}", C.volclust(se.adjusted(R, np.ones(len(se.units)))))
    q = np.quantile(np.abs(R), 0.99, axis=1, keepdims=True)
    put("raw_winsorized_q99", C.volclust(np.sign(R) * np.minimum(np.abs(R), q)))
    se = C.Season(pool, W, "ny15")
    A = se.adjusted(R, np.ones(len(se.units)))
    qa = np.quantile(np.abs(A), 0.99, axis=1, keepdims=True)
    put("adjusted_winsorized_q99", C.volclust(np.sign(A) * np.minimum(np.abs(A), qa)))
    z = W["zero_frac"] <= 0.10
    put("raw_zero_frac_le_0.10", C.volclust(R[z]))
    put("adjusted_zero_frac_le_0.10", C.volclust(A[z]))
    day = (pool["ts"] + 7 * 3600) // 86400
    ud, inv = np.unique(day, return_inverse=True)
    level = np.bincount(inv, weights=np.abs(pool["r"])) / np.bincount(inv)
    wd = (W["ret_ts"] + 7 * 3600) // 86400
    j = np.clip(np.searchsorted(ud, wd), 0, len(ud) - 1)
    scale = np.where(ud[j] == wd, level[j], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        put("day_level_normalized", C.volclust(R / scale))
    return res


def injection(base, ok, se, W, rng):
    """multiply every DDM window by the seasonal profile of a randomly assigned real window's time stamps and recompute volclust_50."""
    P = se.profiles(np.ones(len(se.units)))
    prof = P[se.win_unit[:, None], se.wb]                       # [real windows, L]
    prof = prof / prof.mean(axis=1, keepdims=True)
    out = {}
    for s in ("SA", "SB"):
        before, after = [], []
        for p in base:
            b, a = [], []
            for k in sorted(ok[(p, s)]):
                R = D.ddm_returns(s, p, k, DE.TPC_MAIN, E.L_MAIN)
                j = rng.integers(0, len(prof), len(R))
                b.append(C.volclust(R).mean())
                a.append(C.volclust(R * prof[j]).mean())
            before.append(np.mean(b))
            after.append(np.mean(a))
        before, after = np.array(before), np.array(after)
        out[s] = {"max_before": float(before.max()), "max_after": float(after.max()), "mean_before": float(before.mean()), "mean_after": float(after.mean()), "mean_increase": float((after - before).mean())}
    return out


def main():
    code = E.begin("vc_v2")
    v0a, v1 = E.load("v0a.json"), E.load("v1.json")
    hold, small = v1["independent_held"], v1["small_category_enabled"]
    base, ext, ok = D.ddm_points()
    dbase, dext = D.padded(D.ddm_run_values(base, ok)), D.padded(D.ddm_run_values(ext, ok))
    maxima = {s: float(C.point_means(*dbase[s]).max()) for s in ("SA", "SB")}
    rng = np.random.default_rng(E.SEEDS["real"] + 1)
    out = {"_code": code, "datasets": {}, "sensitivity": {}}
    inj_source = None
    if not hold and "base" in v1["datasets"]["independent_USDJPY"]:
        hist, _ = D.load_hist("USDJPY", E.INDEP[1])
        S = C.prepare(hist)
        W = C.keep(C.build_windows(S, E.L_MAIN, E.INDEP))
        blk, _, se = adjusted_block(S, W, E.INDEP, E.B, rng, dbase, dext, small, t_ref=True)
        out["datasets"]["independent_USDJPY"] = {"role": "primary", **blk}
        E.save("v2.json", out)
        out["sensitivity"]["independent_USDJPY"] = variants(S, W, E.INDEP, maxima)
        inj_source = ("independent_USDJPY", se, W)
    else:
        out["datasets"]["independent_USDJPY"] = {"role": "primary", "not_judged": "held or not judged in V1 (values not computed)"}
    if v0a["time_basis"]["identified_as_fixed_EST"]:
        T = D.b_windows()
        S = C.prepare(D.load_main())
        for name, period, mask in (("R-A", E.ESTIM, (T["period"] == "estimation") & (T["parity"] == 1)), ("validation", E.VALID, T["period"] == "validation")):
            W = C.windows_at(S, T.loc[mask & (~T["excluded"]), "row"].values.astype(np.int64), E.L_MAIN)
            blk, _, se = adjusted_block(S, W, period, E.B, rng, dbase, dext, small)
            out["datasets"][name] = {"role": "descriptive", **blk}
            out["sensitivity"][name] = variants(S, W, period, maxima)
            if inj_source is None and name == "R-A":
                inj_source = (name, se, W)
    else:
        out["datasets"]["main_series"] = {"not_computed": "time basis of the main series not identified (plan V0 step 1)"}
    if inj_source is not None:
        out["injection_control"] = {"profile_from": inj_source[0], **injection(base, ok, inj_source[1], inj_source[2], np.random.default_rng(E.SEEDS["inject"]))}
    raw = v1["datasets"]["independent_USDJPY"].get("combined_classification")
    adj = out["datasets"]["independent_USDJPY"].get("combined_classification")
    key = str(E.DELTA)
    out["v4_condition"] = {"raw": raw[key] if raw else None, "adjusted": adj[key] if adj else None, "run_v4": bool(raw and adj and raw[key] == "差あり" and adj[key] == "差あり")}
    E.save("v2.json", out)
    print(out["v4_condition"], flush=True)
    E.done()


if __name__ == "__main__":
    main()
