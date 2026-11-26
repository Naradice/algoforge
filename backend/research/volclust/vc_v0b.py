"""V0 step 2 (plan 6): source effect -- volclust_50 of the main series versus HistData USDJPY on the same windows (estimation period), plus the check that the
vectorised estimator and the window builder reproduce the stored window table of the previous study. Output: results/v0b.json"""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D


def main():
    code = E.begin("vc_v0b")
    v0a = E.load("v0a.json")
    out = {"_code": code}
    S = C.prepare(D.load_main())
    T = D.b_windows()
    W = C.windows_at(S, T["row"].values.astype(np.int64), E.L_MAIN)
    ok = ~T["excluded"].values
    v = C.volclust(W["R"][ok])
    out["reproduces_stored_windows"] = {"n": int(ok.sum()), "max_abs_diff_volclust_50": float(np.max(np.abs(v - T.loc[ok, "volclust_50"].values))),
                                        "week_equal": bool((W["week"] == T["week"].values).all())}
    built = C.build_windows(S, E.L_MAIN, E.ESTIM)
    est_rows = T.loc[T["period"] == "estimation", "row"].values
    out["reproduces_stored_windows"]["builder_rows_equal_estimation"] = bool(np.array_equal(built["row"], est_rows))
    r = out["reproduces_stored_windows"]
    if not (r["max_abs_diff_volclust_50"] <= 1e-12 and r["week_equal"] and r["builder_rows_equal_estimation"]):
        E.save("v0b.json", out)
        raise SystemExit("the vectorised estimator or the window builder does not reproduce the stored windows")
    # same windows on HistData
    offset = 0 if v0a["time_basis"]["identified_as_fixed_EST"] else v0a["time_basis"]["best"]["all"]["offset_h"]
    hist, _ = D.load_hist("USDJPY", "2021-08-31")
    H = C.prepare(hist)
    hts = H["ts"] + int(offset) * 3600
    sel = (T["period"] == "estimation") & ok
    vm, vh, n_ret = [], [], []
    for row in T.loc[sel, "row"].values.astype(np.int64):
        t0, t1 = S["ts"][row], S["ts"][row + E.L_MAIN]
        a, b = np.searchsorted(hts, t0), np.searchsorted(hts, t1, side="right")
        if b - a - 1 < E.SOURCE_MIN_RETURNS:
            continue
        r = np.diff(np.log(H["close"][a:b]))
        vh.append(C.volclust(r[None, :])[0])
        vm.append(C.volclust(W_row(S, row))[0])
        n_ret.append(len(r))
    vm, vh = np.array(vm), np.array(vh)
    fin = np.isfinite(vm) & np.isfinite(vh)
    vm, vh = vm[fin], vh[fin]
    mad = float(np.mean(np.abs(vh - vm))) if len(vm) else None
    enough = len(vm) >= E.SOURCE_MIN_WINDOWS
    affected = bool((not enough) or mad > E.DELTA / 2)
    out["source_effect"] = {"offset_h_used": int(offset), "n_estimation_windows": int(sel.sum()), "n_matched_windows": int(len(vm)), "min_returns_for_match": E.SOURCE_MIN_RETURNS,
                            "mean_abs_diff": mad, "mean_diff_hist_minus_main": float(np.mean(vh - vm)) if len(vm) else None, "median_abs_diff": float(np.median(np.abs(vh - vm))) if len(vm) else None,
                            "corr": float(np.corrcoef(vm, vh)[0, 1]) if len(vm) > 2 else None, "mean_main": float(vm.mean()) if len(vm) else None, "mean_hist": float(vh.mean()) if len(vm) else None,
                            "threshold_mean_abs_diff": E.DELTA / 2, "enough_matched_windows": bool(enough), "source_effect_flag": affected, "hold_independent_judgement": affected}
    E.save("v0b.json", out)
    print(out["source_effect"], flush=True)
    E.done()


def W_row(S, row):
    return np.diff(np.log(S["close"][row:row + E.L_MAIN + 1]))[None, :]


if __name__ == "__main__":
    main()
