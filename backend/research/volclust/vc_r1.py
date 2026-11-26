"""R1 (plan r1 3): reproduction of the stored values and the revised source criterion -- the mean of volclust_50(HistData) - volclust_50(main) on the same windows,
week-block bootstrap 90% interval within +-delta/4 in both the estimation and the validation period. The original criterion (mean absolute difference) is
reported beside it and judges nothing. The independent period is not read. Output: results_r1/r1.json"""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D


def source_period(S, H, T, period, rng):
    """same matching as the original step: HistData rows inside the time span of the main window, at least SOURCE_MIN_RETURNS returns."""
    sel = (T["period"] == period) & (~T["excluded"])
    vm, vh, week = [], [], []
    for row, wk in zip(T.loc[sel, "row"].values.astype(np.int64), T.loc[sel, "week"].values.astype(np.int64)):
        t0, t1 = S["ts"][row], S["ts"][row + E.L_MAIN]
        a, b = np.searchsorted(H["ts"], t0), np.searchsorted(H["ts"], t1, side="right")
        if b - a - 1 < E.SOURCE_MIN_RETURNS:
            continue
        h = C.volclust(np.diff(np.log(H["close"][a:b]))[None, :])[0]
        m = C.volclust(np.diff(np.log(S["close"][row:row + E.L_MAIN + 1]))[None, :])[0]
        if np.isfinite(h) and np.isfinite(m):
            vm.append(m)
            vh.append(h)
            week.append(wk)
    vm, vh = np.array(vm), np.array(vh)
    n_valid, n = int(sel.sum()), int(len(vm))
    required = max(E.SOURCE_MIN_MATCHED, int(np.ceil(E.SOURCE_MIN_FRACTION * n_valid)))
    out = {"n_valid_windows": n_valid, "n_matched_windows": n, "n_required": required, "enough_matched": bool(n >= required)}
    if n:
        d = vh - vm
        units, unit = np.unique(np.array(week), return_inverse=True)
        iv = C.interval(C.real_layer(d, unit, len(units), E.B, rng)["raw"])
        out.update(n_weeks=int(len(units)), mean_diff_hist_minus_main=float(d.mean()), mean_diff_90=iv, limit=E.SOURCE_BIAS_LIMIT,
                   interval_within_limit=bool(iv[0] >= -E.SOURCE_BIAS_LIMIT and iv[1] <= E.SOURCE_BIAS_LIMIT), mean_main=float(vm.mean()), mean_hist=float(vh.mean()),
                   original_criterion={"mean_abs_diff": float(np.abs(d).mean()), "threshold": E.DELTA / 2, "met": bool(np.abs(d).mean() <= E.DELTA / 2)},
                   median_abs_diff=float(np.median(np.abs(d))), corr=float(np.corrcoef(vm, vh)[0, 1]) if n > 2 else None)
    out["passes"] = bool(out["enough_matched"] and out.get("interval_within_limit", False))
    return out


def main():
    code = E.begin("vc_r1")
    r0 = E.load("r0.json")
    out = {"_code": code}
    S = C.prepare(D.load_main())
    T = D.b_windows()
    ok = ~T["excluded"].values
    W = C.windows_at(S, T["row"].values.astype(np.int64), E.L_MAIN)
    built = C.build_windows(S, E.L_MAIN, E.ESTIM)
    rep = {"n": int(ok.sum()), "max_abs_diff_volclust_50": float(np.max(np.abs(C.volclust(W["R"][ok]) - T.loc[ok, "volclust_50"].values))),
           "week_equal": bool((W["week"] == T["week"].values).all()), "builder_rows_equal_estimation": bool(np.array_equal(built["row"], T.loc[T["period"] == "estimation", "row"].values))}
    out["reproduces_stored_windows"] = rep
    if not (rep["max_abs_diff_volclust_50"] <= 1e-12 and rep["week_equal"] and rep["builder_rows_equal_estimation"]):
        E.save("r1.json", out)
        raise SystemExit("the vectorised estimator or the window builder does not reproduce the stored windows")
    if not r0["usdjpy_ok"]:
        out.update(not_run="USDJPY failed the time-basis check of R0", source_passes=False, hold_independent_judgement=True)
    else:
        H = C.prepare(D.load_hist("USDJPY", E.VALID[1])[0])
        rng = np.random.default_rng(E.SEEDS["source"])
        out["periods"] = {p: source_period(S, H, T, p, rng) for p in ("estimation", "validation")}
        out["source_passes"] = bool(all(v["passes"] for v in out["periods"].values()))
        out["hold_independent_judgement"] = not out["source_passes"]
    E.save("r1.json", out)
    print({"hold_independent": out["hold_independent_judgement"],
           **{p: (v.get("n_matched_windows"), v.get("mean_diff_hist_minus_main"), v.get("mean_diff_90"), v["passes"]) for p, v in out.get("periods", {}).items()}}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
