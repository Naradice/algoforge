"""V0 step 1 (plan 6): input checks, time basis of the main series (by matching HistData USDJPY rows), and the number of valid windows/weeks of the
independent data. No volclust_50 is computed. Output: results/v0a.json"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import vc_env as E
import vc_core as C
import vc_data as D
import dr_env as DE
from dr_a1_stats import IDX


def a1_checks():
    runs, meta = D.a1()["runs"], D.a1()["meta"]
    succ = [(m["set"], m["pid"], m["k"]) for m in meta if m["success"]]
    missing = [x for x in succ if x not in runs]
    bad = [x for x in succ if x in runs and not (len(runs[x]["win"][DE.TPC_MAIN]) == 5 and np.isfinite(runs[x]["win"][DE.TPC_MAIN][:, IDX["volclust_50"]]).all())]
    out = {"n_success": len(succ), "n_in_stats": len(runs), "n_missing": len(missing), "n_bad_windows": len(bad)}
    if not E.SYNTH:
        h = hashlib.sha256(open(E.A1_STATS, "rb").read()).hexdigest()
        out.update(sha256=h, sha256_matches=h == E.A1_STATS_SHA)
    return out


def time_basis(main, hist):
    """correlation of 1-minute returns on matched time stamps for HistData shifted by -12..+12 h (whole period, summer, winter)."""
    def rets(df):
        s = np.log(df["close"])
        r = s.diff()
        dt = df.index.to_series().diff().dt.total_seconds()
        return r[dt.values == 60]
    p0, p1 = pd.Timestamp(E.ESTIM[0]), pd.Timestamp(E.ESTIM[1]) + pd.Timedelta(days=1)
    rm = rets(main[(main.index >= p0) & (main.index < p1)])
    rh = rets(hist[(hist.index >= p0 - pd.Timedelta(days=2)) & (hist.index < p1 + pd.Timedelta(days=2))])
    res = {}
    for off in range(-12, 13):
        sh = rh.copy()
        sh.index = sh.index + pd.Timedelta(hours=off)
        j = pd.concat([rm.rename("m"), sh.rename("h")], axis=1, join="inner").dropna()
        row = {}
        for name, mask in (("all", np.ones(len(j), bool)), ("summer", j.index.month.isin([4, 5, 6, 7, 8, 9, 10])), ("winter", j.index.month.isin([12, 1, 2]))):
            x = j[mask]
            row[name] = {"n": int(len(x)), "corr": float(x["m"].corr(x["h"])) if len(x) > 1000 else None}
        res[off] = row
    best = {}
    for name in ("all", "summer", "winter"):
        c = {o: v[name]["corr"] for o, v in res.items() if v[name]["corr"] is not None}
        order = sorted(c, key=lambda o: -c[o])
        best[name] = {"offset_h": order[0] if order else None, "corr": c[order[0]] if order else None, "second_corr": c[order[1]] if len(order) > 1 else None}
    a = best["all"]
    identified = bool(a["offset_h"] is not None and all(best[n]["offset_h"] == 0 for n in best) and a["corr"] >= 0.5 and a["corr"] >= 3 * max(a["second_corr"] or 0.0, 0.0))
    j = pd.concat([main["close"].rename("m"), hist["close"].rename("h")], axis=1, join="inner")
    j = j[(j.index >= p0) & (j.index < p1)]
    return {"criterion": "fixed EST if the zero offset is best overall, in summer and in winter, with correlation >= 0.5 and >= 3x the second best", "identified_as_fixed_EST": identified,
            "best": best, "corr_by_offset_all": {str(o): v["all"]["corr"] for o, v in res.items()},
            "matched_rows_offset0": int(len(j)), "main_rows_in_period": int(((main.index >= p0) & (main.index < p1)).sum()),
            "fraction_exactly_equal_close": float(np.mean(j["m"].values == j["h"].values)) if len(j) else None,
            "median_abs_close_difference": float(np.median(np.abs(j["m"].values - j["h"].values))) if len(j) else None}


def main():
    code = E.begin("vc_v0a", first=True)
    out = {"_code": code, "a1_stats": a1_checks()}
    main_df = D.load_main()
    hist, info = D.load_hist("USDJPY", "2021-08-31")
    out["hist_usdjpy"] = {**info, "rows_loaded": int(len(hist)), "first": str(hist.index[0]), "last_loaded": str(hist.index[-1])}
    out["time_basis"] = time_basis(main_df, hist)
    S = C.prepare(hist[hist.index < pd.Timestamp(E.INDEP[1]) + pd.Timedelta(days=1)])
    W = C.build_windows(S, E.L_MAIN, E.INDEP)
    K = C.keep(W)
    out["independent_counts"] = {"n_windows_built": int(len(W["row"])), "n_excluded": int(W["excluded"].sum()), "n_valid_windows": int(len(K["row"])),
                                 "n_weeks_with_valid_windows": int(len(np.unique(K["week"]))), "zero_frac_median": float(np.median(W["zero_frac"])), "zero_frac_p90": float(np.percentile(W["zero_frac"], 90)),
                                 "enough_windows_ge_300": bool(len(K["row"]) >= 300)}
    a = out["a1_stats"]
    ok = a["n_missing"] == 0 and a["n_bad_windows"] == 0 and a.get("sha256_matches", True)
    out["inputs_ok"] = bool(ok)
    E.save("v0a.json", out)
    print({"inputs_ok": ok, "time_basis": out["time_basis"]["identified_as_fixed_EST"], "independent": out["independent_counts"]}, flush=True)
    E.done()
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
