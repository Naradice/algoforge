"""R0 (plan r1 2.3): checks of the time basis after conversion, from time stamps and 1-minute return correlations only. No volclust_50 is computed.
(a) share of trading weeks whose first row is Monday 00:00-00:29: independent USDJPY, the 8 other pairs, the main series; (b) USDJPY versus the main series,
estimation and validation: share of weeks in which the correlation at offset 0 exceeds the correlations at -1 h and +1 h. Output: results_r1/r0.json"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vc_env as E
import vc_data as D
from vc_v0a import a1_checks


def weekly_open(df, period):
    """first row of every trading week (ISO week of time stamp + 6 h) whose Monday lies in the period."""
    idx = df.index
    shifted = idx + pd.Timedelta(hours=E.WEEK_SHIFT_HOURS)
    monday = shifted.normalize() - pd.to_timedelta(shifted.dayofweek.values, unit="D")
    m = (monday >= pd.Timestamp(period[0])) & (monday <= pd.Timestamp(period[1]))
    first = pd.Series(idx[m], index=monday[m]).groupby(level=0).min()
    ok = (first.dt.dayofweek.values == 0) & (first.dt.hour.values == 0) & (first.dt.minute.values < E.OPEN_MINUTES)
    share = float(ok.mean()) if len(ok) else 0.0
    return {"period": list(period), "n_weeks": int(len(ok)), "n_conforming": int(ok.sum()), "share": share, "passes": bool(len(ok) > 0 and share >= E.OPEN_CONFORM),
            "nonconforming_first_rows": [str(t) for t in first[~ok].iloc[:60]]}


def returns(df):
    r = np.log(df["close"]).diff()
    dt = df.index.to_series().diff().dt.total_seconds()
    return r[(dt.values == 60) & np.isfinite(r.values)]


def weekly_correlation(main, hist, period):
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    rm = returns(main[(main.index >= a) & (main.index < b)])
    rh = returns(hist[(hist.index >= a - pd.Timedelta(days=2)) & (hist.index < b + pd.Timedelta(days=2))])
    cols = {}
    for off in (-1, 0, 1):
        sh = rh.copy()
        sh.index = sh.index + pd.Timedelta(hours=off)
        j = pd.concat([rm.rename("m"), sh.rename("h")], axis=1, join="inner")
        iso = (j.index + pd.Timedelta(hours=E.WEEK_SHIFT_HOURS)).isocalendar()
        week = iso["year"].values.astype(np.int64) * 100 + iso["week"].values.astype(np.int64)
        g = j.groupby(week)
        cols[f"n{off}"] = g.size()
        cols[f"c{off}"] = pd.Series({w: (x["m"].corr(x["h"]) if len(x) > 2 else np.nan) for w, x in g}, dtype=float)
    t = pd.DataFrame(cols)
    t = t[t["n0"] >= E.CORR_MIN_MATCHED]
    c0 = np.nan_to_num(t["c0"].values.astype(float), nan=-np.inf)
    ok = np.isfinite(c0)
    for off in (-1, 1):
        other = np.where(t[f"n{off}"].fillna(0).values >= E.CORR_MIN_SHIFTED, t[f"c{off}"].values.astype(float), np.nan)
        ok &= ~(np.nan_to_num(other, nan=-np.inf) >= c0)
    share = float(ok.mean()) if len(ok) else 0.0
    med = float(np.nanmedian(t["c0"])) if len(t) else float("nan")
    bad = t[~ok]
    return {"period": list(period), "n_weeks": int(len(t)), "n_offset0_best": int(ok.sum()), "share": share, "median_corr_offset0": med,
            "failing_weeks": {str(w): {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()} for w, row in bad.iloc[:60].iterrows()},
            "passes": bool(len(t) > 0 and share >= E.CORR_CONFORM and med >= E.CORR_MEDIAN)}


def main():
    code = E.begin("vc_r0", first=True)
    out = {"_code": code, "a1_stats": a1_checks()}
    a = out["a1_stats"]
    if not (a["n_missing"] == 0 and a["n_bad_windows"] == 0 and a.get("sha256_matches", True)):
        E.save("r0.json", out)
        raise SystemExit(2)
    main_df = D.load_main()
    main_df = main_df[~main_df.index.duplicated(keep="first")]
    out["main_open"] = weekly_open(main_df, (E.ESTIM[0], E.VALID[1]))
    hist, info = D.load_hist("USDJPY", E.VALID[1])
    out["usdjpy_info"] = info
    out["usdjpy_open_independent"] = weekly_open(hist, E.INDEP)
    out["usdjpy_weekly_correlation"] = weekly_correlation(main_df, hist, (E.ESTIM[0], E.VALID[1]))      # estimation and validation as one set of weeks
    out["usdjpy_ok"] = bool(out["usdjpy_open_independent"]["passes"] and out["usdjpy_weekly_correlation"]["passes"])
    out["main_ok"] = bool(out["main_open"]["passes"])
    out["pairs"] = {}
    for sym in E.PAIRS:
        h, inf = D.load_hist(sym, E.SECOND[1])
        out["pairs"][sym] = {**weekly_open(h, E.SECOND), "info": inf}
    E.save("r0.json", out)
    c = out["usdjpy_weekly_correlation"]
    print({"main_ok": out["main_ok"], "main_share": out["main_open"]["share"], "usdjpy_ok": out["usdjpy_ok"], "usdjpy_open_share": out["usdjpy_open_independent"]["share"],
           "corr_weeks": c["n_weeks"], "corr_share": c["share"], "corr_median": c["median_corr_offset0"],
           "pairs": {k: (round(v["share"], 3), v["passes"]) for k, v in out["pairs"].items()}}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
