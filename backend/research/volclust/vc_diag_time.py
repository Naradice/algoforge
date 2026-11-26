"""Confirmation of the V0 findings (plan V0 step 2: "stop the independent judgement and check"). Written AFTER the results of V0 were seen; it changes no judgement.
(1) monthly return correlation between the main series and HistData USDJPY for offsets 6/7/8 h (estimation period);
(2) time basis of both series from the weekly opening time stamps and from the minute of the largest |return| on first Fridays, US summer versus winter time;
(3) reference: the source difference of volclust_50 when each estimation window is matched with the best offset of its month. The independent period is read for
time stamps only (no returns statistics). Output: results/diag_time.json"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vc_env as E
import vc_core as C
import vc_data as D

OFFSETS = (6, 7, 8)


def returns(df):
    r = np.log(df["close"]).diff()
    dt = df.index.to_series().diff().dt.total_seconds()
    return r[dt.values == 60]


def us_dst(day):
    """True if the calendar day is in US daylight saving time."""
    t = pd.DatetimeIndex(day).normalize() + pd.Timedelta(hours=12)
    return np.array([x.utcoffset().total_seconds() == -4 * 3600 for x in t.tz_localize("America/New_York")])


def weekly_open(df, a, b):
    x = df.index[(df.index >= a) & (df.index < b)]
    gap = np.flatnonzero(np.diff(x.values).astype("timedelta64[h]").astype(int) >= 30) + 1
    first = x[gap]
    first = first[first.dayofweek.isin([6, 0])]
    dst = us_dst(first)
    out = {}
    for name, m in (("us_summer", dst), ("us_winter", ~dst)):
        s = pd.Series([f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][t.dayofweek]} {t.hour:02d}h" for t in first[m]])
        out[name] = {"n_weeks": int(len(s)), "top": {k: int(v) for k, v in s.value_counts().head(4).items()}}
    return out


def release_minute(df, a, b, window):
    r = returns(df[(df.index >= a) & (df.index < b)]).abs().dropna()
    r = r[(r.index.dayofweek == 4) & (r.index.day <= 7)]
    r = r[(r.index.hour >= window[0]) & (r.index.hour < window[1])]
    dst = us_dst(r.index)
    out = {}
    for name, m in (("us_summer", dst), ("us_winter", ~dst)):
        g = r[m].groupby([r[m].index.hour, r[m].index.minute]).mean()
        top = g.sort_values(ascending=False).head(3)
        out[name] = [{"time": f"{h:02d}:{mi:02d}", "mean_abs_r": float(v)} for (h, mi), v in top.items()]
    return out


def main():
    code = E.begin("vc_diag_time")
    main_df = D.load_main()
    main_df = main_df[~main_df.index.duplicated(keep="first")]
    hist, _ = D.load_hist("USDJPY", "2021-08-31")
    a, b = pd.Timestamp(E.ESTIM[0]), pd.Timestamp(E.ESTIM[1]) + pd.Timedelta(days=1)
    out = {"_code": code, "note": "post hoc confirmation; no judgement is changed"}
    rm = returns(main_df[(main_df.index >= a) & (main_df.index < b)])
    rh = returns(hist[(hist.index >= a - pd.Timedelta(days=2)) & (hist.index < b + pd.Timedelta(days=2))])
    corr = {}
    for off in OFFSETS:
        sh = rh.copy()
        sh.index = sh.index + pd.Timedelta(hours=off)
        j = pd.concat([rm.rename("m"), sh.rename("h")], axis=1, join="inner").dropna()
        corr[off] = j.groupby(j.index.strftime("%Y-%m")).apply(lambda g: g["m"].corr(g["h"]) if len(g) > 500 else np.nan)
    tab = pd.DataFrame(corr)
    best = tab.fillna(-1).idxmax(axis=1)
    out["monthly"] = {m: {"best_offset_h": int(best[m]), "corr": {str(o): (None if pd.isna(tab.loc[m, o]) else round(float(tab.loc[m, o]), 4)) for o in OFFSETS}} for m in tab.index}
    out["months_by_best_offset"] = {str(k): int(v) for k, v in best.value_counts().items()}
    out["weekly_open_hist_estimation"] = weekly_open(hist, a, b)
    out["weekly_open_hist_independent_timestamps_only"] = weekly_open(hist, pd.Timestamp(E.INDEP[0]), pd.Timestamp(E.INDEP[1]) + pd.Timedelta(days=1))
    out["weekly_open_main_estimation"] = weekly_open(main_df, a, b)
    out["first_friday_peak_minute_hist"] = release_minute(hist, a, b, (6, 11))
    out["first_friday_peak_minute_main"] = release_minute(main_df, a, b, (13, 18))
    # reference: source difference with the month's best offset
    S, H = C.prepare(main_df), C.prepare(hist)
    S0 = C.prepare(D.load_main())
    T = D.b_windows()
    sel = (T["period"] == "estimation") & (~T["excluded"])
    rows = []
    for row in T.loc[sel, "row"].values.astype(np.int64):
        t0, t1 = S0["ts"][row], S0["ts"][row + E.L_MAIN]
        m0 = str(np.datetime64(int(t0), "s"))[:7]
        m1 = str(np.datetime64(int(t1), "s"))[:7]
        off = int(best.get(m0, 7))
        hts = H["ts"] + off * 3600
        i, k = np.searchsorted(hts, t0), np.searchsorted(hts, t1, side="right")
        if k - i - 1 < E.SOURCE_MIN_RETURNS:
            continue
        vm = C.volclust(np.diff(np.log(S0["close"][row:row + E.L_MAIN + 1]))[None, :])[0]
        vh = C.volclust(np.diff(np.log(H["close"][i:k]))[None, :])[0]
        rows.append((off, m0 == m1 and int(best.get(m1, 7)) == off, vm, vh))
    x = pd.DataFrame(rows, columns=["offset", "single_offset", "main", "hist"])
    x["abs"] = (x["hist"] - x["main"]).abs()
    out["reference_source_difference_month_best_offset"] = {"n": int(len(x)), "mean_abs_diff": float(x["abs"].mean()), "median_abs_diff": float(x["abs"].median()), "mean_diff": float((x["hist"] - x["main"]).mean()),
                                                          "by_offset": {str(o): {"n": int(len(g)), "mean_abs_diff": float(g["abs"].mean())} for o, g in x.groupby("offset")},
                                                          "threshold": E.DELTA / 2}
    E.save("diag_time.json", out)
    print({k: out[k] for k in ("months_by_best_offset", "weekly_open_hist_estimation", "weekly_open_hist_independent_timestamps_only", "weekly_open_main_estimation",
                               "first_friday_peak_minute_hist", "first_friday_peak_minute_main", "reference_source_difference_month_best_offset")}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
