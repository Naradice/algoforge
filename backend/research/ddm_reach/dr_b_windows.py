"""Stage B real-market windows (plan v0.6 section 6). Builds non-overlapping windows from the existing USDJPY 1-minute series (estimation and validation periods only; the
final period is never loaded into the table), with gap/week/quality rules fixed in this file. Functions are reused by later stages with other L and bar widths."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_lib as L  # noqa: E402
from dr_a1_stats import STAT_NAMES  # noqa: E402

GAP_SECONDS = 300                 # a step longer than 5 minutes is a market gap (as in P1-P5)
ZERO_FRAC_MAX = 0.30              # exclusion threshold fixed before any statistic is looked at
PERIODS = {"estimation": ("2016-08-31", "2020-08-31"), "validation": ("2020-09-01", "2021-08-31")}   # same as P1-P5; the final period is not used
WEEK_SHIFT_HOURS = 6              # fixed EST timestamps: the Sunday 18:00 open becomes Monday 00:00 of its trading week


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_raw():
    sha = sha256_file(E.PARQUET)
    assert sha == E.PARQUET_SHA, sha
    df = pd.read_parquet(E.PARQUET, columns=["close"])
    df = df[df.index <= pd.Timestamp(PERIODS["validation"][1]) + pd.Timedelta(days=1)]        # the final period never enters the analysis
    return df, sha


def atr_groups():
    mp = json.load(open(E.MAPPING, encoding="utf-8"))
    e = mp["edges"]
    return [e[4], e[9], e[14]]


def build(df, L_ret, bar, with_stats=True):
    """windows of L_ret bar-returns: rows i .. i + bar*L_ret (1-min rows); within one trading week, no gap > 300 s; stride = bar*L_ret rows from each segment start."""
    close = df["close"].values.astype(float)
    ts = df.index.values.astype("datetime64[s]").astype("int64")
    shifted = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=WEEK_SHIFT_HOURS)
    iso = shifted.isocalendar()
    wk = (iso["year"].values.astype(int) * 100 + iso["week"].values.astype(int))
    brk = np.flatnonzero((np.diff(ts) > GAP_SECONDS) | (np.diff(wk) != 0)) + 1
    seg_start = np.concatenate([[0], brk])
    seg_end = np.concatenate([brk, [len(close)]])
    span = bar * L_ret
    g = atr_groups()
    rows = []
    t_est = (pd.Timestamp(PERIODS["estimation"][0]), pd.Timestamp(PERIODS["estimation"][1]) + pd.Timedelta(days=1))
    t_val = (pd.Timestamp(PERIODS["validation"][0]), pd.Timestamp(PERIODS["validation"][1]) + pd.Timedelta(days=1))
    for a, b in zip(seg_start, seg_end):
        i = a
        while i + span < b:
            t0, t1 = df.index[i], df.index[i + span]
            if t_est[0] <= t0 and t1 < t_est[1]:
                period = "estimation"
            elif t_val[0] <= t0 and t1 < t_val[1]:
                period = "validation"
            else:
                period = None
            if period is not None:
                seg = close[i:i + span + 1]
                c = seg[::bar]
                atr = float(np.mean(np.abs(np.diff(seg))))
                r = {"row": int(i), "start": str(t0), "week": int(wk[i]), "parity": int(wk[i] % 100) % 2, "period": period, "hour": int(pd.Timestamp(t0).hour), "atr": atr,
                     "atr_group": int(np.searchsorted(g, atr, side="right"))}
                if with_stats:
                    if np.all(np.isfinite(c)) and np.all(c > 0):
                        s = L.window_stats(c)
                        for k in STAT_NAMES:
                            r[k] = s[k]
                        if bar == 1 and L_ret == E.L_MAIN:
                            for f in (2, 4):
                                ag = L.agg_stats(c, f)
                                r[f"agg{f}_acf_1"], r[f"agg{f}_kurtosis"] = ag["acf_1"], ag["kurtosis"]
                        r["excluded"] = bool(s["zero_frac"] > ZERO_FRAC_MAX or not np.all(np.isfinite([s[k] for k in ("acf_1", "kurtosis", "volclust_50", "Q")])))
                    else:
                        r["excluded"] = True
                rows.append(r)
            i += span
    return pd.DataFrame(rows)


def main():
    code = E.code_state()
    df, sha = load_raw()
    W = build(df, E.L_MAIN, 1)
    W.to_csv(os.path.join(E.RESULTS, "b_windows_L1000_bar1.csv"), index=False)
    cnt = W.groupby(["period", "parity", "excluded"]).size().unstack(fill_value=0)
    out = {"_code": code, "parquet_sha256": sha, "n_rows_used": int(len(df)), "zero_frac_max": ZERO_FRAC_MAX, "gap_seconds": GAP_SECONDS, "week_shift_hours": WEEK_SHIFT_HOURS,
           "counts": {f"{p}|parity{q}": {str(k): int(v) for k, v in row.items()} for (p, q), row in cnt.iterrows()},
           "atr_group_edges": atr_groups()}
    json.dump(out, open(os.path.join(E.RESULTS, "b_windows_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["counts"]), flush=True)


if __name__ == "__main__":
    main()
