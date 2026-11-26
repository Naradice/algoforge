"""A0 post hoc, time stamps only (written AFTER the results of a0_check.py were seen; it changes no rule and no selection).
(1) weekly first row of the main series by sub-period, Sunday rows; (2) how many C-M1 origins remain under two constraints the A1 plan would need:
the previous origin must be valid (the misfit score uses its forecast), and no Sunday row in the previous window, the origin block or the label window.
Output: results/a0_posthoc.json"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import a0_check as A

PARTS = (("2016-08-31", "2020-08-31"), ("2020-09-01", "2021-06-26"), ("2021-06-27", "2021-08-31"), ("2021-09-01", "2022-08-31"))


def main():
    idx = pd.DatetimeIndex(pd.read_parquet(A.MAIN, columns=["close"]).index)
    out = {"weekly_first_row": {}, "sunday_rows": {}, "C-M1_origins": {}}
    for a, b in PARTS:
        x = idx[(idx >= a) & (idx < pd.Timestamp(b) + pd.Timedelta(days=1))]
        wk = (x + pd.Timedelta(hours=6)).normalize()
        wk = wk - pd.to_timedelta(wk.dayofweek, unit="D")
        first = pd.Series(x, index=wk).groupby(level=0).min()
        key = f"{a}..{b}"
        out["weekly_first_row"][key] = {k: int(v) for k, v in (first.dt.day_name().str[:3] + " " + first.dt.strftime("%H:%M")).value_counts().head(6).items()}
        sun = x[x.dayofweek == 6]
        out["sunday_rows"][key] = {"rows": int(len(sun)), "by_hour": {str(k): int(v) for k, v in sun.strftime("%H").value_counts().sort_index().items()}}
    S = A.Series(idx)
    sunday = np.r_[0, np.cumsum(idx.dayofweek == 6)]
    for per, p in A.PERIODS.items():
        t = S.origins("C-M1", p)
        prev = np.isin(t - 60, t)
        lo = np.maximum(t - 119, 0)
        no_sun = (sunday[t + 61] - sunday[lo]) == 0
        out["C-M1_origins"][per] = {"valid": int(len(t)), "previous_origin_valid": int(prev.sum()), "no_sunday_row": int(no_sun.sum()), "both": int((prev & no_sun).sum())}
    json.dump(out, open(os.path.join(A.HERE, "results", "a0_posthoc.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
