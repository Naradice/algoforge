"""Checks of the downloaded HISTORY archives (plan section 3) from the DATE and TIME columns only; no price is read.
Archives of the evaluation period (2026-12 onward) are refused.

  python a1_history_check.py <folder>      -> results/history_check.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import a1_lib as L
import a1_data as D
from a1_s3 import HISTORY, OPEN_SHARE, month_list, weekly_open

HEADER = ["<DATE>", "<TIME>", "<BID>", "<ASK>"]


def japan_month(stamp):
    """month in Japan time of a broker time stamp (broker time = New York + 7 h)."""
    ny = pd.Timestamp(stamp) - pd.Timedelta(hours=7)
    return ny.tz_localize("America/New_York").tz_convert("Asia/Tokyo").strftime("%Y-%m")


def one(args):
    month, path = args
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        with z.open(names[0]) as f:
            header = f.readline().decode("ascii").strip().split("\t")
    entry = D.ledger_entry(path)
    minutes = np.unique(np.concatenate([np.unique(D._stamps(c).dt.floor("min").values) for c in D._read(path, D.COLUMNS_TIME)]))
    return month, {**entry, "members": names, "header": header, "japan_month_first": japan_month(entry["first"]),
                   "japan_month_last": japan_month(entry["last"])}, minutes


def main(folder):
    need = month_list(*HISTORY)
    found = D.archives(folder)
    use = {m: found[m] for m in need if m in found}                              # history months only
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, use.items()))
    files = {m: e for m, e, _ in res}
    allm = np.concatenate([mm for _, _, mm in res])
    index = pd.DatetimeIndex(np.unique(allm))
    main_close, main_index = L.load_main()
    last_main = main_index.max()
    last_month = need[-1]
    complete = [m for m in files if m != last_month]
    # weekdays (broker date) with few or no minutes: information only, not a stop condition of the plan
    days = pd.Series(1, index=index).resample("D").sum()
    days = days[(days.index >= "2022-09-01") & (days.index <= index.max().normalize() - pd.Timedelta(days=1)) & (days.index.dayofweek < 5)]
    opens_end = (index.max() - pd.Timedelta(days=index.max().dayofweek)).strftime("%Y-%m-%d")
    opens = weekly_open(index, "2022-09-05", opens_end)
    summary = {
        "folder": folder,
        "required_months": len(need),
        "present_months": len(files),
        "missing_months": [m for m in need if m not in files],
        "single_member": all(len(e["members"]) == 1 for e in files.values()),
        "header_as_expected": [m for m, e in files.items() if e["header"][:4] != HEADER] == [],
        "headers": sorted({"|".join(e["header"]) for e in files.values()}),
        "outside_its_japan_month": [m for m, e in files.items() if e["japan_month_first"] != m or e["japan_month_last"] != m],
        "backward_tick_steps": {m: e["backward_steps"] for m, e in files.items() if e["backward_steps"]},
        "minutes_total": int(len(index)),
        "minutes_in_two_files": int(len(allm) - len(index)),
        "files_in_time_order": all(files[a]["last"] < files[b]["first"] for a, b in zip(list(files), list(files)[1:])),
        "main_series_last": str(last_main),
        "first_new_minute": str(index.min()),
        "minutes_not_after_main_series": int((index <= last_main).sum()),
        "weekly_open": {**opens, "required_share": OPEN_SHARE, "passed": bool(opens["share"] >= OPEN_SHARE)},
        "sunday_minutes": int((index.dayofweek == 6).sum()),
        "weekdays_without_minutes": [str(d.date()) for d in days[days == 0].index],
        "weekdays_under_1000_minutes": {str(d.date()): int(n) for d, n in days[(days > 0) & (days < 1000)].items()},
        "last_history_file": {"month": last_month, "last_stamp": files.get(last_month, {}).get("last"),
                              "complete": bool(last_month in files and pd.Timestamp(files[last_month]["last"]) >= pd.Timestamp("2026-11-30 16:00"))},
        "complete_months": complete,
        "run": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(L.RESULTS, exist_ok=True)
    json.dump({"summary": summary, "files": files}, open(os.path.join(L.RESULTS, "history_check.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "complete_months"}, indent=1))
    for m, e in files.items():
        print(m, e["tick_rows"], e["first"], e["last"], e["minutes"], e["backward_steps"], e["gaps_over_300s"], e["sunday_tick_rows"])


if __name__ == "__main__":
    main(sys.argv[1])
