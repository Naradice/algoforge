"""A1 data handling (plan section 3): monthly broker tick archives -> 1-minute closes (last ASK of every minute), and the acquisition ledger.

  python a1_data.py ledger          every archive in the forward folder: hash, size, rows and time stamps (DATE and TIME columns ONLY; no price is read)
  python a1_data.py check202208     the re-downloaded 2022-08 archive in the forward folder reproduces the main series exactly (history month)
  python a1_data.py check202208 <zip>   the same check on another copy of that archive
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

import a1_lib as L

FORWARD_DIR = r"L:\data\fx\anomaly_a1_forward"
NAME = re.compile(r"ticks_USDJPY-oj5k_(\d{4})-(\d{2})\.zip$")
COLUMNS_TIME = ["<DATE>", "<TIME>"]


def archives(folder=FORWARD_DIR):
    return {f"{m.group(1)}-{m.group(2)}": os.path.join(folder, f) for f in sorted(os.listdir(folder)) if (m := NAME.search(f))}


def _read(path, columns, chunksize=2_000_000):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    assert len(names) == 1, names
    with z.open(names[0]) as f:
        yield from pd.read_csv(f, sep="\t", usecols=columns, dtype={c: str for c in COLUMNS_TIME}, chunksize=chunksize)


def _stamps(chunk):
    return pd.to_datetime(chunk["<DATE>"] + " " + chunk["<TIME>"], format="%Y.%m.%d %H:%M:%S.%f")


def ledger_entry(path):
    """time-stamp facts of one archive; the price columns are not loaded."""
    rows = dup = back = gaps = sunday = 0
    first = last = None
    minutes = []
    for chunk in _read(path, COLUMNS_TIME):
        t = _stamps(chunk).values.astype("datetime64[ms]").astype("int64")
        if last is not None:
            t_all = np.concatenate([[last], t])
        else:
            t_all, first = t, t[0]
        d = np.diff(t_all)
        dup += int((d == 0).sum())
        back += int((d < 0).sum())
        gaps += int((d > 300_000).sum())
        rows += len(t)
        sunday += int((((t // 86_400_000) + 4) % 7 == 0).sum())               # 1970-01-01 was a Thursday
        minutes.append(np.unique(t // 60_000))
        last = t[-1]
    stamp = lambda v: str(np.datetime64(int(v), "ms"))
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": L.sha256(path), "tick_rows": rows, "first": stamp(first),
            "last": stamp(last), "minutes": int(len(np.unique(np.concatenate(minutes)))), "equal_consecutive_stamps": dup, "backward_steps": back,
            "gaps_over_300s": gaps, "sunday_tick_rows": sunday}


def minute_closes(path):
    """1-minute closes: the last ASK (file order) of every minute that has an ASK; reads prices."""
    parts = []
    for chunk in _read(path, COLUMNS_TIME + ["<ASK>"]):
        chunk = chunk.dropna(subset=["<ASK>"])
        minute = _stamps(chunk).dt.floor("min")
        parts.append(chunk["<ASK>"].groupby(minute.values, sort=False).last())
    s = pd.concat(parts)
    s = s.groupby(level=0, sort=False).last()                                   # a minute split across two chunks
    assert s.index.is_monotonic_increasing, "minutes out of order inside the archive"
    s.name = "close"
    return s


def check202208(path):
    s = minute_closes(path)
    close, index = L.load_main()
    main = pd.Series(close, index=index)
    beyond = int((s.index > main.index.max()).sum())                            # rows after the main series end are appended later, not compared
    s = s[s.index <= main.index.max()]
    part = main[(main.index >= s.index.min()) & (main.index <= s.index.max())]
    same_index = bool(len(part) == len(s) and (part.index == s.index).all())
    equal = bool(same_index and np.array_equal(part.values, s.values))
    return {"file": os.path.basename(path), "sha256": L.sha256(path), "rows_archive": int(len(s)), "rows_main_same_range": int(len(part)),
            "first": str(s.index.min()), "last": str(s.index.max()), "main_last": str(main.index.max()), "rows_after_main_end": beyond, "same_time_stamps": same_index, "identical": equal}


def main():
    cmd = sys.argv[1]
    os.makedirs(L.RESULTS, exist_ok=True)
    if cmd == "ledger":
        p = os.path.join(L.RESULTS, "ledger.json")
        old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        for month, path in archives().items():
            if month in old:
                assert L.sha256(path) == old[month]["sha256"], f"archive {month} changed after it was recorded"
                continue
            old[month] = {**ledger_entry(path), "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
            print(month, old[month]["tick_rows"], old[month]["first"], old[month]["last"])
        json.dump(old, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print("recorded months:", len(old))
    elif cmd == "check202208":
        path = sys.argv[2] if len(sys.argv) > 2 else archives()["2022-08"]
        res = check202208(path)
        res["source_folder"] = os.path.dirname(path)
        out = "check202208_local_copy.json" if len(sys.argv) > 2 else "check202208.json"
        json.dump(res, open(os.path.join(L.RESULTS, out), "w", encoding="utf-8"), indent=1)
        print(res)
        raise SystemExit(0 if res["identical"] else 2)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
