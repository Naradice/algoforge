"""A0: rows per date of the MT5 USDJPY tick export (date column only; no price is read). Output: results/a0_ticks.json"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = r"L:\data\fx\USDJPY_201112190000_202202280529.csv"


def main():
    t0 = time.time()
    total = None
    for chunk in pd.read_csv(PATH, sep="\t", usecols=["<DATE>"], dtype=str, chunksize=5_000_000):
        c = chunk["<DATE>"].value_counts()
        total = c if total is None else total.add(c, fill_value=0)
    total = total.sort_index()
    days = pd.to_datetime(total.index, format="%Y.%m.%d")
    gaps = [{"after": str(a.date()), "next": str(b.date()), "calendar_days": int((b - a).days)} for a, b in zip(days[:-1], days[1:]) if (b - a).days > 3]
    month = total.groupby(days.strftime("%Y-%m").values).sum()
    out = {"path": PATH, "bytes": os.path.getsize(PATH), "rows": int(total.sum()), "dates": int(len(total)), "first": str(days.min().date()),
           "last": str(days.max().date()), "gaps_over_3_calendar_days": gaps, "rows_per_month": {k: int(v) for k, v in month.items()}, "seconds": time.time() - t0}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "a0_ticks.json"), "w", encoding="utf-8"), indent=1)
    print({k: out[k] for k in ("rows", "dates", "first", "last", "gaps_over_3_calendar_days", "seconds")})


if __name__ == "__main__":
    main()
