"""Post hoc (written AFTER the S1 results were seen; changes nothing frozen): label events per year of the estimation and validation periods,
with the plan's exclusions. Labels only; no score is computed.  Output: results/posthoc_rates.json"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import a1_lib as L

YEARS = (("estimation", "2016-08-31", "2017-08-31"), ("estimation", "2017-09-01", "2018-08-31"), ("estimation", "2018-09-01", "2019-08-31"),
         ("estimation", "2019-09-01", "2020-08-31"), ("validation", "2020-09-01", "2021-08-31"))


def main():
    close, index = L.load_main()
    ctx = L.fm.Context(close, index, L.f1_edges())
    F = L.Features(ctx)
    out = []
    for period, a, b in YEARS:
        inc = L.included_origins(F, period)
        t = ctx.tt[inc]
        inc = inc[np.asarray((t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1)))]
        y = np.array([F.label_value(j) for j in inc]) > L.LABEL_THRESHOLD
        n = len(L.event_onsets(ctx.s.t[inc], y))
        out.append({"from": a, "to": b, "origins": int(len(inc)), "positive_origins": int(y.sum()), "events": n, "events_scaled_to_18_months": n * 1.5})
        print(out[-1])
    json.dump(out, open(os.path.join(L.RESULTS, "posthoc_rates.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
