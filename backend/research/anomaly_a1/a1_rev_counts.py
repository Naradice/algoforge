"""Label-event counts per 12 months for the options of docs/research-anomaly-a1-revision-v0.3.md (definitions fixed there before this was run).
Estimation and validation periods only; labels only, no score is computed. Output: results/revision_counts.json"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import a1_lib as L

YEARS = (("2016-08-31", "2017-08-31"), ("2017-09-01", "2018-08-31"), ("2018-09-01", "2019-08-31"), ("2019-09-01", "2020-08-31"),
         ("2020-09-01", "2021-08-31"))
NORMALISED_THRESHOLD = 2.84941658244219


def main():
    close, index = L.load_main()
    ctx = L.fm.Context(close, index, L.f1_edges())
    F = L.Features(ctx)
    valid = np.flatnonzero(L.valid_origins(ctx, "estimation"))
    V_est = np.array([F.label_value(j) for j in valid])
    q95, q90 = float(np.quantile(V_est, 0.95)), float(np.quantile(V_est, 0.90))
    assert abs(q95 / L.LABEL_THRESHOLD - 1) < 1e-9, q95
    js = np.concatenate([L.included_origins(F, p) for p in ("estimation", "validation")])
    t, row = ctx.tt[js], ctx.s.t[js]
    V = np.array([F.label_value(j) for j in js])
    pos = F.pos[js]
    past = np.array([F.rv59[p - 23:p + 1].mean() if p >= 23 else np.nan for p in pos])
    month = t.to_period("M").to_timestamp()
    rel = np.full(len(js), np.nan)
    for m in month.unique():
        ref = V[np.asarray((t < m) & (t >= m - pd.DateOffset(months=12)))]
        if m >= pd.Timestamp("2017-09-01"):
            rel[month == m] = np.quantile(ref, 0.95)
    flags = {"A_current": (V > L.LABEL_THRESHOLD, 240), "C_q90": (V > q90, 240), "D_relative": (V > rel, 240),
             "E_normalised": (V / past > NORMALISED_THRESHOLD, 240), "F_merge60": (V > L.LABEL_THRESHOLD, 60)}
    out = {"threshold_q95": q95, "threshold_q90": q90, "years": [list(y) for y in YEARS], "options": {}}
    for name, (y, gap) in flags.items():
        rows = []
        for a, b in YEARS:
            m = np.asarray((t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1)))
            if name == "D_relative" and a == YEARS[0][0]:
                rows.append(None)
                continue
            L.MERGE_ROWS = gap
            rows.append({"origins": int(m.sum()), "positive_origins": int(y[m].sum()), "events": int(len(L.event_onsets(row[m], y[m])))})
        out["options"][name] = rows
        print(name, [None if r is None else r["events"] for r in rows], "positives", [None if r is None else r["positive_origins"] for r in rows])
    json.dump(out, open(os.path.join(L.RESULTS, "revision_counts.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
