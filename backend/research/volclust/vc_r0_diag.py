"""Confirmation of the R0 result for the main series. Written AFTER the results of R0-R3 were seen; it changes no judgement and no rule.
Per trading week of the estimation and validation periods: correlation of 1-minute returns, main versus converted HistData USDJPY, for offsets -3..+3 h, and the
first row of each series. The independent period is not read beyond time stamps. Output: results_r1/r0_diag.json"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vc_env as E
import vc_data as D
from vc_r0 import returns

OFFSETS = range(-3, 4)


def week_of(idx):
    iso = (idx + pd.Timedelta(hours=E.WEEK_SHIFT_HOURS)).isocalendar()
    return iso["year"].values.astype(np.int64) * 100 + iso["week"].values.astype(np.int64)


def main():
    code = E.begin("vc_r0_diag")
    m = D.load_main()
    m = m[~m.index.duplicated(keep="first")]
    h, _ = D.load_hist("USDJPY", E.VALID[1])
    a, b = pd.Timestamp(E.ESTIM[0]), pd.Timestamp(E.VALID[1]) + pd.Timedelta(days=1)
    m = m[(m.index >= a) & (m.index < b)]
    h = h[(h.index >= a) & (h.index < b)]
    rm, rh = returns(m), returns(h)
    corr, count = {}, {}
    for off in OFFSETS:
        s = rh.copy()
        s.index = s.index + pd.Timedelta(hours=off)
        j = pd.concat([rm.rename("m"), s.rename("h")], axis=1, join="inner")
        g = j.groupby(week_of(j.index))
        count[off] = g.size()
        corr[off] = pd.Series({w: (x["m"].corr(x["h"]) if len(x) >= E.CORR_MIN_SHIFTED else np.nan) for w, x in g}, dtype=float)
    c, n = pd.DataFrame(corr), pd.DataFrame(count)
    c = c[n[0].reindex(c.index).fillna(0) >= E.CORR_MIN_MATCHED]
    best = c.fillna(-np.inf).idxmax(axis=1)
    first_m = pd.Series(m.index, index=week_of(m.index)).groupby(level=0).min()
    first_h = pd.Series(h.index, index=week_of(h.index)).groupby(level=0).min()
    rows = {}
    for w in c.index:
        if best[w] != 0 or not c.loc[w, 0] >= 0.5 or (w >= 202125):
            rows[str(w)] = {"best_offset_h": int(best[w]), "corr": {str(o): (None if pd.isna(c.loc[w, o]) else round(float(c.loc[w, o]), 4)) for o in OFFSETS},
                            "first_row_main": str(first_m.get(w)), "first_row_hist_converted": str(first_h.get(w))}
    v = (c.index >= 202036)
    out = {"_code": code, "note": "post hoc confirmation; no judgement or rule is changed", "n_weeks": int(len(c)),
           "weeks_by_best_offset": {str(k): int(x) for k, x in best.value_counts().items()}, "n_weeks_corr0_below_0.5": int((~(c[0] >= 0.5)).sum()),
           "validation_weeks_by_best_offset": {str(k): int(x) for k, x in best[v].value_counts().items()}, "listed_weeks": rows}
    E.save("r0_diag.json", out)
    print({k: out[k] for k in ("n_weeks", "weeks_by_best_offset", "n_weeks_corr0_below_0.5", "validation_weeks_by_best_offset")})
    for w, r in rows.items():
        print(w, r["best_offset_h"], r["corr"]["0"], r["corr"][str(r["best_offset_h"])], r["first_row_main"], "|", r["first_row_hist_converted"])
    E.done()


if __name__ == "__main__":
    main()
