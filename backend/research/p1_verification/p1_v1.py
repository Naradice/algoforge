"""P1 V1: independent origin sets, targets, ATR and bucket assignment vs the production arrays (reference only)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_lib as L  # noqa: E402

RUN_DIR = r"C:\tmp\f1_run1_db0926f"
OUT = r"C:\tmp\p1_out"
PROD_RESEARCH = r"C:\Users\ksato\AppData\Local\Temp\claude\715aa9c14b2b3f84\e5122ede-9886-4a11-998b-b3c3060fe575\scratchpad\f1_exec\backend\research"

if __name__ == "__main__":
    CODE = L.code_state()
    sys.path.insert(0, PROD_RESEARCH)
    sys.path.insert(0, os.path.dirname(PROD_RESEARCH))
    import f1_common as fc            # production, reference only
    import f1_models as fm
    pq = os.path.join(os.environ["ARTIFACT_STORE_PATH"], "datasets", "derived", "usdjpy_1min_volatility.parquet")
    raw = L.Raw(pq)
    mp = json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
    prod = fm.Context(raw.close, raw.index, mp["edges"])
    res = {"_code": CODE}
    sets, tg, ok = raw.origin_sets()
    res["origin_ok_equal"] = bool(np.array_equal(ok, prod.origin_ok))
    res["origin_ok_diff_count"] = int((ok != prod.origin_ok).sum())
    res["n_blocks"] = int(raw.nb)
    bad_sets = []
    for (per, h), j in sets.items():
        pj = np.flatnonzero(prod.origin_ok & prod.s.period_mask(per, h) & np.isfinite(prod.targets[h]))
        if not np.array_equal(j, pj):
            bad_sets.append((per, h, len(j), len(pj)))
    res["origin_sets_equal"] = not bad_sets
    res["origin_sets_bad"] = bad_sets
    res["origin_set_sizes"] = {f"{p}|h{h}": int(len(j)) for (p, h), j in sets.items()}
    worst = 0.0
    nan_mismatch = {}
    for h in L.HORIZONS:
        a, b = tg[h], prod.targets[h]
        nan_mismatch[str(h)] = int((np.isnan(a) != np.isnan(b)).sum())
        m = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, float(np.abs(a[m] - b[m]).max()))
    res["target_nan_pattern_mismatch"] = nan_mismatch
    res["target_worst_abs_diff"] = worst
    res["target_within_1e-14"] = bool(worst <= 1e-14 and all(v == 0 for v in nan_mismatch.values()))
    atr_i = raw.atr()
    m = np.isfinite(atr_i) & np.isfinite(prod.atr)
    res["atr_max_abs_diff"] = float(np.max(np.abs(atr_i[m] - prod.atr[m])))
    nz = m & (prod.atr > 0)
    res["atr_max_rel_diff_where_positive"] = float(np.max(np.abs(atr_i[nz] - prod.atr[nz]) / prod.atr[nz]))
    res["atr_n_zero_blocks"] = int((m & (prod.atr == 0)).sum())
    edges = np.asarray(mp["edges"])
    bucket_i = np.searchsorted(edges, atr_i, side="right")
    res["bucket_equal_using_mapping_edges"] = bool(np.array_equal(bucket_i, prod.bucket))
    res["bucket_diff_count"] = int((bucket_i != prod.bucket).sum())
    e_ind = L.edges_from_estimation(raw)
    res["edges_independent_max_rel_diff_vs_mapping"] = float(np.max(np.abs(e_ind - edges) / np.abs(edges)))
    res["bucket_equal_using_independent_edges_on_scored_origins"] = bool(all(
        np.array_equal(np.searchsorted(e_ind, atr_i[j], side="right"), prod.bucket[j]) for j in sets.values()))
    res["pass"] = bool(res["origin_ok_equal"] and res["origin_sets_equal"] and res["target_within_1e-14"]
                       and res["bucket_equal_using_mapping_edges"])
    print(json.dumps(res, indent=1))
    L.save_json(res, os.path.join(OUT, "v1_results.json"))
