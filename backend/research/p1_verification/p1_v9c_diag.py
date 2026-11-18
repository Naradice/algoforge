"""P1 V9c diagnostic (NOT a replacement of the pre-registered criterion): explains the single pit_w mismatch found by p1_v9c.py.
Hypothesis: the independent realisation y = log(C1 / C0) and the production one y = log C1 - log C0 differ by <= ~1e-15 (V1), and a
DDM ensemble sample can equal y exactly, so one formulation sees a tie (pit_w = 1/M) and the other does not."""
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

if __name__ == "__main__":
    CODE = L.code_state()
    pq = os.path.join(os.environ["ARTIFACT_STORE_PATH"], "datasets", "derived", "usdjpy_1min_volatility.parquet")
    raw = L.Raw(pq)
    sets, tg, ok = raw.origin_sets()
    logc = np.log(raw.close)
    mp = json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
    bucket = np.searchsorted(np.asarray(mp["edges"]), raw.atr(), side="right")
    res = {"_code": CODE, "mismatch_cases": [], "per_cell": {}}
    n_y_differs = n_tot = 0
    n_tie_ratio = n_tie_diff = n_tie_mismatch = 0
    for s in (1, 2, 3):
        ens = np.sort(np.load(os.path.join(RUN_DIR, f"ddm_ens_seed{s}.npy")), axis=2)
        for per in ("validation", "final"):
            sc = np.load(os.path.join(RUN_DIR, f"scores_{per}.npz"))
            for hi, h in enumerate(L.HORIZONS):
                j = sets[(per, h)]
                y_ratio = tg[h][j]                                          # independent: log(C1/C0)
                y_diff = logc[raw.t[j] + L.BLOCK * h] - logc[raw.t[j]]      # production-style: log C1 - log C0
                n_y_differs += int((y_ratio != y_diff).sum())
                n_tot += len(j)
                stored_w = sc[f"DDM{s}|h{h}|pit_w"]
                for i in range(len(j)):
                    x = ens[bucket[j[i]], hi]
                    w_r = (np.searchsorted(x, y_ratio[i], side="right") - np.searchsorted(x, y_ratio[i], side="left")) / len(x)
                    w_d = (np.searchsorted(x, y_diff[i], side="right") - np.searchsorted(x, y_diff[i], side="left")) / len(x)
                    n_tie_ratio += int(w_r > 0)
                    n_tie_diff += int(w_d > 0)
                    if w_r != stored_w[i]:
                        n_tie_mismatch += 1
                        res["mismatch_cases"].append({"seed": s, "period": per, "h": h, "j": int(j[i]), "y_ratio": float(y_ratio[i]),
                                                      "y_logdiff": float(y_diff[i]), "y_diff_ulps": float(abs(y_ratio[i] - y_diff[i]) / np.spacing(abs(y_diff[i]))),
                                                      "pit_w_with_ratio_y": w_r, "pit_w_with_logdiff_y": w_d, "stored_pit_w": float(stored_w[i]),
                                                      "equal_sample_value": float(x[np.searchsorted(x, y_ratio[i])]) if w_r > 0 else None})
    res.update({"n_origin_scores": n_tot, "n_y_bitwise_different_between_formulations": n_y_differs,
                "n_ties_with_ratio_y": n_tie_ratio, "n_ties_with_logdiff_y": n_tie_diff, "n_pit_w_mismatch_vs_stored_with_ratio_y": n_tie_mismatch})
    print(json.dumps(res, indent=1))
    L.save_json(res, os.path.join(OUT, "v9c_diag_results.json"))
