"""P1 V9c: full independent recomputation of the DDM per-origin scores (all origins, 3 seeds, both periods, 4 horizons) from the
stored ensembles (ddm_ens_seed*.npy), the mapping and the raw data. Independent: ATR / bucket assignment, targets, fair CRPS (brute
force pair term), hits, pinball losses, PIT bounds. Limit: the ensemble -> forecast-sample step is taken from the stored arrays
as the production run used them (bucket -> 1000 samples per horizon)."""
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

TOL = {"crps": (1e-9, 1e-15), "ql": (1e-9, 1e-15)}


if __name__ == "__main__":
    CODE = L.code_state()
    pq = os.path.join(os.environ["ARTIFACT_STORE_PATH"], "datasets", "derived", "usdjpy_1min_volatility.parquet")
    raw = L.Raw(pq)
    sets, tg, ok = raw.origin_sets()
    mp = json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
    edges = np.asarray(mp["edges"])
    atr = raw.atr()
    bucket = np.searchsorted(edges, atr, side="right")
    res = {"_code": CODE, "cells": {}, "mismatch_examples": []}
    all_ok = True
    for s in (1, 2, 3):
        ens = np.load(os.path.join(RUN_DIR, f"ddm_ens_seed{s}.npy"))             # [bucket, horizon, M]
        sorted_ens = np.sort(ens, axis=2)
        pairs = {(b, hi): L.pair_term(sorted_ens[b, hi]) for b in range(ens.shape[0]) for hi in range(4)}
        qcache = {}
        for b in range(ens.shape[0]):
            for hi in range(4):
                x = sorted_ens[b, hi]
                qcache[(b, hi)] = {"lohi": {c: (L.linear_quantile(x, (1 - c / 100.0) / 2), L.linear_quantile(x, 1 - (1 - c / 100.0) / 2)) for c in L.COVER},
                                   "q": {q: L.linear_quantile(x, q) for q in L.TAILS}}
        for per in ("validation", "final"):
            sc = np.load(os.path.join(RUN_DIR, f"scores_{per}.npz"))
            for hi, h in enumerate(L.HORIZONS):
                j = sets[(per, h)]
                b = bucket[j]
                y = tg[h][j]
                n = len(j)
                got = {k: np.empty(n) for k in ["crps", *[f"hit{c}" for c in L.COVER], *[f"ql{q}" for q in L.TAILS], "pit_lo", "pit_w"]}
                for i in range(n):
                    bi = int(b[i])
                    x = sorted_ens[bi, hi]
                    yi = float(y[i])
                    got["crps"][i] = float(np.abs(x - yi).mean()) - pairs[(bi, hi)]
                    qc = qcache[(bi, hi)]
                    for c in L.COVER:
                        lo, hi_ = qc["lohi"][c]
                        got[f"hit{c}"][i] = 1.0 if (yi >= lo and yi <= hi_) else 0.0
                    for q in L.TAILS:
                        d = yi - qc["q"][q]
                        got[f"ql{q}"][i] = q * d if d >= 0 else (q - 1) * d
                    left = int(np.searchsorted(x, yi, side="left"))
                    right = int(np.searchsorted(x, yi, side="right"))
                    got["pit_lo"][i] = left / len(x)
                    got["pit_w"][i] = (right - left) / len(x)
                model = f"DDM{s}"
                assert np.array_equal(sc[f"{model}|h{h}|j"], j)
                cell = {}
                for k, arr in got.items():
                    stored = sc[f"{model}|h{h}|{k}"]
                    if k.startswith("hit") or k in ("pit_lo", "pit_w"):
                        nbad = int((arr != stored).sum())
                        cell[k] = {"n_mismatch": nbad}
                    else:
                        rt, at = TOL["crps" if k == "crps" else "ql"]
                        tol = rt * np.abs(stored) + at
                        d = np.abs(arr - stored)
                        nbad = int((d > tol).sum())
                        cell[k] = {"n_mismatch": nbad, "worst_abs_diff": float(d.max()), "worst_excess_ratio": float((d / tol).max())}
                    if nbad:
                        all_ok = False
                        if len(res["mismatch_examples"]) < 20:
                            idx = int(np.flatnonzero((arr != stored) if k.startswith("hit") or k in ("pit_lo", "pit_w") else (np.abs(arr - stored) > tol))[0])
                            res["mismatch_examples"].append({"model": model, "period": per, "h": h, "field": k, "j": int(j[idx]),
                                                             "independent": float(arr[idx]), "stored": float(stored[idx])})
                cell["n_origins"] = int(n)
                res["cells"][f"{model}|{per}|h{h}"] = cell
        print("seed", s, "done", flush=True)
    res["pass"] = bool(all_ok)
    res["n_cells"] = len(res["cells"])
    res["n_origin_scores_checked"] = int(sum(c["n_origins"] for c in res["cells"].values()))
    print("V9c pass", all_ok, "cells", res["n_cells"], "origin-scores", res["n_origin_scores_checked"])
    L.save_json(res, os.path.join(OUT, "v9c_results.json"))
