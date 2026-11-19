"""P2 plan section 6 gates 1-3 (input hashes; origin/realised-value/bucket consistency; hit*/pit reproduction)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_data as D  # noqa: E402
import p2_env as E  # noqa: E402
import p2_lib as L  # noqa: E402
import p1_lib as P1  # noqa: E402

if __name__ == "__main__":
    code = E.code_state()
    os.makedirs(E.OUT, exist_ok=True)
    res = {"_code": code, "gate1": {}, "gate2": {}, "gate3": {"mismatches": {}}}
    # gate 1: hashes
    listed = {}
    for line in open(E.SHA_LIST):
        h, f = line.strip().split(" *")
        listed[f] = h
    actual = P1.listdir_hashes(E.RUN_DIR)
    problems = [f for f in listed if actual.get(f) != listed[f]]
    pq = P1.sha256_file(E.PARQUET)
    res["gate1"] = {"n_listed": len(listed), "problems": problems, "parquet_sha256": pq, "parquet_ok": pq == E.PARQUET_SHA}
    g1 = not problems and pq == E.PARQUET_SHA and len(listed) == 58
    res["gate1"]["pass"] = g1
    print("gate1", g1, flush=True)
    if not g1:
        P1.save_json(res, os.path.join(E.OUT, "gates.json"))
        raise SystemExit("gate 1 failed")
    raw = D.load_raw()
    sets = D.origin_sets(raw)
    mp = D.load_mapping()
    bucket = D.buckets_for(raw, mp)
    ens = D.load_ens()
    # gate 2: origin sets == stored j (all 14 models share them; DDM1 and B0 checked)
    g2 = True
    for per in ("validation", "final"):
        sc = np.load(os.path.join(E.RUN_DIR, f"scores_{per}.npz"))
        for h in D.HORIZONS:
            for model in ("DDM1", "DDM2", "DDM3", "B0", "B8"):
                same = bool(np.array_equal(sc[f"{model}|h{h}|j"], sets[(per, h)]))
                res["gate2"][f"{model}|{per}|h{h}"] = same
                g2 &= same
    res["gate2"]["pass"] = g2
    print("gate2", g2, flush=True)
    # gate 3: hit*, pit_lo, pit_w reproduction for all DDM seeds / cells (y = log C1 - log C0)
    g3 = True
    ncell = 0
    for per in ("validation", "final"):
        sc = np.load(os.path.join(E.RUN_DIR, f"scores_{per}.npz"))
        for hi, h in enumerate(D.HORIZONS):
            j = sets[(per, h)]
            y = D.y_target(raw, h, j)
            for s in (1, 2, 3):
                o = L.origin_stats(ens[s][:, hi, :], bucket[j], y)
                for k in ("hit50", "hit80", "hit90", "hit95", "pit_lo", "pit_w"):
                    stored = sc[f"DDM{s}|h{h}|{k}"]
                    n_bad = int((o[k] != stored).sum())
                    res["gate3"]["mismatches"][f"DDM{s}|{per}|h{h}|{k}"] = n_bad
                    g3 &= n_bad == 0
                ncell += 1
    res["gate3"]["n_cells"] = ncell
    res["gate3"]["pass"] = g3
    print("gate3", g3, "cells", ncell, flush=True)
    res["all_pass"] = bool(g1 and g2 and g3)
    P1.save_json(res, os.path.join(E.OUT, "gates.json"))
    print("gates all pass", res["all_pass"])
