"""Stage A1 per-window statistics (plan v0.6 sections 4.5, 5, 6.3). For every successful run: windows of 1000 returns at TPC 10/20/40, the first 4000-return window at TPC 10/20,
aggregation profile (factors 2 and 4 on the TPC=20 windows). Output: OUT/a1_stats.pkl (dict (set, pid, k) -> arrays)."""
from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_lib as L  # noqa: E402

STAT_NAMES = ("std", "skew", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50", "vr20", "vr60", "Q", "zero_frac")
IDX = {k: i for i, k in enumerate(STAT_NAMES)}


def vec(s):
    return [s[k] for k in STAT_NAMES]


def run_stats(args):
    set_name, pid, k = args
    ticks = np.load(E.run_file(set_name, pid, k))
    res = {"win": {}, "L4000": {}, "agg": {}}
    for tpc in E.TPCS:
        c = L.ticks_to_closes(ticks, tpc)
        res["win"][tpc] = np.array([vec(L.window_stats(w)) for w in L.windows(c, E.L_MAIN)])
        if tpc in (10, 20):
            res["L4000"][tpc] = np.array(vec(L.window_stats(c[:E.L_SENS + 1])))
    c20 = L.ticks_to_closes(ticks, E.TPC_MAIN)
    for a in (2, 4):
        res["agg"][a] = np.array([[L.agg_stats(w, a)["acf_1"], L.agg_stats(w, a)["kurtosis"]] for w in L.windows(c20, E.L_MAIN)])
    return (set_name, pid, k), res


def main():
    code = E.code_state()
    metas = [json.loads(line) for line in open(os.path.join(E.A1_DIR, "meta.jsonl"), encoding="utf-8")]
    tasks = [(m["set"], m["pid"], m["k"]) for m in metas if m["success"]]
    with ProcessPoolExecutor(max_workers=E.WORKERS) as ex:
        out = dict(ex.map(run_stats, tasks, chunksize=4))
    pickle.dump({"_code": code, "stat_names": STAT_NAMES, "runs": out}, open(os.path.join(E.OUT, "a1_stats.pkl"), "wb"))
    print("a1 stats for", len(out), "runs", flush=True)


if __name__ == "__main__":
    main()
