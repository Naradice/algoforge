"""P1 V3 + V9b: regenerate forecast samples for the fixed sample of origins (v3_sample.json) with the production samplers from the
frozen code (db0926f worktree) and recompute, independently, fair CRPS (brute force), hits, pinball losses and PIT bounds.

Limit (plan section 1): the production samplers are reused, so defects common to the samplers cannot be detected here.
For every value both the independent CRPS and the production fair_crps applied to the regenerated samples are compared with the
stored CRPS, which separates (a) regenerated samples differing from the original ones from (b) a different CRPS implementation."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_lib as L  # noqa: E402

RUN_DIR = r"C:\tmp\f1_run1_db0926f"
OUT = r"C:\tmp\p1_out"
PROD_RESEARCH = r"C:\Users\ksato\AppData\Local\Temp\claude\715aa9c14b2b3f84\e5122ede-9886-4a11-998b-b3c3060fe575\scratchpad\f1_exec\backend\research"
_G = {}


def _init():
    sys.path.insert(0, PROD_RESEARCH)
    sys.path.insert(0, os.path.dirname(PROD_RESEARCH))
    import f1_common as fc
    import f1_models as fm
    mp = json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
    df = pd.read_parquet(fc.artifact_path(), columns=["close"])
    _G["fc"], _G["fm"] = fc, fm
    _G["ctx"] = fm.Context(df["close"].values, df.index, mp["edges"])
    _G["ens"] = [np.load(os.path.join(RUN_DIR, f"ddm_ens_seed{s}.npy")) for s in (1, 2, 3)]
    _G["stored"] = {per: dict(np.load(os.path.join(RUN_DIR, f"scores_{per}.npz"))) for per in ("validation", "final")}


def month_job(args):
    per, month_start, items, bucket = args           # items: {j: {h: y}}
    fc, fm, ctx, stored = _G["fc"], _G["fm"], _G["ctx"], _G["stored"][per]
    month_start = pd.Timestamp(month_start)
    st = fm.build_state(ctx, month_start, None)
    rows = []
    for name in L.ALL_MODELS:
        for j_s, hy in items.items():
            j = int(j_s)
            if name.startswith("DDM"):
                x4 = _G["ens"][int(name[3:]) - 1][bucket[j_s]]
            else:
                x4 = fm.SAMPLERS[name](st, ctx, j)
            if x4 is None or not np.all(np.isfinite(x4)):
                rows.append({"model": name, "j": j, "failed": True})
                continue
            for h_s, y in hy.items():
                h = int(h_s)
                hi = L.HORIZONS.index(h)
                x = np.sort(np.asarray(x4[hi], float))
                arr = stored[f"{name}|h{h}|j"]
                pos = int(np.searchsorted(arr, j))
                assert arr[pos] == j
                pt = L.point_scores(x, y)
                prod_crps = float(fc.fair_crps(x[None, :], np.array([y]))[0])
                row = {"model": name, "j": j, "h": h, "period": per, "indep_crps": L.fair_crps_brute(x, y), "prod_crps_on_regen": prod_crps,
                       "stored_crps": float(stored[f"{name}|h{h}|crps"][pos])}
                for k, v in pt.items():
                    row[f"{k}"] = v
                    row[f"stored_{k}"] = float(stored[f"{name}|h{h}|{k}"][pos])
                rows.append(row)
    return rows


if __name__ == "__main__":
    CODE = L.code_state()
    t0 = time.time()
    sample = json.load(open(os.path.join(OUT, "v3_sample.json")))
    sha_now = L.sha256_file(os.path.join(OUT, "v3_sample.json"))
    sha_rec = open(os.path.join(OUT, "v3_sample.sha256")).read().split()[0]
    if sha_now != sha_rec:
        raise SystemExit("v3_sample.json changed after it was fixed -> stop")
    pq = os.path.join(os.environ["ARTIFACT_STORE_PATH"], "datasets", "derived", "usdjpy_1min_volatility.parquet")
    raw = L.Raw(pq)
    sets, tg, ok = raw.origin_sets()
    mp = json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
    bucket = np.searchsorted(np.asarray(mp["edges"]), raw.atr(), side="right")
    jobs = []
    for per in ("validation", "final"):
        by_month = {}
        for h in L.HORIZONS:
            for j in sample[f"{per}|h{h}"]["origins"]:
                ms = str(pd.Timestamp(raw.t_time[j]).to_period("M").start_time.date())
                by_month.setdefault(ms, {}).setdefault(str(j), {})[str(h)] = float(tg[h][j])
        for ms, items in sorted(by_month.items()):
            jobs.append((per, ms, items, {k: int(bucket[int(k)]) for k in items}))
    print("jobs", len(jobs), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for k, r in enumerate(ex.map(month_job, jobs)):
            rows += r
            print(f"month job {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    failed = [r for r in rows if r.get("failed")]
    rows = [r for r in rows if not r.get("failed")]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_rows.csv"), index=False)
    res = {"_code": CODE, "n_failed_regeneration": len(failed), "n_origin_scores": int(len(df)), "cells": {}}
    ok_all = True
    for (per, model, h), g in df.groupby(["period", "model", "h"]):
        d_ind = np.abs(g.indep_crps - g.stored_crps)
        tol = 1e-9 * np.abs(g.stored_crps) + 1e-15
        d_prod = np.abs(g.prod_crps_on_regen - g.stored_crps)
        d_ip = np.abs(g.indep_crps - g.prod_crps_on_regen)
        cell = {"n": int(len(g)), "V3_independent_vs_stored_n_bad": int((d_ind > tol).sum()),
                "V3_independent_vs_stored_worst_abs": float(d_ind.max()), "V3_worst_excess_ratio": float((d_ind / tol).max()),
                "regen_vs_stored_prod_fn_n_bad": int((d_prod > tol).sum()), "independent_vs_prod_fn_n_bad": int((d_ip > tol).sum())}
        for c in L.COVER:
            cell[f"hit{c}_n_bad"] = int((g[f"hit{c}"] != g[f"stored_hit{c}"]).sum())
        for k in ("pit_lo", "pit_w"):
            cell[f"{k}_n_bad"] = int((g[k] != g[f"stored_{k}"]).sum())
        for q in L.TAILS:
            dd = np.abs(g[f"ql{q}"] - g[f"stored_ql{q}"])
            tt = 1e-9 * np.abs(g[f"stored_ql{q}"]) + 1e-15
            cell[f"ql{q}_n_bad"] = int((dd > tt).sum())
        bad = sum(v for k, v in cell.items() if k.endswith("_n_bad") and "regen_vs" not in k and "independent_vs_prod" not in k)
        cell["cell_pass"] = bad == 0
        ok_all &= bad == 0
        res["cells"][f"{model}|{per}|h{h}"] = cell
    res["V3_pass"] = bool(ok_all and not failed)
    res["n_cells"] = len(res["cells"])
    res["total_bad_V3"] = int(sum(c["V3_independent_vs_stored_n_bad"] for c in res["cells"].values()))
    res["total_bad_regen_vs_stored_prod_fn"] = int(sum(c["regen_vs_stored_prod_fn_n_bad"] for c in res["cells"].values()))
    res["total_bad_independent_vs_prod_fn"] = int(sum(c["independent_vs_prod_fn_n_bad"] for c in res["cells"].values()))
    res["seconds"] = time.time() - t0
    L.save_json(res, os.path.join(OUT, "v3_v9b_results.json"))
    print("V3/V9b pass", res["V3_pass"], {k: v for k, v in res.items() if k.startswith("total") or k.startswith("n_")})
