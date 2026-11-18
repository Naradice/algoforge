"""P1 step 0: input integrity (58-file SHA256 + raw data hash) and the fixed V3 origin sample (plan v0.3, section 4).
Stops (exit 1) if any hash differs. The sample is written BEFORE any recomputation and its SHA256 is recorded."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_lib as L  # noqa: E402

RUN_DIR = r"C:\tmp\f1_run1_db0926f"
LIST = r"C:\tmp\f1_sha256_58files.txt"
OUT = r"C:\tmp\p1_out"
PARQUET_SHA = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"     # recorded on node #889
SEED = 20261201
N_TARGET = 200


def check_inputs():
    listed = {}
    for line in open(LIST, encoding="utf-8"):
        line = line.strip()
        if line:
            h, name = line.split(" *", 1)
            listed[name] = h
    actual = L.listdir_hashes(RUN_DIR)
    problems = []
    if set(listed) != set(actual):
        problems.append(f"file set differs: only listed {sorted(set(listed) - set(actual))}, only present {sorted(set(actual) - set(listed))}")
    for k in sorted(set(listed) & set(actual)):
        if listed[k] != actual[k]:
            problems.append(f"hash differs: {k}")
    pq = str(__import__("pathlib").Path(os.environ["ARTIFACT_STORE_PATH"]) / "datasets/derived/usdjpy_1min_volatility.parquet")
    pq_sha = L.sha256_file(pq)
    if pq_sha != PARQUET_SHA:
        problems.append(f"raw parquet hash differs: {pq_sha}")
    return {"n_files_listed": len(listed), "n_files_present": len(actual), "problems": problems, "parquet": pq, "parquet_sha256": pq_sha}, pq


def build_sample(raw, sets):
    """Fixed procedure of plan section 4. One origin set per (period, h), shared by all 14 models."""
    rng = np.random.default_rng(SEED)
    out = {}
    gap_rows = raw.gaps
    for per in ("validation", "final"):
        for h in L.HORIZONS:
            valid = sets[(per, h)]
            vset = set(int(x) for x in valid)
            chosen, why = [], {}

            def add(j, tag):
                j = int(j)
                if j not in why:
                    why[j] = tag
                    chosen.append(j)

            for j in valid[:5]:
                add(j, "period_start")
            for j in valid[-5:]:
                add(j, "period_end")
            months = pd.DatetimeIndex(raw.t_time[valid]).to_period("M")
            for mth in sorted(set(months)):
                js = valid[np.asarray(months == mth)]
                for j in js[:2]:
                    add(j, "month_start")
                add(js[-1], "month_end")
            # gap neighbourhood: 2 valid origins after and 2 before each gap inside the period
            cand = []
            lo_row, hi_row = raw.t[valid[0]], raw.t[valid[-1]] + L.BLOCK * h
            for g in gap_rows:
                if g < lo_row - 5000 or g > hi_row:
                    continue
                after = valid[raw.t[valid] > g][:2]
                before = valid[raw.t[valid] <= g][-2:]
                cand += [int(x) for x in after] + [int(x) for x in before]
            cand = sorted(set(cand))
            if len(cand) > 40:
                cand = sorted(int(x) for x in rng.choice(cand, size=40, replace=False))
            for j in cand:
                add(j, "gap_neighbour")
            n_priority = len(chosen)
            rest = np.array([j for j in valid if int(j) not in why])
            need = max(0, N_TARGET - n_priority)
            if need and len(rest):
                for j in sorted(int(x) for x in rng.choice(rest, size=min(need, len(rest)), replace=False)):
                    add(j, "uniform_random")
            out[f"{per}|h{h}"] = {"origins": sorted(chosen), "reason": {str(k): v for k, v in sorted(why.items())},
                                  "n": len(chosen), "n_priority": n_priority, "n_valid": int(len(valid))}
    return out


if __name__ == "__main__":
    CODE = L.code_state()
    os.makedirs(OUT, exist_ok=True)
    info, pq = check_inputs()
    print(json.dumps(info, indent=1))
    info["code"] = CODE
    L.save_json(info, os.path.join(OUT, "step0_input_check.json"))
    if info["problems"]:
        print("INPUT HASH PROBLEMS -> verification NOT started")
        sys.exit(1)
    raw = L.Raw(pq)
    sets, tg, ok = raw.origin_sets()
    for k, v in sets.items():
        print(k, len(v))
    sample = build_sample(raw, sets)
    sample["_meta"] = {"seed": SEED, "target_per_cell": N_TARGET, "plan": "docs/research-f1-p1-verification-plan.md v0.3 section 4",
                       "code": "backend/research/p1_verification/p1_step0.py", "executed_commit": CODE["head"]}
    path = os.path.join(OUT, "v3_sample.json")
    L.save_json(sample, path)
    sha = L.sha256_file(path)
    with open(os.path.join(OUT, "v3_sample.sha256"), "w") as f:
        f.write(sha + " *v3_sample.json\n")
    print("v3_sample.json sha256", sha, {k: (v["n"], v["n_priority"]) for k, v in sample.items() if k != "_meta"})
