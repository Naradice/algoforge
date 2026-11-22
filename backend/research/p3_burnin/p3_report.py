"""P3 reporting (plan v0.3 @ c65cede, sections 6 and 10). Descriptive only: per-condition/seed metrics, paired CRPS differences with
week-block bootstrap intervals, seed-to-seed noise references, attempt-ID correspondence. No p-value-based judgements.
Output: C:\\tmp\\p3_out\\p3_report.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402
import p3_eval as V  # noqa: E402

SEEDS = (1, 2, 3)
# (label, x, ref): paired comparisons from plan section 6 (x minus ref)
COMPARISONS = [
    ("C0s vs C0 (control)", "C0s", "C0"),
    ("C3 vs C0s (shock time structure removed)", "C3", "C0s"),
    ("C4_60 vs C3 (burn-in w/o shocks)", "C4_60", "C3"),
    ("C1_60 vs C0s", "C1_60", "C0s"),
    ("C2_60 vs C0s", "C2_60", "C0s"),
    ("C5_60 vs C0s", "C5_60", "C0s"),
    ("C1_60 vs C5_60 (shock timing, clock runs)", "C1_60", "C5_60"),
    ("C2_60 vs C5_60 (shock timing, clock reset)", "C2_60", "C5_60"),
    ("C2_60 vs C1_60 (clock reset)", "C2_60", "C1_60"),
] + [(f"{a}_{B} vs {b}_{B}", f"{a}_{B}", f"{b}_{B}") for B in (15, 240) for a, b in (("C1", "C5"), ("C2", "C5"), ("C2", "C1"))] \
  + [(f"{a}_{B} vs C0s", f"{a}_{B}", "C0s") for B in (15, 240) for a in ("C1", "C2", "C5")]
PHASE2 = ["C1_15", "C1_240", "C2_15", "C2_240", "C5_15", "C5_240"]


def avail(cond, s):
    if cond == "C0":
        return os.path.exists(os.path.join(E.RUN_DIR, f"ddm_ens_seed{s}.npy"))
    return os.path.exists(os.path.join(E.ENS_DIR, f"{cond}_seed{s}.npz"))


def attempt_rate(a, b):
    """fraction of (bucket, path) slots whose attempt id is identical in both runs; None if not comparable."""
    try:
        A = np.load(os.path.join(E.ENS_DIR, f"{a}_seed1.npz"), allow_pickle=True)["attempt_ids"]
        B = np.load(os.path.join(E.ENS_DIR, f"{b}_seed1.npz"), allow_pickle=True)["attempt_ids"]
    except Exception:
        return None
    if A.shape != B.shape:
        return None
    return float((A == B).mean())


def main():
    code = E.code_state()
    data = V.Data()
    conds = ["C0", "C0s", "C3", "C1_60", "C2_60", "C5_60", "C4_60"] + PHASE2
    M, CR = {}, {}
    for c in conds:
        for s in SEEDS:
            if avail(c, s):
                M[(c, s)], CR[(c, s)] = V.all_metrics(data, c, s, keep_crps=True)
                print("metrics", c, s, flush=True)
    out = {"_code": code, "metrics": {f"{c}|{s}": {k: {kk: vv for kk, vv in v.items()} for k, v in m.items()} for (c, s), m in M.items()},
           "seed_range": {}, "comparisons": [], "stat_ref": {}, "attempt_rate_vs_C0s": {}}
    for c in conds:
        ss = [s for s in SEEDS if (c, s) in M]
        if len(ss) < 2:
            continue
        out["seed_range"][c] = {}
        for cell in M[(c, ss[0])]:
            rng = {}
            for name in ("crps", "cov90", "k90", "P_up"):
                v = np.array([M[(c, s)][cell][name] for s in ss])
                rng[name] = {"mean": float(v.mean()), "range": float(v.max() - v.min())}
            out["seed_range"][c][cell] = rng
    for label, x, ref in COMPARISONS:
        ss = [s for s in SEEDS if (x, s) in M and (ref, s) in M]
        if not ss:
            continue
        row = {"label": label, "x": x, "ref": ref, "seeds": ss, "cells": {}}
        for per in ("validation", "final"):
            for h in V.HZ:
                cell = f"{per}|h{h}"
                per_seed = []
                for s in ss:
                    d = V.crps_rel_diff(data, per, h, CR[(x, s)][cell], CR[(ref, s)][cell], B=2000)
                    per_seed.append({"seed": s, "crps_rel": d, "d_cov90": M[(x, s)][cell]["cov90"] - M[(ref, s)][cell]["cov90"],
                                     "d_k90": M[(x, s)][cell]["k90"] - M[(ref, s)][cell]["k90"], "d_P_up": M[(x, s)][cell]["P_up"] - M[(ref, s)][cell]["P_up"]})
                row["cells"][cell] = {"per_seed": per_seed,
                                      "mean_crps_rel": float(np.mean([p["crps_rel"]["est"] for p in per_seed])),
                                      "mean_d_cov90": float(np.mean([p["d_cov90"] for p in per_seed])),
                                      "mean_d_k90": float(np.mean([p["d_k90"] for p in per_seed])),
                                      "mean_d_P_up": float(np.mean([p["d_P_up"] for p in per_seed]))}
        out["comparisons"].append(row)
    for per in ("validation", "final"):
        for h in V.HZ:
            sc = V.stat_ref_crps(data, per, h)
            out["stat_ref"][f"{per}|h{h}"] = float(np.mean(sc)) if sc is not None else None
    for c in conds:
        if c not in ("C0", "C0s"):
            out["attempt_rate_vs_C0s"][c] = attempt_rate(c, "C0s")
    json.dump(out, open(os.path.join(E.OUT, "p3_report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("written p3_report.json")


if __name__ == "__main__":
    main()
