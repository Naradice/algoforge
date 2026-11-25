"""Stage B frozen rules (plan v0.6 sections 6.2, 7.2, 7.3): R-A/R-B split by trading-week parity, SD_j / caliper / sigma_j from R-A, noise floor tau_j and null exceedance f_0j from R-B.
Real data only (no DDM comparison). Output: results/frozen_rules.json (to be committed before the DDM-real comparison of stage C)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_match as MT  # noqa: E402


def split_est(W):
    W = W[(W["period"] == "estimation") & (~W["excluded"])]
    return W[W["parity"] == 1].reset_index(drop=True), W[W["parity"] == 0].reset_index(drop=True)      # R-A (odd weeks), R-B (even weeks)


def freeze(WA, WB, Mn, En, seed=MT.CROSSFIT_SEED):
    MA, EA = WA[list(Mn)].values, WA[list(En)].values
    MB, EB = WB[list(Mn)].values, WB[list(En)].values
    rl = MT.rules_from_reference(MA, EA)
    mask = MT.box_mask(MB, MA, rl["c"])
    wA, wB = np.ones(len(WA)), np.ones(len(WB))
    d, ok = MT.dn_real_real(EB, EA, mask, wA, rl["sigma"])
    tau = MT.tau_from(d, ok, wB)
    f0, per = MT.crossfit_f0(d, ok, wB, WB["week"].values, MT.K_CROSSFIT, np.random.default_rng(seed))
    ref_n = mask.sum(axis=1)
    return {"M": list(Mn), "E": list(En), "SD": rl["SD"].tolist(), "c": rl["c"].tolist(), "sigma": rl["sigma"].tolist(), "tau": tau.tolist(), "f0": f0.tolist(),
            "f0_split_sd": np.nanstd(per, axis=0).tolist(), "f0_split_min": np.nanmin(per, axis=0).tolist(), "f0_split_max": np.nanmax(per, axis=0).tolist(),
            "n_RA": int(len(WA)), "n_RB": int(len(WB)), "n_RB_valid_ref_ge3": int(ok.sum()), "frac_RB_unreferenceable": float(1 - ok.mean()),
            "ref_count_median": float(np.median(ref_n)), "ref_count_max": int(ref_n.max()), "n_RB_with_ref_ge1": int((ref_n >= 1).sum()),
            "noise_floor_computable": bool(ok.sum() >= 100), "tau_imprecise_lt_200": bool(ok.sum() < 200),
            "f0_outside_0.03_0.08": [bool(x < 0.03 or x > 0.08) for x in f0],
            "n_weeks_RA": int(WA["week"].nunique()), "n_weeks_RB": int(WB["week"].nunique())}


def main():
    code = E.code_state()
    dec = json.load(open(os.path.join(E.RESULTS, "a1_decisions.json"), encoding="utf-8"))
    Mn = tuple(dec["M_components"])
    W = pd.read_csv(os.path.join(E.RESULTS, "b_windows_L1000_bar1.csv"))
    WA, WB = split_est(W)
    out = {"_code": code, "decisions_from_A1": dec, "primary": freeze(WA, WB, Mn, E.E_NAMES)}
    cnt = {}
    for g in range(4):
        cnt[g] = {"RA": int((WA["atr_group"] == g).sum()), "RB": int((WB["atr_group"] == g).sum())}
    out["atr_group_counts"] = cnt
    out["atr_groups_meeting_100_each_half"] = [g for g in range(4) if cnt[g]["RA"] >= 100 and cnt[g]["RB"] >= 100]
    out["gates"] = {"RA_ge_300": bool(len(WA) >= 300), "RB_ge_300": bool(len(WB) >= 300), "noise_floor_ref_windows_ge_100": out["primary"]["noise_floor_computable"]}
    # sensitivity partitions (PB, PC) -- defined only for the sensitivity analysis of plan 7.1
    tail = dec["primary_tail_measure"]
    out["PB"] = freeze(WA, WB, ("acf_1", "volclust_50"), (tail, "skew", "acf_5", "acf_20", "vr20", "vr60"))
    out["PC"] = freeze(WA, WB, ("vr20", tail), ("acf_1", "acf_5", "acf_20", "volclust_50", "skew"))
    json.dump(out, open(os.path.join(E.RESULTS, "frozen_rules.json"), "w", encoding="utf-8"), indent=1)
    p = out["primary"]
    print("gates", out["gates"], "n_valid_RB", p["n_RB_valid_ref_ge3"], "tau", np.round(p["tau"], 2), "f0", np.round(p["f0"], 3), flush=True)


if __name__ == "__main__":
    main()
