"""V3a (plan 6): reachable versus unreachable windows of the previous primary partition (descriptive), and the frozen rules of V3b built from the real data only
(M' = matching without the evaluated statistic, E = volclust_50 adjusted; the raw rules and the raw V3b of the original run are not recomputed, plan r1 5).
Output: results_r1/v3a.json, results_r1/v3b_rules.json (committed before V3b compares)."""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D
from dr_b_rules import freeze, split_est
from dr_c_min import reach_block

VARIANTS = {"primary": ("acf_1", "kurtosis"), "tail_Q": ("acf_1", "Q")}


def estimation_table(identified):
    """stored estimation windows (non-excluded) with raw and, if the time basis is identified, seasonally adjusted volclust_50."""
    T = D.b_windows()
    T = T[(T["period"] == "estimation") & (~T["excluded"])].reset_index(drop=True)
    S = C.prepare(D.load_main())
    W = C.windows_at(S, T["row"].values.astype(np.int64), E.L_MAIN)
    T["volclust_raw"] = C.volclust(W["R"])
    se = None
    if identified:
        se = C.Season(C.pool_returns(S, E.ESTIM), W, "ny15")
        T["volclust_adj"] = C.volclust(se.adjusted(W["R"], np.ones(len(se.units))))
    return T, W, se


def main():
    code = E.begin("vc_v3a")
    if not E.load("r0.json")["main_ok"]:
        E.save("v3a.json", {"_code": code, "not_run": "the main series failed the weekly-open check of R0"})
        E.done()
        return
    T, W, _ = estimation_table(True)
    evals = ["volclust_adj"]
    WA, WB = split_est(T)
    rules = {"_code": code, "variants": {}}
    for name, Mn in VARIANTS.items():
        rules["variants"][name] = {e: freeze(WA, WB, Mn, (e,)) for e in evals}
    E.save("v3b_rules.json", rules)
    # V3a: reachable (previous primary partition, SA matching with eligibility) vs unreachable
    fr = D.dr_json("frozen_rules.json")["primary"]
    PT = D.a1_points()
    P = PT[PT["all10"]].reset_index(drop=True)
    mask, _ = reach_block(WA[fr["M"]].values, P[[f"SA_{m}" for m in fr["M"]]].values, P[[f"SA_{m}_se" for m in fr["M"]]].values, np.array(fr["c"]), np.array(fr["SD"]))
    reach = mask.any(axis=1)
    odd = (T["week"].values % 100) % 2 == 1
    R = W["R"][odd]
    WA = WA.copy()
    WA["jumps"] = (np.abs(R) > 4 * np.sqrt((R ** 2).mean(axis=1, keepdims=True))).sum(axis=1)
    WA["year"] = WA["start"].str[:4]
    out = {"_code": code, "n_RA": int(len(WA)), "n_reachable": int(reach.sum()), "note": "reachability was defined with volclust_50 as a matching component: closeness of the reachable windows' volclust_50 to the DDM follows from the selection"}
    for g, m in (("reachable", reach), ("unreachable", ~reach)):
        x = WA[m]
        d = {c: {"mean": float(x[c].mean()), "median": float(x[c].median())} for c in ["volclust_raw"] + evals + ["jumps", "zero_frac"]}
        d["n"] = int(len(x))
        for key in ("hour", "atr_group", "year"):
            d[f"share_by_{key}"] = {str(k): float(v) for k, v in (x[key].value_counts(normalize=True).sort_index()).items() if v >= 0.02}
        out[g] = d
    E.save("v3a.json", out)
    print({k: (out[k]["n"], out[k]["volclust_raw"]["median"], out[k]["volclust_adj"]["median"]) for k in ("reachable", "unreachable")}, flush=True)
    E.done()


if __name__ == "__main__":
    main()
