"""Stage D (plan v0.6 sections 3, 8, 10): reproducibility and robustness of the reachability result with the frozen rules. Descriptive reachability (and, if stage C ran the residual
analysis, the frozen-rule residuals on the validation period). Output: results/d_robustness.json"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
import dr_match as MT  # noqa: E402
import dr_b_windows as BW  # noqa: E402
from dr_a1_stats import IDX  # noqa: E402
from dr_b_rules import split_est  # noqa: E402
from dr_c_min import descriptive_reach, load_ddm  # noqa: E402


def points_at(R, PT, tpc, Mn, En, l4000=False, exclude_first=False):
    """DDM points (all 10 seeds successful) with M statistics at a given TPC / window definition (all windows unless exclude_first)."""
    P = PT[PT["all10"]].reset_index(drop=True)
    d = {"pid": P["pid"].values, "delta": P["delta"].values, "wma": P["wma"].values}
    for s in ("SA", "SB"):
        M, SE = [], []
        for pid in d["pid"]:
            rr = []
            for k in range(E.SEEDS_PER_SET):
                r = R[(s, pid, k)]
                if l4000:
                    rr.append(r["L4000"][tpc][[IDX[m] for m in Mn]])
                else:
                    w = r["win"][tpc]
                    if exclude_first:
                        w = w[1:]
                    rr.append(w[:, [IDX[m] for m in Mn]].mean(axis=0))
            rr = np.array(rr)
            M.append(rr.mean(axis=0))
            SE.append(rr.std(axis=0, ddof=1) / np.sqrt(len(rr)))
        d[f"M_{s}"], d[f"SEM_{s}"] = np.array(M), np.array(SE)
    return d


def rules_for(WA, Mn):
    sd = WA[list(Mn)].values.std(axis=0, ddof=1)
    return {"M": list(Mn), "SD": sd.tolist(), "c": (0.5 * sd).tolist()}


def reach_only(WA, Mn, d, fr):
    r, _, _ = descriptive_reach(WA, Mn, d, fr)
    keep = ("n_points_matching_eligible", "n_RA_windows", "n_reachable_windows", "fraction_reachable_box", "fraction_reachable_euclid_0.5", "n_weeks_reachable", "fraction_reachable_if_caliper_removed_for")
    out = {k: r[k] for k in keep}
    out["nearest_chebyshev_median"] = r["nearest_chebyshev_distance_SD_units"]["median"]
    return out


def main():
    code = E.code_state()
    fr_all = json.load(open(os.path.join(E.RESULTS, "frozen_rules.json"), encoding="utf-8"))
    fr = fr_all["primary"]
    dec = json.load(open(os.path.join(E.RESULTS, "a1_decisions.json"), encoding="utf-8"))
    PT = pd.read_csv(os.path.join(E.RESULTS, "a1_points.csv"))
    R = pickle.load(open(os.path.join(E.OUT, "a1_stats.pkl"), "rb"))["runs"]
    W = pd.read_csv(os.path.join(E.RESULTS, "b_windows_L1000_bar1.csv"))
    WA, WB = split_est(W)
    Mn = tuple(fr["M"])
    d = load_ddm(fr, dec["exclude_first_window"], R, PT)
    out = {"_code": code}
    # 1. validation period, frozen rules ----------------------------------------------------------------------------------------------------------
    WV = W[(W["period"] == "validation") & (~W["excluded"])].reset_index(drop=True)
    out["validation_frozen"] = {"reach": reach_only(WV, Mn, d, fr), "estimation_RA_reference_fraction": None}
    out["validation_frozen"]["estimation_RA_reference_fraction"] = reach_only(WA, Mn, d, fr)["fraction_reachable_box"]
    # 2. SA/SB swapped -------------------------------------------------------------------------------------------------------------------------------
    ds = dict(d)
    ds["M_SA"], ds["SEM_SA"], ds["M_SB"], ds["SEM_SB"] = d["M_SB"], d["SEM_SB"], d["M_SA"], d["SEM_SA"]
    out["swap_SA_SB"] = reach_only(WA, Mn, ds, fr)
    # 3. sensitivity partitions PB, PC (their own frozen rules) --------------------------------------------------------------------------------
    out["partitions"] = {}
    for name in ("PB", "PC"):
        f = fr_all[name]
        dp = points_at(R, PT, E.TPC_MAIN, tuple(f["M"]), (), exclude_first=dec["exclude_first_window"])
        out["partitions"][name] = {"M": f["M"], **reach_only(WA, tuple(f["M"]), dp, f), "frozen_noise_floor_computable": f["noise_floor_computable"]}
    # 4. unit pairing grid: DDM TPC x real bar width (L = 1000 returns; windows within one trading week) -------------------------------------------
    df, _ = BW.load_raw()
    pairing = {}
    Wb = {1: W}
    for bar in (2, 5):
        Wb[bar] = BW.build(df, E.L_MAIN, bar)
    for bar, Wx in Wb.items():
        WAx, _ = split_est(Wx)
        rl = rules_for(WAx, Mn)
        for tpc in E.TPCS:
            dx = points_at(R, PT, tpc, Mn, ())
            pairing[f"TPC{tpc}_bar{bar}"] = {"n_RA": int(len(WAx)), **reach_only(WAx, Mn, dx, rl)}
    out["unit_pairing_L1000"] = pairing
    # 5. L = 4000 (bar 1 only: 4000 rows fit in a trading week, 2- and 5-minute bars do not) ------------------------------------------------------
    W4 = BW.build(df, E.L_SENS, 1)
    WA4, _ = split_est(W4)
    rl4 = rules_for(WA4, Mn)
    out["L4000_bar1"] = {"n_RA": int(len(WA4)), "note": "L=4000 with 2- or 5-minute bars needs 8,000 / 20,000 rows, longer than a trading week (about 7,200 minutes): not feasible under the within-week rule"}
    for tpc in (10, 20):
        dx = points_at(R, PT, tpc, Mn, (), l4000=True)
        out["L4000_bar1"][f"TPC{tpc}"] = reach_only(WA4, Mn, dx, rl4)
    # 6. sub-periods of the estimation period (calendar year of the window start) ------------------------------------------------------------------
    yr = pd.to_datetime(WA["start"]).dt.year
    out["by_year_RA"] = {str(y): {"n": int((yr == y).sum()), "fraction_reachable": reach_only(WA[yr == y].reset_index(drop=True), Mn, d, fr)["fraction_reachable_box"]} for y in sorted(yr.unique())}
    # 7. time-of-day matched check: windows starting 08-17 (broker-time hours of the series) only ----------------------------------------------------
    sel = WA[(WA["hour"] >= 8) & (WA["hour"] <= 17)].reset_index(drop=True)
    out["hours_08_17_RA"] = reach_only(sel, Mn, d, fr)
    out["not_executed"] = {"other_currency_pairs": "not executed: requires parsing eight HistData CSV files (about 7 million rows each); a reachability check for other markets is left for a follow-up",
                           "residual_checks_on_validation": "only if stage C ran the residual analysis"}
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(out, open(os.path.join(E.RESULTS, "d_robustness.json"), "w", encoding="utf-8"), indent=1, default=conv)
    print(json.dumps({"validation": out["validation_frozen"]["reach"]["fraction_reachable_box"], "swap": out["swap_SA_SB"]["fraction_reachable_box"]}), flush=True)


if __name__ == "__main__":
    main()
