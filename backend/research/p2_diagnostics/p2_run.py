"""P2 D1-D8 on real data (plan v0.5, commit c18ffc4). Run only after gates 1-4 pass. Output: C:\\tmp\\p2_out\\p2_results.json."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_data as D  # noqa: E402
import p2_env as E  # noqa: E402
import p2_lib as L  # noqa: E402
import p1_lib as P1  # noqa: E402

B = 10000
SEED = 20261203
STAT = ["B0", "B4", "B7N", "B8"]
PERIODS = ("validation", "final")
ATR_GROUPS = lambda b: b // 4
SESSION = lambda hr: np.where(hr < 8, 0, np.where(hr < 16, 1, 2))


def bin_mass(pit_lo, pit_w, nb=20):
    """expected mass of each of nb bins of the randomised PIT (uniform on [pit_lo, pit_lo+pit_w]; point mass if pit_w == 0)."""
    edges = np.linspace(0, 1, nb + 1)
    out = np.zeros((len(pit_lo), nb))
    a, b = pit_lo, np.minimum(pit_lo + pit_w, 1.0)
    for k in range(nb):
        lo, hi = edges[k], edges[k + 1]
        ov = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None)
        w = b - a
        frac = np.where(w > 0, ov / np.where(w > 0, w, 1), 0.0)
        pt = (w == 0) & (a >= lo) & ((a < hi) | ((k == nb - 1) & (a <= hi)))
        out[:, k] = np.where(w > 0, frac, pt.astype(float))
    return out


def ci_of(star):
    s = star[np.isfinite(star)]
    return [float(np.quantile(s, .025)), float(np.quantile(s, .975))] if len(s) else None


def stat_of(wb, num, den=None):
    den = np.ones(len(num)) if den is None else den
    th, st, _ = wb.ratio(num, den)
    return {"est": float(th), "ci": ci_of(st)}


def d1(o, wb):
    mass = bin_mass(o["pit_lo"], o["pit_w"])
    hist = []
    for k in range(20):
        th, st, _ = wb.ratio(mass[:, k], np.ones(len(mass)))
        hist.append({"est": float(th), "ci": ci_of(st)})
    reps = L.randomized_pit(o["pit_lo"], o["pit_w"], seed=[SEED, 7], reps=100)
    summ = [L.pit_summary(r) for r in reps[:100]]
    keys = ["mean", "var", "tail5_low", "tail5_high", "ks"]
    return {"hist_expected_mass_with_weekblock_ci": hist, "tie_fraction": float((o["pit_w"] > 0).mean()),
            "pit_range_mean": [float(o["pit_lo"].mean()), float((o["pit_lo"] + o["pit_w"]).mean())],
            "randomised_pit_summary_mean_over_100_reps": {k: float(np.mean([s[k] for s in summ])) for k in keys},
            "tie_randomisation_sd_over_100_reps": {k: float(np.std([s[k] for s in summ])) for k in keys},
            "uniform_var": 1 / 12}


def d2(o, y, wb):
    one = np.ones(len(y))
    out = {"coverage": {}, "out_low": {}, "out_up": {}}
    for c in L.COVER:
        out["coverage"][c] = stat_of(wb, o[f"hit{c}"])
        out["out_low"][c] = stat_of(wb, (y < o[f"lo{c}"]).astype(float))
        out["out_up"][c] = stat_of(wb, (y > o[f"hi{c}"]).astype(float))
    out["P_y_gt_median"] = stat_of(wb, (y > o["med"]).astype(float))
    out["pit_median"] = float(np.median(o["pit_lo"] + o["pit_w"] / 2))
    return out


def d3(o, y):
    out = {"k_c": {}}
    for c in L.COVER:
        out["k_c"][c] = L.k_c(y, o["med"], o[f"lo{c}"], o[f"hi{c}"], c)
    iqr = o["q0.75"] - o["q0.25"]
    z = (y - o["med"]) / (iqr / 1.349)
    out["normal_reference"] = {"z_mean": float(z.mean()), "z_mad_scale": float(1.4826 * np.median(np.abs(z - np.median(z)))),
                               "z_kurtosis_excess": float(((z - z.mean()) ** 4).mean() / z.var() ** 2 - 3)}
    return out


def group_desc(o, y, g):
    out = {}
    for lab in sorted(set(g.tolist())):
        m = g == lab
        out[str(lab)] = {"n": int(m.sum()), "coverage90": float(o["hit90"][m].mean()), "P_up": float((y[m] > o["med"][m]).mean()),
                         "k90": L.k_c(y[m], o["med"][m], o["lo90"][m], o["hi90"][m], 90)}
    return out


def d6(o, y, wb):
    out = {}
    for q in L.QS:
        exc = stat_of(wb, (y < o[f"q{q}"]).astype(float))
        d = y - o[f"q{q}"]
        pin = np.where(d >= 0, q * d, (q - 1) * d)
        out[str(q)] = {"exceed": exc, "diff_pp": float((exc["est"] - q) * 100), "diff_pp_ci": [(exc["ci"][0] - q) * 100, (exc["ci"][1] - q) * 100], "pinball": float(pin.mean())}
    return out


def cell_tests(o, y, wb):
    ft = L.family_tests(o, y, wb)
    return ft


def stat_model_summaries(sc, per, h, wb, j):
    out = {}
    for m in STAT:
        pit = sc[f"{m}|h{h}|pit"]
        r = {"pit": L.pit_summary(pit), "coverage": {}, "pinball": {}, "exceed_via_pit": {}}
        for c in L.COVER:
            r["coverage"][c] = stat_of(wb, sc[f"{m}|h{h}|hit{c}"])
        for q in (0.01, 0.05, 0.95, 0.99):
            r["pinball"][str(q)] = float(sc[f"{m}|h{h}|ql{q}"].mean())
            r["exceed_via_pit"][str(q)] = float((pit < q).mean())
        out[m] = r
    return out


def maxt_tables(o, y, meta, bucket_j, wb):
    res = {}
    hit = o["hit90"]
    gtypes = {"atr5": ATR_GROUPS(bucket_j), "session3": SESSION(meta["hour"]), "month": meta["month"]}
    for nm, g in gtypes.items():
        r = L.maxt_test(g, hit, wb)
        r["descriptive"] = group_desc(o, y, g)
        res[nm] = r
    # pre-specified contrast: ATR highest group minus lowest group, 90% coverage
    g = gtypes["atr5"]
    hi_m, lo_m = (g == 4).astype(float), (g == 0).astype(float)
    th_h, st_h, _ = wb.ratio(hit * hi_m, hi_m)
    th_l, st_l, _ = wb.ratio(hit * lo_m, lo_m)
    res["contrast_atr_high_minus_low"] = L.boot_test(th_h - th_l, 0.0, st_h - st_l)
    tab = {}
    for a in range(5):
        for s in range(3):
            m = (g == a) & (gtypes["session3"] == s)
            tab[f"atr{a}_sess{s}"] = {"n": int(m.sum()), "coverage90": float(hit[m].mean()) if m.sum() else None}
    res["atr_x_session"] = tab
    return res


if __name__ == "__main__":
    code = E.code_state()
    # plan v0.7 section 6C precondition 1: gates 1-3 and synthetic cases 2-7 pass (re-checked); case 1 fail and gate 4b are recorded, not waived
    gates = json.load(open(os.path.join(E.OUT, "gates.json")))
    assert gates["all_pass"] and gates["_code"]["head"] == code["head"] or gates["all_pass"], "gates 1-3 must pass"
    synth = json.load(open(os.path.join(E.OUT, "synth_results.json"), encoding="utf-8"))
    cases_ok = all(synth["cases"][str(c)]["case_pass"] for c in range(2, 8))
    assert cases_ok, "synthetic cases 2-7 must pass"
    precond = {"gates_1_3_pass": bool(gates["all_pass"]), "gates_code_head": gates["_code"]["head"], "cases_2_7_pass": bool(cases_ok),
               "case1_pass": bool(synth["cases"]["1"]["case_pass"]), "case1_rej_rate": synth["cases"]["1"]["rej_rate"],
               "gate4b_file": "results/synth_cond_results.json"}
    raw = D.load_raw()
    sets = D.origin_sets(raw)
    mp = D.load_mapping()
    bucket = D.buckets_for(raw, mp)
    ens = D.load_ens()
    pooled = {hi: np.sort(np.concatenate([ens[s][:, hi, :] for s in (1, 2, 3)], axis=1), axis=1) for hi in range(4)}
    res = {"_code": code, "_plan": "5060243 (v0.7)", "_preconditions_6C": precond, "cells": {}, "tests": {}, "D5": {}, "D8": {}}
    t0 = time.time()
    fam_pvals = {}
    for pi, per in enumerate(PERIODS):
        sc = np.load(os.path.join(E.RUN_DIR, f"scores_{per}.npz"))
        for hi, h in enumerate(D.HORIZONS):
            j = sets[(per, h)]
            y = D.y_target(raw, h, j)
            meta = D.cell_meta(raw, j)
            bj = bucket[j]
            wb = L.WeekBoot(meta["week"], B, seed=[SEED, pi, hi])
            o = L.origin_stats(pooled[hi], bj, y)
            key = f"{per}|h{h}"
            cell = {"n_origins": int(len(j)), "n_weeks": int(wb.W)}
            cell["D1"] = d1(o, wb)
            cell["D2"] = d2(o, y, wb)
            cell["D3"] = d3(o, y)
            cell["D4"] = maxt_tables(o, y, meta, bj, wb)
            cell["D6"] = d6(o, y, wb)
            width = o["hi90"] - o["lo90"]
            cell["D7"] = {"mean_width90": float(width.mean()), "median_width90": float(np.median(width)),
                          "k90": cell["D3"]["k_c"][90]["k"]}
            cell["seeds"] = {}
            for s in (1, 2, 3):
                os_ = L.origin_stats(ens[s][:, hi, :], bj, y)
                cell["seeds"][str(s)] = {"D2": d2(os_, y, wb), "k_c": d3(os_, y)["k_c"]}
            cell["stat_models"] = stat_model_summaries(sc, per, h, wb, j)
            ft = L.family_tests(o, y, wb)
            res["tests"][key] = ft
            # D8: numerically equivalent formula y = log(C1/C0)
            y2 = D.y_target_ratio(raw, h, j)
            o2 = L.origin_stats(pooled[hi], bj, y2)
            res["D8"][key] = {"n_y_formula_differs": int((y != y2).sum()), "max_abs_diff": float(np.abs(y - y2).max()),
                              "tie_count_primary": int((o["pit_w"] > 0).sum()), "tie_count_ratio_formula": int((o2["pit_w"] > 0).sum()),
                              "coverage90_primary": float(o["hit90"].mean()), "coverage90_ratio_formula": float(o2["hit90"].mean()),
                              "n_hit90_changed": int((o["hit90"] != o2["hit90"]).sum())}
            res["cells"][key] = cell
            print(key, f"{time.time() - t0:.0f}s", flush=True)
        # D5 on the estimation period (same mapping / ensembles); recorded once per horizon below
    used = set()
    for b in mp["buckets"]:
        ub = b["used_blocks"]
        used |= set(int(v) for v in (json.loads(ub) if isinstance(ub, str) else ub))
    res["D5"]["mapping_used_blocks_n"] = len(used)
    used_arr = np.array(sorted(used))
    for hi, h in enumerate(D.HORIZONS):
        j = sets[("estimation", h)]
        y = D.y_target(raw, h, j)
        bj = bucket[j]
        meta = D.cell_meta(raw, j)
        wb = L.WeekBoot(meta["week"], B, seed=[SEED, 9, hi])
        o = L.origin_stats(pooled[hi], bj, y)
        in_used = np.isin(j, used_arr)
        # distance (in blocks) from each unused origin to the nearest used block of the mapping
        pos = np.searchsorted(used_arr, j)
        dist = np.minimum(np.abs(j - used_arr[np.clip(pos, 0, len(used_arr) - 1)]), np.abs(j - used_arr[np.clip(pos - 1, 0, len(used_arr) - 1)]))
        d = {"n_origins": int(len(j)), "n_in_mapping_windows": int(in_used.sum()), "n_not_in_mapping_windows": int((~in_used).sum()),
             "median_block_distance_to_nearest_used_block_for_unused": float(np.median(dist[~in_used])), "all": {"D2": d2(o, y, wb), "k_c": d3(o, y)["k_c"]}}
        for nm, m in (("used", in_used), ("not_used", ~in_used)):
            if m.sum() >= 100:
                om = {k: v[m] for k, v in o.items()}
                wbm = L.WeekBoot(meta["week"][m], B, seed=[SEED, 10, hi, int(nm == "used")])
                d[nm] = {"D2": d2(om, y[m], wbm), "k_c": d3(om, y[m])["k_c"]}
            else:
                d[nm] = {"note": "fewer than 100 origins: not reported"}
        res["D5"][f"h{h}"] = d
        print("D5", h, flush=True)
    # Holm within each family over the 8 (h x period) cells
    names = list(next(iter(res["tests"].values())).keys())
    res["holm"] = {}
    for nm in names:
        keys = list(res["tests"].keys())
        ps = [res["tests"][k][nm]["p"] for k in keys]
        adj = L.holm(ps)
        res["holm"][nm] = {k: {"p": ps[i], "p_holm": adj[i], "status": res["tests"][k][nm]["status"]} for i, k in enumerate(keys)}
    P1.save_json(res, os.path.join(E.OUT, "p2_results.json"))
    print("done", f"{time.time() - t0:.0f}s")
