"""P1 evaluation checks V2, V4, V5, V6, V7, V8, V9a (plan v0.3). Independent implementation (p1_lib); the production code is
imported ONLY as a reference to obtain production values that are not stored in the result files (the DM t statistic) and
is never used to compute the independent values."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_lib as L  # noqa: E402

RUN_DIR = r"C:\tmp\f1_run1_db0926f"
OUT = r"C:\tmp\p1_out"
PROD_RESEARCH = r"C:\Users\ksato\AppData\Local\Temp\claude\715aa9c14b2b3f84\e5122ede-9886-4a11-998b-b3c3060fe575\scratchpad\f1_exec\backend\research"
PROD_SEED = 20261117
N_BOOT = 10_000
N_METRICS = 5                                            # c and se_h for four horizons
RES = {}


def load(period):
    z = np.load(os.path.join(RUN_DIR, f"scores_{period}.npz"))
    return {k: z[k] for k in z.files}


def rec(name, ok, **detail):
    RES[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", {k: v for k, v in detail.items() if not isinstance(v, (dict, list)) or len(str(v)) < 160})


def mean_fsum(a):
    return math.fsum(float(x) for x in a) / len(a)


def v2(raw_sets, sc_by_period):
    bad = []
    for per, sc in sc_by_period.items():
        for h in L.HORIZONS:
            ref = raw_sets[(per, h)]
            for m in L.ALL_MODELS:
                j = sc[f"{m}|h{h}|j"]
                if not np.array_equal(j, ref):
                    bad.append((per, h, m, int(len(j)), int(len(ref))))
    rec("V2_origin_sets_identical_across_models_and_equal_to_independent_set", not bad, n_bad=len(bad), examples=bad[:5])


def v5(val, prod_sel):
    sel, rows = {}, {}
    for h in L.HORIZONS:
        vals = {m: mean_fsum(val[f"{m}|h{h}|crps"]) for m in L.SELECT_POOL}
        best_m = min(vals, key=lambda m: vals[m])
        thr = vals[best_m] * (1 + L.TIE_TOL)
        near = [m for m in L.TIE_ORDER if vals[m] <= thr]
        choice = near[0]
        outside = sorted((vals[m] / vals[best_m] - 1, m) for m in vals if vals[m] > thr)[:1]
        closest_gap = min(abs(vals[m] / vals[best_m] - 1 - L.TIE_TOL) for m in vals)
        rows[str(h)] = {"validation_crps": vals, "best": best_m, "threshold": thr, "within_0.5pct": near,
                        "selected": choice, "closest_outside": outside, "min_abs_gap_to_threshold": closest_gap,
                        "reason": f"min={best_m}; candidates within +0.5%={near}; first in tie order {L.TIE_ORDER} -> {choice}"}
        sel[str(h)] = choice
    pool_ok = sorted(prod_sel["validation_crps"]["1"].keys()) == sorted(L.SELECT_POOL)
    # sensitivity: does '<' instead of '<=' change anything / is any candidate within 1e-9 of the threshold
    sens = all(r["min_abs_gap_to_threshold"] > 1e-9 for r in rows.values())
    ok = sel == prod_sel["selected"] and pool_ok and sens
    rec("V5_model_selection", ok, independent=sel, production=prod_sel["selected"], pool_ok=pool_ok, boundary_clear=sens)
    RES["V5_model_selection"]["table"] = rows
    return sel


def v4_v9a(fin, res_final):
    sel = {str(h): res_final["per_horizon"][str(h)]["stat_model"] for h in L.HORIZONS}
    worst_rel, worst_abs = 0.0, 0.0
    details = {}
    bad = []
    for h in L.HORIZONS:
        stat = fin[f"{sel[str(h)]}|h{h}|crps"]
        seeds = [mean_fsum(fin[f"{d}|h{h}|crps"]) for d in L.DDM_MODELS]
        per_origin = (fin[f"DDM1|h{h}|crps"] + fin[f"DDM2|h{h}|crps"] + fin[f"DDM3|h{h}|crps"]) / 3.0
        ddm = mean_fsum(per_origin)
        sd = math.fsum(per_origin)
        ss = math.fsum(stat)
        delta = sd / ss - 1.0
        seed_mean = math.fsum(seeds) / 3.0
        seed_sd_rel = math.sqrt(math.fsum((s - seed_mean) ** 2 for s in seeds) / 3.0) / seed_mean
        p = res_final["per_horizon"][str(h)]
        checks = {"crps_stat": (mean_fsum(stat), p["crps_stat"], 1e-12, 1e-18), "crps_ddm_mean_of_seeds": (seed_mean, p["crps_ddm_mean_of_seeds"], 1e-12, 1e-18),
                  "crps_ddm_per_origin_mean": (ddm, p["crps_ddm_mean_of_seeds"], 1e-12, 1e-18),
                  "delta": (delta, p["delta"], 0.0, 1e-12), "ddm_seed_sd_rel": (seed_sd_rel, p["ddm_seed_sd_rel"], 1e-9, 1e-15),
                  "n_origins": (len(stat), p["n_origins"], 0.0, 0.0)}
        for i, s in enumerate(seeds):
            checks[f"crps_ddm_seed{i + 1}"] = (s, p["crps_ddm_seeds"][i], 1e-12, 1e-18)
        for k, (a, b, rt, at) in checks.items():
            d = abs(a - b)
            ok = d <= rt * abs(b) + at
            worst_abs = max(worst_abs, d)
            if not ok:
                bad.append((h, k, a, b))
        details[str(h)] = {"delta": delta, "n": len(stat)}
    rec("V4_mean_crps_delta_and_seed_sd", not bad, bad=bad, worst_abs_diff=worst_abs, per_h=details)
    # all-model CRPS / plain CRPS / delta vs model / V9a aggregation of stored hits and pinball losses
    bad9, worst9 = [], 0.0
    for m in L.STAT_MODELS:
        for h in L.HORIZONS:
            row = res_final["models_final_crps"][m][str(h)]
            x = fin[f"{m}|h{h}|crps"]
            ddm = (fin[f"DDM1|h{h}|crps"] + fin[f"DDM2|h{h}|crps"] + fin[f"DDM3|h{h}|crps"]) / 3.0
            pairs = [("crps", mean_fsum(x), row["crps"], 1e-12, 1e-18),
                     ("crps_plain", mean_fsum(fin[f"{m}|h{h}|crps_plain"]), row["crps_plain"], 1e-12, 1e-18),
                     ("delta_ddm_vs_model", mean_fsum(ddm) / mean_fsum(x) - 1.0, row["delta_ddm_vs_model"], 0.0, 1e-12)]
            for c in L.COVER:
                pairs.append((f"hit{c}", mean_fsum(fin[f"{m}|h{h}|hit{c}"]), row[f"hit{c}"], 1e-12, 1e-15))
            for q in L.TAILS:
                pairs.append((f"ql{q}", mean_fsum(fin[f"{m}|h{h}|ql{q}"]), row["ql"][str(q)], 1e-12, 1e-18))
            for k, a, b, rt, at in pairs:
                d = abs(a - b)
                worst9 = max(worst9, d)
                if d > rt * abs(b) + at:
                    bad9.append((m, h, k, a, b))
    for h in L.HORIZONS:                      # DDM row: mean over the three seeds
        row = res_final["models_final_crps"]["DDM"][str(h)]
        for c in L.COVER:
            a = math.fsum(mean_fsum(fin[f"{d}|h{h}|hit{c}"]) for d in L.DDM_MODELS) / 3.0
            if abs(a - row[f"hit{c}"]) > 1e-12 * abs(row[f"hit{c}"]) + 1e-15:
                bad9.append(("DDM", h, f"hit{c}", a, row[f"hit{c}"]))
        for q in L.TAILS:
            a = math.fsum(mean_fsum(fin[f"{d}|h{h}|ql{q}"]) for d in L.DDM_MODELS) / 3.0
            if abs(a - row["ql"][str(q)]) > 1e-12 * abs(row["ql"][str(q)]) + 1e-18:
                bad9.append(("DDM", h, f"ql{q}", a, row["ql"][str(q)]))
    rec("V9a_aggregation_of_stored_hits_and_pinball_(not_an_independent_check_of_their_definition)", not bad9, bad=bad9[:8], n_bad=len(bad9),
        worst_abs_diff=worst9)
    return sel


def dm_for(fin, sel, h):
    stat = fin[f"{sel[str(h)]}|h{h}|crps"]
    ddm = (fin[f"DDM1|h{h}|crps"] + fin[f"DDM2|h{h}|crps"] + fin[f"DDM3|h{h}|crps"]) / 3.0
    return ddm - stat


def v6(fin, sel, res_final):
    sys.path.insert(0, PROD_RESEARCH)
    sys.path.insert(0, os.path.dirname(PROD_RESEARCH))
    import f1_analyze as prod                          # reference only: the production t statistic is not stored
    out, logp = {}, {}
    bad = []
    for h in L.HORIZONS:
        d = dm_for(fin, sel, h)
        t, var = L.nw_dm(d, h + 23)
        tp, pp = prod.newey_west_dm(d, h + 23)
        lp = L.log10_p_two_sided(t)
        mimic = float(2 * (1 - stats.norm.cdf(abs(t))))            # the production formula, to document its rounding
        prod_p = res_final["per_horizon"][str(h)]["dm_p"]
        t_ok = abs(t - tp) <= 1e-9 * abs(tp)
        if prod_p >= 1e-12:
            p_ok = abs(10 ** lp - prod_p) <= 1e-6 * prod_p
        else:                                                      # production p is affected by the rounding of 1 - cdf
            p_ok = (mimic == prod_p) and lp < math.log10(1e-12)
        if not (t_ok and p_ok):
            bad.append((h, t, tp, lp, prod_p))
        out[str(h)] = {"t": t, "t_production": tp, "log10_p_independent": lp, "p_production": prod_p, "p_mimic_2(1-cdf)": mimic,
                       "1-cdf(|t|)": float(1 - stats.norm.cdf(abs(t))), "lag": h + 23, "t_ok": t_ok, "p_ok": p_ok}
    # rounding boundary of 2*(1-cdf(|t|)) in double precision
    lo, hi = 5.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1 - stats.norm.cdf(mid) > 0:
            lo = mid
        else:
            hi = mid
    boundary = hi
    # Holm with the exact (log-domain) p values vs production
    lps = {h: out[str(h)]["log10_p_independent"] for h in L.HORIZONS}
    order = sorted(L.HORIZONS, key=lambda h: lps[h])
    run, holm_lp = -1e9, {}
    for rank, h in enumerate(order):
        run = max(run, lps[h] + math.log10(len(order) - rank))
        holm_lp[h] = min(0.0, run)
    holm_bad = []
    for h in L.HORIZONS:
        ph = res_final["per_horizon"][str(h)]["dm_p_holm"]
        if ph >= 1e-12:
            if abs(10 ** holm_lp[h] - ph) > 1e-6 * ph:
                holm_bad.append((h, holm_lp[h], ph))
        elif not (ph == 0.0 and holm_lp[h] < -12):
            holm_bad.append((h, holm_lp[h], ph))
    rec("V6_dm_statistic_and_p_values", not bad and not holm_bad, bad=bad, holm_bad=holm_bad, rounding_boundary_abs_t=boundary,
        per_h=out, holm_log10_independent={str(h): holm_lp[h] for h in L.HORIZONS})


def v7_prepare(raw, sets, fin, sel):
    weeks_by_h, sd_by_h, ss_by_h = {}, {}, {}
    for h in L.HORIZONS:
        j = sets[("final", h)]
        weeks_by_h[h] = L.week_ids(raw.t_time[j])
        ddm = (fin[f"DDM1|h{h}|crps"] + fin[f"DDM2|h{h}|crps"] + fin[f"DDM3|h{h}|crps"]) / 3.0
        sd_by_h[h] = ddm
        ss_by_h[h] = fin[f"{sel[str(h)]}|h{h}|crps"]
    all_weeks = np.unique(np.concatenate([weeks_by_h[h] for h in L.HORIZONS]))
    sd = L.week_sums(sd_by_h, weeks_by_h, all_weeks)
    ss = L.week_sums(ss_by_h, weeks_by_h, all_weeks)
    point = {h: math.fsum(sd_by_h[h]) / math.fsum(ss_by_h[h]) - 1.0 for h in L.HORIZONS}
    return sd, ss, point, len(all_weeks)


def band(point, se, crit):
    return {h: (point[h] - crit * se[i], point[h] + crit * se[i]) for i, h in enumerate(L.HORIZONS)}


def v7(raw, sets, fin, sel, res_final):
    sd, ss, point, W = v7_prepare(raw, sets, fin, sel)
    # V7a: arithmetic reproduction of the production draw procedure (default_rng(20261117), integers(0, W, W) per resample)
    rng_f = lambda: np.random.default_rng(PROD_SEED)                       # noqa: E731
    se_a, crit_a = L.bootstrap(sd, ss, point, N_BOOT, rng_f)
    bd = band(point, se_a, crit_a)
    prod = res_final["per_horizon"]
    cmp, bad = {}, []
    for i, h in enumerate(L.HORIZONS):
        p = prod[str(h)]
        for k, a, b in (("se", se_a[i], p["se"]), ("band_lo", bd[h][0], p["simultaneous_band"][0]), ("band_hi", bd[h][1], p["simultaneous_band"][1])):
            rel = abs(a - b) / abs(b)
            cmp[f"{k}_h{h}"] = {"independent": float(a), "production": b, "rel_diff": rel}
            if rel > 1e-12:
                bad.append((k, h, a, b, rel))
    relc = abs(crit_a - res_final["simultaneous_critical_value"]) / res_final["simultaneous_critical_value"]
    cmp["critical"] = {"independent": crit_a, "production": res_final["simultaneous_critical_value"], "rel_diff": relc}
    if relc > 1e-12:
        bad.append(("crit", crit_a, res_final["simultaneous_critical_value"], relc))
    v7a_ok = not bad
    rec("V7a_arithmetic_reproduction_of_production_bootstrap", v7a_ok, bad=bad, n_weeks=W, compare=cmp,
        note="reproduces the production draw procedure (arithmetic), not an independent design check")
    # V7b: independent seeds, 5 quantities, Bonferroni t prediction interval
    prod_vals = [res_final["simultaneous_critical_value"]] + [prod[str(h)]["se"] for h in L.HORIZONS]
    names = ["c"] + [f"se_h{h}" for h in L.HORIZONS]

    def batch(seeds, tag):
        n = len(seeds)
        arr = np.empty((n, N_METRICS))
        for k, sd_ in enumerate(seeds):
            se, c = L.bootstrap(sd, ss, point, N_BOOT, lambda sd_=sd_: np.random.default_rng(sd_))
            arr[k] = [c, *se]
        tcoef = float(stats.t.ppf(1 - 0.01 / (2 * N_METRICS), n - 1))
        m, s = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        half = tcoef * s * math.sqrt(1 + 1 / n)
        inside = (np.array(prod_vals) >= m - half) & (np.array(prod_vals) <= m + half)
        return {"tag": tag, "seeds": [int(seeds[0]), int(seeds[-1])], "n": n, "t": tcoef,
                "metrics": {nm: {"mean": float(m[i]), "sd": float(s[i]), "lo": float(m[i] - half[i]), "hi": float(m[i] + half[i]),
                                 "production": prod_vals[i], "inside": bool(inside[i])} for i, nm in enumerate(names)},
                "all_inside": bool(inside.all()), "mean_c": float(m[0]), "mean_se": [float(x) for x in m[1:]]}
    b1 = batch(list(range(1, 21)), "batch1_seeds_1-20")
    assert abs(b1["t"] - 3.5794) < 1e-3
    RES["V7b_batch1"] = b1
    print("V7b batch1 all_inside", b1["all_inside"], {k: round(v["mean"], 6) for k, v in b1["metrics"].items()})
    final_b = b1
    b2 = None
    if v7a_ok and not b1["all_inside"]:
        b2 = batch(list(range(21, 121)), "extension_seeds_21-120 (not pooled with batch1)")
        assert abs(b2["t"] - 3.1746) < 1e-3
        RES["V7b_extension"] = b2
        final_b = b2
        print("V7b extension all_inside", b2["all_inside"])
    if not v7a_ok:
        verdict = "FAIL (V7a arithmetic mismatch)"
    elif final_b["all_inside"]:
        verdict = "PASS"
    else:
        verdict = "HOLD (judgement pending)"
    rec("V7_verdict", verdict == "PASS", verdict=verdict, deciding_batch=final_b["tag"])
    return point, se_a, crit_a, b1, b2


def v8(point, se_a, crit_a, b1, res_final):
    bd = band(point, se_a, crit_a)
    bad, out = [], {}
    for i, h in enumerate(L.HORIZONS):
        lab = L.classify(*bd[h])
        mlo = point[h] - b1["mean_c"] * b1["mean_se"][i]
        mhi = point[h] + b1["mean_c"] * b1["mean_se"][i]
        lab_b = L.classify(mlo, mhi)
        prod = res_final["per_horizon"][str(h)]["label"]
        out[str(h)] = {"independent": lab, "independent_from_batch1_mean_band": lab_b, "production": prod,
                       "batch1_mean_band": [mlo, mhi]}
        if lab != prod or lab_b != prod:
            bad.append((h, lab, lab_b, prod))
    rec("V8_classification_exact_match", not bad, bad=bad, per_h=out)


if __name__ == "__main__":
    CODE = L.code_state()
    os.makedirs(OUT, exist_ok=True)
    pq = os.path.join(os.environ["ARTIFACT_STORE_PATH"], "datasets", "derived", "usdjpy_1min_volatility.parquet")
    raw = L.Raw(pq)
    sets, tg, ok = raw.origin_sets()
    val, fin = load("validation"), load("final")
    res_final = json.load(open(os.path.join(RUN_DIR, "final_result.json")))
    prod_sel = json.load(open(os.path.join(RUN_DIR, "selection.json")))
    v2(sets, {"validation": val, "final": fin})
    v5(val, prod_sel)
    sel = v4_v9a(fin, res_final)
    v6(fin, sel, res_final)
    point, se_a, crit_a, b1, b2 = v7(raw, sets, fin, sel, res_final)
    v8(point, se_a, crit_a, b1, res_final)
    RES["_code"] = CODE
    L.save_json(RES, os.path.join(OUT, "eval_results.json"))
    print("\nSUMMARY", {k: v["pass"] for k, v in RES.items() if "pass" in v})
