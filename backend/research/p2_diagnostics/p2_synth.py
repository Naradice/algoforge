"""P2 plan section 6 gate 4: synthetic-data unit tests (plan v0.5, commit c18ffc4). Criteria are fixed in the plan; this script only
evaluates them. Usage: python p2_synth.py [unit|cases]"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "p1_verification"))
import p2_lib as L  # noqa: E402
import p2_env as E  # noqa: E402
import p1_lib as P1  # noqa: E402

N_ORIG, M, N_WEEK, N_BUCKET, B_TEST = 3000, 1000, 53, 20, 1000
S5 = np.sqrt(3.0 / 5.0)                     # standardises t(5) to sd 1
SEED = 20261202
GRID = 0.05


def q_t(p):
    return stats.t.ppf(p, 5) * S5


def q_real4(p):
    """case 4 quantile function (plan 6): equals the predictive t(5) on [0.25, 0.75]; outside, normal increments from the quartiles."""
    p = np.asarray(p, float)
    lo, hi = q_t(0.25), q_t(0.75)
    zl, zh = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    return np.where(p > 0.75, hi + (stats.norm.ppf(p) - zh), np.where(p < 0.25, lo + (stats.norm.ppf(p) - zl), q_t(p)))


def theory_exceed_case4(q):
    """theoretical P(y > Q_t(1-q)) (and lower, by symmetry) under the case-4 truth, from the quantile function, in closed form."""
    x = q_t(1 - q)                               # predictive upper quantile
    z = x - q_t(0.75) + stats.norm.ppf(0.75)     # solve q_real4(p) = x for p > 0.75
    return float(1 - stats.norm.cdf(z))


def theory_exceed_case4_numeric(q):
    """independent numerical check: invert q_real4 by bisection on a fine grid and compare."""
    x = q_t(1 - q)
    ps = np.linspace(0.75, 1 - 1e-12, 4_000_001)
    v = q_real4(ps)
    i = int(np.searchsorted(v, x))
    return float(1 - ps[i])


def gen_case(case, trial):
    rng = np.random.default_rng([SEED, case, trial])
    week = (np.arange(N_ORIG) * N_WEEK) // N_ORIG
    bucket = rng.permutation(np.arange(N_ORIG) % N_BUCKET)            # balanced: 150 origins per bucket
    ens = stats.t.rvs(5, size=(N_BUCKET, M), random_state=rng) * S5
    u = rng.random(N_ORIG)
    y = q_t(u)
    phase = week % 2
    if case == 2:
        y = 1.25 * y
    elif case == 3:
        y = y + 0.3
    elif case == 4:
        y = q_real4(u)
    elif case == 5:
        y = y + np.where(phase == 0, 0.5, -0.5)
    elif case == 6:
        ens = np.round(ens / GRID) * GRID
        y = np.round(y / GRID) * GRID
    elif case == 7:
        ens[:2, :] = 0.0                                              # 2 of 20 buckets (10% of origins) degenerate: identical samples
    ens = np.sort(ens, axis=1)
    return {"y": y, "week": week, "bucket": bucket, "ens": ens, "phase": phase}


def run_trial(case, trial):
    d = gen_case(case, trial)
    y, week, bucket, ens = d["y"], d["week"], d["bucket"], d["ens"]
    o = L.origin_stats(ens, bucket, y)
    wb = L.WeekBoot(week, B_TEST, seed=[SEED, case, trial, 1])
    ft = L.family_tests(o, y, wb)
    rec = {"rej": {k: (v["p"] is not None and v["p"] < 0.05) for k, v in ft.items()},
           "status": {k: v["status"] for k, v in ft.items()},
           "theta": {k: v["theta"] for k, v in ft.items()}}
    ks = {}
    for c in (50, 90):
        r = L.k_c(y, o["med"], o[f"lo{c}"], o[f"hi{c}"], c)
        ks[c] = r
    rec["k"] = {c: ks[c]["k"] for c in ks}
    rec["k_status"] = {c: ks[c]["status"] for c in ks}
    rec["n_excl"] = {c: ks[c]["n_excl"] for c in ks}
    rec["k90_up"] = ks[90]["k_up"]
    rec["k90_dn"] = ks[90]["k_dn"]
    rec["P_up"] = float((y > o["med"]).mean())
    # D4 max-t over 5 bucket groups (reported)
    rec["maxt_p"] = L.maxt_test(bucket // 4, o["hit90"], wb).get("p")
    if case == 5:
        g = {}
        for ph in (0, 1):
            m = d["phase"] == ph
            wbg = L.WeekBoot(week[m], B_TEST, seed=[SEED, case, trial, 2 + ph])
            og = {k: v[m] for k, v in o.items()}
            fc = L.family_tests(og, y[m], wbg)["FC"]
            kk = L.k_c(y[m], og["med"][...], og["lo90"], og["hi90"], 90)
            shift = np.median(y[m] - og["med"])
            kc = L.k_c(y[m] - shift, og["med"], og["lo90"], og["hi90"], 90)
            g[ph] = {"P_up": float((y[m] > og["med"]).mean()), "FC_rej": bool(fc["p"] is not None and fc["p"] < 0.05),
                     "k90": kk["k"], "k90_centre_corrected": kc["k"], "out_up_minus_down": float((y[m] > og["hi90"]).mean() - (y[m] < og["lo90"]).mean())}
        rec["groups"] = g
        rec["output_has_interpretation_fields"] = all(k in rec for k in ("k", "P_up")) and all("P_up" in v and "FC_rej" in v for v in g.values())
    if case == 6:
        pit = L.randomized_pit(o["pit_lo"], o["pit_w"], seed=[SEED, case, trial, 9], reps=100)
        v = pit.var(axis=1)
        rec["tie_frac"] = float((o["pit_w"] > 0).mean())
        rec["pit_var_tie_sd"] = float(v.std())
        rec["pit_var_boot_se"] = ft["FE"]["se"]
    if case == 7:
        rec["finite"] = bool(all(np.all(np.isfinite(v)) for v in o.values()))
        rec["frac_excl"] = {c: ks[c]["n_excl"] / N_ORIG for c in ks}
    return rec


def summarise(case, recs):
    n = len(recs)
    out = {"n_trials": n}
    names = list(recs[0]["rej"].keys())
    out["rej_rate"] = {k: float(np.mean([r["rej"][k] for r in recs])) for k in names}
    out["rej_ci"] = {k: L.clopper_pearson(int(sum(r["rej"][k] for r in recs)), n) for k in names}
    out["k_mean"] = {c: (float(np.mean([r["k"][c] for r in recs if r["k"][c] is not None])) if any(r["k"][c] is not None for r in recs) else None) for c in (50, 90)}
    out["k_status_ok_frac"] = {c: float(np.mean([r["k_status"][c] == "ok" for r in recs])) for c in (50, 90)}
    out["theta_mean"] = {k: float(np.mean([r["theta"][k] for r in recs])) for k in names}
    out["maxt_rej"] = float(np.mean([r["maxt_p"] is not None and r["maxt_p"] < 0.05 for r in recs]))
    crit = {}
    if case == 1:
        for k in ("FA90", "FB", "FC", "FE"):
            crit[f"{k} rej<=0.08"] = out["rej_rate"][k] <= 0.08
        for c in (50, 90):
            crit[f"k{c} mean 1±0.02"] = out["k_mean"][c] is not None and abs(out["k_mean"][c] - 1) <= 0.02
    elif case == 2:
        crit["FA90 rej>=0.90"] = out["rej_rate"]["FA90"] >= 0.90
        crit["k90 mean>=1.10"] = out["k_mean"][90] is not None and out["k_mean"][90] >= 1.10
        crit["FB rej<=0.12"] = out["rej_rate"]["FB"] <= 0.12
    elif case == 3:
        crit["FC rej>=0.90"] = out["rej_rate"]["FC"] >= 0.90
        crit["FB rej>=0.90"] = out["rej_rate"]["FB"] >= 0.90
    elif case == 4:
        th = {q: theory_exceed_case4(q) for q in (0.01,)}
        th_num = theory_exceed_case4_numeric(0.01)
        out["theory_exceed_1pct"] = th[0.01]
        out["theory_exceed_1pct_numeric_check"] = th_num
        crit["theory closed-form == numeric (1e-5)"] = abs(th[0.01] - th_num) < 1e-5
        low = float(np.mean([r["theta"]["FD0.01"] for r in recs]))
        up = float(np.mean([1 - r["theta"]["FD0.99"] for r in recs]))
        out["mean_lower_exceed_q01"], out["mean_upper_exceed_q99"] = low, up
        out["FD_rej_report_only"] = {"0.01": out["rej_rate"]["FD0.01"], "0.99": out["rej_rate"]["FD0.99"]}
        for nm, v in (("lower 1%", low), ("upper 99%", up)):
            crit[f"{nm} exceed < 0.01"] = v < 0.01
            crit[f"{nm} exceed within theory ±0.0015"] = abs(v - th[0.01]) <= 0.0015
        crit["FA50 rej<=0.12"] = out["rej_rate"]["FA50"] <= 0.12
        crit["50% coverage mean 0.50±0.01"] = abs(out["theta_mean"]["FA50"] - 0.50) <= 0.01
    elif case == 5:
        g = [r["groups"] for r in recs]
        out["group_P_up_mean"] = {ph: float(np.mean([x[ph]["P_up"] for x in g])) for ph in (0, 1)}
        out["group_FC_rej"] = {ph: float(np.mean([x[ph]["FC_rej"] for x in g])) for ph in (0, 1)}
        out["group_k90_mean"] = {ph: float(np.mean([x[ph]["k90"] for x in g])) for ph in (0, 1)}
        out["group_k90_centre_corrected_mean"] = {ph: float(np.mean([x[ph]["k90_centre_corrected"] for x in g])) for ph in (0, 1)}
        out["k90_pooled_mean"] = out["k_mean"][90]
        crit["FE rej>=0.90"] = out["rej_rate"]["FE"] >= 0.90
        crit["group P_up opposite signs"] = (out["group_P_up_mean"][0] - 0.5) * (out["group_P_up_mean"][1] - 0.5) < 0
        for ph in (0, 1):
            crit[f"group{ph} FC rej>=0.90"] = out["group_FC_rej"][ph] >= 0.90
        crit["output has interpretation fields"] = all(r["output_has_interpretation_fields"] for r in recs)
        out["centre_correction_moves_k90_toward_1 (record only)"] = bool(
            all(abs(out["group_k90_centre_corrected_mean"][ph] - 1) < abs(out["group_k90_mean"][ph] - 1) for ph in (0, 1)))
    elif case == 6:
        for c in (50, 90):
            crit[f"k{c} mean 1±0.05"] = out["k_mean"][c] is not None and abs(out["k_mean"][c] - 1) <= 0.05
        out["tie_frac_mean"] = float(np.mean([r["tie_frac"] for r in recs]))
        out["pit_var_tie_sd_mean"] = float(np.mean([r["pit_var_tie_sd"] for r in recs]))
        out["pit_var_boot_se_mean"] = float(np.mean([r["pit_var_boot_se"] for r in recs if r["pit_var_boot_se"] is not None]))
        crit["tie randomisation variation reported in a separate column (>0)"] = out["pit_var_tie_sd_mean"] > 0
    elif case == 7:
        crit["no NaN/inf in outputs"] = all(r["finite"] for r in recs)
        fe = float(np.mean([r["frac_excl"][90] for r in recs]))
        out["excluded_fraction_c90"] = fe
        crit["excluded count recorded 10%±1%"] = abs(fe - 0.10) <= 0.01
        crit["k_c reported 推定不能 when >0.1% excluded"] = all(r["k_status"][c] == "推定不能" for r in recs for c in (50, 90))
    out["criteria"] = crit
    out["case_pass"] = bool(all(crit.values()))
    return out


# ---- exception-handling and implementation checks (small synthetic data) ------------------------------------------------------------
def unit_checks():
    res = {}
    rng = np.random.default_rng([SEED, 99, 0])
    # (a) implementation vs brute force reference (P1 independent definitions)
    nb, mm = 5, 200
    ens = np.sort(rng.normal(size=(nb, mm)), axis=1)
    bucket = rng.integers(0, nb, 300)
    y = rng.normal(size=300)
    y[:20] = ens[bucket[:20], 7]                                       # force exact ties
    o = L.origin_stats(ens, bucket, y)
    bad = 0
    for i in range(300):
        ps = P1.point_scores(ens[bucket[i]], y[i])
        for c in (50, 80, 90, 95):
            bad += ps[f"hit{c}"] != o[f"hit{c}"][i]
        bad += abs(ps["pit_lo"] - o["pit_lo"][i]) > 0
        bad += abs(ps["pit_w"] - o["pit_w"][i]) > 1e-15
    res["origin_stats vs P1 point_scores: mismatches"] = int(bad)
    # (b) k_c hand example
    y1 = np.array([2.0, -1.0, 0.5, 0.0]); med = np.zeros(4); lo = -np.ones(4); hi = np.ones(4) * 2
    u, side, valid = L.u_values(y1, med, lo, hi)
    res["u_values hand example"] = bool(np.allclose(u, [1.0, 1.0, 0.25, 0.0]) and list(side) == [1, -1, 1, 0] and valid.all())
    r = L.k_c(y1, med, lo, np.zeros(4), 90)                              # hi - med = 0 -> exclusions on the up side
    res["k_c denominators <=0 excluded and 推定不能"] = bool(r["status"] == "推定不能" and r["n_excl"] == 2)
    # (c) boot_test: se = 0, exclusions >5%, centred p-value formula
    t0 = L.boot_test(0.5, 0.4, np.full(1000, 0.5))
    res["se=0 -> 判定不能"] = bool(t0["status"] == "判定不能" and t0["se"] == 0.0)
    ts = np.concatenate([np.full(70, np.nan), rng.normal(0.5, 0.01, 930)])
    res["exclusion>5% -> 判定不能"] = bool(L.boot_test(0.5, 0.4, ts)["status"] == "判定不能")
    ts = rng.normal(0.5, 0.01, 1000)
    tt = L.boot_test(0.5, 0.5, ts)
    cnt = int((np.abs(ts - 0.5) >= 0 - 1e-12).sum())
    res["p formula (1+#)/(B+1) at theta=theta0"] = bool(abs(tt["p"] - (1 + cnt) / 1001) < 1e-12)
    res["p minimum 1/(B+1)"] = bool(abs(L.boot_test(0.9, 0.5, rng.normal(0.9, 0.01, 1000))["p"] - 1 / 1001) < 1e-12)
    # (d) Holm with 判定不能 counted in m
    adj = L.holm([0.001, None, 0.02, 0.5])
    res["Holm counts 判定不能"] = bool(adj[1] is None and abs(adj[0] - 0.004) < 1e-12 and abs(adj[2] - 0.06) < 1e-12 and abs(adj[3] - 1.0) < 1e-12)
    # (e) max-t groups: small cell, rare group (missing in bootstrap), se = 0 group
    n = 3000
    week = (np.arange(n) * 53) // n
    group = np.zeros(n, int)
    group[:3] = 1                                                      # n<100 -> descriptive
    group[2400:2480] = 2                                               # 80 origins: n<100
    group[(week == 5)] = 3                                             # one week only (~57): small too
    group[(week >= 10) & (week < 12)] = 4                              # 2 weeks (~113): present >=100 but missing often
    hit = (rng.random(n) < 0.9).astype(float)
    hit[group == 0] = hit[group == 0]
    wb = L.WeekBoot(week, 2000, seed=[SEED, 99, 1])
    mt = L.maxt_test(group, hit, wb)
    notes = {k: v.get("note") for k, v in mt["groups"].items()}
    res["max-t: small and rare groups descriptive-only"] = bool(notes["1"] and notes["2"] and notes["3"] and notes["4"] and mt["target_groups"] == ["0"] and mt["status"] == "判定不能")
    hit2 = np.ones(n)                                                  # coverage 1 everywhere -> se = 0 for all groups
    g2 = (np.arange(n) // 1000)
    mt2 = L.maxt_test(g2, hit2, wb)
    res["max-t: se=0 -> 判定不能"] = bool(mt2["status"] == "判定不能" and mt2["target_groups"] == [])
    g3 = (np.arange(n) // 1000)
    hit3 = (rng.random(n) < 0.9).astype(float)
    mt3 = L.maxt_test(g3, hit3, wb)
    res["max-t: normal case runs, fixed target set"] = bool(mt3["status"] == "ok" and mt3["target_groups"] == ["0", "1", "2"] and 0 < mt3["p"] <= 1)
    res["all_pass"] = all(v is True or v == 0 for v in res.values())
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cases"
    code = E.code_state()
    os.makedirs(E.OUT, exist_ok=True)
    if mode == "unit":
        r = unit_checks()
        print(json.dumps(r, indent=1, ensure_ascii=False))
        json.dump({"code": code, "unit_checks": r}, open(os.path.join(E.OUT, "synth_unit_checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
        raise SystemExit(0 if r["all_pass"] else 1)
    plan = {1: 1000, 2: 200, 3: 200, 4: 200, 5: 200, 6: 200, 7: 200}
    only = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else list(plan)
    results = {"_code": code, "_plan": "docs/research-f1-p2-calibration-diagnostics-plan.md @ c18ffc4 section 6", "cases": {}}
    for case in only:
        t0 = time.time()
        recs = [run_trial(case, t) for t in range(plan[case])]
        results["cases"][str(case)] = summarise(case, recs)
        print("case", case, "pass", results["cases"][str(case)]["case_pass"], f"{time.time() - t0:.0f}s", flush=True)
        print(json.dumps(results["cases"][str(case)], indent=1, ensure_ascii=False, default=float), flush=True)
        json.dump(results, open(os.path.join(E.OUT, "synth_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    results["all_cases_pass"] = all(v["case_pass"] for v in results["cases"].values())
    json.dump(results, open(os.path.join(E.OUT, "synth_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("ALL PASS", results["all_cases_pass"])
