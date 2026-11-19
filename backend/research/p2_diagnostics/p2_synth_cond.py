"""P2 gate 4b (plan v0.6.1, commit 703b49a, section 6B): conditions A-E on synthetic case 1. Does not change p2_lib / family_tests.
Order: (1) independent full-enumeration test of the condition-B null values; (2) condition A reproduces the stored case-1 result exactly;
(3) conditions B-E. Stops on failure of (1) or (2). Usage: python p2_synth_cond.py"""
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
import p2_synth as S  # noqa: E402
import p1_lib as P1  # noqa: E402

SEED = 20261204
TRIALS = 1000
KEYS = ("FA90", "FB", "FC", "FE", "FA50")
LIMIT = 0.08


# ---- condition B nulls ------------------------------------------------------------------------------------------------------------
def theta0_B(ens, counts):
    """exact expectations under y ~ uniform over the M stored values of the bucket (full enumeration), same definitions as production:
    linear-interpolation quantiles, inclusive interval, y > m, y < q. ens [nb, M] sorted; counts [nb] origins per bucket."""
    nb, M = ens.shape
    w = np.asarray(counts, float) / np.sum(counts)
    yy = ens.ravel()
    bb = np.repeat(np.arange(nb), M)
    o = L.origin_stats(ens, bb, yy)

    def avg(v):
        return float((np.asarray(v, float).reshape(nb, M).mean(axis=1) * w).sum())
    out = {f"FA{c}": avg(o[f"hit{c}"]) for c in L.COVER}
    out["FB"] = avg((yy < o["lo90"]).astype(float) - (yy > o["hi90"]).astype(float))
    out["FC"] = avg((yy > o["med"]).astype(float))
    for q in L.QS:
        out[f"FD{q}"] = avg((yy < o[f"q{q}"]).astype(float))
    e1, e2 = L.pit_moments(o["pit_lo"], o["pit_w"])
    out["FE"] = avg(e2) - avg(e1) ** 2
    return out


def theta0_B_bruteforce(ens, counts):
    """independent: plain double loops with P1 definitions (linear_quantile, searchsorted); E2 via (lo^2 + lo*hi + hi^2)/3."""
    nb, M = ens.shape
    tot = float(np.sum(counts))
    acc = {}
    s1 = s2 = 0.0
    for b in range(nb):
        x = ens[b]
        wb_ = counts[b] / tot
        lo = {c: P1.linear_quantile(x, (1 - c / 100.0) / 2) for c in L.COVER}
        hi = {c: P1.linear_quantile(x, 1 - (1 - c / 100.0) / 2) for c in L.COVER}
        med = P1.linear_quantile(x, 0.5)
        qq = {q: P1.linear_quantile(x, q) for q in L.QS}
        for k in range(M):
            v = float(x[k])
            for c in L.COVER:
                acc[f"FA{c}"] = acc.get(f"FA{c}", 0.0) + wb_ / M * (1.0 if lo[c] <= v <= hi[c] else 0.0)
            acc["FB"] = acc.get("FB", 0.0) + wb_ / M * ((1.0 if v < lo[90] else 0.0) - (1.0 if v > hi[90] else 0.0))
            acc["FC"] = acc.get("FC", 0.0) + wb_ / M * (1.0 if v > med else 0.0)
            for q in L.QS:
                acc[f"FD{q}"] = acc.get(f"FD{q}", 0.0) + wb_ / M * (1.0 if v < qq[q] else 0.0)
            left = int(np.searchsorted(x, v, side="left"))
            right = int(np.searchsorted(x, v, side="right"))
            plo, phi = left / M, right / M
            s1 += wb_ / M * (plo + phi) / 2
            s2 += wb_ / M * (plo ** 2 + plo * phi + phi ** 2) / 3
    acc["FE"] = s2 - s1 ** 2
    return acc


def enumeration_test():
    rng = np.random.default_rng([SEED, 0, 0, 0])
    res = {}
    ok = True
    for label, rnd in (("no_ties", None), ("ties", 0.1)):
        ens = np.sort(stats.t.rvs(5, size=(3, 50), random_state=rng), axis=1)
        if rnd:
            ens = np.sort(np.round(ens / rnd) * rnd, axis=1)
        counts = [10, 20, 30]
        a = theta0_B(ens, counts)
        b = theta0_B_bruteforce(ens, counts)
        worst = max(abs(a[k] - b[k]) for k in b)
        fe_dev = abs(a["FE"] - 1 / 12)
        res[label] = {"max_abs_diff_vs_bruteforce": worst, "FE_minus_1_over_12": fe_dev, "n_stats": len(b)}
        ok &= worst <= 1e-12 and fe_dev <= 1e-12
    res["pass"] = bool(ok)
    return res


# ---- tests with an arbitrary null value (same primitives as family_tests) -----------------------------------------------------
def tests_with_theta0(o, y, wb, th0):
    one = np.ones(len(y))
    res = {}
    for c in (50, 90):
        th, st, _ = wb.ratio(o[f"hit{c}"], one)
        res[f"FA{c}"] = L.boot_test(th, th0[f"FA{c}"], st)
    th, st, _ = wb.ratio((y < o["lo90"]).astype(float) - (y > o["hi90"]).astype(float), one)
    res["FB"] = L.boot_test(th, th0["FB"], st)
    th, st, _ = wb.ratio((y > o["med"]).astype(float), one)
    res["FC"] = L.boot_test(th, th0["FC"], st)
    e1, e2 = L.pit_moments(o["pit_lo"], o["pit_w"])
    m1, _, _ = wb.ratio(e1, one)
    m2, _, _ = wb.ratio(e2, one)
    _, r1, _ = wb.ratio(e1, one)
    _, r2, _ = wb.ratio(e2, one)
    res["FE"] = L.boot_test(m2 - m1 ** 2, th0["FE"], r2 - r1 ** 2)
    return res


NOMINAL = {"FA50": 0.5, "FA90": 0.9, "FB": 0.0, "FC": 0.5, "FE": 1 / 12}


def stats_peroriginal(ens, y):
    """origin_stats for one independent ensemble per origin (vectorised; same definitions)."""
    tab = L.ens_table(ens)
    o = dict(tab)
    M = ens.shape[1]
    left = (ens < y[:, None]).sum(axis=1)
    right = (ens <= y[:, None]).sum(axis=1)
    o["pit_lo"] = left / M
    o["pit_w"] = (right - left) / M
    for c in L.COVER:
        o[f"hit{c}"] = ((y >= o[f"lo{c}"]) & (y <= o[f"hi{c}"])).astype(float)
    return o


def rej_of(ft):
    return {k: bool(ft[k]["p"] is not None and ft[k]["p"] < 0.05) for k in KEYS}


def rate_ci(x):
    n = len(x)
    k = int(np.sum(x))
    lo, hi = L.clopper_pearson(k, n)
    return {"rate": k / n, "ci": [lo, hi], "k": k, "n": n}


def paired_diff(x, a):
    """rate(x) - rate(a), paired binary; CI per plan: (b-c)/n +- 1.96 sqrt((b+c) - (b-c)^2/n)/n."""
    x, a = np.asarray(x, bool), np.asarray(a, bool)
    n = len(x)
    b = int((x & ~a).sum())
    c = int((~x & a).sum())
    d = (b - c) / n
    h = 1.96 * np.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
    return {"diff": d, "ci": [d - h, d + h], "b_x_only": b, "c_a_only": c}


def newcombe_diff(x, a):
    def wilson(k, n):
        z = 1.96
        p = k / n
        den = 1 + z * z / n
        cen = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return p, cen - h, cen + h
    p1, l1, u1 = wilson(int(np.sum(x)), len(x))
    p2, l2, u2 = wilson(int(np.sum(a)), len(a))
    d = p1 - p2
    return {"diff": d, "ci": [d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]}


if __name__ == "__main__":
    code = E.code_state()
    out = {"_code": code, "_plan": "docs/research-f1-p2-calibration-diagnostics-plan.md @ 703b49a section 6B", "trials": TRIALS}
    t0 = time.time()
    # (1) enumeration test
    out["enumeration_test"] = enumeration_test()
    print("enumeration test", out["enumeration_test"], flush=True)
    json.dump(out, open(os.path.join(E.OUT, "synth_cond_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    if not out["enumeration_test"]["pass"]:
        raise SystemExit("enumeration test failed: stop (plan 6B)")
    # (2) condition A reproduces the stored v0.5 case-1 result exactly (before B-E)
    stored = json.load(open(os.path.join(E.OUT, "synth_results.json"), encoding="utf-8"))["cases"]["1"]["rej_rate"]
    pre = {k: 0 for k in KEYS}
    for t in range(TRIALS):
        d = S.gen_case(1, t)
        o_ = L.origin_stats(d["ens"], d["bucket"], d["y"])
        ft_ = L.family_tests(o_, d["y"], L.WeekBoot(d["week"], S.B_TEST, seed=[S.SEED, 1, t, 1]))
        for k, v in rej_of(ft_).items():
            pre[k] += int(v)
    repro0 = {k: [pre[k] / TRIALS, float(stored[k])] for k in KEYS}
    out["A_reproduction_prepass"] = {"rates_new_vs_stored": repro0, "exact": bool(all(abs(v[0] - v[1]) < 1e-12 for v in repro0.values()))}
    print("A reproduction", out["A_reproduction_prepass"], flush=True)
    json.dump(out, open(os.path.join(E.OUT, "synth_cond_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    if not out["A_reproduction_prepass"]["exact"]:
        raise SystemExit("condition A did not reproduce the stored case-1 result: stop (plan 6B rule 1)")
    # per-trial results: conditions
    R = {c: {k: [] for k in KEYS} for c in ("A", "B", "B_nominal", "C", "D4000", "D20000", "E")}
    tie = {c: [] for c in R}
    th0b_mean = {k: [] for k in KEYS}
    unit_pairs_ok = True
    for t in range(TRIALS):
        d = S.gen_case(1, t)                        # condition A data (v0.5 case-1 seeds)
        y, week, bucket, ens = d["y"], d["week"], d["bucket"], d["ens"]
        counts = np.bincount(bucket, minlength=S.N_BUCKET)
        wbA = L.WeekBoot(week, S.B_TEST, seed=[S.SEED, 1, t, 1])
        oA = L.origin_stats(ens, bucket, y)
        ftA = L.family_tests(oA, y, wbA)
        if t == 0:                                  # tests_with_theta0 equals family_tests at the nominal values
            chk = tests_with_theta0(oA, y, wbA, NOMINAL)
            unit_pairs_ok = all((chk[k]["p"] == ftA[k]["p"]) for k in KEYS)
        for k, v in rej_of(ftA).items():
            R["A"][k].append(v)
        tie["A"].append(float((oA["pit_w"] > 0).mean()))
        # E: same data, B_test = 10000
        wbE = L.WeekBoot(week, 10000, seed=[S.SEED, 1, t, 1])
        ftE = L.family_tests(oA, y, wbE)
        for k, v in rej_of(ftE).items():
            R["E"][k].append(v)
        tie["E"].append(tie["A"][-1])
        if t == 49 or t == TRIALS - 1:
            pass
        # B: realised values drawn from the stored ensemble of the bucket
        rng = np.random.default_rng([SEED, 2, 0, t])
        ensB = np.sort(stats.t.rvs(5, size=(S.N_BUCKET, S.M), random_state=rng) * S.S5, axis=1)
        kk = rng.integers(0, S.M, S.N_ORIG)
        yB = ensB[bucket, kk]
        oB = L.origin_stats(ensB, bucket, yB)
        wbB = L.WeekBoot(week, S.B_TEST, seed=[SEED, 2, 0, t, 1])
        th0 = theta0_B(ensB, counts)
        for k in KEYS:
            th0b_mean[k].append(th0[k])
        ftB = tests_with_theta0(oB, yB, wbB, th0)
        for k, v in rej_of(ftB).items():
            R["B"][k].append(v)
        ftBn = tests_with_theta0(oB, yB, wbB, NOMINAL)
        for k, v in rej_of(ftBn).items():
            R["B_nominal"][k].append(v)
        tie["B"].append(float((oB["pit_w"] > 0).mean()))
        tie["B_nominal"].append(tie["B"][-1])
        # C: independent ensemble per origin, same realised y as A
        rngC = np.random.default_rng([SEED, 3, 0, t])
        ensC = np.sort(stats.t.rvs(5, size=(S.N_ORIG, S.M), random_state=rngC) * S.S5, axis=1)
        oC = stats_peroriginal(ensC, y)
        wbC = L.WeekBoot(week, S.B_TEST, seed=[SEED, 3, 0, t, 1])
        ftC = L.family_tests(oC, y, wbC)
        for k, v in rej_of(ftC).items():
            R["C"][k].append(v)
        tie["C"].append(float((oC["pit_w"] > 0).mean()))
        # D: shared ensembles with larger M, same y as A
        for mi, Mx in enumerate((4000, 20000)):
            rngD = np.random.default_rng([SEED, 4, mi, t])
            ensD = np.sort(stats.t.rvs(5, size=(S.N_BUCKET, Mx), random_state=rngD) * S.S5, axis=1)
            oD = L.origin_stats(ensD, bucket, y)
            wbD = L.WeekBoot(week, S.B_TEST, seed=[SEED, 4, mi, t, 1])
            ftD = L.family_tests(oD, y, wbD)
            nm = f"D{Mx}"
            for k, v in rej_of(ftD).items():
                R[nm][k].append(v)
            tie[nm].append(float((oD["pit_w"] > 0).mean()))
        if t % 50 == 0:
            print("trial", t, f"{time.time() - t0:.0f}s", flush=True)
    out["tests_with_theta0_equals_family_tests_at_nominal"] = bool(unit_pairs_ok)
    # (2) condition A reproduces the stored v0.5 case-1 result exactly
    stored = json.load(open(os.path.join(E.OUT, "synth_results.json"), encoding="utf-8"))["cases"]["1"]["rej_rate"]
    repro = {k: [float(np.mean(R["A"][k])), float(stored[k])] for k in KEYS}
    out["A_reproduction"] = {"rates_new_vs_stored": repro, "exact": bool(all(abs(v[0] - v[1]) < 1e-12 for v in repro.values()))}
    print("A reproduction", out["A_reproduction"], flush=True)
    out["conditions"] = {}
    for c in R:
        out["conditions"][c] = {"rej": {k: rate_ci(R[c][k]) for k in KEYS}, "tie_rate_mean": float(np.mean(tie[c])),
                                "all_four_within_0.08": bool(all(np.mean(R[c][k]) <= LIMIT for k in ("FA90", "FB", "FC", "FE")))}
    for c in ("C", "D4000", "D20000", "E"):
        out["conditions"][c]["paired_diff_vs_A"] = {k: paired_diff(R[c][k], R["A"][k]) for k in KEYS}
    out["conditions"]["B"]["unpaired_diff_vs_A"] = {k: newcombe_diff(R["B"][k], R["A"][k]) for k in KEYS}
    out["conditions"]["B"]["theta0_B_mean_over_trials"] = {k: float(np.mean(v)) for k, v in th0b_mean.items()}
    out["conditions"]["B_nominal"]["note"] = "same data as B, nominal theta0; record of the offset of the discrete empirical distribution; not used for the criteria"
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(E.OUT, "synth_cond_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    np.savez_compressed(os.path.join(E.OUT, "synth_cond_trials.npz"), **{f"{c}|{k}": np.array(R[c][k]) for c in R for k in KEYS})
    print(json.dumps({c: {k: round(v["rate"], 3) for k, v in out["conditions"][c]["rej"].items()} for c in R}, indent=1))
