"""DP3 (Study Manager #1155): does the sign of the trend term change the cancellation on 60 s bars?

Plan: docs/research-ddm-trend-term-plan.md v0.1 (approved 2026-11-28). The DP1 and DP2 code is reused unchanged.

    cd backend/research/ddm_persist3
    python dp3_run.py s0     # checks 1-5 (once)
    python dp3_run.py s1     # zero / mix runs and the evaluation (once; needs a passing s0 and unchanged code)

Choices the plan does not spell out, fixed before any run:
  - the retention rho of each condition is reported with a 95% interval (not judged);
  - if both zero and mix fail the pilot, S0 stops (there is nothing to compare);
  - the overall 60 s variance ratio uses the bars with at least one trade.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
for _d in ("ddm_persist", "ddm_persist2"):
    _p = os.path.join(RESEARCH, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dp_lib as L  # noqa: E402  (sets the BLAS thread variables before numpy)
import dp_run as R1  # noqa: E402
import dp2_run as D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = os.path.join(HERE, "results")
WORK = os.path.join(L.REPO, "artifacts", "research", "ddm_persist3")
CODE_PATHS = ["backend/research/ddm_persist3", "backend/research/ddm_persist2/dp2_run.py",
              "backend/research/ddm_persist/dp_lib.py", "backend/research/ddm_persist/dp_run.py",
              "backend/research/tdm_fig3_audit.py", "backend/research/tdm_fig3_t0c.py",
              "backend/data/collectors/ddm_simulator.py", "docs/research-ddm-trend-term-plan.md"]
EXCLUDE = [":(exclude)backend/research/ddm_persist3/results"]
CONDS = ("base", "zero", "mix")
TREATMENTS = ("zero", "mix")
ORDERS = ("ord", "shuf")

CFG = dict(D.CFG)
CFG.update({
    "main_seeds": list(range(401, 421)), "pilot_seeds": list(range(391, 396)), "equiv_seeds": [7, 8, 9],
    "boot_seed": 20261130, "coef_offset": 100_000, "coef_lo": 1.5, "coef_hi": 3.5,
    "delta_min": 0.2, "level_delta": 0.975, "cpu_cap_s": 3600.0,
})


def code_state():
    dirty = R1.git("status", "--porcelain", "--", *CODE_PATHS, *EXCLUDE)
    if dirty:
        raise SystemExit("uncommitted changes -> commit first:\n" + dirty)
    got = {k: R1.md.version(k) for k in R1.VERSIONS}
    if got != R1.VERSIONS:
        raise SystemExit(f"library versions differ: {got}")
    return {"head": R1.git("rev-parse", "HEAD"), "versions": got, "python": sys.version.split()[0],
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def code_changed_since(head):
    return R1.git("diff", "--stat", head, "HEAD", "--", *CODE_PATHS, *EXCLUDE)


# ------------------------------------------------------------------------------------------- model
def coefficients(cfg, seed, cond):
    """Trend coefficients d_i; the magnitudes are shared by the three conditions of a seed."""
    n = cfg["model"]["num_agent"]
    rng = np.random.default_rng(cfg["coef_offset"] + seed)
    m = rng.uniform(cfg["coef_lo"], cfg["coef_hi"], n)
    s = rng.permutation(np.concatenate([np.ones(n // 2), -np.ones(n - n // 2)]))
    return {"base": -m, "zero": np.zeros(n), "mix": s * m}[cond]


def equivalence(cfg, seed, n_trades):
    """S0 check 1: passing the coefficients DDMv3 drew itself must reproduce its prices bit for bit."""
    kw = D.uncoupled_model_kwargs(cfg)
    random.seed(seed)
    np.random.seed(seed)
    first = L.DDMv3(**kw)
    d = first.dealer_sensitive.copy()
    a = L.collect_prices(first, n_trades, cfg)
    random.seed(seed)
    np.random.seed(seed)
    b = L.collect_prices(L.DDMv3(**kw, dealer_sensitive=d), n_trades, cfg)
    return {"seed": seed, "n": n_trades, "equal": bool(np.array_equal(a, b)),
            "price_changes": int(np.count_nonzero(np.diff(a))), "d_min": float(d.min()), "d_max": float(d.max())}


def run_job(args):
    """One simulation of one condition. Pilot runs (save_dir None) keep nothing but success/failure and time."""
    cfg, cond, seed, save_dir = args
    c0, w0 = time.process_time(), time.perf_counter()
    out = {"cond": cond, "seed": seed, "ok": True, "error": None}
    try:
        random.seed(seed)
        np.random.seed(seed)
        n = cfg["burn"] + cfg["k"]
        if cfg["fake"]:
            if cond in cfg.get("fake_fail", []):
                raise RuntimeError("test failure")
            rng = np.random.default_rng(seed * 10 + CONDS.index(cond))
            prices = cfg["model"]["initial_price"] + cfg["model"]["spread"] + np.cumsum(rng.normal(0.0, 0.002, n))
        else:
            model = L.DDMv3(**D.uncoupled_model_kwargs(cfg), dealer_sensitive=coefficients(cfg, seed, cond))
            prices = L.collect_prices(model, n, cfg)
        if save_dir is not None:
            np.savez(os.path.join(save_dir, f"{cond}_{seed}.npz"), prices=prices)
    except Exception as e:  # failure of the run as defined in the DP1 plan
        out.update(ok=False, error=repr(e))
    out.update(cpu=time.process_time() - c0, wall=time.perf_counter() - w0)
    return out


def run_jobs(jobs, cfg, budget):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    ex = ProcessPoolExecutor(max_workers=cfg["workers"])
    try:
        for f in as_completed([ex.submit(run_job, j) for j in jobs]):
            r = f.result()
            out.append(r)
            budget.add(r["cpu"])
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return sorted(out, key=lambda r: (r["cond"], r["seed"]))


# ------------------------------------------------------------------------------------------- statistics
def run_stats(prices, t, cfg):
    """Fixed-time statistics of one price series on one clock (DP2 definitions plus the overall variance ratio)."""
    out = D.series_stats(prices, t, cfg)
    r, n, q = D.bars(prices, t, cfg)
    has = n >= 1
    out["vr_all"] = float((r[has] ** 2).sum() / q[has].sum())
    out["kurtosis_60s"] = float(pd.Series(r).kurt())
    return out


def price_stats(prices, cfg):
    b = cfg["burn"]
    d = np.diff(np.log(prices[b - 1:]))
    return {"trade_ret_acf": {str(k): L.acf(d, k) for k in range(1, 6)},
            "bar20_ret_acf1": L.acf(L.bar_returns(prices, cfg), 1),
            "price_min": float(prices.min()), "price_max": float(prices.max())}


def judge_delta(lo, hi, cfg):
    m = cfg["delta_min"]
    if lo > m:
        return "打ち消しが大きく減る"
    if hi < -m:
        return "打ち消しが大きく増える"
    if lo > -m and hi < m:
        return "大きな変化なし"
    return "判定保留"


def hypothesis_from(verdict):
    if verdict == "打ち消しが大きく減る":
        return "支持"
    if verdict in ("大きな変化なし", "打ち消しが大きく増える"):
        return "不支持"
    return "保留"


def compare(base, treat, cfg):
    """base / treat: per-seed effects {"g": G of the series, "rw": G of its shuffled benchmark} over common seeds."""
    g = {k: np.asarray(v["g"], float) for k, v in (("base", base), ("treat", treat))}
    rw = {k: np.asarray(v["rw"], float) for k, v in (("base", base), ("treat", treat))}
    out = {"G": {k: L.boot_ci(g[k], cfg["level_pre"], cfg) for k in g},
           "G_RW": {k: L.boot_ci(rw[k], cfg["level_pre"], cfg) for k in rw}}
    out["precondition"] = bool(all(out["G_RW"][k]["lo"] > cfg["pre_min"] for k in rw))
    if not out["precondition"]:
        out["verdict"] = "判定しない(前提を満たさない)"
        return out
    n = len(g["base"])
    idx = np.random.default_rng(cfg["boot_seed"]).integers(0, n, size=(cfg["n_boot"], n))
    reps = {k: g[k][idx].mean(axis=1) / rw[k][idx].mean(axis=1) for k in g}
    point = {k: float(g[k].mean() / rw[k].mean()) for k in g}

    def interval(x, level):
        a = (1 - level) / 2
        lo, hi = np.quantile(x, [a, 1 - a])
        return float(lo), float(hi)

    for k in g:
        lo, hi = interval(reps[k], cfg["level_pre"])
        out.setdefault("rho", {})[k] = {"value": point[k], "lo": lo, "hi": hi, "level": cfg["level_pre"]}
    lo, hi = interval(reps["treat"] - reps["base"], cfg["level_delta"])
    out["delta"] = {"value": point["treat"] - point["base"], "lo": lo, "hi": hi, "level": cfg["level_delta"]}
    out["verdict"] = judge_delta(lo, hi, cfg)
    return out


# ------------------------------------------------------------------------------------------- S0
def stage_s0(cfg, work, results, state):
    if os.path.exists(os.path.join(results, "s0.json")):
        raise SystemExit("s0.json exists: S0 is run once")
    os.makedirs(work, exist_ok=True)
    res = {"code": state, "cfg": cfg, "checks": {}, "proceed": False, "stopped": None}
    budget = R1.Budget(cfg["cpu_cap_s"])
    try:
        _s0_body(cfg, work, res, budget)
    except R1.Stop as e:
        res["stopped"] = str(e)
    res["cpu_s"] = budget.used
    R1.save(results, "s0.json", res)
    print("S0 proceed:", res["proceed"], "| stopped:", res["stopped"], f"| CPU {budget.used:.0f} s")


def _s0_body(cfg, work, res, budget):
    ch = res["checks"]
    # 1 pool and coefficient equivalence
    cal, sha = D.load_reference(cfg)
    res["calib"] = cal
    got = R1.sha256(cfg["pool_path"])
    c0 = time.process_time()
    rows = [equivalence(cfg, s, cfg["equiv_trades"]) for s in cfg["equiv_seeds"]] if got == sha else []
    ch["1_pool_equivalence"] = {"pool_sha256": got, "dp1_sha256": sha, "rows": rows,
                                "pass": bool(got == sha and rows and all(r["equal"] for r in rows))}
    budget.add(time.process_time() - c0)
    print("check 1", ch["1_pool_equivalence"], flush=True)
    if not ch["1_pool_equivalence"]["pass"]:
        raise R1.Stop("check 1 (pool / equivalence) failed")
    lf = np.load(cfg["pool_path"])

    # 2 clock, 3 operation (no prices)
    c0 = time.process_time()
    positive, diffs, counts = True, [], {o: [] for o in ORDERS}
    for s in cfg["main_seeds"]:
        for o in ORDERS:
            t = D.clock(lf, cal, s, o, cfg)
            positive &= bool(np.all(t > 0) and np.all(np.isfinite(t)))
            counts[o].append(L.p_stat(D.bar_counts(t, cfg), cfg))
            if o == "ord":
                diffs.append(L.half_difference(t, cfg))
    half = float(np.mean(diffs))
    ch["2_clock"] = {"all_positive": positive, "half_diff_mean": half, "half_diff_by_seed": diffs,
                     "pass": bool(positive and abs(half) <= cfg["half_tol"])}
    diff = float(np.mean(counts["ord"]) - np.mean(counts["shuf"]))
    ch["3_operation"] = {"ord": float(np.mean(counts["ord"])), "shuf": float(np.mean(counts["shuf"])), "diff": diff,
                         "pass": bool(diff >= cfg["op_min"])}
    budget.add(time.process_time() - c0)
    print("check 2", {k: v for k, v in ch["2_clock"].items() if k != "half_diff_by_seed"}, "\ncheck 3", ch["3_operation"],
          flush=True)
    if not ch["2_clock"]["pass"]:
        raise R1.Stop("check 2 (clock) failed")
    if not ch["3_operation"]["pass"]:
        raise R1.Stop("check 3 (operation) failed")

    # 4 stability pilot (no statistic is computed)
    rows = run_jobs([(cfg, c, s, None) for c in CONDS for s in cfg["pilot_seeds"]], cfg, budget)
    fails = {c: sum(1 for r in rows if r["cond"] == c and not r["ok"]) for c in CONDS}
    bad = [c for c, n in fails.items() if n > cfg["pilot_max_fail"]]
    ch["4_pilot"] = {"runs": rows, "failures": fails, "failed_conditions": bad}
    print("check 4 failures", fails, flush=True)
    if "base" in bad:
        raise R1.Stop("check 4: base is unstable")
    res["run_conditions"] = [c for c in TREATMENTS if c not in bad]
    if not res["run_conditions"]:
        raise R1.Stop("check 4: no comparison left")

    # 5 base runs and negative control
    rows = run_jobs([(cfg, "base", s, work) for s in cfg["main_seeds"]], cfg, budget)
    res["base_runs"] = rows
    ok = [r["seed"] for r in rows if r["ok"]]
    if len(ok) < cfg["min_pairs"]:
        raise R1.Stop(f"check 5: only {len(ok)} valid base runs")
    c0 = time.process_time()
    perm = []
    for s in ok:
        t = D.clock(lf, cal, s, "ord", cfg)
        r, _, _ = D.bars(D.make_rw(R1.load_prices(work, "base", s), s, cfg), t, cfg)
        perm.append(D.f_stat(np.random.default_rng(cfg["barperm_offset"] + s).permutation(r), cfg))
    neg = L.boot_ci(perm, cfg["level_pre"], cfg)
    neg["pass"] = bool(neg["lo"] >= -cfg["neg_band"] and neg["hi"] <= cfg["neg_band"])
    ch["5_negative"] = neg
    budget.add(time.process_time() - c0)
    print("check 5", neg, flush=True)
    if not neg["pass"]:
        raise R1.Stop("check 5 (negative control) failed")
    res["proceed"] = True


# ------------------------------------------------------------------------------------------- S1
def stage_s1(cfg, work, results, state, changed=""):
    if os.path.exists(os.path.join(results, "s1_started.json")) or os.path.exists(os.path.join(results, "s1.json")):
        raise SystemExit("S1 was already started: it is run once")
    s0 = R1.load(results, "s0.json")
    if not s0["proceed"]:
        raise SystemExit("S0 did not pass")
    if s0["cfg"] != json.loads(json.dumps(cfg)):
        raise SystemExit("configuration differs from S0")
    if changed:
        raise SystemExit("code or plan changed since S0:\n" + changed)
    if R1.sha256(cfg["pool_path"]) != s0["checks"]["1_pool_equivalence"]["pool_sha256"]:
        raise SystemExit("pool changed")
    R1.save(results, "s1_started.json", state)
    res = {"code": state, "s0_head": s0["code"]["head"], "stopped": None}
    budget = R1.Budget(cfg["cpu_cap_s"], s0["cpu_s"])
    try:
        _s1_body(cfg, work, res, budget, s0)
    except R1.Stop as e:
        res["stopped"] = str(e)
    res["cpu_s_total"] = budget.used
    R1.save(results, "s1.json", res)
    print("S1 done | stopped:", res["stopped"], f"| CPU total {budget.used:.0f} s")


def _s1_body(cfg, work, res, budget, s0):
    lf = np.load(cfg["pool_path"])
    cal = s0["calib"]
    conds = s0["run_conditions"]
    rows = run_jobs([(cfg, c, s, work) for c in conds for s in cfg["main_seeds"]], cfg, budget)
    res["runs"] = rows
    ok = {(r["cond"], r["seed"]) for r in rows + s0["base_runs"] if r["ok"]}

    c0 = time.process_time()
    per_run = {}
    for c in ["base"] + conds:
        per_run[c] = {}
        for s in cfg["main_seeds"]:
            if (c, s) not in ok:
                continue
            p = R1.load_prices(work, c, s)
            rw = D.make_rw(p, s, cfg)
            entry = {"price": price_stats(p, cfg)}
            for o in ORDERS:
                t = D.clock(lf, cal, s, o, cfg)
                entry[o] = run_stats(p, t, cfg)
                entry[f"rw_{o}"] = run_stats(rw, t, cfg)
            per_run[c][s] = entry

    def effects(c, seeds):
        return {"g": [per_run[c][s]["ord"]["f"] - per_run[c][s]["shuf"]["f"] for s in seeds],
                "rw": [per_run[c][s]["rw_ord"]["f"] - per_run[c][s]["rw_shuf"]["f"] for s in seeds]}

    comparisons = {}
    for c in TREATMENTS:
        if c not in conds:
            comparisons[c] = {"verdict": "評価不能(安定性パイロットで除外)"}
            continue
        seeds = [s for s in cfg["main_seeds"] if s in per_run["base"] and s in per_run[c]]
        if len(seeds) < cfg["min_pairs"]:
            comparisons[c] = {"verdict": "評価不能(有効な seed が不足)", "seeds": seeds}
            continue
        comparisons[c] = {"seeds": seeds, **compare(effects("base", seeds), effects(c, seeds), cfg)}
    res["comparisons"] = comparisons
    zero = comparisons["zero"]["verdict"]
    res["hypothesis"] = hypothesis_from(zero)
    res["summary"] = {c: L.mean_tree(list(v.values())) for c, v in per_run.items() if v}
    res["per_run"] = per_run
    budget.add(time.process_time() - c0)
    print(json.dumps({c: {k: v for k, v in row.items() if k != "seeds"} for c, row in comparisons.items()},
                     ensure_ascii=False, indent=1))
    print("hypothesis:", res["hypothesis"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("s0", "s1"):
        raise SystemExit(__doc__)
    state = code_state()
    if sys.argv[1] == "s0":
        stage_s0(CFG, WORK, RESULTS, state)
    else:
        changed = code_changed_since(R1.load(RESULTS, "s0.json")["code"]["head"])
        stage_s1(CFG, WORK, RESULTS, state, changed)
