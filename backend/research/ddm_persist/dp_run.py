"""DP1 stages. Plan: docs/research-ddm-volatility-persistence-plan.md v0.1 (approved).

    cd backend/research/ddm_persist
    python dp_run.py s0     # instrument checks 1-6 (once)
    python dp_run.py s1     # main experiment (once; needs a passing s0 and unchanged code)

Results: results/s0.json, results/s1.json. Price series: artifacts/research/ddm_persist/.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import dp_lib as L
import numpy as np

RESULTS = os.path.join(L.HERE, "results")
WORK = os.path.join(L.REPO, "artifacts", "research", "ddm_persist")
CODE_PATHS = ["backend/research/ddm_persist", "backend/research/tdm_fig3_audit.py", "backend/research/tdm_fig3_t0c.py",
              "backend/data/collectors/ddm_simulator.py", "docs/research-ddm-volatility-persistence-plan.md"]
EXCLUDE = ":(exclude)backend/research/ddm_persist/results"
VERSIONS = {"numpy": "2.3.3", "scipy": "1.17.1", "pandas": "3.0.1"}


class Stop(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", "-C", L.REPO, *a], text=True).strip()


def code_state():
    dirty = git("status", "--porcelain", "--", *CODE_PATHS, EXCLUDE)
    if dirty:
        raise SystemExit("uncommitted changes -> commit first:\n" + dirty)
    got = {k: md.version(k) for k in VERSIONS}
    if got != VERSIONS:
        raise SystemExit(f"library versions differ: {got}")
    return {"head": git("rev-parse", "HEAD"), "versions": got, "python": sys.version.split()[0],
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def save(results, name, obj):
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def load(results, name):
    with open(os.path.join(results, name), encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Budget:
    def __init__(self, cap, used=0.0):
        self.cap, self.used = cap, used

    def add(self, s):
        self.used += s
        if self.used > self.cap:
            raise Stop(f"CPU time cap exceeded: {self.used:.0f} s > {self.cap:.0f} s")


def run_jobs(jobs, cfg, budget):
    out = []
    ex = ProcessPoolExecutor(max_workers=cfg["workers"])
    try:
        futures = [ex.submit(L.run_job, j) for j in jobs]
        for f in as_completed(futures):
            r = f.result()
            out.append(r)
            budget.add(r["cpu"])
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return sorted(out, key=lambda r: (r["cond"], r["seed"]))


def clock_for(lf, calib, cond, seed, cfg):
    name, order = L.CONDITIONS[cond]
    return None if name is None else L.make_clock(lf, calib[name], seed, order, cfg)


def load_prices(work, cond, seed):
    return np.load(os.path.join(work, f"{cond}_{seed}.npz"))["prices"]


# ------------------------------------------------------------------------------------------- S0
def stage_s0(cfg, work, results, state):
    if os.path.exists(os.path.join(results, "s0.json")):
        raise SystemExit("s0.json exists: S0 is run once")
    os.makedirs(work, exist_ok=True)
    res = {"code": state, "cfg": cfg, "checks": {}, "proceed": False, "stopped": None}
    budget = Budget(cfg["cpu_cap_s"])
    try:
        _s0_body(cfg, work, res, budget)
    except Stop as e:
        res["stopped"] = str(e)
    res["cpu_s"] = budget.used
    save(results, "s0.json", res)
    print("S0 proceed:", res["proceed"], "| stopped:", res["stopped"], f"| CPU {budget.used:.0f} s")


def _s0_body(cfg, work, res, budget):
    ch = res["checks"]
    # 1 equivalence
    rows = []
    for s in cfg["equiv_seeds"]:
        c0 = time.process_time()
        rows.append(L.equivalence(cfg, s, cfg["equiv_trades"]))
        budget.add(time.process_time() - c0)
    ch["1_equivalence"] = {"rows": rows, "pass": all(r["equal"] for r in rows)}
    print("check 1", ch["1_equivalence"], flush=True)
    if not ch["1_equivalence"]["pass"]:
        raise Stop("check 1 (equivalence) failed")

    # 2 clock calibration and checks (no prices)
    with ProcessPoolExecutor(max_workers=len(cfg["pool_seeds"])) as ex:
        parts = list(ex.map(L.pool_part, [(cfg, s) for s in cfg["pool_seeds"]]))
    for _, _, cpu in parts:
        budget.add(cpu)
    lf = L.make_logf([p for _, p, _ in parts], cfg)
    lf_path = os.path.join(work, "lf_pool.npy")
    np.save(lf_path, lf)
    c2 = {"pool": {"n": int(len(lf)), "var_logF": float(lf.var()), "sha256": sha256(lf_path)}, "clocks": {}}
    calib, kept = {}, []
    for name, (n, lam) in cfg["clocks"].items():
        c0 = time.process_time()
        calib[name] = L.calibrate(lf, n, lam, cfg)
        diag = L.transient_diag(lf, n, lam, calib[name]["m0"], cfg)
        positive, diffs, means = True, [], []
        for s in cfg["pilot_seeds"] + cfg["main_seeds"]:
            t = L.make_clock(lf, calib[name], s, "ord", cfg)
            positive &= bool(np.all(t > 0) and np.all(np.isfinite(t)))
            if s in cfg["main_seeds"]:
                diffs.append(L.half_difference(t, cfg))
                means.append(float(t[cfg["burn"]:].mean()))
        row = {**calib[name], "transient": diag, "all_positive": positive,
               "half_diff_mean": float(np.mean(diffs)), "half_diff_by_seed": diffs,
               "mean_interval": float(np.mean(means))}
        row["pass"] = bool(positive and diag["pass"] and abs(row["half_diff_mean"]) <= cfg["half_tol"])
        c2["clocks"][name] = row
        if row["pass"]:
            kept.append(name)
        budget.add(time.process_time() - c0)
        print("check 2", name, {k: row[k] for k in ("m0", "kappa", "half_diff_mean", "mean_interval", "pass")}, diag, flush=True)
    ch["2_clock"] = c2
    res["calib"] = calib
    res["kept_clocks"] = kept

    # 3 operation check (no prices)
    c3, judged = {}, []
    for name in kept:
        c0 = time.process_time()
        vals = {o: [L.p_stat(L.bar_durations(L.make_clock(lf, calib[name], s, o, cfg), cfg), cfg)
                    for s in cfg["main_seeds"]] for o in ("ord", "shuf")}
        diff = float(np.mean(vals["ord"]) - np.mean(vals["shuf"]))
        c3[name] = {"ord": float(np.mean(vals["ord"])), "shuf": float(np.mean(vals["shuf"])), "diff": diff,
                    "pass": bool(diff >= cfg["op_min"])}
        if c3[name]["pass"]:
            judged.append(name)
        budget.add(time.process_time() - c0)
        print("check 3", name, c3[name], flush=True)
    ch["3_operation"] = c3
    res["judged_clocks"] = judged

    # 4 stability pilot (no statistic is computed)
    conds = ["B0"] + [c for name in kept for c in L.COMPARISONS[name]]
    jobs = [(cfg, c, s, clock_for(lf, calib, c, s, cfg), None) for c in conds for s in cfg["pilot_seeds"]]
    rows = run_jobs(jobs, cfg, budget)
    fails = {c: sum(1 for r in rows if r["cond"] == c and not r["ok"]) for c in conds}
    bad = [c for c, n in fails.items() if n > cfg["pilot_max_fail"]]
    ch["4_pilot"] = {"runs": rows, "failures": fails, "failed_conditions": bad}
    print("check 4 failures", fails, flush=True)
    if "B0" in bad:
        raise Stop("check 4: B0 is unstable (needed for checks 5 and 6)")
    res["run_clocks"] = [n for n in kept if not set(L.COMPARISONS[n]) & set(bad)]

    # 5 and 6 on the main B0 runs
    rows = run_jobs([(cfg, "B0", s, None, work) for s in cfg["main_seeds"]], cfg, budget)
    ok = [r["seed"] for r in rows if r["ok"]]
    res["b0_runs"] = rows
    if len(ok) < cfg["min_pairs"]:
        raise Stop(f"only {len(ok)} valid B0 runs")
    c0 = time.process_time()
    base, perm, pos = [], [], []
    for s in ok:
        p = load_prices(work, "B0", s)
        base.append(L.p_of_run(p, cfg))
        perm.append(L.permuted_p(p, s, cfg))
        pos.append(L.p_of_run(L.positive_control(p, s, cfg), cfg))
    neg = L.boot_ci(perm, cfg["level_check"], cfg)
    neg["pass"] = bool(neg["lo"] >= -cfg["neg_band"] and neg["hi"] <= cfg["neg_band"])
    dif = L.boot_ci(np.array(pos) - np.array(base), cfg["level_check"], cfg)
    dif["pass"] = bool(dif["lo"] > cfg["pos_min"])
    ch["5_negative"] = neg
    ch["6_positive"] = {**dif, "p_positive_mean": float(np.mean(pos))}
    budget.add(time.process_time() - c0)
    print("check 5", neg, "\ncheck 6", ch["6_positive"], flush=True)
    if not neg["pass"]:
        raise Stop("check 5 (negative control) failed")
    if not dif["pass"]:
        raise Stop("check 6 (positive control) failed")
    if not res["run_clocks"]:
        raise Stop("no clock left to compare")
    res["proceed"] = True


# ------------------------------------------------------------------------------------------- S1
def stage_s1(cfg, work, results, state):
    marker = os.path.join(results, "s1_started.json")
    if os.path.exists(marker) or os.path.exists(os.path.join(results, "s1.json")):
        raise SystemExit("S1 was already started: it is run once")
    s0 = load(results, "s0.json")
    if not s0["proceed"]:
        raise SystemExit("S0 did not pass")
    if s0["cfg"] != json.loads(json.dumps(cfg)):
        raise SystemExit("configuration differs from S0")
    lf_path = os.path.join(work, "lf_pool.npy")
    if sha256(lf_path) != s0["checks"]["2_clock"]["pool"]["sha256"]:
        raise SystemExit("lf_pool changed")
    lf = np.load(lf_path)
    calib = s0["calib"]
    save(results, "s1_started.json", state)
    res = {"code": state, "s0_head": s0["code"]["head"], "stopped": None}
    budget = Budget(cfg["cpu_cap_s"], s0["cpu_s"])
    try:
        _s1_body(cfg, work, res, budget, s0, lf, calib)
    except Stop as e:
        res["stopped"] = str(e)
    res["cpu_s_total"] = budget.used
    save(results, "s1.json", res)
    print("S1 done | stopped:", res["stopped"], f"| CPU total {budget.used:.0f} s")


def _s1_body(cfg, work, res, budget, s0, lf, calib):
    conds = [c for name in s0["run_clocks"] for c in L.COMPARISONS[name]]
    jobs = [(cfg, c, s, clock_for(lf, calib, c, s, cfg), work) for c in conds for s in cfg["main_seeds"]]
    rows = run_jobs(jobs, cfg, budget)
    res["runs"] = rows
    ok = {(r["cond"], r["seed"]) for r in rows + s0["b0_runs"] if r["ok"]}

    c0 = time.process_time()
    per_run, summary = {}, {}
    for c in ["B0"] + conds:
        seeds = [s for s in cfg["main_seeds"] if (c, s) in ok]
        returns, stats = {}, {}
        for s in seeds:
            p = load_prices(work, c, s)
            stats[s] = L.secondary_run(p, clock_for(lf, calib, c, s, cfg), cfg)
            returns[s] = L.bar_returns(p, cfg)
        per_run[c] = stats
        if seeds:
            absr = [np.abs(returns[s]) for s in seeds]
            summary[c] = {"n_runs": len(seeds), "mean": L.mean_tree([stats[s] for s in seeds]),
                          "abs_acf_def32": {str(k): L.pooled_acf_32(absr, k) for k in cfg["lags_sec"]},
                          "har": L.har_r2(returns, cfg)}
    comparisons = {}
    for name, (a, b) in L.COMPARISONS.items():
        row = {"conditions": [a, b]}
        if name not in s0["run_clocks"]:
            row["verdict"] = "評価不能(S0 の検査で除外)"
        else:
            seeds = [s for s in cfg["main_seeds"] if (a, s) in ok and (b, s) in ok]
            row["pairs"] = len(seeds)
            if len(seeds) < cfg["min_pairs"]:
                row["verdict"] = "評価不能(有効な対が不足)"
            else:
                pa = [per_run[a][s]["p"] for s in seeds]
                pb = [per_run[b][s]["p"] for s in seeds]
                row.update(p_ord=float(np.mean(pa)), p_shuf=float(np.mean(pb)),
                           d=L.boot_ci(np.array(pa) - np.array(pb), cfg["level_main"], cfg),
                           d_by_seed=dict(zip(map(str, seeds), (np.array(pa) - np.array(pb)).tolist())))
                if name in s0["judged_clocks"]:
                    row["verdict"] = L.judge(row["d"]["lo"], row["d"]["hi"], cfg["delta"])
                else:
                    row["verdict"] = "判定しない(操作確認に不合格。値は記述)"
        comparisons[name] = row
    res["comparisons"] = comparisons
    res["summary"] = summary
    res["per_run"] = per_run
    budget.add(time.process_time() - c0)
    for name, row in comparisons.items():
        print(name, {k: v for k, v in row.items() if k != "d_by_seed"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("s0", "s1"):
        raise SystemExit(__doc__)
    {"s0": stage_s0, "s1": stage_s1}[sys.argv[1]](L.CFG, WORK, RESULTS, code_state())
