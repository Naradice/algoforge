"""DP2 (Study Manager #1153): at which stage is the clock's persistence lost on 60 s bars?

Plan: docs/research-ddm-fixed-time-persistence-plan.md v0.1 (approved 2026-11-28). The DP1 code
(research/ddm_persist/dp_lib.py) is reused unchanged.

    cd backend/research/ddm_persist2
    python dp2_run.py s0     # checks 1-4 (once)
    python dp2_run.py s1     # coupled runs and the evaluation (once; needs a passing s0 and unchanged code)

Choices for descriptive items that the plan does not spell out, fixed before any run:
  - the remaining share R and the effects G_U, G_C are reported with 95% intervals (not judged);
  - the quartile bins of the trade count N (bars with N >= 1, per run) are N <= q25, (q25, q50], (q50, q75], > q75.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DP1 = os.path.join(os.path.dirname(HERE), "ddm_persist")
if DP1 not in sys.path:
    sys.path.insert(0, DP1)

import dp_lib as L  # noqa: E402  (sets the BLAS thread variables before numpy)
import dp_run as R1  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

RESULTS = os.path.join(HERE, "results")
WORK = os.path.join(L.REPO, "artifacts", "research", "ddm_persist2")
CODE_PATHS = ["backend/research/ddm_persist2", "backend/research/ddm_persist/dp_lib.py",
              "backend/research/ddm_persist/dp_run.py", "backend/research/tdm_fig3_audit.py",
              "backend/research/tdm_fig3_t0c.py", "backend/data/collectors/ddm_simulator.py",
              "docs/research-ddm-fixed-time-persistence-plan.md"]
EXCLUDE = [":(exclude)backend/research/ddm_persist2/results"]
CLOCK = "K-long"
SERIES = ("RW", "U", "C")
ORDERS = ("ord", "shuf")

CFG = dict(L.CFG)
CFG.update({
    "main_seeds": list(range(301, 321)), "equiv_seeds": [4, 5, 6], "equiv_trades": 20_000,
    "rw_offset": 90_000, "barperm_offset": 60_000, "boot_seed": 20261129, "width": 60.0,
    "pre_min": 0.03, "share_major": 0.5, "share_some": 0.2, "level_pre": 0.95, "level_share": 0.975,
    "cpu_cap_s": 3600.0,
    "dp1_s0": os.path.join(DP1, "results", "s0.json"),
    "pool_path": os.path.join(L.REPO, "artifacts", "research", "ddm_persist", "lf_pool.npy"),
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


# ------------------------------------------------------------------------------------------- series
def clock(lf, cal, seed, order, cfg):
    return L.make_clock(lf, cal, seed, order, cfg)


def uncoupled_model_kwargs(cfg):
    return dict(cfg["model"], wma=[1, 1, 1, 1, 1])


def run_job(args):
    """One simulation: "U" (uncoupled) or "C-ord" / "C-shuf" (DP1 time-window model)."""
    cfg, cond, seed, intervals, save_dir = args
    c0, w0 = time.process_time(), time.perf_counter()
    out = {"cond": cond, "seed": seed, "ok": True, "error": None}
    try:
        random.seed(seed)
        np.random.seed(seed)
        n = cfg["burn"] + cfg["k"]
        if cfg["fake"]:
            rng = np.random.default_rng(seed * 10 + ["U", "C-ord", "C-shuf"].index(cond))
            prices = cfg["model"]["initial_price"] + cfg["model"]["spread"] + np.cumsum(rng.normal(0.0, 0.002, n))
        else:
            if cond == "U":
                model = L.DDMv3(**uncoupled_model_kwargs(cfg))
            else:
                model = L.DDMWindow(intervals, cfg["eta"], **cfg["model"])
            prices = L.collect_prices(model, n, cfg)
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


def make_rw(prices, seed, cfg):
    """The post-burn-in per-trade log returns of the uncoupled series in random order (burn-in untouched)."""
    b = cfg["burn"]
    lp = np.log(prices[b - 1:])
    d = np.random.default_rng(cfg["rw_offset"] + seed).permutation(np.diff(lp))
    out = prices.copy()
    out[b:] = np.exp(lp[0] + np.cumsum(d))
    return out


# ------------------------------------------------------------------------------------------- fixed-time bars
def grid(t, cfg):
    """The 60 s grid of DP1 / T1: trade times, first and one-past-last trade index inside it, bar of each."""
    times = np.cumsum(t)
    w = cfg["width"]
    start = np.ceil(times[cfg["burn"] - 1] / w) * w
    end = np.floor(times[-1] / w) * w
    nb = int(round((end - start) / w))
    i0, i1 = np.searchsorted(times, [start, end], "left")
    bins = ((times[i0:i1] - start) // w).astype(np.int64)     # a trade exactly on a boundary goes to the later bar
    return nb, i0, i1, bins


def bar_counts(t, cfg):
    nb, _, _, bins = grid(t, cfg)
    return np.bincount(bins, minlength=nb).astype(float)


def bars(prices, t, cfg):
    """Bar log return r, trade count n and sum of squared per-trade log returns q."""
    nb, i0, i1, bins = grid(t, cfg)
    r = L.fixed_time_returns(prices, t, cfg, cfg["width"])
    d = np.diff(np.log(prices[i0 - 1:i1]))
    n = np.bincount(bins, minlength=nb).astype(float)
    q = np.bincount(bins, weights=d * d, minlength=nb)
    return r, n, q


def f_stat(r, cfg):
    return L.p_stat(np.abs(r), cfg)


def series_stats(prices, t, cfg):
    r, n, q = bars(prices, t, cfg)
    a = np.abs(r)
    has = n >= 1
    nh, rh, qh = n[has], r[has], q[has]
    bins = np.searchsorted(np.quantile(nh, [0.25, 0.5, 0.75]), nh, side="left")
    vr = []
    for j in range(4):
        m = bins == j
        s = float(qh[m].sum())
        vr.append(float((rh[m] ** 2).sum() / s) if s > 0 else float("nan"))
    return {"f": f_stat(r, cfg), "abs_acf": {str(k): L.acf(a, k) for k in (1, 5, 20)},
            "f_count": L.p_stat(n, cfg), "f_sqrtq": L.p_stat(np.sqrt(q), cfg), "vr_by_n_quartile": vr,
            "spearman_n_q_per_trade": float(sps.spearmanr(nh, qh / nh).statistic),
            "mean_count": float(n.mean()), "empty_share": float(np.mean(~has)), "n_bars": int(len(r))}


# ------------------------------------------------------------------------------------------- shares
def judge_share(lo, hi, cfg):
    if lo > cfg["share_major"]:
        return "主な原因"
    if lo > cfg["share_some"]:
        return "寄与あり"
    if hi < cfg["share_some"]:
        return "小さい"
    return "判定保留"


def share_ci(num, den, level, cfg):
    """Bootstrap over seeds of mean(num) / mean(den): resample seeds, average, then take the ratio."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(cfg["boot_seed"])
    idx = rng.integers(0, len(num), size=(cfg["n_boot"], len(num)))
    reps = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a])
    return {"value": float(num.mean() / den.mean()), "lo": float(lo), "hi": float(hi), "level": level}


def evaluate(f, cfg):
    """f[series][order] = per-seed F arrays over the same valid seeds."""
    d = {x: np.asarray(f[x]["ord"]) - np.asarray(f[x]["shuf"]) for x in SERIES}
    g = {x: L.boot_ci(d[x], cfg["level_pre"], cfg) for x in SERIES}
    out = {"G": g, "d_by_seed": {x: d[x].tolist() for x in SERIES},
           "F_mean": {x: {o: float(np.mean(f[x][o])) for o in ORDERS} for x in SERIES}}
    out["precondition"] = {"threshold": cfg["pre_min"], "pass": bool(g["RW"]["lo"] > cfg["pre_min"])}
    if not out["precondition"]["pass"]:
        out["conclusion"] = "前提を満たさない: 約定間の依存がなくても、この時計の持続は 60 秒足にほとんど出ない(割合は判定しない)"
        return out
    shares = {"L_A": share_ci(d["RW"] - d["U"], d["RW"], cfg["level_share"], cfg),
              "L_B": share_ci(d["U"] - d["C"], d["RW"], cfg["level_share"], cfg),
              "R": share_ci(d["C"], d["RW"], cfg["level_pre"], cfg)}
    for k in ("L_A", "L_B"):
        shares[k]["verdict"] = judge_share(shares[k]["lo"], shares[k]["hi"], cfg)
    out["shares"] = shares
    return out


# ------------------------------------------------------------------------------------------- S0
def load_reference(cfg):
    with open(cfg["dp1_s0"], encoding="utf-8") as fh:
        ref = json.load(fh)
    return ref["calib"][CLOCK], ref["checks"]["2_clock"]["pool"]["sha256"]


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
    # 1 pool and equivalence
    cal, sha = load_reference(cfg)
    res["calib"] = cal
    got = R1.sha256(cfg["pool_path"])
    c0 = time.process_time()
    rows = []
    if got == sha:
        for s in cfg["equiv_seeds"]:
            rows.append(L.equivalence(cfg, s, cfg["equiv_trades"]))
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
            t = clock(lf, cal, s, o, cfg)
            positive &= bool(np.all(t > 0) and np.all(np.isfinite(t)))
            counts[o].append(L.p_stat(bar_counts(t, cfg), cfg))
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

    # 4 uncoupled runs and negative control
    rows = run_jobs([(cfg, "U", s, None, work) for s in cfg["main_seeds"]], cfg, budget)
    res["u_runs"] = rows
    ok = [r["seed"] for r in rows if r["ok"]]
    if len(ok) < cfg["min_pairs"]:
        raise R1.Stop(f"check 4: only {len(ok)} valid U runs")
    c0 = time.process_time()
    perm = []
    for s in ok:
        t = clock(lf, cal, s, "ord", cfg)
        r, _, _ = bars(make_rw(R1.load_prices(work, "U", s), s, cfg), t, cfg)
        perm.append(f_stat(np.random.default_rng(cfg["barperm_offset"] + s).permutation(r), cfg))
    neg = L.boot_ci(perm, cfg["level_pre"], cfg)
    neg["pass"] = bool(neg["lo"] >= -cfg["neg_band"] and neg["hi"] <= cfg["neg_band"])
    ch["4_negative"] = neg
    budget.add(time.process_time() - c0)
    print("check 4", neg, flush=True)
    if not neg["pass"]:
        raise R1.Stop("check 4 (negative control) failed")
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
    jobs = [(cfg, f"C-{o}", s, clock(lf, cal, s, o, cfg), work) for o in ORDERS for s in cfg["main_seeds"]]
    rows = run_jobs(jobs, cfg, budget)
    res["c_runs"] = rows
    ok = {(r["cond"], r["seed"]) for r in rows + s0["u_runs"] if r["ok"]}
    seeds = [s for s in cfg["main_seeds"] if all((c, s) in ok for c in ("U", "C-ord", "C-shuf"))]
    res["valid_seeds"] = seeds
    if len(seeds) < cfg["min_pairs"]:
        res["evaluation"] = {"conclusion": f"評価不能(有効な seed が {len(seeds)})"}
        return
    c0 = time.process_time()
    per_run = {x: {o: [] for o in ORDERS} for x in SERIES}
    for s in seeds:
        u = R1.load_prices(work, "U", s)
        rw = make_rw(u, s, cfg)
        for o in ORDERS:
            t = clock(lf, cal, s, o, cfg)
            prices = {"RW": rw, "U": u, "C": R1.load_prices(work, f"C-{o}", s)}
            for x in SERIES:
                per_run[x][o].append(series_stats(prices[x], t, cfg))
    f = {x: {o: [p["f"] for p in per_run[x][o]] for o in ORDERS} for x in SERIES}
    res["evaluation"] = evaluate(f, cfg)
    res["summary"] = {x: {o: L.mean_tree(per_run[x][o]) for o in ORDERS} for x in SERIES}
    res["per_run"] = per_run
    budget.add(time.process_time() - c0)
    ev = res["evaluation"]
    print(json.dumps({k: v for k, v in ev.items() if k != "d_by_seed"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("s0", "s1"):
        raise SystemExit(__doc__)
    state = code_state()
    if sys.argv[1] == "s0":
        stage_s0(CFG, WORK, RESULTS, state)
    else:
        changed = code_changed_since(R1.load(RESULTS, "s0.json")["code"]["head"])
        stage_s1(CFG, WORK, RESULTS, state, changed)
