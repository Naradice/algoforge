"""T1 (Study Manager #895, plan v0.4, approved 2026-11-17): effect of the trade-interval clock on fixed-time OHLC.

The price event stream of DDMv3 (v3_shock) is held fixed; different clocks assign times to the SAME events:
  C0  current repository clock (tick_time, i.i.d. exp-weighted noise per step), as-is
  C0s C0 multiplied by a calibration constant so that its mean interval is 3.0 s (scale-only diagnostic)
  C1  M2-S2 (n=16, lam=0.1), independent of the dynamics, mean interval 3.0 s via a pre-fixed m0
  C2  i.i.d. exponential, mean 3.0 s (fixed, not the realised mean of C1)
  C3  C1 with the intervals after the burn-in permuted (20 independent shuffles per price stream, averaged within a
      price stream: conditional replications, NOT independent runs)

All constants below are the pre-registered specification (plan v0.4 S1-S8) and are fixed before any result is seen.
Independent unit of every confidence interval = the price-event seed.

Stages:  python research/t1_clock_ohlc.py calib <workdir>
         python research/t1_clock_ohlc.py pilot <workdir>     (price seeds 1-5, G1 gate)
         python research/t1_clock_ohlc.py full  <workdir>     (price seeds 1-20, replacing failed seeds)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from tdm_fig3_audit import model1_intervals  # noqa: E402
from tdm_fig3_t0c import simulate_ar  # noqa: E402

# ----------------------------------------------------------------------------- pre-registered constants
LAM, NWIN, ALPHA = 0.1, 16, 1.25
TARGET_MEAN = 3.0                         # seconds per trade (research convention "20 trades = 1 minute")
BURN = 500                                # events excluded everywhere (= 50 / lam)
K_EVENTS = 100_000
BAR = 60.0
N_SHUF = 20
POOL_SEEDS, POOL_TRADES, POOL_N = (9001, 9002, 9003), 20_000, 100
CAL_CLOCK_SEEDS, CAL_K = tuple(range(8001, 8021)), 400_000
S0_SEEDS = tuple(range(7001, 7006))
DIAG_CHAINS, DIAG_LEN, DIAG_TOL = 4000, 2000, 0.05
CLOCK_SEED_OFFSET, SHUFFLE_SEED_BASE = 1000, 50_000
N_BOOT = 10_000
MIN_BARS = 30                             # amended before any OHLC comparison was seen: C0 as-is yields only ~40 M1 bars per 100k trades
COUNT_CANDLE = 20                         # trades per count-based candle used by all earlier research
G1_ACF_RANGE, G1_CV_RANGE = (0.20, 0.50), (0.9, 1.8)
G1_QUARTER_MEAN_TOL, G1_QUARTER_STD_RATIO = 0.25, (0.8, 1.25)
G1_MIN_INTERVAL_FRAC = 1e-3               # fraction of intervals below 1e-3 s must be < 1e-3
G1_SOFT_PASS_FRAC = 0.90                  # criteria (iii)-(iv) must hold in >= 90% of the runs (all runs for the 5-run pilot)


# ----------------------------------------------------------------------------- helpers
def acf1(x, lag=1):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-lag], x[lag:]) / d) if d > 0 else float("nan")


def _pool_job(seed):
    _, Tp, _ = model1_intervals(POOL_TRADES, POOL_N, 1.0, 0.01, 0.01, "signed", seed)
    return Tp


def gen_events(seed):
    """DDMv3 v3_shock (transfer-experiment configuration); returns prices and C0 tick_time per trade."""
    from data.collectors.ddm_simulator import DDMv3
    random.seed(seed)
    np.random.seed(seed)
    try:
        model = DDMv3(num_agent=500, max_volatility=0.02, min_volatility=0.01, wma=5,
                      dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5,
                      exogenous_shock_probability=0.0015, exogenous_shock_size=0.3)
        prices, ticks = [], []
        for p, t in model.simulate_stream(n_trades=K_EVENTS):
            prices.append(p)
            ticks.append(t)
        return seed, np.array(prices, float), np.array(ticks, float), None
    except Exception as e:  # divergence / stall guard of the simulator
        return seed, None, None, repr(e)


def load_calib(workdir):
    c = json.load(open(os.path.join(workdir, "calib.json")))
    c["lf_pool"] = np.load(os.path.join(workdir, "lf_pool.npy"))
    return c


# ----------------------------------------------------------------------------- stage: calibration
def stage_calib(workdir):
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        pools = list(ex.map(_pool_job, POOL_SEEDS))
    Tp = np.concatenate(pools)
    Tp = Tp[Tp > 0]
    x = Tp ** ALPHA
    lf = np.log(x / x.mean())
    lf = lf - lf.mean()                                   # E[log F] = 0 exactly
    np.save(os.path.join(workdir, "lf_pool.npy"), lf)
    print(f"pool: {len(lf)} F values, var(logF)={lf.var():.3f}  ({time.time() - t0:.0f}s)")

    kap = []
    for s in CAL_CLOCK_SEEDS:
        rng = np.random.default_rng(s)
        T = simulate_ar(rng.choice(lf, size=CAL_K), NWIN, LAM, m0=1.0)
        kap.append(T[BURN:].mean())
    kappa = float(np.mean(kap))
    m0 = TARGET_MEAN / kappa
    print(f"kappa=E[T]/m0={kappa:.4f} (sd across chains {np.std(kap):.4f})  ->  m0={m0:.5f}")

    # transient diagnostic: displaced initial history, many chains
    diag = {}
    rng = np.random.default_rng(6001)
    for disp in (+2.0, -2.0):
        vals = []
        for _ in range(DIAG_CHAINS):
            T = simulate_ar(rng.choice(lf, size=DIAG_LEN), NWIN, LAM, m0=m0, init_log=np.log(m0) + disp)
            vals.append(np.log(T[BURN - 50:BURN]).mean() - np.log(m0))
        diag[f"bias_at_burn_disp{disp:+.0f}"] = float(np.mean(vals))
        diag[f"se_disp{disp:+.0f}"] = float(np.std(vals) / np.sqrt(DIAG_CHAINS))
    diag["pass"] = bool(all(abs(diag[k]) < DIAG_TOL for k in diag if k.startswith("bias")))
    print("transient diagnostic:", diag)

    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(gen_events, S0_SEEDS))
    mean_iv = []
    for seed, p, t, err in res:
        if err:
            print("calibration event seed failed:", seed, err)
            continue
        mean_iv.append(np.diff(t)[BURN:].mean())
    s0 = TARGET_MEAN / float(np.mean(mean_iv))
    print(f"C0 mean interval {np.mean(mean_iv):.4f} s -> s0={s0:.2f}")
    json.dump({"kappa": kappa, "m0": m0, "s0": s0, "diag": diag, "c0_mean_interval": float(np.mean(mean_iv)),
               "pool_var_logF": float(lf.var())}, open(os.path.join(workdir, "calib.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- clock + OHLC statistics
def clock_stats(T):
    T = np.asarray(T, float)
    lg = np.log(T)
    q = len(T) // 4
    quarters = [(float(lg[i * q:(i + 1) * q].mean()), float(lg[i * q:(i + 1) * q].std())) for i in range(4)]
    return {
        "acf_logT1": acf1(lg), "acf_logT10": acf1(lg, 10), "acf_T1": acf1(T), "cv": float(T.std() / T.mean()),
        "mean": float(T.mean()), "geo": float(np.exp(lg.mean())), "max_mean": float(T.max() / T.mean()),
        "min": float(T.min()), "frac_below_1ms": float(np.mean(T < 1e-3)),
        "q_mean_logT": [a for a, _ in quarters], "q_std_logT": [b for _, b in quarters],
        "mean_logT": float(lg.mean()), "std_logT": float(lg.std()),
    }


def bars_stats(times, prices):
    """Continuous grid (main) and dropna (compat) statistics on 60 s bars after the burn-in window."""
    start = np.ceil(times[BURN - 1] / BAR) * BAR
    end = np.floor(times[-1] / BAR) * BAR
    nb = int(round((end - start) / BAR))
    if nb < MIN_BARS:
        return None
    i0 = int(np.searchsorted(times, start, "left"))
    i1 = int(np.searchsorted(times, end, "left"))
    t, p = times[i0:i1], prices[i0:i1]
    pc = prices[i0 - 1]                                    # last event price before the window
    bins = ((t - start) // BAR).astype(np.int64)
    counts = np.bincount(bins, minlength=nb).astype(float)
    ne = np.flatnonzero(counts)
    first = np.searchsorted(bins, ne, "left")
    high = np.maximum.reduceat(p, first)
    low = np.minimum.reduceat(p, first)
    cls = p[np.append(first[1:], len(p)) - 1]
    # continuous grid: forward-filled flat bars
    idx = np.where(counts > 0, np.arange(nb), -1)
    ff = np.maximum.accumulate(idx)
    cls_full = np.full(nb, np.nan)
    cls_full[ne] = cls
    close_c = np.where(ff >= 0, cls_full[np.maximum(ff, 0)], pc)
    r_c = np.diff(np.log(np.concatenate([[pc], close_c])))
    rng_c = np.zeros(nb)
    rng_c[ne] = high - low
    # compatibility (dropna) variant
    r_d = np.diff(np.log(np.concatenate([[pc], cls])))

    def ret_stats(r):
        a = np.abs(r)
        return {"var": float(r.var()), "skew": float(sps.skew(r)), "kurt": float(sps.kurtosis(r)),
                "acf1": acf1(r) if r.std() > 0 else float("nan"),
                "absacf1": acf1(a) if a.std() > 0 else float("nan"),
                "absacf10": float(np.nanmean([acf1(a, l) for l in range(1, 11)])) if a.std() > 0 else float("nan"),
                "maxabs": float(a.max()), "n": int(len(r))}

    cont = ret_stats(r_c)
    cont.update({"vmr": float(counts.var() / counts.mean()), "mean_count": float(counts.mean()),
                 "zero_bar_rate": float(np.mean(counts == 0)), "count_acf1": acf1(counts),
                 "range_mean": float(rng_c.mean()), "n_bars": nb})
    comp = ret_stats(r_d)
    comp.update({"vmr": float(counts[ne].var() / counts[ne].mean()), "n_bars": int(len(ne)),
                 "gap_bars_dropped": int(nb - len(ne))})
    return {"cont": cont, "compat": comp}


def count_ohlc(prices):
    """Event-count based candles (20 trades = 1 candle) from the prices after the burn-in, as in all earlier research."""
    p = prices[BURN:]
    n = (len(p) // COUNT_CANDLE) * COUNT_CANDLE
    g = p[:n].reshape(-1, COUNT_CANDLE)
    return np.column_stack([g[:, 0], g.max(1), g.min(1), g[:, -1]])


def numeric_mean(dicts):
    out = {}
    for k in dicts[0]:
        if isinstance(dicts[0][k], dict):
            out[k] = numeric_mean([d[k] for d in dicts])
        elif isinstance(dicts[0][k], (int, float)):
            out[k] = float(np.mean([d[k] for d in dicts]))
    return out


def analyse_seed(seed, prices, ticks, calib):
    lf_pool, m0, s0 = calib["lf_pool"], calib["m0"], calib["s0"]
    rng = np.random.default_rng(seed + CLOCK_SEED_OFFSET)
    T0 = np.diff(np.concatenate([[0.0], ticks]))
    T1 = simulate_ar(rng.choice(lf_pool, size=len(prices)), NWIN, LAM, m0=m0)
    T2 = rng.exponential(TARGET_MEAN, size=len(prices))
    clocks = {"C0": ticks, "C0s": s0 * ticks, "C1": np.cumsum(T1), "C2": np.cumsum(T2)}
    ivs = {"C0": T0, "C0s": s0 * T0, "C1": T1, "C2": T2}
    out = {"seed": seed, "clock": {}, "grid": {}, "count_ohlc_equal": True}
    ref = count_ohlc(prices)
    for name in clocks:
        out["clock"][name] = clock_stats(ivs[name][BURN:])
        out["grid"][name] = bars_stats(clocks[name], prices)
        out["count_ohlc_equal"] &= bool(np.array_equal(ref, count_ohlc(prices)))
    # C3: conditional shuffles of the post-burn-in C1 intervals
    sh_clock, sh_grid, multiset_ok, mono_ok = [], [], True, True
    for j in range(N_SHUF):
        r2 = np.random.default_rng(SHUFFLE_SEED_BASE + seed * 100 + j)
        T3 = T1.copy()
        T3[BURN:] = r2.permutation(T1[BURN:])
        multiset_ok &= bool(np.array_equal(np.sort(T3[BURN:]), np.sort(T1[BURN:])))
        t3 = np.cumsum(T3)
        mono_ok &= bool(np.all(np.diff(t3) > 0))
        sh_clock.append(clock_stats(T3[BURN:]))
        sh_grid.append(bars_stats(t3, prices))
    out["clock"]["C3"] = numeric_mean([{k: v for k, v in c.items() if not isinstance(v, list)} for c in sh_clock])
    out["grid"]["C3"] = numeric_mean(sh_grid)
    out["shuffle_sd"] = {m: float(np.std([g["cont"][m] for g in sh_grid])) for m in ("vmr", "absacf1")}
    out["g1"] = {"multiset_equal": multiset_ok, "monotonic": bool(mono_ok and all(np.all(np.diff(clocks[c]) > 0) for c in clocks))}
    return out


def g1_row(res):
    c1 = res["clock"]["C1"]
    qm = np.array(c1["q_mean_logT"]) - c1["mean_logT"]
    qs = np.array(c1["q_std_logT"]) / c1["std_logT"]
    return {
        "i_positive_monotonic": bool(res["g1"]["monotonic"] and c1["min"] > 0 and c1["frac_below_1ms"] < G1_MIN_INTERVAL_FRAC),
        "ii_multiset": bool(res["g1"]["multiset_equal"]),
        "iii_acf_cv": bool(G1_ACF_RANGE[0] <= c1["acf_logT1"] <= G1_ACF_RANGE[1] and G1_CV_RANGE[0] <= c1["cv"] <= G1_CV_RANGE[1]),
        "iv_stationarity": bool(np.all(np.abs(qm) < G1_QUARTER_MEAN_TOL) and np.all((qs >= G1_QUARTER_STD_RATIO[0]) & (qs <= G1_QUARTER_STD_RATIO[1]))),
        "count_ohlc_equal": bool(res["count_ohlc_equal"]),
    }


def g1_decide(rows, diag_pass, n_target):
    n = len(rows)
    frac = lambda k: float(np.mean([r[k] for r in rows]))
    need = 1.0 if n <= 5 else G1_SOFT_PASS_FRAC
    decision = {
        "i_all": frac("i_positive_monotonic") == 1.0, "ii_all": frac("ii_multiset") == 1.0,
        "iii_frac": frac("iii_acf_cv"), "iv_frac": frac("iv_stationarity"),
        "count_ohlc_all": frac("count_ohlc_equal") == 1.0, "v_transient_diag": bool(diag_pass), "n_runs": n,
    }
    decision["pass"] = bool(decision["i_all"] and decision["ii_all"] and decision["count_ohlc_all"] and decision["v_transient_diag"]
                            and decision["iii_frac"] >= need and decision["iv_frac"] >= need)
    return decision


# ----------------------------------------------------------------------------- aggregation
def paired(results, getter, a, b, rng, rel=False):
    d = []
    for r in results:
        xa, xb = getter(r, a), getter(r, b)
        d.append((xa - xb) / xb if rel else xa - xb)
    d = np.array(d)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_seeds": len(d)}


def aggregate(results):
    rng = np.random.default_rng(424242)
    gets = {
        "clock_acf_logT1": lambda r, c: r["clock"][c]["acf_logT1"],
        "clock_cv": lambda r, c: r["clock"][c]["cv"],
        "clock_mean": lambda r, c: r["clock"][c]["mean"],
        "cont_vmr": lambda r, c: r["grid"][c]["cont"]["vmr"],
        "cont_absacf1": lambda r, c: r["grid"][c]["cont"]["absacf1"],
        "cont_absacf10": lambda r, c: r["grid"][c]["cont"]["absacf10"],
        "cont_count_acf1": lambda r, c: r["grid"][c]["cont"]["count_acf1"],
        "cont_zero_bar_rate": lambda r, c: r["grid"][c]["cont"]["zero_bar_rate"],
        "cont_mean_count": lambda r, c: r["grid"][c]["cont"]["mean_count"],
        "cont_ret_var": lambda r, c: r["grid"][c]["cont"]["var"],
        "cont_kurt": lambda r, c: r["grid"][c]["cont"]["kurt"],
        "cont_skew": lambda r, c: r["grid"][c]["cont"]["skew"],
        "cont_ret_acf1": lambda r, c: r["grid"][c]["cont"]["acf1"],
        "cont_range_mean": lambda r, c: r["grid"][c]["cont"]["range_mean"],
        "cont_maxabs": lambda r, c: r["grid"][c]["cont"]["maxabs"],
        "compat_vmr": lambda r, c: r["grid"][c]["compat"]["vmr"],
        "compat_absacf1": lambda r, c: r["grid"][c]["compat"]["absacf1"],
        "compat_ret_var": lambda r, c: r["grid"][c]["compat"]["var"],
        "compat_kurt": lambda r, c: r["grid"][c]["compat"]["kurt"],
        "compat_gap_bars_dropped": lambda r, c: r["grid"][c]["compat"]["gap_bars_dropped"],
    }
    agg = {"levels": {}, "diffs": {}}
    for name, g in gets.items():
        agg["levels"][name] = {c: float(np.mean([g(r, c) for r in results])) for c in ("C0", "C0s", "C1", "C2", "C3")}
        agg["diffs"][name] = {}
        for a, b in (("C1", "C3"), ("C1", "C0"), ("C1", "C0s"), ("C1", "C2")):
            agg["diffs"][name][f"{a}-{b}"] = paired(results, g, a, b, rng)
    agg["diffs"]["cont_vmr_rel_C1_vs_C3"] = paired(results, gets["cont_vmr"], "C1", "C3", rng, rel=True)
    agg["diffs"]["compat_vmr_rel_C1_vs_C3"] = paired(results, gets["compat_vmr"], "C1", "C3", rng, rel=True)
    return agg


# ----------------------------------------------------------------------------- stages
def get_events(seeds, workdir):
    """Event streams (cached). Returns {seed: (prices, ticks)} for seeds that did not fail, plus failure log."""
    cache = os.path.join(workdir, "events")
    os.makedirs(cache, exist_ok=True)
    ok, failed, todo = {}, {}, []
    for s in seeds:
        f = os.path.join(cache, f"seed{s}.npz")
        if os.path.exists(f):
            z = np.load(f)
            ok[s] = (z["prices"], z["ticks"])
        elif os.path.exists(f + ".failed"):
            failed[s] = open(f + ".failed").read()
        else:
            todo.append(s)
    if todo:
        with ProcessPoolExecutor(max_workers=10) as ex:
            for seed, p, t, err in ex.map(gen_events, todo):
                f = os.path.join(cache, f"seed{seed}.npz")
                if err:
                    failed[seed] = err
                    open(f + ".failed", "w").write(err)
                else:
                    np.savez(f, prices=p, ticks=t)
                    ok[seed] = (p, t)
    return ok, failed


def run_stage(stage, workdir):
    calib = load_calib(workdir)
    n_target = 5 if stage == "pilot" else 20
    seeds, valid, failures, nxt = list(range(1, n_target + 1)), {}, {}, n_target + 1
    while True:
        ok, failed = get_events(seeds, workdir)
        valid.update(ok)
        failures.update(failed)
        need = n_target - len(valid)
        if need <= 0:
            break
        seeds = list(range(nxt, nxt + need))
        nxt += need
    used = sorted(valid)[:n_target]
    results = []
    for s in used:
        t0 = time.time()
        results.append(analyse_seed(s, valid[s][0], valid[s][1], calib))
        print(f"seed {s} analysed ({time.time() - t0:.1f}s)", flush=True)
    rows = [g1_row(r) for r in results]
    g1 = g1_decide(rows, calib["diag"]["pass"], n_target)
    print("G1:", json.dumps(g1))
    for r, row in zip(results, rows):
        if not all(row.values()):
            print("  seed", r["seed"], "G1 flags:", {k: v for k, v in row.items() if not v},
                  "acf_logT1=%.3f cv=%.2f" % (r["clock"]["C1"]["acf_logT1"], r["clock"]["C1"]["cv"]))
    out = {"stage": stage, "seeds": used, "failed_seeds": failures, "g1": g1, "g1_rows": rows, "per_seed": results}
    if g1["pass"]:
        out["aggregate"] = aggregate(results)
    json.dump(out, open(os.path.join(workdir, f"{stage}_result.json"), "w"), indent=1)
    if g1["pass"]:
        for name in ("clock_acf_logT1", "cont_vmr", "cont_absacf1", "cont_count_acf1", "cont_zero_bar_rate", "cont_ret_var", "cont_kurt",
                     "compat_vmr", "compat_absacf1"):
            print(name, {k: round(v, 4) for k, v in out["aggregate"]["levels"][name].items()})
            print("   diffs", {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in out["aggregate"]["diffs"][name].items()})
        for k in ("cont_vmr_rel_C1_vs_C3", "compat_vmr_rel_C1_vs_C3"):
            v = out["aggregate"]["diffs"][k]
            print(k, round(v["mean"], 4), [round(x, 4) for x in v["ci95"]])


if __name__ == "__main__":
    stage, workdir = sys.argv[1], sys.argv[2]
    if stage == "calib":
        stage_calib(workdir)
    else:
        run_stage(stage, workdir)
