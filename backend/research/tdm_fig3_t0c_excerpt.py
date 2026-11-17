"""T0-D check: do paper-faithful (P) and stabilised (S-OU, S-AR) clocks produce a 49-interval excerpt like the
paper's Fig.3 model-2 excerpt (mean 9.2 s, CV 1.43, ACF1 +0.31, max/mean 7.6; read from the PDF drawing)?

Reports, per model, the median and 10-90% range of excerpt statistics and the percentile rank of the paper's
observed values (single 50-event excerpt, so this is a plausibility check, not a fit).

Usage: python research/tdm_fig3_t0c_excerpt.py <cache_dir> [out_json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdm_fig3_t0c import ALPHA, PAPER_MEAN_T1, load_pool, make_F, simulate, simulate_ar  # noqa: E402

OBS = {"cv": 1.43, "acf1": 0.31, "max_mean": 7.6, "mean": 9.18}


def stats49(x):
    x = np.asarray(x)
    d = x - x.mean()
    return {"cv": x.std() / x.mean(), "acf1": float(np.dot(d[:-1], d[1:]) / np.dot(d, d)),
            "max_mean": x.max() / x.mean(), "mean": x.mean()}


def excerpts(T, rng, n_ex=3000, L=49):
    starts = rng.integers(0, len(T) - L, size=n_ex)
    rows = [stats49(T[s:s + L]) for s in starts]
    return {k: np.array([r[k] for r in rows]) for k in OBS}


def summarize(name, ex):
    out = {"name": name}
    for k in OBS:
        v = ex[k]
        out[k] = {"p10": float(np.percentile(v, 10)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                  "obs_percentile": float(np.mean(v <= OBS[k]) * 100)}
    # joint plausibility: fraction of excerpts at least as clustered (acf1>=obs) AND cv>=obs*0.8 AND cv<=obs*1.25
    out["joint_acf_cv"] = float(np.mean((ex["acf1"] >= OBS["acf1"] - 0.15) & (ex["acf1"] <= OBS["acf1"] + 0.15) &
                                       (ex["cv"] >= OBS["cv"] * 0.8) & (ex["cv"] <= OBS["cv"] * 1.25)))
    return out


def main():
    cache = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    Tp = load_pool(cache)
    rng = np.random.default_rng(7)
    logF = np.log(Tp ** ALPHA / np.mean(Tp ** ALPHA))
    rows = []
    # paper model-1 reference excerpt-like stats
    rows.append(summarize("Model-1 clock (T')", excerpts(Tp, rng)))
    # the paper's model-2 excerpt starts at t=30 s with long intervals (47, 42, 70 s): test run STARTS (burn-in)
    for init in (1.0, 2.0, 3.0, 5.0):
        rows_b = []
        for r in range(2000):
            Fb = make_F(Tp, 60, np.random.default_rng(10_000 + r))
            Tb, _ = simulate(Fb, 150.0, "sec", init)
            rows_b.append(stats49(Tb[3:52]))          # skip the first 3 events (paper's first event is at 30 s)
        rows.append(summarize(f"P  tau=150s run START (events 4-52), init {init:g}x",
                              {k: np.array([r_[k] for r_ in rows_b]) for k in OBS}))
    K = 60000
    F = make_F(Tp, K, np.random.default_rng(1))
    T, _ = simulate(F, 150.0, "sec", 1.0)
    rows.append(summarize("P  tau=150s, events 0-2000 (burn-in, init 1x)", excerpts(T[:2000], rng)))
    rows.append(summarize("P  tau=150s, events 20000-60000 (after level collapse)", excerpts(T[20000:], rng)))
    for lam in (0.03, 0.1):
        T, _ = simulate(F, 150.0, "sec", 1.0, lam=lam)
        rows.append(summarize(f"S-OU lam={lam} (arith window 150 s + log mean reversion)", excerpts(T[10000:], rng)))
    for n, lam in ((16, 0.03), (16, 0.1), (16, 0.3), (50, 0.03), (50, 0.1)):
        lf = np.random.default_rng(3).choice(logF, size=200000)
        lf = lf - lf.mean()
        T = simulate_ar(lf, n, lam)[20000:]
        T = T / T.mean() * 9.18   # excerpt statistics other than 'mean' are scale free
        rows.append(summarize(f"S-AR n={n} lam={lam}", excerpts(T, rng)))
    print(f"paper observation: {OBS}")
    for r in rows:
        print(f"{r['name']:<62} cv {r['cv']['median']:.2f} [{r['cv']['p10']:.2f},{r['cv']['p90']:.2f}] pct {r['cv']['obs_percentile']:.0f} | "
              f"acf1 {r['acf1']['median']:.2f} [{r['acf1']['p10']:.2f},{r['acf1']['p90']:.2f}] pct {r['acf1']['obs_percentile']:.0f} | "
              f"max/mean {r['max_mean']['median']:.1f} pct {r['max_mean']['obs_percentile']:.0f} | joint {r['joint_acf_cv']:.3f}")
    if out:
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
