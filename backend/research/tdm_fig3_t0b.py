"""T0-B (Study Manager #903): which reading of Yamada et al. eq.(3) yields interval clustering?

Model-1 interval series T'_k (paper eq.1-2, via tdm_fig3_audit.model1_intervals) are generated once per
(c_mode, seed) and cached; each variant of the self-modulation clock is then applied to the SAME T'.
One factor is changed at a time from the baseline B0 (the T0-A literal reading):
  B0: tau=150 seconds of modulated time, <T> averages past modulated T, initial <T>=mean(T'),
      empty window -> last T, alpha=1.25.

Clustering metrics (per variant):
  vr20      variance of 20-event block means / (same after shuffling T)  (>1 => clustering)
  acf1/acf10 ACF of T at lag 1 / 10
  drift     log(mean T last quarter / first quarter)
  max_mean  max(T)/mean(T);  p5  fraction of T > 5*mean(T)
  level     mean(T)/mean(T')  (scale relative to model-1 clock)

Usage: python research/tdm_fig3_t0b.py [n_trades] [out_json] [cache_dir]
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdm_fig3_audit import model1_intervals  # noqa: E402


def modulate(Tp, tau=150.0, tau_unit="sec", init=1.0, empty="last", avg_over="T", alpha=1.25):
    """init is a multiple of mean(T'). tau_unit 'sec' = modulated seconds, 'ticks' = last tau events."""
    mean_tp = float(np.mean(Tp))
    F = Tp ** alpha / np.mean(Tp ** alpha)
    out = np.empty_like(F)
    win: deque = deque()   # (end_time, value)
    s = 0.0
    now = 0.0
    avg = init * mean_tp
    for k, f in enumerate(F):
        Tk = f * avg
        now += Tk
        v = Tk if avg_over == "T" else Tp[k]
        win.append((now, v))
        s += v
        if tau_unit == "sec":
            while win and win[0][0] < now - tau:
                s -= win.popleft()[1]
        else:
            while len(win) > tau:
                s -= win.popleft()[1]
        # running sums lose all precision once T collapses by many orders of magnitude (observed:
        # negative T); re-sum exactly for short windows and resync periodically for long ones.
        if len(win) <= 64 or k % 64 == 0:
            s = sum(w[1] for w in win)
        if win:
            avg = s / len(win)
        else:
            avg = Tk if empty == "last" else mean_tp
        out[k] = Tk
    return out


def metrics(T, Tp, rng):
    T = np.asarray(T, float)
    n = len(T) // 20 * 20
    blocks = T[:n].reshape(-1, 20).mean(1)
    sh = rng.permutation(T[:n]).reshape(-1, 20).mean(1)
    x = T - T.mean()

    def acf(l):
        return float(np.dot(x[:-l], x[l:]) / np.dot(x, x))

    q = len(T) // 4
    return {
        "vr20": float(blocks.var() / sh.var()),
        "acf1": acf(1), "acf10": acf(10),
        "drift": float(np.log(T[-q:].mean() / T[:q].mean())),
        "max_mean": float(T.max() / T.mean()),
        "p5": float(np.mean(T > 5 * T.mean())),
        "cv": float(T.std() / T.mean()),
        "level": float(T.mean() / Tp.mean()),
        "finite": bool(np.all(np.isfinite(T)) and T.min() >= 0),
        "frac_collapsed": float(np.mean(T < 1e-3 * Tp.mean())),
    }


VARIANTS = {
    "B0_literal": {},
    "tau_5s": {"tau": 5.0},
    "tau_15s": {"tau": 15.0},
    "tau_50s": {"tau": 50.0},
    "tau_500s": {"tau": 500.0},
    "tau_20ticks": {"tau": 20, "tau_unit": "ticks"},
    "tau_150ticks": {"tau": 150, "tau_unit": "ticks"},
    "init_0.1x": {"init": 0.1},
    "init_10x": {"init": 10.0},
    "empty_meanTp": {"empty": "meanTp"},
    "avg_over_Tprime": {"avg_over": "Tp"},
    "alpha_1.0": {"alpha": 1.0},
    "alpha_1.5": {"alpha": 1.5},
    "tau_5s_empty_meanTp": {"tau": 5.0, "empty": "meanTp"},
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    out = sys.argv[2] if len(sys.argv) > 2 else None
    cache = sys.argv[3] if len(sys.argv) > 3 else "."
    res = {}
    for mode in ("signed", "abs"):
        for seed in (1, 2, 3):
            f = os.path.join(cache, f"tp_{mode}_s{seed}_n{n}.npy")
            if os.path.exists(f):
                Tp = np.load(f)
            else:
                _, Tp, _ = model1_intervals(n, 500, 1.0, 0.01, 0.01, mode, seed)
                np.save(f, Tp)
            rng = np.random.default_rng(0)
            res[f"{mode}_s{seed}"] = {"model1_ref": metrics(Tp, Tp, rng)}
            for name, kw in VARIANTS.items():
                res[f"{mode}_s{seed}"][name] = metrics(modulate(Tp, **kw), Tp, rng)
    s = json.dumps(res, indent=1)
    if out:
        open(out, "w").write(s)
    # compact table: median over the 6 series
    names = ["model1_ref", *VARIANTS]
    keys = ["vr20", "acf1", "acf10", "drift", "max_mean", "p5", "cv", "level", "frac_collapsed"]
    print("variant".ljust(22), *[k.rjust(8) for k in keys], "  nonfinite")
    for nm in names:
        vals = {k: [res[r][nm][k] for r in res] for k in keys}
        nf = sum(not res[r][nm]["finite"] for r in res)
        print(nm.ljust(22), *[f"{np.median(vals[k]):8.3f}" for k in keys], f"  {nf}/6")
    print("range of level across series (min..max):")
    for nm in names:
        lv = [res[r][nm]["level"] for r in res]
        print(" ", nm.ljust(22), f"{min(lv):.3g} .. {max(lv):.3g}")


if __name__ == "__main__":
    main()
