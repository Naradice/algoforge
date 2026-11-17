"""T0-C/D/E (Study Manager #886 sub-goals): stationarity of the self-modulation clock (eq.3).

C: decompose the log-change of the interval level; when does the negative drift persist / get repaired?
D: which time scale does tau=150 s correspond to?  Fig.3 of the paper was read from the PDF vector drawing
   (event ticks, 500 s excerpt): model-1 mean interval 7.3 s (CV 0.75, ACF1 -0.15, 69 events); model-2
   mean 9.2 s (CV 1.43, ACF1 +0.31, 50 events, max/mean 7.6)  =>  tau / mean(T) ~ 16 events per window.
E: stabilised extensions are SEPARATE models (M2-S*), never called a reproduction of the paper.

Model-1 intervals T' (tdm_fig3_audit.model1_intervals, cached) are rescaled to mean 7.27 s and bootstrapped
(T' are ~white) to build iid F_k = T'^alpha / E[T'^alpha] (alpha=1.25).

Models
  P   (paper eq.3)       T_k = F_k * A_k, A_k = arithmetic mean of T over the last tau seconds (E F = 1)
  P-c (normalisation)    same with F scaled by c (c != 1 is NOT the paper)
  S-OU (M2-S1)           P plus mean reversion in log:  A = exp((1-lam) log(mean) + lam log m0)
  S-AR (M2-S2)           log-domain, n-event window:  log T_k = log F_k + (1-lam) mean_{j<=n} log T_{k-j} + lam log m0,
                         log F normalised to E log F = 0  (a linear AR(n) process: stationary iff lam > 0)

Usage: python research/tdm_fig3_t0c.py <cache_dir> <out_json> [K] [reps]
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPHA = 1.25
PAPER_MEAN_T1 = 7.27     # model-1 mean interval read from Fig.3 (s)


def load_pool(cache, n=20000):
    pool = []
    for mode in ("signed", "abs"):
        for seed in (1, 2, 3):
            pool.append(np.load(os.path.join(cache, f"tp_{mode}_s{seed}_n{n}.npy")))
    Tp = np.concatenate([p / p.mean() * PAPER_MEAN_T1 for p in pool])
    return Tp[Tp > 0]


def make_F(Tp, K, rng, c=1.0):
    x = rng.choice(Tp, size=K) ** ALPHA
    return c * x / np.mean(Tp ** ALPHA)


def simulate(F, tau, unit="sec", init=1.0, lam=0.0, m0=PAPER_MEAN_T1):
    """Arithmetic-window clock. Returns T and the number of events in the window when T_k was drawn."""
    K = len(F)
    T = np.empty(K)
    nwin = np.empty(K, dtype=np.int32)
    win: deque = deque()
    s = 0.0
    now = 0.0
    A = init * m0
    lm0 = np.log(m0)
    for k in range(K):
        Tk = F[k] * A
        T[k] = Tk
        nwin[k] = len(win)
        now += Tk
        win.append((now, Tk))
        s += Tk
        if unit == "sec":
            while win and win[0][0] < now - tau:
                s -= win.popleft()[1]
        else:
            while len(win) > tau:
                s -= win.popleft()[1]
        if len(win) <= 48 or k % 64 == 0:
            s = sum(w[1] for w in win)
        mean = s / len(win) if win else Tk
        if lam > 0 and mean > 0:
            A = float(np.exp((1 - lam) * np.log(mean) + lam * lm0))
        else:
            A = mean
    return T, nwin


def simulate_ar(logF0, n, lam, m0=PAPER_MEAN_T1, init_log=None):
    """S-AR: log T_k = logF0_k + (1-lam)*mean(log T_{k-1..k-n}) + lam*log m0  (logF0 has E=0).
    Exact recursion via lfilter with the initial history log T_{-j} = init_log (default log m0)."""
    phi = (1.0 - lam) / n
    a = np.concatenate([[1.0], -phi * np.ones(n)])
    lm0 = np.log(m0)
    y0 = lm0 if init_log is None else init_log
    zi = np.array([phi * y0 * (n - m) for m in range(n)])   # transposed direct-form-II state for constant history
    y, _ = lfilter([1.0], a, logF0 + lam * lm0, zi=zi)
    return np.exp(y)


def seg_metrics(T, nwin=None):
    if len(T) < 40 or T.max() < 1e-100 or not np.all(np.isfinite(T)) or T.max() > 1e100:
        return {"acf1": float("nan"), "vr20": float("nan"), "mean_T": float("nan"), "geo_T": float("nan"),
                "n_window": float("nan"), "cv": float("nan"), "max_mean": float("nan")}
    x = T - T.mean()
    n = len(T) // 20 * 20
    blocks = T[:n].reshape(-1, 20).mean(1)
    sh = np.random.default_rng(0).permutation(T[:n]).reshape(-1, 20).mean(1)
    return {
        "acf1": float(np.dot(x[:-1], x[1:]) / np.dot(x, x)),
        "vr20": float(blocks.var() / sh.var()),
        "mean_T": float(T.mean()),
        "geo_T": float(np.exp(np.log(T).mean())),
        "n_window": float(np.median(nwin)) if nwin is not None else float("nan"),
        "cv": float(T.std() / T.mean()),
        "max_mean": float(T.max() / T.mean()),
    }


def med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def run_norm_scan(Tp, K, reps):
    """Drift of the level for the paper model as the constant normalisation c varies (tick windows)."""
    out = {}
    for n in (16, 150):
        for c in (0.97, 0.99, 1.0, 1.01, 1.03):
            dl = []
            for r in range(reps):
                F = make_F(Tp, K, np.random.default_rng(300 + r), c)
                T, _ = simulate(F, n, "ticks")
                q = K // 4
                dl.append(float(np.log(max(T[-q:].mean(), 1e-300) / T[:q].mean())))
            out[f"n{n}_c{c}"] = {"dlogLevel_median": med(dl), "n_up": int(sum(d > 0 for d in dl)), "reps": reps}
    return out


def run_fixed_ticks(Tp, K, reps):
    rows = {}
    for n in (2, 3, 5, 8, 12, 16, 25, 40, 80, 150, 300):
        acf, vr, dr = [], [], []
        for r in range(reps):
            F = make_F(Tp, K, np.random.default_rng(100 + r))
            T, nw = simulate(F, n, "ticks")
            m = seg_metrics(T[: K // 10] if T[K // 10:].max() < 1e-100 else T[K // 10: K // 10 + 20000], nw[:1])
            acf.append(m["acf1"]); vr.append(m["vr20"])
            q = K // 4
            dr.append(float(np.log(max(T[-q:].mean(), 1e-300) / T[:q].mean())))
        rows[str(n)] = {"acf1_after_burnin": med(acf), "vr20": med(vr), "dlogLevel_Q1_to_Q4": med(dr),
                        "frac_runs_collapsed(dlog<-20)": float(np.mean(np.array(dr) < -20))}
    return rows


def main():
    cache, out = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    Tp = load_pool(cache)
    logF = np.log(Tp ** ALPHA / np.mean(Tp ** ALPHA))
    res = {"E_logF": float(logF.mean()), "var_logF": float(logF.var()),
           "cv_F": float((Tp ** ALPHA).std() / (Tp ** ALPHA).mean())}
    print("E[log F] = %.4f  Var(log F) = %.4f  CV(F) = %.3f" % (res["E_logF"], res["var_logF"], res["cv_F"]))

    Kf = min(K, 60000)
    res["E1_fixed_ticks_paper"] = run_fixed_ticks(Tp, Kf, max(4, reps // 2))
    print("E1 paper model, fixed n-event window (E F = 1):")
    for n, m in res["E1_fixed_ticks_paper"].items():
        print(f"  n={n:>4} acf1={m['acf1_after_burnin']:.3f} vr20={m['vr20']:.2f} dlogLevel(Q1->Q4)={m['dlogLevel_Q1_to_Q4']:.2f} "
              f"collapsed={m['frac_runs_collapsed(dlog<-20)']:.2f}")
    res["E1b_norm_scan"] = run_norm_scan(Tp, Kf, max(4, reps // 2))
    print("E1b constant normalisation c (dlogLevel, #runs rising):")
    for k, m in res["E1b_norm_scan"].items():
        print(f"  {k:>12} {m['dlogLevel_median']:9.2f}  up {m['n_up']}/{m['reps']}")

    segs = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 12000), (12000, 24000), (24000, 50000), (K - 10000, K)]
    configs = {
        "P_tau150s_init1": dict(tau=150.0, unit="sec", init=1.0),
        "P_tau150s_init0.3": dict(tau=150.0, unit="sec", init=0.3),
        "P_tau150s_init3": dict(tau=150.0, unit="sec", init=3.0),
        "SOU_lam0.002": dict(tau=150.0, unit="sec", init=1.0, lam=0.002),
        "SOU_lam0.01": dict(tau=150.0, unit="sec", init=1.0, lam=0.01),
        "SOU_lam0.01_init3": dict(tau=150.0, unit="sec", init=3.0, lam=0.01),
        "SOU_lam0.05": dict(tau=150.0, unit="sec", init=1.0, lam=0.05),
    }
    res["E2_E3"] = {}
    for name, cfg in configs.items():
        per = []
        for r in range(reps):
            F = make_F(Tp, K, np.random.default_rng(500 + r))
            T, nw = simulate(F, **cfg)
            per.append({f"{a}-{b}": seg_metrics(T[a:b], nw[a:b]) for a, b in segs})
        agg = {}
        for sg in per[0]:
            agg[sg] = {k: med([p[sg][k] for p in per]) for k in per[0][sg]}
            agg[sg]["sd_log_geoT_reps"] = float(np.nanstd([np.log(p[sg]["geo_T"]) for p in per]))
        res["E2_E3"][name] = agg
        print(name)
        for sg, m in agg.items():
            print(f"   {sg:>12} geoT={m['geo_T']:.4g} nwin={m['n_window']:.0f} acf1={m['acf1']:.3f} vr20={m['vr20']:.2f} "
                  f"cv={m['cv']:.2f} max/mean={m['max_mean']:.1f} sd(logGeoT|reps)={m['sd_log_geoT_reps']:.2f}")

    # S-AR: stationary log-domain AR family, E log F = 0
    KA = 400000
    res["E4_SAR"] = {}
    print("E4 S-AR (n, lam): acf1(T) vr20 cv max/mean | sd(log level) quarters | init dependence")
    for n in (16, 50):
        for lam in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
            rows = []
            for r in range(8):
                rng = np.random.default_rng(900 + r)
                lf = rng.choice(logF, size=KA)
                lf = lf - lf.mean()                         # E log F = 0 exactly
                lo = simulate_ar(lf, n, lam, init_log=np.log(PAPER_MEAN_T1) + 2.0)
                hi = simulate_ar(lf, n, lam, init_log=np.log(PAPER_MEAN_T1) - 2.0)
                q = KA // 4
                lg = np.log(lo)
                x = lo[q:] - lo[q:].mean()
                m = seg_metrics(lo[q: q + 100000])
                m["quarter_mean_logT"] = [float(lg[i * q:(i + 1) * q].mean()) for i in range(4)]
                m["var_logT_late"] = float(lg[q:].var())
                m["init_gap_end"] = float(np.log(lo[-q:].mean()) - np.log(hi[-q:].mean()))
                m["acf_logT"] = {str(l): float(np.corrcoef(lg[q:-l], lg[q + l:])[0, 1]) for l in (1, 10, 100)}
                rows.append(m)
            agg = {k: med([r_[k] for r_ in rows]) for k in ("acf1", "vr20", "cv", "max_mean", "var_logT_late", "init_gap_end")}
            agg["quarter_drift_logT"] = med([r_["quarter_mean_logT"][3] - r_["quarter_mean_logT"][0] for r_ in rows])
            agg["acf_logT"] = {l: med([r_["acf_logT"][l] for r_ in rows]) for l in ("1", "10", "100")}
            res["E4_SAR"][f"n{n}_lam{lam}"] = agg
            print(f"  n={n:>3} lam={lam:<7} acf1={agg['acf1']:.3f} vr20={agg['vr20']:.2f} cv={agg['cv']:.2f} max/mean={agg['max_mean']:.1f} | "
                  f"var(logT)={agg['var_logT_late']:.2f} dQ4-Q1={agg['quarter_drift_logT']:.2f} | init gap end={agg['init_gap_end']:.2f} "
                  f"| ACF(logT) 1/10/100 = {agg['acf_logT']['1']:.2f}/{agg['acf_logT']['10']:.2f}/{agg['acf_logT']['100']:.2f}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
