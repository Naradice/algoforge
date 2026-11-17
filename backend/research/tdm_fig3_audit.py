"""T0-A (Study Manager #894): Yamada-Takayasu-Takayasu threshold-dealer-model Fig.3 audit.

Separate from data/collectors/ddm_simulator.py (which is NOT modified). Implements paper
Model-1 (eq.1-2) and Model-2 self-modulation (eq.3) literally, and runs the current DDMv1
code as-is for comparison, then reports interval statistics for:
  C0 = Model-1 internal time t'            (intervals T'_k)
  C1 = Model-2 internal dynamics           (must be identical to C0: eq.3 only rescales time)
  C2 = Model-2 transformed time t          (intervals T_k = F_k * <T>_tau)
  CODE = current DDMv1 (tick_time with iid time noise), N=100 and N=500

Usage: python research/tdm_fig3_audit.py [n_trades] [out_json]
"""
from __future__ import annotations

import json
import random
import sys
from collections import deque

import numpy as np

ALPHA = 1.25
BETA = ALPHA - 1.0


def model1_intervals(n_trades, N, L, dt, c_max, c_mode, seed, trade_unit=0.001):
    """Paper eq.(1)-(2). Returns (prices, T' intervals in t' time, steps-per-interval)."""
    rng = np.random.default_rng(seed)
    if c_mode == "signed":      # literal reading of "ci=[-c,c]"
        c = rng.uniform(-c_max, c_max, N)
    else:                        # |c| reading (what the repo code does, [min,max] >= 0)
        c = rng.uniform(0.0, c_max, N)
    pos = np.array([1] * (N // 2) + [-1] * (N - N // 2))
    rng.shuffle(pos)
    b = rng.uniform(100.0, 100.0 + L, N)
    prices, T, steps_hist = [], [], []
    t = 0.0
    last_t = 0.0
    steps = 0
    guard = 0
    while len(prices) < n_trades:
        buy = pos == 1
        sell = ~buy
        bmax = b[buy].max() if buy.any() else None
        amin = (b[sell].min() + L) if sell.any() else None
        if bmax is not None and amin is not None and bmax - amin >= 0:      # eq.(2)
            ib = np.where(buy)[0][np.argmax(b[buy])]
            js = np.where(sell)[0][np.argmin(b[sell])]
            prices.append(np.floor((bmax + amin) / 2 / trade_unit) * trade_unit)
            pos[ib], pos[js] = -1, 1
            T.append(t - last_t)
            steps_hist.append(steps)
            last_t, steps = t, 0
            guard = 0
        else:
            b += pos * c * dt                                                # eq.(1)
            t += dt
            steps += 1
            guard += 1
            if guard > 2_000_000:
                raise RuntimeError("stalled")
    return np.array(prices), np.array(T), np.array(steps_hist)


def self_modulate(Tp, tau, alpha=ALPHA):
    """Paper eq.(3): T_k = G_k <T>_tau T'_k with G_k = T'^beta_k / Tbar, Tbar = mean(T'^alpha)
    => T_k = F_k <T>_tau, F_k = T'^alpha_k / mean(T'^alpha). <T>_tau = mean of T over the last
    tau seconds of modulated time (falls back to the last T if the window is empty)."""
    Tbar = np.mean(Tp ** alpha)
    F = Tp ** alpha / Tbar
    out = np.empty_like(F)
    win: deque[tuple[float, float]] = deque()   # (end_time, T)
    now = 0.0
    avg = float(np.mean(Tp))                     # initial <T>: unspecified in paper -> mean T'
    for k, f in enumerate(F):
        Tk = f * avg
        now += Tk
        win.append((now, Tk))
        while win and win[0][0] < now - tau:
            win.popleft()
        avg = float(np.mean([w[1] for w in win])) if win else Tk
        out[k] = Tk
    return out


def run_length(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def acf(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def stats(T):
    T = np.asarray(T, float)
    T = T[T > 0]
    med = float(np.median(T))
    q = np.quantile(T, [0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
    lg = np.log(T)
    return {
        "n": int(len(T)), "mean": float(T.mean()), "median": med, "max": float(T.max()),
        "max_over_median": float(T.max() / med), "q": [float(v) for v in q],
        "cv": float(T.std() / T.mean()),
        "frac_gt_5x_mean": float(np.mean(T > 5 * T.mean())),
        "frac_gt_10x_mean": float(np.mean(T > 10 * T.mean())),
        "longest_short_run(<0.2*med)": run_length(T < 0.2 * med),
        "longest_long_run(>3*med)": run_length(T > 3 * med),
        "acf_T": {str(l): acf(T, l) for l in (1, 2, 5, 10, 50)},
        "acf_logT": {str(l): acf(lg, l) for l in (1, 2, 5, 10, 50)},
        "trend_first_vs_last_quarter_mean": [float(T[: len(T) // 4].mean()), float(T[-len(T) // 4:].mean())],
    }


def code_ddmv1(n_trades, N, seed):
    sys.path.insert(0, ".")
    from data.collectors.ddm_simulator import DDMv1
    random.seed(seed)
    np.random.seed(seed)
    m = DDMv1(num_agent=N, max_volatility=0.02, min_volatility=0.01, trade_unit=0.001,
              initial_price=100.0, spread=1.0, tick_time=0.001, time_noise_method="exp",
              max_noise_factor=100)
    ts = [tick for _, tick in m.simulate_stream(n_trades=n_trades)]
    return np.diff(np.array(ts))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    out = sys.argv[2] if len(sys.argv) > 2 else None
    res = {}
    # Fig.3 conditions: N=500, L=1.0, dt=0.01, c=[-0.01,0.01], tau=150
    for mode in ("signed", "abs"):
        for seed in (1, 2, 3):
            prices, Tp, _ = model1_intervals(n, 500, 1.0, 0.01, 0.01, mode, seed)
            T2 = self_modulate(Tp, tau=150.0)
            res[f"fig3_{mode}_seed{seed}"] = {
                "C0_model1_t'": stats(Tp),
                "C1_model2_internal_identical_to_C0": True,   # eq.(3) rescales time only
                "C2_model2_t": stats(T2),
                "price_changes_abs_mean": float(np.mean(np.abs(np.diff(prices)))),
            }
    for N in (100, 500):
        res[f"CODE_DDMv1_N{N}"] = stats(code_ddmv1(n, N, 1))
    s = json.dumps(res, indent=1)
    if out:
        open(out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
