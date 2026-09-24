"""
Lyapunov-dial characterization (pre-training check for conditions N1-N*).

The periodic-forcing handoff concluded that DDM+X transfers to USDJPY vol_20 iff X's future
volatility is a low-noise function of its recent window, and suggested the per-bar Lyapunov
exponent as the dial: Delay (Mackey-Glass, ~0.009/bar) transfers, Lorenz (dt=0.02, ~0.017/bar)
does not. But Delay and Lorenz differ in far more than their exponent (attractor geometry,
dimension, lobe switching), so that comparison alone can't isolate it.

Time-rescaling can: Lorenz with a smaller RK4 step (lorenz_dt) and Mackey-Glass sampled every
`stride` steps keep each attractor unchanged while moving the per-bar exponent. If the per-bar
exponent is what matters, Lorenz at dt=0.01 should look like Delay here (and transfer), and
Delay at stride=2 should look like Lorenz (and not).

Measured per generator, model-free:
  - lyapunov_per_step: Rosenstein largest exponent (characterize_synthetic_generators.py)
  - knn_r2_next: kNN R^2 of the pipeline's actual target (vol_20 at the bar after a 60-bar close
    window) -- mostly overlaps the window, so expected high for most generators
  - knn_r2_ahead: kNN R^2 of vol_20 ending AHEAD bars after the window (no overlap with it) --
    the "is genuinely-future volatility predictable from the window" quantity the handoff's
    mechanism is about
Train/test split is chronological within each series; kNN on the raw close window (the
pipeline's zscore is dataset-level, so it only rescales distances uniformly).

Usage: python characterize_lyapunov_dial.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from characterize_synthetic_generators import lyapunov_rosenstein
from data.collectors.synthetic_function import _ar1, _lfsr, _lorenz, _mackey_glass

LENGTH = 60_000   # per-component rows in every DDM+X mixture
OBS_LEN = 60      # BASE_HP obs_len
VOL_PERIOD = 20   # BASE_HP target vol_{PERIOD}
AHEAD = 20        # non-overlapping horizon for knn_r2_ahead
BASE_PRICE = 100.0

N_TRAIN, N_TEST = 20_000, 5_000

GENERATORS = {
    # name -> (fn, known outcome or "?" for untested)
    "sine_p15": (lambda: np.sin(2 * np.pi * np.arange(LENGTH) / 15.0), "transfers (M1)"),
    "sine_p50": (lambda: np.sin(2 * np.pi * np.arange(LENGTH) / 50.0), "transfers (D1)"),
    "sine_p200": (lambda: np.sin(2 * np.pi * np.arange(LENGTH) / 200.0), "transfers (M2)"),
    "delay_s1": (lambda: _mackey_glass(LENGTH, 17), "transfers (D2)"),
    "delay_s2": (lambda: _mackey_glass(LENGTH, 17, stride=2), "?"),
    "delay_s3": (lambda: _mackey_glass(LENGTH, 17, stride=3), "?"),
    "lorenz_dt0.005": (lambda: _lorenz(LENGTH, dt=0.005), "?"),
    "lorenz_dt0.0075": (lambda: _lorenz(LENGTH, dt=0.0075), "?"),
    "lorenz_dt0.01": (lambda: _lorenz(LENGTH, dt=0.01), "?"),
    "lorenz_dt0.0125": (lambda: _lorenz(LENGTH, dt=0.0125), "?"),
    "lorenz_dt0.015": (lambda: _lorenz(LENGTH, dt=0.015), "?"),
    "lorenz_dt0.02": (lambda: _lorenz(LENGTH, dt=0.02), "no effect (H1)"),
    "ar1": (lambda: _ar1(LENGTH, phi=0.98, sigma=1.0, seed=2005), "no effect (G1)"),
    "lfsr": (lambda: 2 * _lfsr(LENGTH, bits=8, seed=2004) - 1, "no effect (D4)"),
}


def _windows_and_targets(x: np.ndarray, ahead: int) -> tuple[np.ndarray, np.ndarray]:
    close = BASE_PRICE + x
    ret = np.diff(close) / close[:-1]
    # vol[t] = std of the VOL_PERIOD returns ending at close index t (pandas rolling std, ddof=1)
    rs = np.lib.stride_tricks.sliding_window_view(ret, VOL_PERIOD).std(axis=1, ddof=1)
    vol = np.full(len(close), np.nan)
    vol[VOL_PERIOD:] = rs
    starts = np.arange(VOL_PERIOD, len(close) - OBS_LEN - ahead)
    win = np.lib.stride_tricks.sliding_window_view(close, OBS_LEN)[starts]
    tgt = vol[starts + OBS_LEN - 1 + ahead]
    return win, tgt


def knn_r2(x: np.ndarray, ahead: int, k: int = 10) -> float:
    win, tgt = _windows_and_targets(x, ahead)
    n = len(tgt)
    split = int(n * 0.75)
    rng = np.random.default_rng(0)
    tr = rng.choice(split - ahead - OBS_LEN, size=min(N_TRAIN, split - ahead - OBS_LEN), replace=False)
    te = split + rng.choice(n - split, size=min(N_TEST, n - split), replace=False)
    mu, sd = win[tr].mean(), win[tr].std()
    model = KNeighborsRegressor(n_neighbors=k).fit((win[tr] - mu) / sd, tgt[tr])
    pred = model.predict((win[te] - mu) / sd)
    resid = tgt[te] - pred
    return float(1 - resid.var() / tgt[te].var()) if tgt[te].var() > 0 else float("nan")


def main() -> None:
    rows = []
    for name, (fn, status) in GENERATORS.items():
        x = fn()
        lyap = lyapunov_rosenstein(x)
        row = {
            "name": name, "status": status,
            "lyapunov_per_step": round(lyap, 5) if lyap is not None else None,
            "knn_r2_next": round(knn_r2(x, ahead=1), 4),
            "knn_r2_ahead": round(knn_r2(x, ahead=AHEAD), 4),
        }
        rows.append(row)
        print(f"{name:<17}{status:<17}lyap={str(row['lyapunov_per_step']):>9}  "
              f"r2_next={row['knn_r2_next']:>7}  r2_ahead{AHEAD}={row['knn_r2_ahead']:>7}", flush=True)

    with open("lyapunov_dial_characteristics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved lyapunov_dial_characteristics.json")


if __name__ == "__main__":
    main()
