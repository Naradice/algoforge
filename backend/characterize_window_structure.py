"""
Phase 6b: what does Lorenz gain between lorenz_dt=0.02 (H1, no transfer) and 0.01 (N1, transfers)?

Same attractor, only the per-bar time step differs, and the per-bar Lyapunov exponent was ruled
out as the dial (N2, Delay stride=2, transfers at 0.021/bar). This measures window-scale structure
per generator, model-free, on the exact close series the pipeline sees (base_price + x):

  - ret_ac1: lag-1 autocorrelation of pct returns (per-bar smoothness)
  - turns_per_win: mean number of return sign changes (local extrema) per 60-bar window
  - zero_cross_per_win: mean number of crossings of the series' own mean per 60-bar window
    (for Lorenz: lobe switches; for oscillators: half-cycles)
  - vol_ratio: median vol_20 relative to the DDM segments' median (1.13e-4, measured on the
    mixture datasets)
  - vol_cv: coefficient of variation of vol_20 within the segment (how much the target moves)
  - vol_ac_obs: autocorrelation of vol_20 at lag OBS_LEN (does target level persist a window)

Usage: python characterize_window_structure.py
"""

from __future__ import annotations

import json

import numpy as np

from characterize_lyapunov_dial import GENERATORS as DIAL_GENERATORS
from data.collectors.synthetic_function import _lorenz, _mackey_glass

LENGTH = 60_000
OBS_LEN = 60
VOL_PERIOD = 20
BASE_PRICE = 100.0
DDM_MEDIAN_VOL = 1.13e-4

GENERATORS = dict(DIAL_GENERATORS)
GENERATORS["lorenz_dt0.0175"] = (lambda: _lorenz(LENGTH, dt=0.0175), "?")
GENERATORS["delay_s4"] = (lambda: _mackey_glass(LENGTH, 17, stride=4), "?")
# Outcomes measured since characterize_lyapunov_dial.py was written
GENERATORS["lorenz_dt0.01"] = (GENERATORS["lorenz_dt0.01"][0], "transfers (N1)")
GENERATORS["delay_s2"] = (GENERATORS["delay_s2"][0], "transfers (N2)")


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x[np.isfinite(x)]
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    return float((a * b).mean() / x.var())


def measure(x: np.ndarray) -> dict:
    close = BASE_PRICE + x
    ret = np.diff(close) / close[:-1]
    vol = np.lib.stride_tricks.sliding_window_view(ret, VOL_PERIOD).std(axis=1, ddof=1)

    sign = np.sign(ret)
    turns = (sign[1:] * sign[:-1] < 0).astype(np.float64)
    above = x > x.mean()
    cross = (above[1:] != above[:-1]).astype(np.float64)
    per_win = lambda ev: float(np.convolve(ev, np.ones(OBS_LEN), "valid").mean())

    return {
        "ret_ac1": round(_autocorr(ret, 1), 4),
        "turns_per_win": round(per_win(turns), 2),
        "zero_cross_per_win": round(per_win(cross), 2),
        "vol_ratio": round(float(np.median(vol)) / DDM_MEDIAN_VOL, 2),
        "vol_cv": round(float(vol.std() / vol.mean()), 3),
        "vol_ac_obs": round(_autocorr(vol, OBS_LEN), 3),
    }


def main() -> None:
    rows = []
    print(f"{'name':<17}{'status':<17}{'ret_ac1':>9}{'turns/w':>9}{'xings/w':>9}"
          f"{'vol_ratio':>10}{'vol_cv':>8}{'vol_ac60':>9}")
    for name, (fn, status) in GENERATORS.items():
        r = {"name": name, "status": status, **measure(fn())}
        rows.append(r)
        print(f"{name:<17}{status:<17}{r['ret_ac1']:>9}{r['turns_per_win']:>9}"
              f"{r['zero_cross_per_win']:>9}{r['vol_ratio']:>10}{r['vol_cv']:>8}{r['vol_ac_obs']:>9}",
              flush=True)
    with open("window_structure_characteristics.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
