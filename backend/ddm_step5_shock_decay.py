"""Step 5: does an exponentially-decaying shock rate fix Step 4's h=24 over-dispersion while
keeping h=1's calibration win?

This is a CHEAP, DDM-only characterization -- no neural model, no real USDJPY data loading.
Step 4's full pipeline (real-vs-DDM coverage/tail-probability) is expensive (~100 min per
setting) specifically because of the real-data/model side; the problem being diagnosed here
(DDM's OWN dispersion growing too fast with horizon) is an internal DDM property that can be
checked directly against the ALREADY-KNOWN real-data growth rate from Step 4's full run:

    real cum-return std:  h=1=0.001087  h=4=0.002113  h=12=0.003501  h=24=0.004832
    real growth ratio from h=1:  h=4/h=1=1.94  h=12/h=1=3.22  h=24/h=1=4.44  (~ sqrt(24)=4.90)

Step 4's constant-rate shock blew this up to 18.1x (p=0.002/size=0.3) and 8.95x
(p=0.0005/size=0.8) growth from h=1 to h=24. Target: get DDM's own h=24/h=1 growth ratio back
near ~4.4-4.9x while keeping the ABSOLUTE h=1 dispersion at roughly the same level Step 4 already
validated works for short-horizon calibration (i.e. don't also suppress the h=1 win in the
process of fixing h=24).

Only once a decay setting looks promising here is it worth re-running the full, expensive
real-vs-DDM pipeline (ddm_horizon_analysis.py + ddm_calibration_check.py) to confirm.
"""
from __future__ import annotations

import random

import numpy as np

from calibrate_ddm import TRADES_PER_CANDLE
from data.collectors.ddm_simulator import DDMv3

CANDIDATES = [(-0.8, 7), (-0.5, 10), (-0.1, 15), (0.0, 20)]
HORIZONS_WINDOWS = [1, 4, 12, 24]
MAX_HORIZON = max(HORIZONS_WINDOWS)
CANDLES_PER_WINDOW = 60  # OBS_LEN, matches ddm_ensemble_selection.py
# NOTE: decay_tau is denominated in RAW STEP count (self._step_count, incremented every
# _common_step() call including no-trade steps), not trade count. Empirically ~1.45 raw steps
# occur per trade for this candidate family (measured directly, match rate ~69%), so
# "1 window" (=CANDLES_PER_WINDOW*TRADES_PER_CANDLE=1200 trades) is roughly 1740 steps -- the
# DECAY_TAUS grid below (300-3000) is chosen to span sub-window to ~1.7-window decay scales using
# this approximate conversion, not an exact one.
N_SIMS_PER_CANDIDATE = 10
N_AGENT = 300

# Real-data reference (from Step 4's full pipeline, ddm_horizon_analysis.py phase2 baseline run)
REAL_CUM_STD = {1: 0.001087, 4: 0.002113, 12: 0.003501, 24: 0.004832}
REAL_GROWTH_FROM_H1 = {h: REAL_CUM_STD[h] / REAL_CUM_STD[1] for h in HORIZONS_WINDOWS}

# Step 4's own two tested initial rates, reused rather than re-searched from scratch.
INITIAL_PROBABILITIES = {"lowmed": 0.002, "verylowlarge": 0.0005}
SHOCK_SIZES = {"lowmed": 0.3, "verylowlarge": 0.8}
FLOOR_PROBABILITIES = [0.0, 0.0001, 0.0005]
DECAY_TAUS = [300, 600, 1200, 3000]  # in steps; 1 window = 1200 steps


def simulate_long_path(delta: float, wma: int, seed: int, shock_prob: float, shock_size: float,
                        decay_tau: float | None, floor_prob: float) -> np.ndarray | None:
    n_trades = MAX_HORIZON * CANDLES_PER_WINDOW * TRADES_PER_CANDLE
    random.seed(seed); np.random.seed(seed)
    model = DDMv3(
        num_agent=N_AGENT, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
        exogenous_shock_probability=shock_prob, exogenous_shock_size=shock_size,
        exogenous_shock_decay_tau=decay_tau, exogenous_shock_floor_probability=floor_prob,
    )
    try:
        prices = model.simulate(n_trades=n_trades)["price"].values
    except (ValueError, RuntimeError):
        return None
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return None
    n = (len(prices) // TRADES_PER_CANDLE) * TRADES_PER_CANDLE
    grouped = prices[:n].reshape(-1, TRADES_PER_CANDLE)
    closes = grouped[:, -1]
    if len(closes) < MAX_HORIZON * CANDLES_PER_WINDOW:
        return None
    return closes[: MAX_HORIZON * CANDLES_PER_WINDOW]


def cum_returns_by_horizon(closes: np.ndarray) -> dict[int, float]:
    out = {}
    for h in HORIZONS_WINDOWS:
        n_candles = h * CANDLES_PER_WINDOW
        out[h] = float(np.log(closes[n_candles - 1] / closes[0]))
    return out


def run_setting(shock_prob: float, shock_size: float, decay_tau: float | None, floor_prob: float) -> dict:
    by_h = {h: [] for h in HORIZONS_WINDOWS}
    for delta, wma in CANDIDATES:
        for s in range(N_SIMS_PER_CANDIDATE):
            seed = 90000 + hash((delta, wma, decay_tau, floor_prob, s)) % 10000
            closes = simulate_long_path(delta, wma, seed, shock_prob, shock_size, decay_tau, floor_prob)
            if closes is None:
                continue
            crs = cum_returns_by_horizon(closes)
            for h in HORIZONS_WINDOWS:
                by_h[h].append(crs[h])
    stds = {h: float(np.std(by_h[h])) if by_h[h] else float("nan") for h in HORIZONS_WINDOWS}
    growth = {h: stds[h] / stds[1] if stds[1] else float("nan") for h in HORIZONS_WINDOWS}
    n_ok = len(by_h[1])
    return {"stds": stds, "growth": growth, "n_ok": n_ok}


def main():
    print(f"Target growth from h=1 (real): {REAL_GROWTH_FROM_H1}\n")
    results = []

    for label, initial_prob in INITIAL_PROBABILITIES.items():
        shock_size = SHOCK_SIZES[label]
        print(f"{'='*100}\n{label}: initial_prob={initial_prob}, size={shock_size}\n{'='*100}")

        # No-decay reference (Step 4's own result, recomputed here for an apples-to-apples check
        # against this script's own N_SIMS/candidate set before judging the decay variants).
        base = run_setting(initial_prob, shock_size, None, 0.0)
        print(f"  no-decay (n_ok={base['n_ok']}): std={base['stds']} growth={base['growth']}")
        results.append({"label": label, "decay_tau": None, "floor": 0.0, **base})

        for tau in DECAY_TAUS:
            for floor in FLOOR_PROBABILITIES:
                r = run_setting(initial_prob, shock_size, tau, floor)
                growth_err = abs(r["growth"][24] - REAL_GROWTH_FROM_H1[24]) if not np.isnan(r["growth"][24]) else float("nan")
                print(f"  tau={tau:>5} floor={floor:.4f} (n_ok={r['n_ok']:>3}): "
                      f"std_h1={r['stds'][1]:.6f} growth={{" +
                      ", ".join(f"{h}:{r['growth'][h]:.2f}" for h in HORIZONS_WINDOWS) +
                      f"}}  |growth_h24-target|={growth_err:.2f}")
                results.append({"label": label, "decay_tau": tau, "floor": floor, **r})

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    results = main()
    out_path = Path(__file__).resolve().parent / "results" / "ddm_step5_shock_decay.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
