"""4 metrics x 4 horizons, real vs. DDM ensemble, same out-of-sample quantile assignment.

CRITICAL FIX vs. the previous ddm_ensemble_backtest.py: that script's DDM-side horizon was in
raw DDM "candles" (1 candle = TRADES_PER_CANDLE=20 trades), while the real-side horizon was in
60-real-minute "windows" (matching OBS_LEN). Both were called "24," but 24 DDM candles is only
~24 real-minutes-equivalent (per the 1-candle~=1-real-minute convention this whole investigation
has used since generate_regime_datasets.py), NOT 24*60=1440 minutes (1 day) like the real side.
That is a 60x horizon mismatch -- entirely capable of explaining the previously observed 5-15x
real/DDM dispersion gap on its own, before concluding anything about DDM's actual path dynamics.
Fixed here: DDM horizon in "windows" (60 candles each), matching the real side exactly.

For each candidate, ONE long simulation (covering the longest horizon) is run per seed, and
shorter horizons are read off as PREFIXES of that same path -- avoids resimulating from scratch
per horizon.

Metrics (both sides), per horizon h in WINDOWS:
  A. windowed future volatility AT t+h  (std of returns within just the single window at t+h --
     matches delta_forward_validation.py's original, already-validated statistic)
  B. cumulative return std, t -> t+h    (dispersion of the whole path's endpoint return)
  C. return acf_1 over the t->t+h path
  D. volatility clustering (acf of |returns|, lag lower of 50 or path_length//2-1) over the path
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibrate_ddm import TRADES_PER_CANDLE
from data.collectors.ddm_simulator import DDMv3
from ddm_ensemble_selection import (
    CALIB_FRAC, CHAR_WINDOW, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT,
    predict_all_windows,
)
from ddm_ensemble_backtest import DISTRIBUTIONS_CACHE  # noqa: F401 (kept for path reference only)
from real_data_transfer_test import load_models
import random

HORIZONS_WINDOWS = [1, 4, 12, 24]  # in 60-real-minute windows
MAX_HORIZON = max(HORIZONS_WINDOWS)
N_SIMS_PER_CANDIDATE = 20  # reduced from 100: each sim now covers a MUCH longer path (24x60 candles)
CANDLES_PER_WINDOW = OBS_LEN  # 1 "window" = 60 candles, by the established 1-candle~=1-real-minute convention

# Spread-feedback experiment knobs (both None/default reproduce the feedback-free baseline
# exactly -- see ddm_baseline_snapshot.json). Set via CLI in __main__, not hardcoded, so the
# baseline cache file is never silently overwritten by an experimental run.
SPREAD_FEEDBACK_A: float | None = None
SPREAD_FEEDBACK_WINDOW: int = 150

# Step 4 exogenous-shock knobs (both 0.0/default reproduce the shock-free baseline exactly).
# Independent of the spread-feedback knobs above -- Step 4 isolates the shock mechanism alone,
# same "one mechanism at a time" discipline as Steps 2/3.
SHOCK_PROBABILITY: float = 0.0
SHOCK_SIZE: float = 0.0


def simulate_long_path(delta: float, wma: int, seed: int, num_agent: int = 300) -> np.ndarray | None:
    """One simulation covering MAX_HORIZON windows worth of candles. Returns the close-price
    array (length MAX_HORIZON * CANDLES_PER_WINDOW), or None if the simulation diverged."""
    n_trades = MAX_HORIZON * CANDLES_PER_WINDOW * TRADES_PER_CANDLE
    random.seed(seed); np.random.seed(seed)
    model = DDMv3(
        num_agent=num_agent, max_volatility=0.02, min_volatility=0.01, wma=wma,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
        spread_feedback_a=SPREAD_FEEDBACK_A, spread_feedback_window=SPREAD_FEEDBACK_WINDOW,
        exogenous_shock_probability=SHOCK_PROBABILITY, exogenous_shock_size=SHOCK_SIZE,
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


def path_metrics(closes: np.ndarray, horizon_windows: int) -> dict:
    """Metrics for the sub-path covering the first `horizon_windows` windows of `closes`."""
    n_candles = horizon_windows * CANDLES_PER_WINDOW
    sub = closes[:n_candles]
    log_r = np.diff(np.log(sub))
    cum_return = float(np.log(sub[-1] / sub[0]))
    # metric A: volatility of just the LAST window (the single window AT t+horizon)
    last_window_r = np.diff(np.log(sub[-CANDLES_PER_WINDOW:]))
    windowed_vol = float(last_window_r.std()) if len(last_window_r) > 1 else np.nan
    acf1 = np.nan
    if len(log_r) > 4:
        from statsmodels.tsa.stattools import acf as _acf
        vals = _acf(log_r, nlags=1, fft=True)
        acf1 = float(vals[1]) if len(vals) > 1 else np.nan
    return {"windowed_vol": windowed_vol, "cum_return": cum_return, "acf_1": acf1}


def build_ddm_horizon_stats(step3_results: list[dict]) -> dict:
    """{q: {horizon: {"windowed_vol": array, "cum_return": array, "acf_1": array}}}"""
    out = {}
    for r in step3_results:
        q = r["q"]
        by_horizon = {h: {"windowed_vol": [], "cum_return": [], "acf_1": []} for h in HORIZONS_WINDOWS}
        for _, cand in r["candidates"].iterrows():
            for s in range(N_SIMS_PER_CANDIDATE):
                seed = 70000 + q * 10000 + int(cand.name) * 100 + s
                closes = simulate_long_path(cand["delta"], int(cand["wma"]), seed)
                if closes is None:
                    continue
                for h in HORIZONS_WINDOWS:
                    m = path_metrics(closes, h)
                    for key in ("windowed_vol", "cum_return", "acf_1"):
                        by_horizon[h][key].append(m[key])
        out[q] = {h: {k: np.array(v) for k, v in d.items()} for h, d in by_horizon.items()}
        n_ok = len(by_horizon[HORIZONS_WINDOWS[0]]["cum_return"])
        print(f"  q={q}: {n_ok} successful long-path simulations")
    return out


def get_real_horizon_stats() -> dict:
    """{q: {horizon: {"windowed_vol": array, "cum_return": array}}} from the out-of-sample test
    period, using the exact same quantile assignment as delta_forward_validation.py."""
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes_extended = df["close"].values[START_ROW : START_ROW + N_ROWS + CHAR_WINDOW].astype(np.float64)
    closes = closes_extended[:N_ROWS]
    log_returns = np.diff(np.log(closes))
    mu, sigma = log_returns.mean(), log_returns.std()

    models = asyncio.run(load_models())
    delta_sliding = predict_all_windows(models, log_returns, mu, sigma)

    n_windows = len(log_returns) // OBS_LEN
    window_start_idx = np.arange(n_windows) * OBS_LEN
    delta_per_window = delta_sliding[window_start_idx]

    n_calib = int(n_windows * CALIB_FRAC)
    calib_deltas = delta_per_window[:n_calib]
    quantile_edges = np.quantile(calib_deltas, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])

    test_idx = np.arange(n_calib, n_windows - MAX_HORIZON)
    test_delta = delta_per_window[test_idx]
    test_quantile = np.digitize(test_delta, quantile_edges)

    out = {q: {h: {"windowed_vol": [], "cum_return": []} for h in HORIZONS_WINDOWS} for q in range(N_QUANTILES)}
    for i, q in zip(test_idx, test_quantile):
        cur_start = window_start_idx[i]
        for h in HORIZONS_WINDOWS:
            fwd_window_start = window_start_idx[i + h]
            windowed_r = np.diff(np.log(closes[fwd_window_start : fwd_window_start + OBS_LEN + 1]))
            out[q][h]["windowed_vol"].append(float(windowed_r.std()) if len(windowed_r) > 1 else np.nan)
            out[q][h]["cum_return"].append(float(np.log(closes[fwd_window_start] / closes[cur_start])))
    return {q: {h: {k: np.array(v) for k, v in d.items()} for h, d in hd.items()} for q, hd in out.items()}


def _cache_path() -> Path:
    """Baseline (no mechanism enabled) keeps the original filename, already produced this
    session; any experimental run gets its own filename encoding which knob(s) are active, so it
    never overwrites the baseline or another experiment's cache."""
    base = Path(__file__).resolve().parent
    if SPREAD_FEEDBACK_A is None and SHOCK_PROBABILITY == 0.0:
        return base / "ddm_horizon_stats.npz"
    parts = []
    if SPREAD_FEEDBACK_A is not None:
        parts.append(f"spreadA{SPREAD_FEEDBACK_A}_w{SPREAD_FEEDBACK_WINDOW}")
    if SHOCK_PROBABILITY != 0.0:
        parts.append(f"shockP{SHOCK_PROBABILITY}_S{SHOCK_SIZE}")
    return base / f"ddm_horizon_stats_{'_'.join(parts)}.npz"


DDM_STATS_CACHE = _cache_path()  # kept as a module attribute for ddm_calibration_check.py's import;
                                  # refreshed in __main__ after CLI args are parsed, before use.


def _flatten_ddm_stats(ddm_stats: dict) -> dict:
    flat = {}
    for q, by_h in ddm_stats.items():
        for h, metrics in by_h.items():
            for key, arr in metrics.items():
                flat[f"q{q}_h{h}_{key}"] = arr
    return flat


def _unflatten_ddm_stats(flat) -> dict:
    ddm_stats: dict = {}
    for name in flat.files:
        q_part, h_part, key = name.split("_", 2)
        q, h = int(q_part[1:]), int(h_part[1:])
        ddm_stats.setdefault(q, {}).setdefault(h, {})[key] = flat[name]
    return ddm_stats


def run_phase1():
    from ddm_ensemble_selection import main as step3_main
    print("=== Step 3 (candidate sets) ===")
    step3_results, real_df, atlas = step3_main()

    print(f"\n=== Building DDM long-path ensemble stats (spread_feedback_a={SPREAD_FEEDBACK_A}, "
          f"window={SPREAD_FEEDBACK_WINDOW}, shock_probability={SHOCK_PROBABILITY}, "
          f"shock_size={SHOCK_SIZE}) -- this takes a while ===")
    ddm_stats = build_ddm_horizon_stats(step3_results)
    cache_path = _cache_path()
    np.savez(cache_path, **_flatten_ddm_stats(ddm_stats))
    print(f"\nSaved DDM horizon stats to {cache_path}")
    return ddm_stats


def run_phase2(ddm_stats: dict):
    print("\n=== Computing real out-of-sample horizon stats ===")
    real_stats = get_real_horizon_stats()

    print("\n=== A. Windowed future volatility AT t+h (matches delta_forward_validation.py's metric) ===")
    print(f"{'h':>3} {'Spearman(q, DDM windowed_vol)':>30} {'Spearman(q, REAL windowed_vol)':>32}")
    for h in HORIZONS_WINDOWS:
        qs = sorted(ddm_stats)
        ddm_vals = [np.nanmean(ddm_stats[q][h]["windowed_vol"]) for q in qs]
        real_vals = [np.nanmean(real_stats[q][h]["windowed_vol"]) for q in qs]
        r_ddm, _ = spearmanr(qs, ddm_vals)
        r_real, _ = spearmanr(qs, real_vals)
        print(f"{h:>3} {r_ddm:>30.4f} {r_real:>32.4f}")

    print("\n=== B. Cumulative return std, t -> t+h, and DDM/Real ratio ===")
    print(f"{'h':>3} {'ddm_std':>10} {'real_std':>10} {'ratio(ddm/real)':>16}")
    ratios = []
    for h in HORIZONS_WINDOWS:
        qs = sorted(ddm_stats)
        ddm_all = np.concatenate([ddm_stats[q][h]["cum_return"] for q in qs])
        real_all = np.concatenate([real_stats[q][h]["cum_return"] for q in qs])
        ddm_std, real_std = np.nanstd(ddm_all), np.nanstd(real_all)
        ratio = ddm_std / real_std
        ratios.append(ratio)
        print(f"{h:>3} {ddm_std:>10.6f} {real_std:>10.6f} {ratio:>16.3f}")

    print("\n=== C. Return acf_1 over the t->t+h path (DDM only -- diagnostic) ===")
    for h in HORIZONS_WINDOWS:
        qs = sorted(ddm_stats)
        ddm_acf1 = np.nanmean(np.concatenate([ddm_stats[q][h]["acf_1"] for q in qs]))
        print(f"  h={h:>3}: DDM acf_1 = {ddm_acf1:+.4f}")

    print(f"\n=== Summary: DDM/Real cumulative-return-std ratio vs horizon ===")
    for h, r in zip(HORIZONS_WINDOWS, ratios):
        print(f"  {h:>3} windows: ratio = {r:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["phase1", "phase2"], nargs="?", default="phase1")
    parser.add_argument("--spread-a", type=float, default=None,
                         help="spread_feedback_a; omit for the feedback-free baseline")
    parser.add_argument("--spread-window", type=int, default=150, help="spread_feedback_window")
    parser.add_argument("--shock-prob", type=float, default=0.0,
                         help="exogenous_shock_probability; 0.0 for the shock-free baseline")
    parser.add_argument("--shock-size", type=float, default=0.0, help="exogenous_shock_size")
    args = parser.parse_args()

    SPREAD_FEEDBACK_A = args.spread_a
    SPREAD_FEEDBACK_WINDOW = args.spread_window
    SHOCK_PROBABILITY = args.shock_prob
    SHOCK_SIZE = args.shock_size
    DDM_STATS_CACHE = _cache_path()

    if args.mode == "phase1":
        run_phase1()
    elif args.mode == "phase2":
        cached = np.load(DDM_STATS_CACHE)
        print(f"Loaded {DDM_STATS_CACHE}")
        run_phase2(_unflatten_ddm_stats(cached))
