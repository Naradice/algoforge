"""Consolidation-phase statistics for the Step 5 predictive-usefulness result. Three checks,
one model load:

1. CRPS significance -- paired per-window CRPS difference (baseline vs §11 decayed-shock),
   iid bootstrap CI + moving-block bootstrap CI (block ~ horizon, since forward windows overlap
   at h>1) + Diebold-Mariano stat with Newey-West HAC variance.
2. Horizon-wise robustness of the improvement -- bootstrap over the simulation draws inside each
   quantile's ensemble (the sims ARE the seed variation within one phase1 run; candidate
   selection is deterministic and unaffected), giving a CI on the ~7.5% pooled improvement.
3. Regime-conditioning contribution -- paired per-window CRPS difference, §11 conditioned on the
   predicted-delta quantile vs. §11 pooled over all quantiles (regime label discarded), same
   bootstrap + DM battery.

Target conclusion this is meant to support or refute: "the calibrated DDM is a modest but
consistently better probabilistic forecast than the baseline, while the inferred regime label
provides little incremental forecasting value."
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ddm_ensemble_selection import (
    CALIB_FRAC, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT, predict_all_windows,
)
from ddm_horizon_analysis import HORIZONS_WINDOWS, MAX_HORIZON, _unflatten_ddm_stats
from real_data_transfer_test import load_models

BASELINE_CACHE = "ddm_horizon_stats.npz"
STEP5_CACHE = "ddm_horizon_stats_shockP0.002_S0.3_tau300.0_floor0.0.npz"
N_BOOT = 5000
RNG = np.random.default_rng(12345)
RESULTS: dict = {}


def crps_scalar(samples: np.ndarray, y: float, e_xx_half: float) -> float:
    """CRPS(F, y) = E|X-y| - 0.5 E|X-X'|. e_xx_half = 0.5 E|X-X'| is precomputed per ensemble."""
    return float(np.mean(np.abs(samples - y)) - e_xx_half)


def e_xx_half(samples: np.ndarray) -> float:
    s = np.sort(samples)
    m = len(s)
    i = np.arange(1, m + 1)
    return float((1.0 / m**2) * np.sum((2 * i - m - 1) * s))  # = 0.5 * E|X-X'|


def get_real_windows() -> dict:
    """{h: (quantile_array, realized_cum_return_array)} in TIME ORDER (needed for block bootstrap)."""
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    from ddm_ensemble_selection import CHAR_WINDOW
    closes_ext = df["close"].values[START_ROW : START_ROW + N_ROWS + CHAR_WINDOW].astype(np.float64)
    closes = closes_ext[:N_ROWS]
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
    test_q = np.digitize(delta_per_window[test_idx], quantile_edges)

    out = {}
    for h in HORIZONS_WINDOWS:
        ys = np.array([np.log(closes[window_start_idx[i + h]] / closes[window_start_idx[i]]) for i in test_idx])
        out[h] = (test_q.copy(), ys)
    return out


def _boot_ci(x: np.ndarray, block: int, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Moving-block bootstrap 95% CI for mean(x). block=1 -> ordinary iid bootstrap."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts_pool = np.arange(0, n - block + 1) if n > block else np.array([0])
    for b in range(n_boot):
        starts = RNG.choice(starts_pool, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _dm_stat(d: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano statistic for the loss-differential series d, Newey-West HAC variance
    with `lag` lags. Returns (stat, two-sided normal p-value)."""
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.mean(dc * dc)
    var = gamma0
    for k in range(1, lag + 1):
        gk = np.mean(dc[k:] * dc[:-k])
        var += 2 * (1 - k / (lag + 1)) * gk
    stat = dbar / np.sqrt(var / n)
    from math import erfc, sqrt
    p = erfc(abs(stat) / sqrt(2))
    return float(stat), float(p)


def paired_report(name: str, d: np.ndarray, base_level: float, h: int) -> dict:
    """d_i = loss_worse_i - loss_better_i  (positive mean => `better` really is better)."""
    n = len(d)
    mean_d = d.mean()
    pct = 100 * mean_d / base_level
    block = max(1, h)
    ci_iid = _boot_ci(d, 1)
    ci_blk = _boot_ci(d, block)
    dm_stat, dm_p = _dm_stat(d, lag=max(1, h))
    frac_better = float(np.mean(d > 0))
    print(f"  [{name}] h={h:>2}  mean ΔCRPS={mean_d:+.3e} ({pct:+.2f}% of {base_level:.3e})  "
          f"frac_better={frac_better:.3f}")
    print(f"        iid boot 95% CI: [{ci_iid[0]:+.3e}, {ci_iid[1]:+.3e}]  "
          f"({100*ci_iid[0]/base_level:+.2f}%, {100*ci_iid[1]/base_level:+.2f}%)")
    print(f"        block(={block}) boot 95% CI: [{ci_blk[0]:+.3e}, {ci_blk[1]:+.3e}]  "
          f"({100*ci_blk[0]/base_level:+.2f}%, {100*ci_blk[1]/base_level:+.2f}%)")
    print(f"        Diebold-Mariano (NW lag {max(1,h)}): stat={dm_stat:+.2f}  p={dm_p:.4f}")
    return {
        "n": n, "mean_delta": mean_d, "pct_of_base": pct, "frac_better": frac_better,
        "ci_iid": list(ci_iid), "ci_iid_pct": [100 * c / base_level for c in ci_iid],
        "ci_block": list(ci_blk), "ci_block_pct": [100 * c / base_level for c in ci_blk],
        "dm_stat": dm_stat, "dm_p": dm_p,
    }


def main():
    baseline = _unflatten_ddm_stats(np.load(BASELINE_CACHE))
    step5 = _unflatten_ddm_stats(np.load(STEP5_CACHE))
    print(f"baseline={BASELINE_CACHE}  step5={STEP5_CACHE}  N_BOOT={N_BOOT}")
    print("Computing real out-of-sample windows (single model load)...")
    real = get_real_windows()

    for h in HORIZONS_WINDOWS:
        q_arr, y_arr = real[h]
        n = len(y_arr)

        # Precompute the ensemble self-distance term per quantile (both variants) + pooled step5.
        base_exx = {q: e_xx_half(baseline[q][h]["cum_return"]) for q in range(N_QUANTILES)}
        s5_exx = {q: e_xx_half(step5[q][h]["cum_return"]) for q in range(N_QUANTILES)}
        s5_pool = np.concatenate([step5[q][h]["cum_return"] for q in range(N_QUANTILES)])
        s5_pool_exx = e_xx_half(s5_pool)

        crps_base = np.array([crps_scalar(baseline[q][h]["cum_return"], y, base_exx[q]) for q, y in zip(q_arr, y_arr)])
        crps_s5 = np.array([crps_scalar(step5[q][h]["cum_return"], y, s5_exx[q]) for q, y in zip(q_arr, y_arr)])
        crps_pool = np.array([crps_scalar(s5_pool, y, s5_pool_exx) for y in y_arr])

        print(f"\n{'='*88}\nHORIZON h={h}  (n={n} windows)  "
              f"pooled CRPS: baseline={crps_base.mean():.6f}  step5={crps_s5.mean():.6f}  "
              f"step5_pooled={crps_pool.mean():.6f}\n{'='*88}")

        # --- Check 1: baseline vs step5 (conditioned) ---
        d1 = crps_base - crps_s5  # >0 => step5 better
        r1 = paired_report("1: baseline - step5", d1, crps_base.mean(), h)

        # --- Check 3: step5_pooled vs step5_conditioned ---
        d3 = crps_pool - crps_s5  # >0 => conditioning helps
        r3 = paired_report("3: pooled - conditioned", d3, crps_pool.mean(), h)

        # --- Check 2: sim-bootstrap CI on the pooled improvement % ---
        improvements = np.empty(N_BOOT)
        obs_by_q = {q: y_arr[q_arr == q] for q in range(N_QUANTILES)}
        by_q_base = {q: baseline[q][h]["cum_return"] for q in range(N_QUANTILES)}
        by_q_s5 = {q: step5[q][h]["cum_return"] for q in range(N_QUANTILES)}
        for b in range(N_BOOT):
            cb = cs = 0.0
            for q in range(N_QUANTILES):
                obs_q = obs_by_q[q]
                if len(obs_q) == 0:
                    continue
                sb = by_q_base[q]; ss = by_q_s5[q]
                rb = sb[RNG.integers(0, len(sb), len(sb))]
                rs = ss[RNG.integers(0, len(ss), len(ss))]
                cb += np.abs(rb[None, :] - obs_q[:, None]).mean(axis=1).sum() - e_xx_half(rb) * len(obs_q)
                cs += np.abs(rs[None, :] - obs_q[:, None]).mean(axis=1).sum() - e_xx_half(rs) * len(obs_q)
            improvements[b] = 100 * (1 - cs / cb)
        imp_lo, imp_hi = np.percentile(improvements, [2.5, 97.5])
        point = 100 * (1 - crps_s5.mean() / crps_base.mean())
        print(f"  [2: sim-bootstrap] h={h:>2}  pooled improvement point={point:+.2f}%  "
              f"95% CI [{imp_lo:+.2f}%, {imp_hi:+.2f}%]  (>0 in {100*np.mean(improvements>0):.1f}% of resamples)")

        RESULTS[str(h)] = {
            "n": n,
            "pooled_crps": {"baseline": float(crps_base.mean()), "step5": float(crps_s5.mean()),
                            "step5_pooled": float(crps_pool.mean())},
            "check1_baseline_vs_step5": r1,
            "check3_pooled_vs_conditioned": r3,
            "check2_sim_bootstrap": {"point_pct": float(point), "ci_pct": [float(imp_lo), float(imp_hi)],
                                      "frac_positive": float(np.mean(improvements > 0))},
        }


if __name__ == "__main__":
    main()
    out = Path(__file__).resolve().parent / "results" / "ddm_step5_significance.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nWrote {out}")
