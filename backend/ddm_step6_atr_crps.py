"""Step 6 Phase B: does regime-conditioning on the Transformer latent delta buy more forecast
skill than conditioning on a trivial close-only volatility indicator (ATR proxy)?

Paired per-window CRPS over the same 1,476 out-of-sample windows, against the same realized
returns, at h=1/4/12/24. Five forecasts scored:
  baseline            shock-free DDM, Transformer-quantile candidate selection
  transformer_s11     §11 decayed-shock DDM, conditioned on Transformer delta quantile
  atr_s11             §11 decayed-shock DDM, conditioned on close-only ATR-proxy quantile
  transformer_pooled  §11 Transformer ensemble pooled over all quantiles (regime label discarded)
  atr_pooled          §11 ATR ensemble pooled over all quantiles

Key paired tests (moving-block bootstrap CI + Diebold-Mariano, reusing ddm_step5_significance):
  atr_s11 vs transformer_s11     -- is the learned regime worth more than close-only ATR?
  atr_s11 vs atr_pooled          -- does ATR-conditioning add skill over its own pooled dist?
  transformer_s11 vs transformer_pooled  -- (known ~0 from §12; included for completeness)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ddm_ensemble_selection import (
    CALIB_FRAC, CHAR_WINDOW, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT,
)
from ddm_horizon_analysis import HORIZONS_WINDOWS, MAX_HORIZON, _unflatten_ddm_stats
from ddm_step5_significance import _boot_ci, _dm_stat, crps_scalar, e_xx_half, get_real_windows

BASELINE_CACHE = "ddm_horizon_stats.npz"
TRANSFORMER_CACHE = "ddm_horizon_stats_shockP0.002_S0.3_tau300.0_floor0.0.npz"
ATR_CACHE = "ddm_horizon_stats_ATR_shockP0.002_S0.3_tau300.0_floor0.0.npz"
RESULTS: dict = {}


def atr_quantile_for_test_windows() -> np.ndarray:
    """Per-test-window ATR-proxy quantile, using ddm_step6_atr_phase1's exact conventions
    (n_windows = len(closes)//OBS_LEN, calib edges from the first 70%). test_idx is reconstructed
    to match ddm_step5_significance.get_real_windows (n_windows = len(log_returns)//OBS_LEN)."""
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes = df["close"].values[START_ROW : START_ROW + N_ROWS].astype(np.float64)
    log_returns = np.diff(np.log(closes))

    n_windows_eval = len(log_returns) // OBS_LEN
    n_calib_eval = int(n_windows_eval * CALIB_FRAC)
    test_idx = np.arange(n_calib_eval, n_windows_eval - MAX_HORIZON)

    n_windows_p1 = len(closes) // OBS_LEN
    n_calib_p1 = int(n_windows_p1 * CALIB_FRAC)
    atr_pw = np.array([
        np.abs(np.diff(closes[i * OBS_LEN : i * OBS_LEN + OBS_LEN + 1])).mean()
        for i in range(n_windows_p1)
    ])
    edges = np.quantile(atr_pw[:n_calib_p1], np.linspace(0, 1, N_QUANTILES + 1)[1:-1])
    return np.digitize(atr_pw[test_idx], edges)


def paired(name: str, worse: np.ndarray, better: np.ndarray, base_level: float, h: int) -> dict:
    d = worse - better  # >0 => `better` really is better
    block = max(1, h)
    ci_blk = _boot_ci(d, block)
    dm_stat, dm_p = _dm_stat(d, lag=max(1, h))
    mean_d, pct = d.mean(), 100 * d.mean() / base_level
    print(f"  [{name}] h={h:>2}  mean={mean_d:+.3e} ({pct:+.2f}%)  frac_better={np.mean(d > 0):.3f}  "
          f"block95%CI [{100*ci_blk[0]/base_level:+.2f}%, {100*ci_blk[1]/base_level:+.2f}%]  "
          f"DM stat={dm_stat:+.2f} p={dm_p:.4f}")
    return {"mean_delta": float(mean_d), "pct": float(pct), "frac_better": float(np.mean(d > 0)),
            "ci_block_pct": [100 * c / base_level for c in ci_blk], "dm_stat": dm_stat, "dm_p": dm_p}


def main():
    baseline = _unflatten_ddm_stats(np.load(BASELINE_CACHE))
    transf = _unflatten_ddm_stats(np.load(TRANSFORMER_CACHE))
    atr = _unflatten_ddm_stats(np.load(ATR_CACHE))
    print(f"caches: baseline={BASELINE_CACHE}\n        transformer={TRANSFORMER_CACHE}\n        atr={ATR_CACHE}")

    print("\nReal out-of-sample windows: Transformer quantile + realized returns (one model load)...")
    real = get_real_windows()  # {h: (q_transformer_array, y_array)}
    q_atr = atr_quantile_for_test_windows()
    print(f"n test windows: transformer={len(real[1][0])}, atr={len(q_atr)}")

    for h in HORIZONS_WINDOWS:
        q_tr, y = real[h]
        n = len(y)

        tr_exx = {q: e_xx_half(transf[q][h]["cum_return"]) for q in range(N_QUANTILES)}
        at_exx = {q: e_xx_half(atr[q][h]["cum_return"]) for q in range(N_QUANTILES)}
        bl_exx = {q: e_xx_half(baseline[q][h]["cum_return"]) for q in range(N_QUANTILES)}
        tr_pool = np.concatenate([transf[q][h]["cum_return"] for q in range(N_QUANTILES)])
        at_pool = np.concatenate([atr[q][h]["cum_return"] for q in range(N_QUANTILES)])
        tr_pool_exx, at_pool_exx = e_xx_half(tr_pool), e_xx_half(at_pool)

        c_base = np.array([crps_scalar(baseline[q][h]["cum_return"], yi, bl_exx[q]) for q, yi in zip(q_tr, y)])
        c_tr = np.array([crps_scalar(transf[q][h]["cum_return"], yi, tr_exx[q]) for q, yi in zip(q_tr, y)])
        c_at = np.array([crps_scalar(atr[q][h]["cum_return"], yi, at_exx[q]) for q, yi in zip(q_atr, y)])
        c_tr_pool = np.array([crps_scalar(tr_pool, yi, tr_pool_exx) for yi in y])
        c_at_pool = np.array([crps_scalar(at_pool, yi, at_pool_exx) for yi in y])

        print(f"\n{'='*92}\nHORIZON h={h}  (n={n})  pooled CRPS: "
              f"baseline={c_base.mean():.6f}  transformer_s11={c_tr.mean():.6f}  atr_s11={c_at.mean():.6f}  "
              f"transformer_pooled={c_tr_pool.mean():.6f}  atr_pooled={c_at_pool.mean():.6f}\n{'='*92}")

        rec = {"n": n, "pooled_crps": {
            "baseline": float(c_base.mean()), "transformer_s11": float(c_tr.mean()),
            "atr_s11": float(c_at.mean()), "transformer_pooled": float(c_tr_pool.mean()),
            "atr_pooled": float(c_at_pool.mean())}}
        rec["baseline_vs_transformer"] = paired("baseline - transformer_s11", c_base, c_tr, c_base.mean(), h)
        rec["baseline_vs_atr"] = paired("baseline - atr_s11", c_base, c_at, c_base.mean(), h)
        rec["transformer_vs_atr"] = paired("transformer_s11 - atr_s11 (>0: ATR better)", c_tr, c_at, c_tr.mean(), h)
        rec["atr_vs_atr_pooled"] = paired("atr_pooled - atr_s11 (>0: conditioning helps)", c_at_pool, c_at, c_at_pool.mean(), h)
        rec["transformer_vs_transformer_pooled"] = paired(
            "transformer_pooled - transformer_s11 (>0: conditioning helps)", c_tr_pool, c_tr, c_tr_pool.mean(), h)
        RESULTS[str(h)] = rec

    out = Path(__file__).resolve().parent / "results" / "ddm_step6_atr_crps.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
