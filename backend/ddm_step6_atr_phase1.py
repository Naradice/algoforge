"""Step 6 Phase B, phase-1 for the atr_proxy regime: build a 20-quantile ATR regime map, assign
each quantile a DDM candidate set with the SAME Step-3 rule (match the quantile's forward
1000-candle real statistics to the closest atlas candidate), then simulate the §11 decayed-shock
ensemble per quantile. Mirrors ddm_horizon_analysis.py phase1 but with the ATR-proxy quantile
assignment substituted for the Transformer delta.

No neural model needed -- ATR is close-only. Candidate selection is cheap; the cost is the
~3200 long-path DDM simulations (build_ddm_horizon_stats), same as any other phase1.

Output: ddm_horizon_stats_ATR_shockP0.002_S0.3_tau300.0_floor0.0.npz
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import ddm_horizon_analysis as dha
from ddm_ensemble_selection import (
    CALIB_FRAC, CHAR_WINDOW, N_QUANTILES, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT,
    full_characterize, select_ensemble,
)

# §11 decayed-shock mechanism -- identical to the adopted Transformer-regime setting.
dha.SHOCK_PROBABILITY = 0.002
dha.SHOCK_SIZE = 0.3
dha.SHOCK_DECAY_TAU = 300.0
dha.SHOCK_FLOOR_PROBABILITY = 0.0

CACHE = Path(__file__).resolve().parent / "ddm_horizon_stats_ATR_shockP0.002_S0.3_tau300.0_floor0.0.npz"


def atr_proxy_per_window(closes: np.ndarray, n_windows: int) -> np.ndarray:
    """Close-only ATR: mean |Δclose| over each non-overlapping OBS_LEN window."""
    out = np.empty(n_windows)
    for i in range(n_windows):
        c0 = i * OBS_LEN
        out[i] = np.abs(np.diff(closes[c0 : c0 + OBS_LEN + 1])).mean()
    return out


def build_atr_quantile_candidates(atlas: pd.DataFrame) -> list[dict]:
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes_ext = df["close"].values[START_ROW : START_ROW + N_ROWS + CHAR_WINDOW].astype(np.float64)
    closes = closes_ext[:N_ROWS]

    n_windows = len(closes) // OBS_LEN
    window_start_idx = np.arange(n_windows) * OBS_LEN
    atr_pw = atr_proxy_per_window(closes, n_windows)

    n_calib = int(n_windows * CALIB_FRAC)
    calib = atr_pw[:n_calib]
    edges = np.quantile(calib, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])
    test_atr = atr_pw[n_calib:]
    test_q = np.digitize(test_atr, edges)
    test_starts = window_start_idx[n_calib:]

    results = []
    for q in range(N_QUANTILES):
        idx = np.where(test_q == q)[0]
        if len(idx) == 0:
            continue
        stats_list = []
        for i in idx:
            fwd = test_starts[i]
            wc = closes_ext[fwd : fwd + CHAR_WINDOW + 1]
            if len(wc) < CHAR_WINDOW // 2:
                continue
            stats_list.append(full_characterize(pd.DataFrame({"close": wc})))
        avg = {name: float(np.nanmean([s[name] for s in stats_list])) for name in
               ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]}
        real_row = pd.Series(avg)
        sel = select_ensemble(real_row, atlas)
        best = sel["candidates"].iloc[0]
        status = "OUT OF RANGE" if sel["out_of_range"] else "ok"
        print(f"  q={q:>2} n={len(stats_list):>4} real_std={avg['std']:.6f} "
              f"best=({best['delta']:+.2f},{best['wma']:.0f}) ratio={sel['best_std_ratio']:.2f} {status}")
        results.append({"q": q, "real_std": avg["std"], **sel})
    return results


def main():
    atlas = pd.read_csv(Path(__file__).resolve().parent / "ddm_regime_atlas.csv")
    print(f"Loaded atlas: {len(atlas)} points")
    print("\n=== ATR-proxy 20-quantile regime -> DDM candidate sets ===")
    atr_results = build_atr_quantile_candidates(atlas)
    n_oor = sum(1 for r in atr_results if r["out_of_range"])
    print(f"\n{len(atr_results)}/{N_QUANTILES} quantiles mapped; {n_oor} flagged OUT OF RANGE")

    print(f"\n=== Building ATR-regime DDM long-path ensemble (§11 decayed shock) -- takes a while ===")
    ddm_stats = dha.build_ddm_horizon_stats(atr_results)
    np.savez(CACHE, **dha._flatten_ddm_stats(ddm_stats))
    print(f"\nSaved {CACHE}")


if __name__ == "__main__":
    main()
