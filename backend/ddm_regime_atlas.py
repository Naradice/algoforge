"""DDM regime atlas: dealer_sensitive (delta) x wma grid, mapped to full summary statistics plus
a per-point stability score (fraction of seeds that didn't diverge).

This is deliberately NOT a point-calibration tool -- it's the lookup table candidate-selection
draws from: for a given real regime's (std, acf_1, vol_clustering), select the SET of (delta, wma)
points whose own statistics are close, rather than a single "best" point. Unstable points are
naturally down-weighted by a low success_rate rather than excluded outright -- a point that
diverges in 4 of 5 seeds but succeeds once still contributes a (noisy) sample to the atlas, and
the atlas records exactly how much to trust it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_ddm import ACF_LAGS, VOL_CLUSTER_LAG, simulate_ohlc
from data.characteristics import compute_acf, compute_return_dist, compute_vol_clustering

FULL_STAT_NAMES = ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]

DELTA_VALUES = np.round(np.linspace(-1.5, 0.0, 16), 3).tolist()
WMA_VALUES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40]
N_SEEDS = 5
N_TRADES = 20_000
BASE_SEED = 5000


def full_characterize(df: pd.DataFrame) -> dict:
    rd = compute_return_dist(df)["stats"]
    acf = compute_acf(df)["returns"]
    vc = compute_vol_clustering(df)["values"]
    out = {"std": rd["std"], "skewness": rd["skewness"], "kurtosis": rd["kurtosis"]}
    for lag in ACF_LAGS:
        out[f"acf_{lag}"] = acf[lag] if lag < len(acf) else np.nan
    out[f"volclust_{VOL_CLUSTER_LAG}"] = vc[VOL_CLUSTER_LAG] if VOL_CLUSTER_LAG < len(vc) else np.nan
    return out


def build_atlas() -> pd.DataFrame:
    rows = []
    total = len(DELTA_VALUES) * len(WMA_VALUES)
    done = 0
    for delta in DELTA_VALUES:
        for wma in WMA_VALUES:
            seed_stats = []
            for s in range(N_SEEDS):
                try:
                    df = simulate_ohlc(delta, wma, n_trades=N_TRADES, seed=BASE_SEED + s)
                    seed_stats.append(full_characterize(df))
                except (ValueError, RuntimeError):
                    continue
            success_rate = len(seed_stats) / N_SEEDS
            row = {"delta": delta, "wma": wma, "success_rate": success_rate, "n_success": len(seed_stats)}
            if seed_stats:
                for name in FULL_STAT_NAMES:
                    row[name] = float(np.nanmean([s[name] for s in seed_stats]))
                    row[f"{name}_std_across_seeds"] = float(np.nanstd([s[name] for s in seed_stats])) if len(seed_stats) > 1 else np.nan
            else:
                for name in FULL_STAT_NAMES:
                    row[name] = np.nan
                    row[f"{name}_std_across_seeds"] = np.nan
            rows.append(row)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} grid points done...")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"Building atlas: {len(DELTA_VALUES)} delta x {len(WMA_VALUES)} wma = "
          f"{len(DELTA_VALUES)*len(WMA_VALUES)} points, {N_SEEDS} seeds each...")
    atlas = build_atlas()
    out_path = Path(__file__).resolve().parent / "ddm_regime_atlas.csv"
    atlas.to_csv(out_path, index=False)
    print(f"\nSaved atlas to {out_path} ({len(atlas)} rows)")

    print("\n=== Coverage summary ===")
    print(f"Fully stable (success_rate=1.0): {(atlas['success_rate']==1.0).sum()}/{len(atlas)}")
    print(f"Fully diverged (success_rate=0.0): {(atlas['success_rate']==0.0).sum()}/{len(atlas)}")
    print(f"Partial: {((atlas['success_rate']>0) & (atlas['success_rate']<1.0)).sum()}/{len(atlas)}")

    print("\n=== Volatility (std) range covered ===")
    valid = atlas.dropna(subset=["std"])
    print(f"std range: [{valid['std'].min():.6f}, {valid['std'].max():.6f}]")
    print(f"(real USDJPY quantile bins ranged roughly [0.000038, 0.000221] per the earlier ordinal check)")
