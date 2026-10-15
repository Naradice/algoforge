"""Step 3: for each USDJPY predicted-delta quantile bin, select the DDM (delta, wma) candidate
SET from the regime atlas that is most consistent with that bin's observed (std, acf_1,
vol_clustering) -- not a single best-fit point, and explicitly not a claim that the match is good
everywhere.

Selection weighting (agreed): std is primary, acf_1 and vol_clustering are secondary (small
weight). skewness/kurtosis/acf_20 are deliberately excluded from selection -- see
regime_atlas_comparison.py's finding that matching them pulls the fit in the WRONG direction
(their real-vs-DDM relationship to the regime axis has opposite sign).

Explicitly does NOT try to force a match for every quantile: low-volatility USDJPY quantiles sit
below the atlas's achievable std floor (see ddm_regime_atlas.py's coverage summary, and the
num_agent / volatility-bounds pilots showing no clean independent scale knob exists to reach
lower). Those quantiles are flagged as OUT OF RANGE rather than assigned a misleading "best
available" ensemble.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from real_data_transfer_test import load_models
from ddm_regime_atlas import full_characterize

USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
OBS_LEN = 60
N_ROWS = 300_000
START_ROW = 300_000
CHAR_WINDOW = 1000
CALIB_FRAC = 0.7
N_QUANTILES = 20

# Selection weights: std primary, acf_1 + vol_clustering secondary (small).
# skewness/kurtosis/acf_20 excluded -- diagnostic-only, tracked but not selected on.
SELECTION_STATS = ["std", "acf_1", "volclust_50"]
SELECTION_WEIGHTS = {"std": 1.0, "acf_1": 0.25, "volclust_50": 0.25}
TOP_K = 8
# A quantile is flagged OUT OF RANGE if even its best-matching candidate's std differs from the
# real value by more than this multiplicative factor (in either direction).
OOR_STD_RATIO_THRESHOLD = 2.0


def predict_all_windows(models, log_returns: np.ndarray, mu: float, sigma: float, batch_size: int = 2000) -> np.ndarray:
    n = len(log_returns) - OBS_LEN + 1
    windows = np.stack([log_returns[i : i + OBS_LEN] for i in range(n)])
    z = ((windows - mu) / sigma).astype(np.float32)
    all_preds = [[] for _ in models]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = torch.tensor(z[start : start + batch_size, :, None], dtype=torch.float32)
            tgt = torch.zeros(chunk.size(0), 0, 1)
            for mi, model in enumerate(models):
                all_preds[mi].append(model(chunk, tgt)[:, -1, 0].numpy())
    return np.mean([np.concatenate(p) for p in all_preds], axis=0)


def compute_real_quantile_stats() -> pd.DataFrame:
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

    test_delta = delta_per_window[n_calib:]
    test_quantile = np.digitize(test_delta, quantile_edges)
    test_window_starts = window_start_idx[n_calib:]

    rows = []
    for q in range(N_QUANTILES):
        idx = np.where(test_quantile == q)[0]
        if len(idx) == 0:
            continue
        stats_list = []
        for i in idx:
            fwd_start = test_window_starts[i]
            window_closes = closes_extended[fwd_start : fwd_start + CHAR_WINDOW + 1]
            if len(window_closes) < CHAR_WINDOW // 2:
                continue
            stats_list.append(full_characterize(pd.DataFrame({"close": window_closes})))
        avg = {name: float(np.nanmean([s[name] for s in stats_list])) for name in
               ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]}
        d_lo = calib_deltas.min() if q == 0 else quantile_edges[q - 1]
        d_hi = calib_deltas.max() if q == N_QUANTILES - 1 else quantile_edges[q]
        rows.append({"q": q, "delta_mid": (d_lo + d_hi) / 2, "n": len(stats_list), **avg})
    return pd.DataFrame(rows)


SHORTLIST_SIZE = 30  # candidates kept at the std-filtering stage, before acf_1/vol_clustering refine it


def select_ensemble(real_row: pd.Series, atlas: pd.DataFrame) -> dict:
    valid = atlas.dropna(subset=SELECTION_STATS).copy()

    # Stage 1 (PRIMARY, dominant): shortlist purely by std proximity. This is a hierarchical
    # filter, not an additive weighted term -- an earlier z-scored-weighted-sum version let acf_1
    # override std entirely, because acf_1 can't get close to real ANYWHERE in the atlas (its
    # best achievable value is still far off -- see regime_atlas_comparison.py), so the "least
    # bad acf_1" candidate kept winning regardless of how bad its std was. Filtering by std FIRST
    # makes that structurally impossible: acf_1/vol_clustering can only choose among candidates
    # that already fit the real regime's volatility level.
    valid["std_distance"] = (np.log(valid["std"]) - np.log(real_row["std"])).abs()
    shortlist = valid.nsmallest(SHORTLIST_SIZE, "std_distance").copy()

    # Stage 2 (SECONDARY, tie-breaker only): within the std-shortlist, rank by acf_1 +
    # vol_clustering proximity (z-scored against the SHORTLIST's own spread, not the whole
    # atlas -- comparing against the whole atlas's spread was part of the original bug, since the
    # whole atlas's acf_1 range includes points nothing like the shortlist's regime).
    dist_sq = np.zeros(len(shortlist))
    for name in ["acf_1", "volclust_50"]:
        mu, sigma = shortlist[name].mean(), shortlist[name].std()
        sigma = sigma if sigma > 0 else 1.0
        z_atlas = (shortlist[name] - mu) / sigma
        z_real = (real_row[name] - mu) / sigma
        dist_sq += SELECTION_WEIGHTS[name] * (z_atlas - z_real) ** 2
    shortlist["distance"] = np.sqrt(dist_sq)
    shortlist = shortlist.sort_values("distance")
    top = shortlist.head(TOP_K)
    best = top.iloc[0]
    std_ratio = max(best["std"] / real_row["std"], real_row["std"] / best["std"])
    return {
        "candidates": top[["delta", "wma", "std", "acf_1", "volclust_50", "distance", "success_rate"]],
        "best_std_ratio": std_ratio,
        "out_of_range": std_ratio > OOR_STD_RATIO_THRESHOLD,
    }


def main():
    atlas = pd.read_csv(Path(__file__).resolve().parent / "ddm_regime_atlas.csv")
    print(f"Loaded atlas: {len(atlas)} points, {atlas['success_rate'].gt(0).sum()} with >=1 successful seed")

    print("\nComputing real USDJPY quantile stats (this reruns the neural model + characterization)...")
    real_df = compute_real_quantile_stats()

    print(f"\n{'q':>3} {'real_std':>10} {'best_std':>10} {'ratio':>7} {'best_delta':>11} {'best_wma':>9} {'n_cand':>7} {'status':>12}")
    results = []
    for _, row in real_df.iterrows():
        sel = select_ensemble(row, atlas)
        best = sel["candidates"].iloc[0]
        status = "OUT OF RANGE" if sel["out_of_range"] else "ok"
        print(f"{int(row['q']):>3} {row['std']:>10.6f} {best['std']:>10.6f} {sel['best_std_ratio']:>7.2f} "
              f"{best['delta']:>11.2f} {best['wma']:>9.0f} {len(sel['candidates']):>7} {status:>12}")
        results.append({"q": int(row["q"]), "real_std": row["std"], **sel})

    n_ok = sum(1 for r in results if not r["out_of_range"])
    print(f"\nQuantiles with a representable DDM ensemble: {n_ok}/{len(results)}")
    print(f"Quantiles flagged OUT OF RANGE (no DDM candidate within {OOR_STD_RATIO_THRESHOLD}x std): "
          f"{len(results)-n_ok}/{len(results)}")

    return results, real_df, atlas


if __name__ == "__main__":
    main()
