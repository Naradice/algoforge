"""Throwaway: does the DDM-regime-trained delta-recovery model (ml_model 115, Pre-LN + LayerScale,
runs 1417-1419) transfer to real USDJPY data at all?

There's no ground-truth delta for real data, so this can't check "is the prediction correct" --
instead it checks the one falsifiable claim the model's own training implies: since delta drives
realized volatility in DDM (established in the original dealer_sensitive sensitivity sweep), if
the model's learned mapping means anything on real data, its output should correlate with an
independently-computed realized-volatility statistic on the SAME windows. No correlation would
mean the tokenization/scale-fit to synthetic DDM data didn't transfer; a real correlation would
be the first evidence the underlying regime signal generalizes beyond the synthetic domain.

Uses returns_zscore normalization computed on USDJPY's OWN return distribution (not DDM's) --
that's the whole point of z-scoring: the network expects unit-scale input regardless of the
source series' native volatility, so re-standardizing per-dataset is how this transfer test is
supposed to work, not a departure from it.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

RUN_IDS = [1417, 1418, 1419]
USDJPY_ARTIFACT = "datasets/derived/usdjpy_1min_volatility.parquet"
OBS_LEN = 60
N_WINDOWS_SAMPLE = 20000  # subsample from the full sliding-window set for speed


async def load_models():
    import database
    from sqlalchemy import select
    from model.models import TrainingRun, MLModel
    from model_core.architectures import build_model

    models = []
    async with database.async_session_factory() as db:
        for run_id in RUN_IDS:
            run = (await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()
            mlm = (await db.execute(select(MLModel).where(MLModel.id == run.model_id))).scalar_one()
            hp = run.hyperparams
            effective_config = {
                **mlm.config, "input_dim": 1, "output_dim": 1,
                "obs_len": hp["obs_len"], "pred_len": hp["pred_len"], "seq_len": hp["obs_len"],
            }
            model = build_model(mlm.architecture, effective_config, device="cpu")
            store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
            ckpt = torch.load(store / run.artifact_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            models.append(model)
    return models


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    closes = df["close"].values.astype(np.float64)
    print(f"USDJPY rows: {len(closes)}")

    log_returns = np.diff(np.log(closes))
    n_total_windows = len(log_returns) - OBS_LEN + 1
    print(f"total possible windows: {n_total_windows}")

    rng = np.random.default_rng(42)
    starts = rng.choice(n_total_windows, size=min(N_WINDOWS_SAMPLE, n_total_windows), replace=False)
    starts.sort()

    raw_windows = np.stack([log_returns[s : s + OBS_LEN] for s in starts])  # [n, 60], RAW scale
    ret_std = raw_windows.std(axis=1)  # independent realized-volatility statistic, raw scale

    # z-score using THIS dataset's own return distribution (same convention as training's
    # returns_zscore -- self-contained per-dataset standardization, not DDM's constants).
    mu, sigma = log_returns.mean(), log_returns.std()
    z_windows = (raw_windows - mu) / sigma
    print(f"USDJPY return scale: mean={mu:.6f} std={sigma:.6f} (vs. DDM's ~1e-3 scale)")

    models = asyncio.run(load_models())

    x = torch.tensor(z_windows[:, :, None], dtype=torch.float32)
    tgt = torch.zeros(x.size(0), 0, 1)
    preds = []
    with torch.no_grad():
        for model in models:
            out = model(x, tgt)[:, -1, 0].numpy()
            preds.append(out)
    pred_mean = np.mean(preds, axis=0)
    pred_agreement = np.std(preds, axis=0)

    print(f"\npredicted delta: mean={pred_mean.mean():.4f} std={pred_mean.std():.4f} "
          f"range=[{pred_mean.min():.4f}, {pred_mean.max():.4f}]")
    print(f"cross-seed agreement (mean std across seeds per window): {pred_agreement.mean():.4f}")
    print(f"ret_std: mean={ret_std.mean():.6f} std={ret_std.std():.6f} "
          f"range=[{ret_std.min():.6f}, {ret_std.max():.6f}]")

    pear_r, pear_p = pearsonr(pred_mean, ret_std)
    spear_r, spear_p = spearmanr(pred_mean, ret_std)
    print(f"\nPearson  r={pear_r:.4f}  p={pear_p:.3e}")
    print(f"Spearman r={spear_r:.4f}  p={spear_p:.3e}")


if __name__ == "__main__":
    main()
