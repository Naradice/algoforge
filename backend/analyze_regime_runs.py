"""Throwaway: evaluate the 3-seed ddm_regime_delta_regression batch (TrainingRun ids 1389-1391)
on the held-out regimes -- per-window predictions aggregated per regime, R^2 against Var(delta)
over the held-out set, and a predicted-vs-true table per held-out regime to see whether error is
uniform or concentrated somewhere in the delta range.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_IDS = [1389, 1390, 1391]
REGIME_MAP_PATH = "../artifacts/datasets/derived/ddm_regime_delta_regression_regime_map.json"


async def load_run_and_model(run_id: int):
    import database
    from sqlalchemy import select
    from model.models import TrainingRun, MLModel
    from model_core.architectures import build_model
    from model_core.trainers.dataset import OHLCWindowDataset

    async with database.async_session_factory() as db:
        run = (await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()
        mlm = (await db.execute(select(MLModel).where(MLModel.id == run.model_id))).scalar_one()

    hp = run.hyperparams
    ds = OHLCWindowDataset(
        artifact_path="datasets/derived/ddm_regime_delta_regression.parquet",
        obs_len=hp["obs_len"], pred_len=hp["pred_len"], feature_cols=hp["feature_cols"],
        normalize=hp["normalize"], src_normalize=hp.get("src_normalize"), val_split=hp["val_split"],
        preprocessing=None, max_rows=hp["max_rows"], token_level=hp.get("token_level"),
        tgt_feature_cols=hp["tgt_feature_cols"], split_mode=hp["split_mode"],
        require_contiguous=hp["require_contiguous"],
    )
    ds.eval()

    effective_config = {
        **mlm.config,
        "input_dim": ds.n_features, "output_dim": ds.n_features,
        "obs_len": hp["obs_len"], "pred_len": hp["pred_len"],
    }
    if ds.vocab_size is not None:
        effective_config["vocab_size"] = ds.vocab_size
        effective_config["embedding_dim"] = hp.get("embedding_dim")
    effective_config["seq_len"] = ds.effective_seq_len

    device = "cpu"
    model = build_model(mlm.architecture, effective_config, device=device)
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    ckpt = torch.load(store / run.artifact_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return run, ds, model


def predict_all(ds, model, batch_size=256) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred, true) arrays over every held-out val window, length len(ds)."""
    preds, trues = [], []
    n = len(ds)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            src, tgt = ds[i : min(i + batch_size, n)]
            input_tgt, output_tgt = tgt[:, :-1, :], tgt[:, 1:, :]
            logits = model(src, input_tgt)
            preds.append(logits[:, -1, 0].cpu().numpy())
            trues.append(output_tgt[:, -1, 0].cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


async def main_async():
    regime_map = json.load(open(REGIME_MAP_PATH))
    heldout_regimes = [r for r in regime_map["regimes"] if r["split"] == "heldout"]
    heldout_deltas = np.array(sorted(r["delta"] for r in heldout_regimes))
    var_delta = heldout_deltas.var()
    print(f"Var(delta) over {len(heldout_deltas)} held-out regimes: {var_delta:.6f}\n")

    all_seed_r2 = []
    per_regime_preds: dict[float, list[float]] = {}

    for run_id in RUN_IDS:
        # One shared event loop for the whole batch -- a separate asyncio.run() per iteration
        # leaves database.async_session_factory's pooled connection bound to an already-closed
        # loop on the next iteration (same failure mode hit in submit_regime_training.py).
        run, ds, model = await load_run_and_model(run_id)
        pred, true = predict_all(ds, model)
        mse = float(np.mean((pred - true) ** 2))
        r2 = 1 - mse / var_delta
        all_seed_r2.append(r2)
        print(f"run_id={run_id} seed={run.hyperparams.get('seed')} n_val_windows={len(pred)} "
              f"MSE={mse:.6f} R2={r2:.4f} (stored val_loss={run.val_loss:.6f})")

        # Aggregate predictions per regime (true delta identifies which regime each window
        # belongs to -- exact match since delta is float32-quantized but constant per regime).
        for d in np.unique(np.round(true, 6)):
            mask = np.round(true, 6) == d
            per_regime_preds.setdefault(float(d), []).append(float(pred[mask].mean()))

    print(f"\nMean R2 across {len(RUN_IDS)} seeds: {np.mean(all_seed_r2):.4f} (seeds: {[round(r,4) for r in all_seed_r2]})\n")

    print(f"{'true_delta':>12} {'pred_mean(3 seeds)':>20} {'pred_std':>10} {'abs_err':>10}")
    for d in sorted(per_regime_preds):
        preds = per_regime_preds[d]
        pmean, pstd = np.mean(preds), np.std(preds)
        print(f"{d:12.4f} {pmean:20.4f} {pstd:10.4f} {abs(pmean - d):10.4f}")


if __name__ == "__main__":
    asyncio.run(main_async())
