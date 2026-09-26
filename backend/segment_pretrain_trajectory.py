"""
Phase 6d: per-segment pretraining trajectory, N1 (Lorenz dt=0.01, transfers 8/8) vs
N3 (Lorenz dt=0.0125, 0/3).

Frozen layer-3 volatility probes already showed transfer is decided during pretraining (N1's
pretrain checkpoint carries a USDJPY-volatility feature, N3's doesn't). This asks HOW the two
pretrains diverge, using the 20 intermediate checkpoints (epoch_0001..0020, one per 2,000 steps)
each run saved. For every checkpoint:

  - pretrain val loss (MSE on the z-scored vol_20 target, same as the trainer's eval) split by
    source segment: DDM windows vs Lorenz windows, plus the overall mixture loss
  - per-segment R^2 = 1 - MSE / var(target within that segment's val windows)
  - USDJPY future-vol probe: Ridge R^2 on each layer's frozen last-position hidden state

Separates: (a) N3 fits Lorenz at DDM's expense (loss competition), (b) both segments train
normally but the feature doesn't form (representation formation), (c) near-identical loss
trajectories but different probe outcome (optimizer path / feature formation).

The pretrain dataset is rebuilt with the exact BASE_HP the runs used (regime_controlled split,
split_seed default) so the val windows are the ones the trainer validated on. The synthetic
component is the last ABLATION_COMPONENT_ROWS rows of each mixture dataset.

Usage:
    python segment_pretrain_trajectory.py                 # N1 + N3 -> segment_pretrain_trajectory.json
    python segment_pretrain_trajectory.py B_ddm_only ...  # chosen RUNS labels -> ..._<labels>.json
Per-segment keys are "ddm_*" and "lorenz_*" (the synthetic segment, whatever its generator; a
DDM-only run like B has no synthetic keys).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_representations import (
    PROBE_HP, USDJPY_DATASET_ID, _artifact_store, _build_model, _dataset_artifact_path,
    _forward_with_hidden_states, _load_checkpoint,
)
from submit_transfer_experiment import ABLATION_COMPONENT_ROWS, BASE_HP

RUNS = {
    # label: (pretrain run id, model id, dataset id, synthetic rows at the tail)
    "N1_dt0.01": (1579, 175, 80, ABLATION_COMPONENT_ROWS),
    "N3_dt0.0125": (1599, 179, 82, ABLATION_COMPONENT_ROWS),
    "B_ddm_only": (1442, 125, 52, 0),  # Phase 6e: DDM-only pretrain ends at DDM R^2 = 0.000
}
DEFAULT_LABELS = ["N1_dt0.01", "N3_dt0.0125"]
N_EPOCHS = 20
N_SEG_SAMPLES = 8_000      # val windows per segment per checkpoint
N_PROBE_SAMPLES = 10_000   # USDJPY val windows for the probe
SEED = 123
BATCH = 1024


def _predict(model, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (head predictions for the 1-step target, per-layer pooled hidden states)."""
    import torch

    preds, layers = [], None
    model.eval()
    with torch.no_grad():
        for i in range(0, len(src), BATCH):
            s = torch.tensor(src[i:i + BATCH])
            t = torch.tensor(tgt[i:i + BATCH])
            preds.append(model(s, t[:, :-1, :]).numpy())
            hs = _forward_with_hidden_states(model, s)
            layers = hs if layers is None else [np.concatenate([a, b]) for a, b in zip(layers, hs)]
    return np.concatenate(preds), layers


def _probe_r2(features: np.ndarray, target: np.ndarray) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    n_tr = int(len(target) * 0.8)
    m = Ridge(alpha=1.0).fit(features[:n_tr], target[:n_tr])
    return float(r2_score(target[n_tr:], m.predict(features[n_tr:])))


async def main() -> None:
    from model_core.trainers import OHLCWindowDataset

    rng = np.random.default_rng(SEED)

    # USDJPY probe windows (shared across all checkpoints)
    usd = OHLCWindowDataset(artifact_path=await _dataset_artifact_path(USDJPY_DATASET_ID), **PROBE_HP)
    usd.eval()
    usd_idx = rng.choice(len(usd._val_src), size=N_PROBE_SAMPLES, replace=False)
    usd_src = usd._val_src[usd_idx]
    usd_vol = usd._val_tgt[usd_idx][:, 1, PROBE_HP["tgt_feature_cols"].index("vol_20")]

    labels = sys.argv[1:] or DEFAULT_LABELS
    results = {}
    for label in labels:
        run_id, model_id, dataset_id, syn_rows = RUNS[label]
        artifact = await _dataset_artifact_path(dataset_id)

        hp = {k: BASE_HP[k] for k in (
            "obs_len", "pred_len", "feature_cols", "tgt_feature_cols", "preprocessing", "normalize",
            "split_mode", "require_contiguous", "max_rows", "val_split",
        )}
        ds = OHLCWindowDataset(artifact_path=artifact, **hp)
        ds.eval()
        if syn_rows:
            raw_index = pd.read_parquet(_artifact_store() / artifact, columns=["close"]).index
            boundary = raw_index[len(raw_index) - syn_rows]
            is_syn = pd.DatetimeIndex(ds.window_start_timestamps) >= boundary
        else:
            is_syn = np.zeros(len(ds), dtype=bool)
        seg_idx = {"ddm": rng.choice(np.where(~is_syn)[0], size=N_SEG_SAMPLES, replace=False)}
        if syn_rows:
            seg_idx["lorenz"] = rng.choice(np.where(is_syn)[0], size=N_SEG_SAMPLES, replace=False)
        syn_frac = float(is_syn.mean())
        print(f"\n=== {label} (run {run_id}) === val windows={len(is_syn)}, lorenz share={syn_frac:.3f}")
        print(f"{'ckpt':>5}{'step':>7}{'ddm_mse':>9}{'lor_mse':>9}{'all_mse':>9}"
              f"{'ddm_r2':>8}{'lor_r2':>8}  usd_vol_probe_r2 L0..L3")

        rows = []
        for epoch in range(1, N_EPOCHS + 1):
            model = _build_model()
            _load_checkpoint(model, f"models/{model_id}/training_{run_id}/epoch_{epoch:04d}.pt")
            row = {"epoch": epoch, "step": epoch * 2000}
            for seg, idx in seg_idx.items():
                src, tgt = ds._val_src[idx], ds._val_tgt[idx]
                pred, _ = _predict(model, src, tgt)
                y = tgt[:, 1:, :]
                mse = float(((pred - y) ** 2).mean())
                row[f"{seg}_mse"] = mse
                row[f"{seg}_r2"] = 1 - mse / float(y.var())
            row["all_mse"] = (1 - syn_frac) * row["ddm_mse"] + syn_frac * row.get("lorenz_mse", 0.0)
            _, usd_layers = _predict(model, usd_src, usd._val_tgt[usd_idx][:, :, :1])
            row["usd_probe_r2"] = [_probe_r2(h, usd_vol) for h in usd_layers]
            rows.append(row)
            print(f"{epoch:>5}{row['step']:>7}{row['ddm_mse']:>9.4f}{row.get('lorenz_mse', float('nan')):>9.4f}"
                  f"{row['all_mse']:>9.4f}{row['ddm_r2']:>8.3f}{row.get('lorenz_r2', float('nan')):>8.3f}  "
                  + " ".join(f"{v:.3f}" for v in row["usd_probe_r2"]), flush=True)
        results[label] = {"run_id": run_id, "lorenz_val_share": syn_frac, "trajectory": rows}

    suffix = "" if labels == DEFAULT_LABELS else "_" + "_".join(labels)
    out = Path(__file__).resolve().parent / f"segment_pretrain_trajectory{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
