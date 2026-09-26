"""
Phase 6e: does the Phase 6d hypothesis hold across EVERY existing pretrain?

Phase 6d found N3 (no transfer) never learns the DDM segment of its mixture while N1 (transfers)
does, and proposed that dataset-level z-scoring shrinks DDM's vol_20 variance below the synthetic
segment's residual loss. This evaluates each condition's pretrain best.pt (the checkpoint every
fine-tune warm-started from) on its own pretrain val windows, split by segment:

  - ddm_r2:   R^2 on DDM-segment windows (does the pretrain learn DDM at all?)
  - ddm_var:  variance of the z-scored target on DDM windows (how compressed DDM is)
  - syn_mse:  residual MSE on synthetic-segment windows
  - syn_r2:   R^2 on synthetic-segment windows
  - ddm_var / syn_mse: DDM's whole signal relative to what the synthetic part leaves unexplained

Test: does ddm_r2 > 0 (or ddm_var / syn_mse above some level) separate every transferring
condition from every non-transferring one, including M1 (Sine p15, 26x scale, transfers) and
K1/L1 (~8x scale, fail)? No training involved.

Usage: python cross_condition_segment_check.py   (writes cross_condition_segment_check.json)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_representations import _artifact_store, _build_model, _dataset_artifact_path, _load_checkpoint
from segment_pretrain_trajectory import _predict
from submit_transfer_experiment import BASE_HP

# label: (dataset id, pretrain run id, model id, synthetic rows at the tail, outcome)
# Outcome = fine-tune mean best val_loss and transfer count, from the handoff memo and
# docs/research-periodic-forcing-transfer.md.
CONDITIONS = {
    "B ddm only":       (52, 1442, 125, 0,       "no (0.837, 0/3)"),
    "D1 sine p50":      (54, 1450, 127, 60_000,  "YES (0.507, 3/3)"),
    "D2 delay":         (55, 1458, 130, 60_000,  "YES (0.516, 3/3)"),
    "D3 xor":           (56, 1462, 131, 60_000,  "no (~0.83, 0/3)"),
    "D4 lfsr":          (57, 1466, 132, 60_000,  "no (~0.83, 0/3)"),
    "E1 sine 120K":     (58, 1470, 133, 120_000, "YES (flat)"),
    "E2 sine 180K":     (59, 1474, 134, 180_000, "YES (flat)"),
    "E3 sine 240K":     (60, 1478, 135, 240_000, "YES (flat)"),
    "G1 ar1":           (63, 1493, 139, 60_000,  "no (0.825, 0/3)"),
    "H1 lorenz .02":    (64, 1499, 142, 60_000,  "no (0.826, 0/3)"),
    "H2 ar1f s42":      (65, 1505, 147, 60_000,  "mixed (0.692, 2/3)"),
    "H2 ar1f s99":      (65, 1557, 166, 60_000,  "no (0.834, 0/3)"),
    "I1 ar1f p70":      (68, 1516, 150, 60_000,  "no (0.832, 0/5)"),
    "I2 ar1f p140":     (69, 1522, 151, 60_000,  "no (0.828, 0/5)"),
    "J1 ar1f 2p":       (74, 1534, 158, 60_000,  "no (0.825, 0/5)"),
    "K1 ar1f rand5":    (76, 1540, 159, 60_000,  "no (0.827, 0/5)"),
    "L1 ar1f gap+10":   (77, 1548, 162, 60_000,  "no (0.830, 0/5)"),
    "M1 sine p15":      (78, 1564, 170, 60_000,  "YES (0.599, 5/5)"),
    "M2 sine p200":     (79, 1571, 172, 60_000,  "YES (0.535, 5/5)"),
    "N1 lorenz .01":    (80, 1579, 175, 60_000,  "YES (0.615, 5/5)"),
    "N1 lorenz .01 s99": (80, 1591, 177, 60_000, "YES (0.577, 3/3)"),
    "N2 delay s2":      (81, 1580, 176, 60_000,  "YES (0.324, 5/5)"),
    "N2 delay s2 s99":  (81, 1592, 178, 60_000,  "YES (0.420, 3/3)"),
    "N3 lorenz .0125":  (82, 1599, 179, 60_000,  "no (0.825, 0/3)"),
    "N4 lorenz .015":   (83, 1600, 180, 60_000,  "no (0.827, 0/3)"),
    "N5 lorenz .0175":  (84, 1601, 181, 60_000,  "no (0.824, 0/3)"),
}
N_SEG_SAMPLES = 8_000
SEED = 123

HP_KEYS = ("obs_len", "pred_len", "feature_cols", "tgt_feature_cols", "preprocessing", "normalize",
           "split_mode", "require_contiguous", "max_rows", "val_split")


def _seg_stats(model, ds, idx) -> dict:
    src, tgt = ds._val_src[idx], ds._val_tgt[idx]
    pred, _ = _predict(model, src, tgt)
    y = tgt[:, 1:, :]
    mse = float(((pred - y) ** 2).mean())
    var = float(y.var())
    return {"mse": mse, "var": var, "r2": 1 - mse / var}


async def main() -> None:
    from model_core.trainers import OHLCWindowDataset

    rng = np.random.default_rng(SEED)
    ds_cache: dict[int, tuple] = {}
    rows = []
    print(f"{'condition':<19}{'outcome':<20}{'ddm_r2':>8}{'ddm_var':>9}{'syn_mse':>9}"
          f"{'syn_r2':>8}{'ddm_var/syn_mse':>16}")
    for label, (dataset_id, run_id, model_id, syn_rows, outcome) in CONDITIONS.items():
        if dataset_id not in ds_cache:
            artifact = await _dataset_artifact_path(dataset_id)
            ds = OHLCWindowDataset(artifact_path=artifact, **{k: BASE_HP[k] for k in HP_KEYS})
            ds.eval()
            if syn_rows:
                raw_index = pd.read_parquet(_artifact_store() / artifact, columns=["close"]).index
                boundary = raw_index[len(raw_index) - syn_rows]
                is_syn = pd.DatetimeIndex(ds.window_start_timestamps) >= boundary
            else:
                is_syn = np.zeros(len(ds), dtype=bool)
            ds_cache[dataset_id] = (ds, is_syn)
        ds, is_syn = ds_cache[dataset_id]

        model = _build_model()
        _load_checkpoint(model, f"models/{model_id}/training_{run_id}/best.pt")
        ddm_pool = np.where(~is_syn)[0]
        ddm = _seg_stats(model, ds, rng.choice(ddm_pool, size=min(N_SEG_SAMPLES, len(ddm_pool)), replace=False))
        row = {"condition": label, "outcome": outcome, "pretrain_run": run_id,
               "ddm_r2": ddm["r2"], "ddm_var": ddm["var"], "ddm_mse": ddm["mse"]}
        if is_syn.any():
            syn_pool = np.where(is_syn)[0]
            syn = _seg_stats(model, ds, rng.choice(syn_pool, size=min(N_SEG_SAMPLES, len(syn_pool)), replace=False))
            row.update({"syn_mse": syn["mse"], "syn_var": syn["var"], "syn_r2": syn["r2"],
                        "ddm_var_over_syn_mse": ddm["var"] / syn["mse"]})
        rows.append(row)
        fmt = lambda k, w, p: f"{row[k]:>{w}.{p}f}" if k in row else f"{'-':>{w}}"
        print(f"{label:<19}{outcome:<20}{fmt('ddm_r2', 8, 3)}{fmt('ddm_var', 9, 4)}"
              f"{fmt('syn_mse', 9, 4)}{fmt('syn_r2', 8, 3)}{fmt('ddm_var_over_syn_mse', 16, 2)}", flush=True)

    out = Path(__file__).resolve().parent / "cross_condition_segment_check.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
