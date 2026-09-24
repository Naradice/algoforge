"""Phase 5 (user-requested): frozen representation probing -- "what is transferring, at which
depth, and is it a genuinely new feature or a simple combination of DDM's and Sine's own
representations?"

Phase 5e-period-representation (user-requested) extends this with DDM+Sine at three different
periods (15, 50, 200 -- all confirmed to TRANSFER, spanning >13x) plus DDM+LFSR (exactly
periodic, confirmed NOT to transfer) as checkpoints "ddm_sine_p15"/"ddm_sine_p200"/"ddm_lfsr".
Question: does the SAME kind of late-layer representation form regardless of period (supporting
"periodic signals steer DDM representation formation in a specific direction", a real mechanistic
story) or do different periods produce different/no representations despite all transferring
downstream (which would mean the fine-tune-level transfer effect isn't explained by a shared
frozen-representation phenomenon)? ddm_lfsr is the key negative control: exactly periodic but not
smooth/single-tone, and confirmed NOT to transfer -- if it also lacks the late-layer volatility
representation, that strengthens "the representation, not just periodicity-as-a-label" is what's
shared by the three transferring Sine periods.

Loads six pretrain checkpoints from the DDM/USDJPY transfer experiment (Scratch = untrained
random init, DDM only, Sine only, DDM+Sine, DDM+Delay, DDM+XOR), freezes them, runs the SAME
USDJPY validation windows through each, and extracts each Transformer layer's pooled ("last"
position, matching MODEL_CONFIG's pooling="last") hidden state. A small linear probe (Ridge
regression, fit on 80% of the extracted samples, scored on the held-out 20%) is then trained per
(checkpoint, layer, target) to see how well that FROZEN representation alone predicts four
different USDJPY future targets:

    future return          (returns_1)          -- directional information
    future volatility      (vol_20)             -- volatility dynamics (the task every run in
                                                    this investigation was actually fine-tuned on)
    future kurtosis        (kurtosis_20)        -- return-distribution statistics
    future autocorrelation (autocorr_20_lag1)   -- local temporal structure

The key comparison isn't any single number -- it's whether DDM+Sine's row of this table shows a
distinctive jump on some probes but not others (narrows down WHAT transferred), and at which
layer that jump first appears (narrows down WHERE/WHEN it's formed). A follow-up script
(representation_similarity.py) reuses the same saved per-layer arrays for CKA/cosine comparisons,
to check whether DDM+Sine's representation is actually NEW (not a simple mix of DDM-only's and
Sine-only's).

Usage:
    python probe_representations.py extract   # forward pass + save per-checkpoint .npz files
    python probe_representations.py probe     # fit linear probes on the saved arrays, print table
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np

# ---------------------------------------------------------------------------
# Checkpoints -- artifact-store-relative paths, resolved against ARTIFACT_STORE_PATH below.
# All six pretrain runs used pretrain_seed=42, so a single canonical checkpoint per condition is
# used here (not the D1 pretrain-seed replication set) -- consistent with how every fine-tune
# condition after D1's replication check reused a single pretrain seed.
# ---------------------------------------------------------------------------
CHECKPOINTS: dict[str, str | None] = {
    "scratch":       None,  # untrained random init -- built fresh with a fixed seed, no checkpoint
    "ddm_only":      "models/125/training_1442/best.pt",   # condition B pretrain
    "sine_only":     "models/137/training_1485/best.pt",   # E4 pretrain
    "ddm_sine":      "models/127/training_1450/best.pt",   # D1 pretrain (Sine period=50)
    "ddm_delay":     "models/130/training_1458/best.pt",   # D2 pretrain
    "ddm_xor":       "models/131/training_1462/best.pt",   # D3 pretrain
    "ddm_lfsr":      "models/132/training_1466/best.pt",   # D4 pretrain -- exactly periodic
                                                             # (period 255) but NOT smooth/single-
                                                             # tone; confirmed NOT to transfer
                                                             # (0/3) -- the key negative control
    "ddm_sine_p15":  "models/170/training_1564/best.pt",   # M1 pretrain (Sine period=15, transfers 5/5)
    "ddm_sine_p200": "models/172/training_1571/best.pt",   # M2 pretrain (Sine period=200, transfers 5/5)
}
SCRATCH_INIT_SEED = 42

USDJPY_DATASET_ID = 29
MAX_ROWS = 1_000_000
OBS_LEN = 60
PRED_LEN = 1

# Same decoder_only architecture config used throughout the transfer experiment
# (submit_transfer_experiment.py's MODEL_CONFIG) -- required to load each checkpoint's state_dict
# correctly (shapes must match what it was trained with).
MODEL_CONFIG = {
    "layernorm_mode": "pre",
    "layerscale_init": 1e-2,
    "pooling": "last",
    "d_model": 64,
    "nhead": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "input_dim": 1,     # feature_cols=["close"] during actual training
    "output_dim": 1,    # tgt_feature_cols=["vol_20"] (single column) during actual training --
                         # irrelevant here since only frozen hidden states are read, never the head
    "obs_len": OBS_LEN,
    "pred_len": PRED_LEN,
    "seq_len": OBS_LEN,
}

# The 4 probe targets, computed via preprocessing.py indicators on the SAME dataframe/windows the
# models were fine-tuned on -- see model_core/trainers/preprocessing.py's kurtosis/autocorr
# additions (added specifically for this analysis).
PROBE_TARGET_COLS = ["returns_1", "vol_20", "kurtosis_20", "autocorr_20_lag1"]
PROBE_TARGET_LABELS = {
    "returns_1": "future_return",
    "vol_20": "future_volatility",
    "kurtosis_20": "future_kurtosis",
    "autocorr_20_lag1": "future_local_structure",
}
PROBE_HP = {
    "obs_len": OBS_LEN,
    "pred_len": PRED_LEN,
    "feature_cols": ["close"],
    "tgt_feature_cols": PROBE_TARGET_COLS,
    "preprocessing": {"indicators": [
        {"type": "returns", "period": 1, "column": "close"},
        {"type": "volatility", "period": 20, "column": "close"},
        {"type": "kurtosis", "period": 20, "column": "close"},
        {"type": "autocorr", "period": 20, "lag": 1, "column": "close"},
    ]},
    "normalize": "zscore",
    "split_mode": "regime_controlled",
    "require_contiguous": True,
    "val_split": 0.2,
    "max_rows": MAX_ROWS,
}

# Validation set is large (~200K windows); subsample to a fixed, seeded subset for tractable
# CPU forward passes + probe fitting -- 20K samples is already generous for a linear probe on a
# 64-dim representation, and this machine has recently hit real memory pressure (see
# analyze_transfer_experiment.py's _VARIANCE_MAX_ROWS fix from the same investigation).
N_PROBE_SAMPLES = 20_000
PROBE_SAMPLE_SEED = 123
PROBE_TRAIN_FRACTION = 0.8

OUTPUT_DIR = Path(__file__).resolve().parent / "probe_artifacts"


def _artifact_store() -> Path:
    import os
    return Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()


async def _dataset_artifact_path(dataset_id: int) -> str:
    import database
    from sqlalchemy import select
    from data.models import Dataset

    async with database.async_session_factory() as db:
        dataset = (await db.execute(select(Dataset).where(Dataset.id == dataset_id))).scalar_one()
        return dataset.artifact_path


def _build_model():
    from model_core.architectures import build_model
    return build_model("decoder_only", MODEL_CONFIG, device="cpu")


def _load_checkpoint(model, checkpoint_path: str) -> None:
    import torch

    ckpt_path = _artifact_store() / checkpoint_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)


def _forward_with_hidden_states(model, src) -> list:
    """Replicates DecoderOnlyTransformer.forward() up through each block, returning the pooled
    ("last" position, matching model.pooling) hidden state after EVERY block -- the model's own
    forward() only returns the final head output, with no hook for intermediate layers."""
    import torch

    with torch.no_grad():
        x = model.src_proj(src)
        x = model.pos_enc(x) if model.pooling != "mean" else model.pos_enc.dropout(x)
        hidden_states = []
        for block in model.blocks:
            x = block(x, model.causal_mask)
            pooled = x.mean(dim=1) if model.pooling == "mean" else x[:, -1, :]
            hidden_states.append(pooled.numpy())
    return hidden_states


async def extract() -> None:
    import torch
    from model_core.trainers import OHLCWindowDataset

    OUTPUT_DIR.mkdir(exist_ok=True)

    ds = OHLCWindowDataset(artifact_path=(await _dataset_artifact_path(USDJPY_DATASET_ID)), **PROBE_HP)
    ds.eval()
    n_val = len(ds)
    print(f"Validation set: {n_val} windows total")

    rng = np.random.default_rng(PROBE_SAMPLE_SEED)
    sample_idx = rng.choice(n_val, size=min(N_PROBE_SAMPLES, n_val), replace=False)
    sample_idx.sort()

    src_all, tgt_all = ds[0:n_val]
    src = src_all[sample_idx]
    # tgt window layout is [pred_len+1, n_targets]; index 0 is the teacher-forced "current" value,
    # index 1: is the actual future target (see model_core/trainers/dataset.py's _make_windows) --
    # pred_len=1 here so this is exactly the one-step-ahead value.
    targets = tgt_all[sample_idx][:, 1, :].numpy()  # [n_samples, len(PROBE_TARGET_COLS)]
    print(f"Probing on {len(sample_idx)} sampled windows, targets shape={targets.shape}")

    for label, checkpoint_path in CHECKPOINTS.items():
        print(f"\n=== {label} ===")
        if label == "scratch":
            torch.manual_seed(SCRATCH_INIT_SEED)
        model = _build_model()
        if checkpoint_path is not None:
            _load_checkpoint(model, checkpoint_path)
            print(f"  loaded checkpoint {checkpoint_path}")
        else:
            print(f"  untrained random init (seed={SCRATCH_INIT_SEED})")
        model.eval()

        batch_size = 2000
        n_layers = MODEL_CONFIG["num_layers"]
        layer_reps = [[] for _ in range(n_layers)]
        for start in range(0, len(src), batch_size):
            batch = src[start:start + batch_size]
            hidden_states = _forward_with_hidden_states(model, batch)
            for layer_idx, h in enumerate(hidden_states):
                layer_reps[layer_idx].append(h)
            print(f"  processed {min(start + batch_size, len(src))}/{len(src)}", end="\r")
        print()

        save_dict = {f"layer_{i}": np.concatenate(layer_reps[i], axis=0) for i in range(n_layers)}
        save_dict["targets"] = targets
        save_dict["sample_idx"] = sample_idx
        out_path = OUTPUT_DIR / f"{label}.npz"
        np.savez(out_path, **save_dict)
        print(f"  saved {out_path} (layers 0-{n_layers - 1}, each shape {save_dict['layer_0'].shape})")


def probe() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(PROBE_SAMPLE_SEED)
    results = {}  # (label, layer) -> {target_label: r2}

    labels = list(CHECKPOINTS.keys())
    first = np.load(OUTPUT_DIR / f"{labels[0]}.npz")
    n_samples = first["targets"].shape[0]
    n_layers = sum(1 for k in first.files if k.startswith("layer_"))

    # Same train/test split (by row position within the extracted sample set) reused for every
    # checkpoint/layer/target -- fair comparison, and identical across checkpoints since
    # extract() used the same sample_idx (same PROBE_SAMPLE_SEED) for all of them.
    perm = rng.permutation(n_samples)
    split = int(n_samples * PROBE_TRAIN_FRACTION)
    train_idx, test_idx = perm[:split], perm[split:]

    for label in labels:
        data = np.load(OUTPUT_DIR / f"{label}.npz")
        targets = data["targets"]
        for layer_idx in range(n_layers):
            reps = data[f"layer_{layer_idx}"]
            row = {}
            for col_idx, col in enumerate(PROBE_TARGET_COLS):
                y = targets[:, col_idx]
                valid = ~np.isnan(y) & np.isfinite(y)
                tr = train_idx[valid[train_idx]]
                te = test_idx[valid[test_idx]]
                probe_model = Ridge(alpha=1.0)
                probe_model.fit(reps[tr], y[tr])
                pred = probe_model.predict(reps[te])
                row[PROBE_TARGET_LABELS[col]] = r2_score(y[te], pred)
            results[(label, layer_idx)] = row

    # Print one table per probe target: rows=checkpoint, cols=layer
    for col in PROBE_TARGET_COLS:
        target_label = PROBE_TARGET_LABELS[col]
        print(f"\n=== probe target: {target_label} (R^2 on held-out 20%) ===")
        header = "checkpoint".ljust(14) + "".join(f"layer{i}".rjust(10) for i in range(n_layers))
        print(header)
        for label in labels:
            row_vals = "".join(f"{results[(label, i)][target_label]:.4f}".rjust(10) for i in range(n_layers))
            print(label.ljust(14) + row_vals)

    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if mode == "extract":
        import asyncio
        asyncio.run(extract())
    elif mode == "probe":
        probe()
    else:
        raise SystemExit("usage: probe_representations.py <extract|probe>")


if __name__ == "__main__":
    main()
