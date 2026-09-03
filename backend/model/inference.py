"""
Inference service — loads a deployed model checkpoint and runs prediction.

Usage:
    result = predict(model_record, features, feature_names)

`features` is a list of rows [[f1, f2, ...], ...] with length >= obs_len.
Returns a list of prediction dicts, one per pred_len step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_loaded_models: dict[int, tuple] = {}  # cache: model_id → (model, config, obs_len, pred_len, normalize)


def _cache_key(model_id: int, artifact_path: str) -> str:
    return f"{model_id}:{artifact_path}"


def predict(
    model_id: int,
    architecture: str,
    model_config: dict,
    artifact_path: str,
    hyperparams: dict,
    features: list[list[float]],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """
    Load the model (cached) and run a single forward pass.

    features: list of rows, length must be >= obs_len
    Returns: list of pred_len dicts with keys matching feature_names + "direction", "step"
    """
    from model_core.architectures import build_model
    from model_core.trainers.arima_trainer import ARIMA_ARCHITECTURES

    if architecture in ARIMA_ARCHITECTURES:
        raise ValueError(
            f"Live inference for {architecture!r} models isn't supported yet — "
            "use the training run's val_loss for baseline comparison."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = _cache_key(model_id, artifact_path)

    if cache_key not in _loaded_models:
        model = build_model(architecture, model_config, device=device)
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        ckpt = torch.load(store / artifact_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _loaded_models[cache_key] = model

    model = _loaded_models[cache_key]

    obs_len: int = hyperparams.get("obs_len", 60)
    pred_len: int = hyperparams.get("pred_len", 10)
    normalize: str = hyperparams.get("normalize", "returns")

    data = np.array(features, dtype=np.float32)[-obs_len:]  # take last obs_len rows
    if len(data) < obs_len:
        raise ValueError(f"Need at least {obs_len} rows of features, got {len(data)}")

    if normalize == "returns":
        data = np.diff(np.log(data + 1e-8), axis=0)
        if len(data) < obs_len:
            data = np.pad(data, ((obs_len - len(data), 0), (0, 0)), mode="edge")
    elif normalize == "minmax":
        mn, mx = data.min(axis=0), data.max(axis=0)
        rng = np.where(mx - mn == 0, 1.0, mx - mn)
        data = (data - mn) / rng

    src = torch.tensor(data[:obs_len], dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_len, features]

    with torch.no_grad():
        if architecture in ("lstm", "cnn_lstm", "tcn", "nbeats"):
            # Single-pass models: tgt is unused
            output = model(src, src[:, -pred_len:, :])  # [1, pred_len, features]
        elif architecture == "timegan":
            # GAN generates synthetic sequences
            noise = torch.randn(1, model.latent_dim, device=device)
            output = model(src, noise)                  # [1, pred_len, features]
        elif architecture == "vae":
            # VAE returns (pred, mu, log_var); use pred only
            output, _mu, _lv = model(src)               # [1, pred_len, features]
        else:
            # Seq2Seq: start decoding from last obs step
            tgt_seed = src[:, -1:, :]                   # [1, 1, features]
            output = model(src, tgt_seed)               # [1, 1..pred_len, features]

    preds = output[0].cpu().numpy()  # [pred_len, features]

    results = []
    close_idx = feature_names.index("close") if "close" in feature_names else 0
    for step, row in enumerate(preds):
        entry: dict[str, Any] = {"step": step + 1}
        for j, name in enumerate(feature_names):
            if j < len(row):
                entry[name] = float(row[j])
        # Direction: sign of the close return prediction
        close_val = float(row[close_idx]) if close_idx < len(row) else 0.0
        entry["direction"] = 1 if close_val > 0 else (-1 if close_val < 0 else 0)
        results.append(entry)

    return results


def evict_cache(model_id: int) -> None:
    """Remove a model from the inference cache (call after re-deploy)."""
    keys_to_remove = [k for k in _loaded_models if k.startswith(f"{model_id}:")]
    for k in keys_to_remove:
        del _loaded_models[k]
