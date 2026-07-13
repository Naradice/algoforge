"""
Post-training validation metrics for ML models.

For supervised models (LSTM, Transformer):
    - directional_accuracy: % of correct direction predictions
    - mae, mse, rmse: error on normalised returns
    - sharpe_proxy: mean / std of returns implied by predictions

For GAN (TimeGAN):
    - acf_match: correlation between ACF of real vs generated returns
    - hurst_diff: |hurst(real) - hurst(generated)|
    - kurtosis_real, kurtosis_generated: fat-tails comparison
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.architectures import build_model
from model.trainers.dataset import OHLCWindowDataset


def _hurst(r: np.ndarray) -> float:
    max_lag = min(50, len(r) // 4)
    if max_lag < 2:
        return float("nan")
    lags = range(2, max_lag)
    tau = [float(np.std(r[lg:] - r[:-lg])) for lg in lags]
    if any(t <= 0 for t in tau):
        return float("nan")
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])


def validate_supervised(
    model: torch.nn.Module,
    ds: OHLCWindowDataset,
    batch_size: int = 64,
) -> dict:
    model.eval()
    ds.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(ds) - batch_size, batch_size):
            src, tgt = ds[i : i + batch_size]
            input_tgt = tgt[:, :-1, :]
            pred = model(src, input_tgt)         # [batch, pred_len, features]
            output_tgt = tgt[:, 1:, :]
            preds.append(pred.cpu().numpy())
            targets.append(output_tgt.cpu().numpy())

    if not preds:
        return {"error": "no_data"}

    p = np.concatenate(preds, axis=0)    # [N, pred_len, features]
    t = np.concatenate(targets, axis=0)

    # Directional accuracy: sign of last-step prediction vs target
    p_dir = np.sign(p[:, -1, 0])
    t_dir = np.sign(t[:, -1, 0])
    dir_acc = float(np.mean(p_dir == t_dir))

    mae = float(np.mean(np.abs(p - t)))
    mse = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(mse))

    implied_returns = p[:, -1, 0]
    sharpe_proxy = float(implied_returns.mean() / (implied_returns.std() + 1e-8))

    return {
        "directional_accuracy": dir_acc,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "sharpe_proxy": sharpe_proxy,
    }


def validate_gan(
    model: torch.nn.Module,
    ds: OHLCWindowDataset,
    batch_size: int = 64,
    n_samples: int = 1000,
) -> dict:
    from statsmodels.tsa.stattools import acf

    model.eval()
    ds.eval()

    # Sample real returns
    real_returns = []
    with torch.no_grad():
        for i in range(0, min(len(ds), n_samples), batch_size):
            src, tgt = ds[i : i + batch_size]
            real_returns.append(tgt[:, :-1, 0].cpu().numpy().flatten())
    real_r = np.concatenate(real_returns) if real_returns else np.array([])

    # Generate fake sequences
    fake_returns = []
    with torch.no_grad():
        for i in range(0, min(len(ds), n_samples), batch_size):
            src, _ = ds[i : i + batch_size]
            src = src.to(model.device)
            noise = torch.randn(src.size(0), model.latent_dim, device=model.device)
            fake = model.generator(src, noise)   # [batch, pred_len, features]
            fake_returns.append(fake[:, :, 0].cpu().numpy().flatten())
    fake_r = np.concatenate(fake_returns) if fake_returns else np.array([])

    if len(real_r) < 10 or len(fake_r) < 10:
        return {"error": "insufficient_data"}

    # ACF match (first 20 lags)
    n_lags = min(20, len(real_r) // 4 - 1, len(fake_r) // 4 - 1)
    real_acf = acf(real_r, nlags=n_lags, fft=True)[1:]
    fake_acf = acf(fake_r, nlags=n_lags, fft=True)[1:]
    acf_match = float(np.corrcoef(real_acf, fake_acf)[0, 1]) if n_lags > 1 else float("nan")

    return {
        "acf_match": acf_match,
        "hurst_real": _hurst(real_r),
        "hurst_generated": _hurst(fake_r),
        "hurst_diff": abs(_hurst(real_r) - _hurst(fake_r)),
        "kurtosis_real": float(pd.Series(real_r).kurtosis()),
        "kurtosis_generated": float(pd.Series(fake_r).kurtosis()),
        "std_real": float(np.std(real_r)),
        "std_generated": float(np.std(fake_r)),
    }


def run_validation(
    artifact_path: str,
    architecture: str,
    model_config: dict,
    hyperparams: dict,
    dataset_artifact_path: str,
) -> dict:
    """Load a checkpoint and run the appropriate validation. Returns metrics dict."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(architecture, model_config, device=device)

    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    checkpoint = torch.load(store / artifact_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    hp = hyperparams
    ds = OHLCWindowDataset(
        dataset_artifact_path,
        obs_len=hp.get("obs_len", 60),
        pred_len=hp.get("pred_len", 10),
        feature_cols=hp.get("feature_cols", ["close"]),
        normalize=hp.get("normalize", "returns"),
        val_split=hp.get("val_split", 0.2),
        device=device,
        preprocessing=hp.get("preprocessing"),
    )

    if architecture == "timegan":
        return validate_gan(model, ds, batch_size=hp.get("batch_size", 64))
    else:
        return validate_supervised(model, ds, batch_size=hp.get("batch_size", 64))
