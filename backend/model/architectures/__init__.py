"""
Model factory — builds a model instance from architecture name + config dict.

Supported architectures:
    "lstm"                 → LSTMModel
    "seq2seq_transformer"  → Seq2SeqTransformer
    "decoder_only"         → DecoderOnlyTransformer -- GPT-style causal decoder; use_attention=False
                             swaps self-attention for a fixed-shape learned linear mix, an ablation
                             for isolating attention's contribution (see decoder_only.py)
    "timegan"              → TimeGAN
    "cnn_lstm"             → CNNLSTMModel
    "tcn"                  → TCNModel
    "vae"                  → VAEModel
    "nbeats"               → NBEATSModel
    "rl_agent"             → raises ValueError (handled by ml_worker, Python 3.8)
    "ar" / "ma" / "arma"   → raises ValueError — not torch.nn.Module; fit via statsmodels in
                             celery_worker.py's _run_arima_training, see model/trainers/arima_trainer.py
"""

from __future__ import annotations

import torch

from .lstm import LSTMModel
from .transformer import Seq2SeqTransformer
from .decoder_only import DecoderOnlyTransformer
from .gan import TimeGAN
from .cnn_lstm import CNNLSTMModel
from .tcn import TCNModel
from .vae import VAEModel
from .nbeats import NBEATSModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCHITECTURE_DEFAULTS: dict[str, dict] = {
    "lstm": {
        "input_dim": 1,
        "hidden_dim": 128,
        "output_dim": 1,
        "pred_len": 10,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "seq2seq_transformer": {
        "input_dim": 1,
        "output_dim": 1,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "decoder_only": {
        "input_dim": 1,
        "output_dim": 1,
        "seq_len": 60,
        "pred_len": 10,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        # The ablation switch: True = standard causal self-attention (GPT-style decoder). False =
        # self-attention replaced by CausalLinearMix, a fixed-shape learned linear mixing layer --
        # isolates whether attention's content-dependent weighting specifically matters for a
        # result (e.g. a scaling-law curve), vs. "any learned cross-position mixing at all".
        # Everything else in the block (FFN, LayerNorm, residual, positional encoding) is
        # identical between the two settings -- see model/architectures/decoder_only.py.
        "use_attention": True,
    },
    "timegan": {
        "input_dim": 1,
        "latent_dim": 32,
        "hidden_dim": 64,
        "output_len": 60,
        "output_dim": 1,
        "num_layers": 2,
    },
    "cnn_lstm": {
        "input_dim": 1,
        "output_dim": 1,
        "pred_len": 10,
        "cnn_filters": 64,
        "kernel_size": 3,
        "cnn_layers": 2,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "dropout": 0.2,
    },
    "tcn": {
        "input_dim": 1,
        "output_dim": 1,
        "pred_len": 10,
        "num_channels": 64,
        "num_levels": 4,
        "kernel_size": 3,
        "dropout": 0.2,
    },
    "vae": {
        "input_dim": 1,
        "output_dim": 1,
        "pred_len": 10,
        "latent_dim": 32,
        "encoder_hidden": 128,
        "decoder_hidden": 128,
        "encoder_layers": 2,
        "dropout": 0.1,
    },
    "nbeats": {
        "input_dim": 1,
        "output_dim": 1,
        "obs_len": 60,
        "pred_len": 10,
        "hidden_units": 256,
        "nb_blocks": 3,
        "theta_dim": 64,
    },
    # ar/ma/arma are statsmodels ARIMA fits, not torch models — build_model() rejects them
    # below. Defaults kept here so this stays the single source of truth per architecture;
    # model/trainers/arima_trainer.py's order_from_config() merges these with MLModel.config.
    "ar": {"p": 2, "d": 0},
    "ma": {"q": 2, "d": 0},
    "arma": {"p": 2, "d": 0, "q": 2},
}

TRAINING_DEFAULTS: dict[str, dict] = {
    "lstm": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "seq2seq_transformer": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "decoder_only": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "timegan": {"obs_len": 60, "pred_len": 60, "epochs": 100, "batch_size": 32, "lr": 0.0002, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "cnn_lstm": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "tcn": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "vae": {"obs_len": 60, "pred_len": 10, "epochs": 80, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "nbeats": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    # No epochs/batch_size/lr — fitting is a single MLE optimization, not gradient descent.
    "ar":   {"pred_len": 10, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "ma":   {"pred_len": 10, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "arma": {"pred_len": 10, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
}


def build_model(architecture: str, config: dict, device: str = _DEVICE) -> torch.nn.Module:
    arch = architecture.lower()
    defaults = ARCHITECTURE_DEFAULTS.get(arch, {})
    merged = {**defaults, **config}

    if arch == "lstm":
        return LSTMModel(**merged, device=device)
    elif arch == "seq2seq_transformer":
        return Seq2SeqTransformer(**merged, device=device)
    elif arch == "decoder_only":
        return DecoderOnlyTransformer(**merged, device=device)
    elif arch == "timegan":
        return TimeGAN(**merged, device=device)
    elif arch == "cnn_lstm":
        return CNNLSTMModel(**merged, device=device)
    elif arch == "tcn":
        return TCNModel(**merged, device=device)
    elif arch == "vae":
        return VAEModel(**merged, device=device)
    elif arch == "nbeats":
        return NBEATSModel(**merged, device=device)
    elif arch == "rl_agent":
        raise ValueError("rl_agent training must be submitted to ml_worker (Python 3.8 container)")
    elif arch in ("ar", "ma", "arma"):
        raise ValueError(
            f"{arch!r} is fit via statsmodels, not build_model() — see "
            "celery_worker.py's _run_arima_training / model/trainers/arima_trainer.py"
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")