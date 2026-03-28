"""
Model factory — builds a model instance from architecture name + config dict.

Supported architectures:
    "lstm"                 → LSTMModel
    "seq2seq_transformer"  → Seq2SeqTransformer
    "timegan"              → TimeGAN
    "rl_agent"             → raises ValueError (handled by ml_worker, Python 3.8)
"""

from __future__ import annotations

import torch

from .lstm import LSTMModel
from .transformer import Seq2SeqTransformer
from .gan import TimeGAN

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
    "timegan": {
        "input_dim": 1,
        "latent_dim": 32,
        "hidden_dim": 64,
        "output_len": 60,
        "output_dim": 1,
        "num_layers": 2,
    },
}

TRAINING_DEFAULTS: dict[str, dict] = {
    "lstm": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "seq2seq_transformer": {"obs_len": 60, "pred_len": 10, "epochs": 50, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
    "timegan": {"obs_len": 60, "pred_len": 60, "epochs": 100, "batch_size": 32, "lr": 0.0002, "val_split": 0.2, "feature_cols": ["close"], "normalize": "returns"},
}


def build_model(architecture: str, config: dict, device: str = _DEVICE) -> torch.nn.Module:
    if architecture == "lstm":
        return LSTMModel(**{**ARCHITECTURE_DEFAULTS["lstm"], **config}, device=device)
    elif architecture == "seq2seq_transformer":
        return Seq2SeqTransformer(**{**ARCHITECTURE_DEFAULTS["seq2seq_transformer"], **config}, device=device)
    elif architecture == "timegan":
        return TimeGAN(**{**ARCHITECTURE_DEFAULTS["timegan"], **config}, device=device)
    elif architecture == "rl_agent":
        raise ValueError("rl_agent training must be submitted to ml_worker (Python 3.8 container)")
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")