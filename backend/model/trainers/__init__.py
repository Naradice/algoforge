"""Trainer dispatch — picks supervised, GAN, or VAE loop based on architecture."""

from __future__ import annotations

import torch.nn as nn

from .supervised import train_epoch as sl_train, eval_epoch as sl_eval, train_steps as sl_train_steps
from .gan_trainer import train_epoch as gan_train, eval_epoch as gan_eval
from .vae_trainer import train_epoch as vae_train, eval_epoch as vae_eval
from .dataset import OHLCWindowDataset, compute_effective_characteristics
from .arima_trainer import ARIMA_ARCHITECTURES, order_from_config, load_series_for_arima, fit_and_evaluate_arima


def get_trainer_fns(architecture: str):
    if architecture == "timegan":
        return gan_train, gan_eval
    if architecture == "vae":
        return vae_train, vae_eval
    return sl_train, sl_eval


def get_step_trainer_fn(architecture: str):
    """Step-based training (see supervised.train_steps) is only implemented for the supervised
    trainer -- gan/vae have their own multi-part loss structure that doesn't decompose into an
    infinite single-batch stream the same way. Callers must check architecture before opting a
    run into max_steps mode."""
    if architecture in ("timegan", "vae"):
        raise ValueError(f"Step-based training (max_steps) isn't supported for architecture={architecture!r}")
    return sl_train_steps
