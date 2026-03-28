"""Trainer dispatch — picks supervised or GAN loop based on architecture."""

from __future__ import annotations

import torch.nn as nn

from .supervised import train_epoch as sl_train, eval_epoch as sl_eval
from .gan_trainer import train_epoch as gan_train, eval_epoch as gan_eval
from .dataset import OHLCWindowDataset


def get_trainer_fns(architecture: str):
    if architecture == "timegan":
        return gan_train, gan_eval
    return sl_train, sl_eval