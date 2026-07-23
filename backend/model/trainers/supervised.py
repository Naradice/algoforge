"""
Supervised training loop — adapts stocknet/trainer/sltrainer.py seq2seq_train/eval.

Works for LSTM and Seq2SeqTransformer architectures.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dataset import OHLCWindowDataset


def _split_tgt(tgt: torch.Tensor):
    """Teacher-forced split: input_tgt = tgt[:, :-1], output_tgt = tgt[:, 1:]"""
    return tgt[:, :-1, :], tgt[:, 1:, :]


def train_epoch(
    model: nn.Module,
    ds: OHLCWindowDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    shuffle: bool = False,
) -> float:
    model.train()
    ds.train()
    losses = []
    n = len(ds)
    # Windows are highly overlapping/redundant for small or strongly periodic datasets, and
    # without shuffling every epoch walks them in the exact same order — for a small dataset
    # trained many epochs, that's a perfectly periodic gradient sequence that can lock Adam's
    # momentum/adaptive-LR state into a plateau tied to that specific batch order rather than
    # converging toward the true optimum. Shuffling breaks that periodicity.
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n - batch_size, batch_size):
        batch_idx = order[i : i + batch_size]
        src, tgt = ds[batch_idx]
        input_tgt, output_tgt = _split_tgt(tgt)
        logits = model(src, input_tgt)
        loss = criterion(logits, output_tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def train_steps(
    model: nn.Module,
    ds: OHLCWindowDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    n_steps: int,
) -> float:
    """Runs exactly n_steps gradient updates over an infinite, reshuffled stream of training
    windows -- unlike train_epoch, there is no "one pass through the data" boundary here, so
    nothing (validation, early stopping, LR schedule) can be keyed to it. Exists to test whether
    an epoch-length-dependent effect is really about epoch structure at all, by removing the
    concept of an epoch from the training loop entirely (see docs/model-layer.md, "Baseline
    Models" / mechanism-hunt notes)."""
    model.train()
    ds.train()
    n = len(ds)
    losses = []
    order = np.random.permutation(n)
    pos = 0
    for _ in range(n_steps):
        if pos + batch_size > n:
            order = np.random.permutation(n)
            pos = 0
        batch_idx = order[pos : pos + batch_size]
        pos += batch_size
        src, tgt = ds[batch_idx]
        input_tgt, output_tgt = _split_tgt(tgt)
        logits = model(src, input_tgt)
        loss = criterion(logits, output_tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def eval_epoch(model: nn.Module, ds: OHLCWindowDataset, criterion: nn.Module, batch_size: int) -> float:
    model.eval()
    ds.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(ds) - batch_size, batch_size):
            src, tgt = ds[i : i + batch_size]
            input_tgt, output_tgt = _split_tgt(tgt)
            logits = model(src, input_tgt)
            loss = criterion(logits, output_tgt)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")
