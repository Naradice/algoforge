"""
VAE training loop — ELBO loss (reconstruction MSE + β·KL divergence).

Works for the VAEModel architecture where forward() returns (pred, mu, log_var).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import OHLCWindowDataset

_BETA = 0.001  # KL weight; small β keeps reconstruction quality high


def _split_tgt(tgt: torch.Tensor):
    return tgt[:, :-1, :], tgt[:, 1:, :]


def train_epoch(model, ds: OHLCWindowDataset, optimizer: torch.optim.Optimizer, criterion, batch_size: int, shuffle: bool = False) -> float:
    # shuffle: accepted for call-signature uniformity with the supervised trainer
    # (celery_worker.py dispatches to whichever train_epoch generically) — not yet applied here.
    model.train()
    ds.train()
    losses = []

    for i in range(0, len(ds) - batch_size, batch_size):
        src, tgt = ds[i : i + batch_size]
        input_tgt, output_tgt = _split_tgt(tgt)

        pred, mu, log_var = model(src, input_tgt)
        recon_loss = F.mse_loss(pred, output_tgt)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + _BETA * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("inf")


def eval_epoch(model, ds: OHLCWindowDataset, criterion, batch_size: int) -> float:
    model.eval()
    ds.eval()
    losses = []

    with torch.no_grad():
        for i in range(0, len(ds) - batch_size, batch_size):
            src, tgt = ds[i : i + batch_size]
            input_tgt, output_tgt = _split_tgt(tgt)

            pred, mu, log_var = model(src, input_tgt)
            recon_loss = F.mse_loss(pred, output_tgt)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + _BETA * kl_loss
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("inf")
