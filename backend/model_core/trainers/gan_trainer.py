"""
GAN training loop — thin wrapper around stocknet/trainer/gantrainer.py logic.

Works for TimeGAN architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dataset import OHLCWindowDataset


def train_epoch(model, ds: OHLCWindowDataset, optimizer: torch.optim.Optimizer, criterion, batch_size: int, shuffle: bool = False) -> float:
    # shuffle: accepted for call-signature uniformity with the supervised trainer
    # (celery_worker.py dispatches to whichever train_epoch generically) — not yet applied here.
    model.train()
    ds.train()

    if not hasattr(model, "_d_optimizer"):
        lr = optimizer.param_groups[0]["lr"]
        model._d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    adversarial_loss = criterion if criterion is not None else nn.BCELoss()
    device = model.device
    d_losses, g_losses = [], []

    for i in range(0, len(ds) - batch_size, batch_size):
        src, tgt = ds[i : i + batch_size]
        src, tgt = src.to(device), tgt[:, :-1, :].to(device)   # tgt shape: [batch, pred_len, features]
        bs = src.size(0)
        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # Discriminator step
        model._d_optimizer.zero_grad()
        real_pred = model.discriminator(src, tgt)
        d_real_loss = adversarial_loss(real_pred, real_labels)
        noise = torch.randn(bs, model.latent_dim, device=device)
        fake_tgt = model.generator(src, noise).detach()
        fake_pred = model.discriminator(src, fake_tgt)
        d_fake_loss = adversarial_loss(fake_pred, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        model._d_optimizer.step()
        d_losses.append(d_loss.item())

        # Generator step
        optimizer.zero_grad()
        noise = torch.randn(bs, model.latent_dim, device=device)
        fake_tgt = model.generator(src, noise)
        fake_pred = model.discriminator(src, fake_tgt)
        g_loss = adversarial_loss(fake_pred, real_labels)
        g_loss.backward()
        optimizer.step()
        g_losses.append(g_loss.item())

    return float(np.mean(g_losses) + np.mean(d_losses)) if g_losses else float("inf")


def eval_epoch(model, ds: OHLCWindowDataset, criterion, batch_size: int) -> float:
    model.eval()
    ds.eval()
    adversarial_loss = criterion if criterion is not None else nn.BCELoss()
    device = model.device
    g_losses = []
    with torch.no_grad():
        for i in range(0, len(ds) - batch_size, batch_size):
            src, _ = ds[i : i + batch_size]
            src = src.to(device)
            bs = src.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            noise = torch.randn(bs, model.latent_dim, device=device)
            fake_tgt = model.generator(src, noise)
            fake_pred = model.discriminator(src, fake_tgt)
            g_loss = adversarial_loss(fake_pred, real_labels)
            g_losses.append(g_loss.item())
    return float(np.mean(g_losses)) if g_losses else float("inf")
