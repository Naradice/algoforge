"""
TimeGAN — adapted from stocknet/stocknet/nets/gan.py.

Generator:     src [batch, obs_len, input_dim]  → fake_tgt [batch, output_len, output_dim]
Discriminator: (src, tgt) → probability [batch, 1]

At inference:  model.forward(src, noise=None) → generated sequences
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeGANGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_len: int,
        output_dim: int,
        num_layers: int = 2,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.device = device

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch = src.size(0)
        _, (h, c) = self.encoder(src)
        if noise is None:
            noise = torch.randn(batch, self.latent_dim, device=src.device)
        z = self.latent_proj(noise).unsqueeze(1).expand(-1, self.output_len, -1)
        out, _ = self.decoder(z, (h, c))
        return self.out(out)


class TimeGANDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, device: str = "cpu", **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([src, tgt], dim=1)
        out, _ = self.lstm(combined)
        return self.head(out[:, -1, :])


class TimeGAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_len: int,
        output_dim: int,
        num_layers: int = 2,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.generator = TimeGANGenerator(input_dim, latent_dim, hidden_dim, output_len, output_dim, num_layers, device)
        self.discriminator = TimeGANDiscriminator(input_dim, hidden_dim, num_layers, device)
        self.to(device)

    def forward(self, src: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        return self.generator(src, noise)

    def parameters(self, recurse: bool = True):
        """Expose only generator parameters to the external optimizer."""
        return self.generator.parameters(recurse=recurse)
