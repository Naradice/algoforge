"""
Sequence Variational Autoencoder (Seq-VAE).

Encoder:  LSTM processes the observation window → (mu, log_var) in latent space.
Decoder:  Projects z back to a hidden sequence of pred_len steps → future values.

At training time forward() returns (prediction, mu, log_var) so the trainer can
compute the ELBO loss (reconstruction MSE + β·KL divergence).
At inference time only `prediction` is used.

Input:  src [batch, obs_len, input_dim]
Output: pred [batch, pred_len, output_dim], mu, log_var
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 10,
        latent_dim: int = 32,
        encoder_hidden: int = 128,
        decoder_hidden: int = 128,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.device = device

        # Encoder: LSTM → mu / log_var
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )
        self.mu_proj = nn.Linear(encoder_hidden, latent_dim)
        self.logvar_proj = nn.Linear(encoder_hidden, latent_dim)

        # Decoder: z → sequence of pred_len future steps
        self.z_proj = nn.Linear(latent_dim, decoder_hidden)
        self.decoder = nn.LSTM(input_size=decoder_hidden, hidden_size=decoder_hidden, batch_first=True)
        self.out_proj = nn.Linear(decoder_hidden, output_dim)

        self.to(device)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        src: [batch, obs_len, input_dim]
        Returns: (pred [batch, pred_len, output_dim], mu, log_var)
        """
        _, (h, _) = self.encoder(src)          # h: [num_layers, batch, encoder_hidden]
        h_last = h[-1]                          # [batch, encoder_hidden]
        mu = self.mu_proj(h_last)               # [batch, latent_dim]
        log_var = self.logvar_proj(h_last)      # [batch, latent_dim]
        z = self._reparameterize(mu, log_var)   # [batch, latent_dim]

        # Expand z over pred_len steps and decode
        z_seq = self.z_proj(z).unsqueeze(1).expand(-1, self.pred_len, -1)  # [batch, pred_len, decoder_hidden]
        dec_out, _ = self.decoder(z_seq)                                    # [batch, pred_len, decoder_hidden]
        pred = self.out_proj(dec_out)                                       # [batch, pred_len, output_dim]

        return pred, mu, log_var
