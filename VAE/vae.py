"""MNIST VAE core extracted from the original train.py.

Includes the encoder, reparameterization, decoder, and BCE + beta * KL loss.
The decoder returns logits; apply torch.sigmoid for reconstructed images.
Loss values sum over pixels/latent dimensions and average over the batch.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )
        self.mu = nn.Linear(200, latent_dim)
        self.logvar = nn.Linear(200, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_terms(
    logits: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float
) -> Dict[str, Tensor]:
    # Sum over pixels, then mean over batch: ELBO units are nats / image.
    reconstruction = nn.functional.binary_cross_entropy_with_logits(
        logits, x, reduction="none"
    ).flatten(1).sum(1).mean()
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl = kl_per_dim.sum(1).mean()
    return {
        "reconstruction_loss": reconstruction,
        "kl": kl,
        "beta_kl": beta * kl,
        "loss": reconstruction + beta * kl,
        "kl_per_dim": kl_per_dim.mean(0),
    }
