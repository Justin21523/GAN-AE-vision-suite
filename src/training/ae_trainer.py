"""
Deprecated training helper for Autoencoders (AE) and Variational Autoencoders (VAE).

This file is kept for backward-compatibility with older notebooks/snippets.
Prefer the canonical, config-driven trainer:
  `python -m src.scripts.train_ae --config <yaml>`
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from src.metrics.image import compute_psnr, compute_ssim


class AETrainer:
    """Lightweight trainer wrapper used by some scripts/notebooks."""

    def __init__(self, model, optimizer, device):
        warnings.warn(
            "`src.training.ae_trainer.AETrainer` is deprecated; prefer "
            "`python -m src.scripts.train_ae`.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def loss_ae(self, recon_x, x):
        """AE reconstruction loss (MSE)."""
        return F.mse_loss(recon_x, x)

    def loss_vae(self, recon_x, x, mu, logvar):
        """VAE loss split into (total, recon, kl) components."""
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kld_loss, recon_loss, kld_loss

    def train_epoch(self, loader, model_type="ae"):
        """
        Train for one epoch.

        Returns:
            (avg_loss, avg_psnr, avg_ssim). Note that PSNR/SSIM are returned as
            tensors if the underlying metric functions return per-image tensors.
        """
        self.model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for x, _ in loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()

            if model_type == "ae":
                recon = self.model(x)
                loss = self.loss_ae(recon, x)
            else:
                recon, mu, logvar = self.model(x)
                loss, _, _ = self.loss_vae(recon, x, mu, logvar)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_psnr += compute_psnr(recon, x)
            total_ssim += compute_ssim(recon, x)

        return (
            total_loss / len(loader),
            total_psnr / len(loader),
            total_ssim / len(loader),
        )

    def eval_epoch(self, loader, model_type="ae"):
        """Evaluate for one epoch (no gradients)."""
        self.model.eval()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                if model_type == "ae":
                    recon = self.model(x)
                    loss = self.loss_ae(recon, x)
                else:
                    recon, mu, logvar = self.model(x)
                    loss, _, _ = self.loss_vae(recon, x, mu, logvar)

                total_loss += loss.item()
                total_psnr += compute_psnr(recon, x)
                total_ssim += compute_ssim(recon, x)

        return (
            total_loss / len(loader),
            total_psnr / len(loader),
            total_ssim / len(loader),
        )
