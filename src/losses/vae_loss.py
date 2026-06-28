"""
Classic VAE loss (reconstruction + KL divergence).

This is a simple reference implementation:
- Reconstruction term uses summed MSE.
- KL term matches the standard Normal prior N(0, I).
"""

import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):
    """Compute per-batch VAE loss normalized by batch size."""
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return (recon_loss + kld) / x.size(0)
