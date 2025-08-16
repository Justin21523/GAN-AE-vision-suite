# src/losses/vae_loss.py

import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return (recon_loss + kld) / x.size(0)
