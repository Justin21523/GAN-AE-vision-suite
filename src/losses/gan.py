"""
GAN loss utilities.

This module provides:
- Hinge losses (common for DCGAN-like training without sigmoid)
- BCE-with-logits losses (classic GAN objective)
- WGAN-GP gradient penalty

All losses operate on discriminator/critic outputs (logits/scores).
"""

import torch
import torch.nn.functional as F
from torch import nn


def d_hinge_loss(real_logits, fake_logits):
    """Hinge loss for the discriminator."""
    # max(0, 1 - D(real)) + max(0, 1 + D(fake))
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    """Hinge loss for the generator."""
    # -D(fake)
    return -fake_logits.mean()


def d_bce_loss(real_logits, fake_logits):
    """BCE-with-logits loss for the discriminator (targets: real=1, fake=0)."""
    bce = nn.BCEWithLogitsLoss()
    real_loss = bce(real_logits, torch.ones_like(real_logits))
    fake_loss = bce(fake_logits, torch.zeros_like(fake_logits))
    return real_loss + fake_loss


def g_bce_loss(fake_logits):
    """BCE-with-logits loss for the generator (targets: fake=1)."""
    bce = nn.BCEWithLogitsLoss()
    return bce(fake_logits, torch.ones_like(fake_logits))


def compute_gradient_penalty(discriminator, real, fake, device="cuda"):
    """
    Compute WGAN-GP gradient penalty term.

    WGAN-GP enforces a 1-Lipschitz constraint by penalizing the gradient norm of
    the critic on random interpolations between real and fake samples.
    """
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates.requires_grad_(True)
    d_inter = discriminator(interpolates)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
