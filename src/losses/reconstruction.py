"""
Reconstruction losses for AE/VAE-style models.

These are thin wrappers around PyTorch's built-in loss functions to keep the
training code readable and consistent.
"""

import torch.nn.functional as F


def mse_loss(pred, target):
    """Mean-squared error (MSE) reconstruction loss."""
    return F.mse_loss(pred, target)


def l1_loss(pred, target):
    """Mean absolute error (L1) reconstruction loss."""
    return F.l1_loss(pred, target)
