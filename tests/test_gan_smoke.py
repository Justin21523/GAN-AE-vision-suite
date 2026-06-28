"""
GAN module smoke tests.

These tests validate basic tensor shapes and a non-negative gradient penalty value.
They are designed to be fast and CPU-friendly.
"""

from __future__ import annotations

import torch
import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gan.generator import DCGANGenerator
from src.models.gan.discriminator import DCGANDiscriminator
from src.losses.gan import compute_gradient_penalty


def test_shapes():
    G = DCGANGenerator(
        latent_dim=16, img_channels=3, channels=(128, 64, 32, 16, 8), img_size=128
    )
    D = DCGANDiscriminator(img_channels=3, channels=(8, 16, 32, 64, 128), img_size=128)
    z = torch.randn(4, 16)
    x = G(z)
    assert x.shape == (4, 3, 128, 128)
    y = D(x)
    assert y.shape == (4,)


def test_gp():
    D = DCGANDiscriminator(img_channels=3, channels=(8, 16, 32, 64, 128), img_size=128)
    real = torch.randn(2, 3, 128, 128)
    fake = torch.randn(2, 3, 128, 128)
    gp = compute_gradient_penalty(D, real, fake, device="cpu")
    assert gp >= 0.0
