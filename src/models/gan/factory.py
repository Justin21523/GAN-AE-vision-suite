"""
GAN model factory helpers shared by sampling/evaluation entrypoints.
"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.gan.generator import DCGANGenerator
from src.models.gan.resnet import ResNetGenerator


def build_generator_from_cfg(mcfg: Mapping[str, Any]) -> nn.Module:
    """Build the generator architecture described by `cfg["model"]`."""
    arch = str(mcfg.get("arch", "dcgan")).lower()
    if arch in {"resnet", "sngan", "biggan"}:
        return ResNetGenerator(
            latent_dim=int(mcfg["latent_dim"]),
            img_channels=int(mcfg["img_channels"]),
            img_size=int(mcfg["img_size"]),
            base_channels=int(mcfg.get("base_channels", 64)),
        )
    return DCGANGenerator(
        latent_dim=int(mcfg["latent_dim"]),
        img_channels=int(mcfg["img_channels"]),
        channels=tuple(mcfg["g_channels"]),
        img_size=int(mcfg["img_size"]),
    )
