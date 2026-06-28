"""
ResNet-style GAN blocks (SNGAN/BigGAN-ish) for stronger image generation than DCGAN.

This module provides:
- `ResNetGenerator`: residual upsampling generator with BatchNorm
- `ResNetDiscriminator`: residual downsampling discriminator with SpectralNorm

Both models output/expect images in [-1, 1] (tanh-normalized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn(module: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


class GBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # GroupNorm is more stable than BatchNorm for small per-GPU batches (common at 256/512px).
        self.n1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.n2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.conv_sc = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(x), inplace=False)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.n2(h), inplace=False)
        h = self.conv2(h)

        sc = F.interpolate(x, scale_factor=2, mode="nearest")
        sc = self.conv_sc(sc)
        return h + sc


class DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sn: bool = True):
        super().__init__()
        self.conv1 = _sn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), sn)
        self.conv2 = _sn(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), sn)
        self.conv_sc = _sn(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0), sn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x, inplace=False)
        h = self.conv1(h)
        h = F.relu(h, inplace=False)
        h = self.conv2(h)
        h = F.avg_pool2d(h, kernel_size=2)

        sc = F.avg_pool2d(x, kernel_size=2)
        sc = self.conv_sc(sc)
        return h + sc


@dataclass
class ResNetSpec:
    img_size: int = 128
    img_channels: int = 3
    latent_dim: int = 128
    base_channels: int = 64
    sn: bool = True


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        img_size: int = 128,
        base_channels: int = 64,
        channels: Sequence[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.img_channels = int(img_channels)
        self.img_size = int(img_size)
        self.base_channels = int(base_channels)

        if self.img_size not in {32, 64, 128, 256, 512}:
            raise ValueError("ResNetGenerator supports img_size in {32, 64, 128, 256, 512}.")

        if channels is None:
            # 4->8->16->32->64->128
            channels = {
                32: (base_channels * 8, base_channels * 4, base_channels * 2),
                64: (base_channels * 8, base_channels * 4, base_channels * 2, base_channels),
                128: (
                    base_channels * 16,
                    base_channels * 8,
                    base_channels * 4,
                    base_channels * 2,
                    base_channels,
                ),
                256: (
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 8,
                    base_channels * 4,
                    base_channels * 2,
                    base_channels,
                ),
                512: (
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 8,
                    base_channels * 4,
                    base_channels * 2,
                    base_channels,
                    base_channels,
                ),
            }[self.img_size]

        self.ch = list(map(int, channels))
        self.fc = nn.Linear(self.latent_dim, self.ch[0] * 4 * 4)
        blocks = []
        for a, b in zip(self.ch[:-1], self.ch[1:]):
            blocks.append(GBlock(a, b))
        self.blocks = nn.Sequential(*blocks)
        self.n_out = nn.GroupNorm(num_groups=min(32, self.ch[-1]), num_channels=self.ch[-1])
        self.conv_out = nn.Conv2d(self.ch[-1], self.img_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)  # type: ignore[arg-type]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z shape (B,{self.latent_dim}), got {tuple(z.shape)}")
        h = self.fc(z).view(z.size(0), self.ch[0], 4, 4)
        h = self.blocks(h)
        h = F.relu(self.n_out(h), inplace=False)
        out = torch.tanh(self.conv_out(h))
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 128,
        base_channels: int = 64,
        channels: Sequence[int] | None = None,
        sn: bool = True,
    ):
        super().__init__()
        self.img_channels = int(img_channels)
        self.img_size = int(img_size)
        self.base_channels = int(base_channels)
        self.sn = bool(sn)

        if self.img_size not in {32, 64, 128, 256, 512}:
            raise ValueError("ResNetDiscriminator supports img_size in {32, 64, 128, 256, 512}.")

        if channels is None:
            channels = {
                32: (base_channels, base_channels * 2, base_channels * 4),
                64: (base_channels, base_channels * 2, base_channels * 4, base_channels * 8),
                128: (
                    base_channels,
                    base_channels * 2,
                    base_channels * 4,
                    base_channels * 8,
                    base_channels * 16,
                ),
                256: (
                    base_channels,
                    base_channels * 2,
                    base_channels * 4,
                    base_channels * 8,
                    base_channels * 16,
                    base_channels * 16,
                ),
                512: (
                    base_channels,
                    base_channels * 2,
                    base_channels * 4,
                    base_channels * 8,
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 16,
                ),
            }[self.img_size]

        self.ch = list(map(int, channels))
        self.conv_in = _sn(
            nn.Conv2d(self.img_channels, self.ch[0], kernel_size=3, padding=1),
            self.sn,
        )
        blocks = []
        for a, b in zip(self.ch[:-1], self.ch[1:]):
            blocks.append(DBlock(a, b, sn=self.sn))
        self.blocks = nn.Sequential(*blocks)
        self.linear = _sn(nn.Linear(self.ch[-1], 1), self.sn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.blocks(h)
        h = F.relu(h, inplace=False)
        h = h.sum(dim=(2, 3))  # global sum pooling
        out = self.linear(h).view(x.size(0))
        return out
