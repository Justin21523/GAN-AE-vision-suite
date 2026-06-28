"""
Differentiable augmentation (DiffAugment) for GAN training.

Lightweight, torchvision-free, and works on GPU tensors.
Policies are comma-separated strings like: "color,translation,cutout".
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _rand_uniform(shape, device, dtype, lo: float, hi: float) -> torch.Tensor:
    return (hi - lo) * torch.rand(shape, device=device, dtype=dtype) + lo


def _color(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x + _rand_uniform((b, 1, 1, 1), x.device, x.dtype, -0.2, 0.2)  # brightness
    x_mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - x_mean) * _rand_uniform((b, 1, 1, 1), x.device, x.dtype, 0.8, 1.2) + x_mean  # contrast
    if c == 3:
        x_mean_c = x.mean(dim=1, keepdim=True)
        x = (x - x_mean_c) * _rand_uniform((b, 1, 1, 1), x.device, x.dtype, 0.8, 1.2) + x_mean_c  # saturation
    return x


def _translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    b, c, h, w = x.shape
    shift_x = int(round(w * ratio))
    shift_y = int(round(h * ratio))
    if shift_x == 0 and shift_y == 0:
        return x
    tx = torch.randint(-shift_x, shift_x + 1, (b,), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, (b,), device=x.device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid_x = grid_x[None, :, :].repeat(b, 1, 1) + tx[:, None, None]
    grid_y = grid_y[None, :, :].repeat(b, 1, 1) + ty[:, None, None]
    grid_x = grid_x.clamp(0, w - 1)
    grid_y = grid_y.clamp(0, h - 1)
    idx = (grid_y * w + grid_x).view(b, 1, h * w).repeat(1, c, 1)
    x_flat = x.view(b, c, h * w)
    out = torch.gather(x_flat, 2, idx).view(b, c, h, w)
    return out


def _cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    b, c, h, w = x.shape
    cut_h = int(round(h * ratio))
    cut_w = int(round(w * ratio))
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    y0 = (cy - cut_h // 2).clamp(0, h)
    y1 = (cy + cut_h // 2).clamp(0, h)
    x0 = (cx - cut_w // 2).clamp(0, w)
    x1 = (cx + cut_w // 2).clamp(0, w)

    mask = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    for i in range(b):
        mask[i, :, int(y0[i]) : int(y1[i]), int(x0[i]) : int(x1[i])] = 0
    return x * mask


def diff_augment(x: torch.Tensor, policy: str, p: float = 1.0) -> torch.Tensor:
    if not policy:
        return x
    if p <= 0:
        return x
    if p < 1.0:
        if torch.rand(()) > p:
            return x

    pol = [s.strip().lower() for s in policy.split(",") if s.strip()]
    out = x
    for name in pol:
        if name == "color":
            out = _color(out)
        elif name in {"translation", "translate"}:
            out = _translation(out)
        elif name == "cutout":
            out = _cutout(out)
        else:
            raise ValueError(f"Unknown DiffAugment policy token: {name}")
    return out

