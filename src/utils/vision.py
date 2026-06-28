"""
Small vision utilities that avoid hard dependencies on `torchvision`.

Why this exists:
- Some environments may have a torch/torchvision mismatch (import errors).
- The core pipelines in this repo (GAN sampling/API/UI) only need a minimal
  "make grid + save PNG" implementation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image


def make_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """
    Create an image grid tensor from a batch.

    Args:
        images: Tensor of shape (N, C, H, W) in [0, 1].
        nrow: Number of images per row.

    Returns:
        Grid tensor of shape (C, grid_h, grid_w) in [0, 1].
    """
    if images.ndim != 4:
        raise ValueError("Expected images in NCHW format (N, C, H, W).")

    n, c, h, w = images.shape
    if n == 0:
        raise ValueError("Cannot make a grid from an empty batch.")

    nrow = max(1, int(nrow))
    ncol = min(nrow, n)
    nrows = int(math.ceil(n / ncol))

    grid = images.new_zeros((c, nrows * h, ncol * w))
    for idx in range(n):
        row = idx // ncol
        col = idx % ncol
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = images[idx]
    return grid


def to_pil_rgb(img_chw: torch.Tensor) -> Image.Image:
    """
    Convert a CHW tensor in [0,1] to a PIL RGB image.

    Args:
        img_chw: Tensor of shape (C, H, W) in [0, 1].
    """
    if img_chw.ndim != 3:
        raise ValueError("Expected a CHW tensor (C, H, W).")

    x = img_chw.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[0]}.")

    nd = (x * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(nd)


def _normalize_to_01(
    images: torch.Tensor, value_range: Optional[Tuple[float, float]]
) -> torch.Tensor:
    if value_range is None:
        # Heuristic: if negatives exist, assume [-1,1] (tanh); else assume [0,1].
        value_range = (-1.0, 1.0) if images.min().item() < 0 else (0.0, 1.0)

    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        raise ValueError(f"Invalid value_range: {value_range}")

    x = images
    x = (x - lo) / (hi - lo)
    return x.clamp(0, 1)


def save_image_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 8,
    value_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Save a batch of images as a single PNG grid.

    Args:
        images: Tensor (N, C, H, W).
        path: Output path (PNG recommended).
        nrow: Grid columns.
        value_range: Explicit range of `images` (e.g., (-1,1) or (0,1)).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x01 = _normalize_to_01(images, value_range=value_range)
    grid = make_grid(x01, nrow=nrow)
    to_pil_rgb(grid).save(str(path), format="PNG")
