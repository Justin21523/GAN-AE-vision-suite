# src/metrics/image.py
"""
PSNR / SSIM implemented in PyTorch (NCHW, values in [0,1])
"""

import torch
import torch.nn.functional as F


def _ensure_same_shape(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Make y same shape as x by bilinear interpolation if needed.
    Assumes tensors are [B, C, H, W]. Keeps dtype/device of y but aligns size.
    """
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("psnr/ssim expect 4D tensors [B, C, H, W].")
    if x.shape == y.shape:
        return y
    # Interpolate y to x's spatial size; if channel mismatch, try to expand gray to RGB
    if y.shape[1] == 1 and x.shape[1] == 3:
        y = y.repeat(1, 3, 1, 1)
    elif y.shape[1] == 3 and x.shape[1] == 1:
        # convert y to luma as simple average (fallback)
        y = y.mean(dim=1, keepdim=True)
    y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return y


# 高峰噪訊比 Peak Signal-to-Noise Ratio（PSNR）
def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) in dB.
    x, y: [B, C, H, W] in the same scale (e.g., [0,1] or [-1,1] with max_val adjusted).
    Handles minor spatial/channel mismatches by aligning y to x.
    """
    # align dtype/device
    y = y.to(dtype=x.dtype, device=x.device)
    y = _ensure_same_shape(x, y)
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3))  # [B]
    # Avoid log of zero
    mse = torch.clamp(mse, min=1e-12)
    psnr_b = 20.0 * torch.log10(
        torch.tensor(max_val, dtype=x.dtype, device=x.device)
    ) - 10.0 * torch.log10(mse)
    return psnr_b  # [B], per-image PSNR


def _gaussian_window(
    channels: int, window_size: int = 11, sigma: float = 1.5, device=None, dtype=None
):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    w2d = g[:, None] @ g[None, :]
    w2d = w2d / w2d.sum()
    window = w2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


# 結構相似度 Structural Similarity（SSIM）
def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Minimal SSIM (uniform window) for smoke checks; not production-grade.
    Aligns shapes like PSNR. Returns [B].
    """
    y = y.to(dtype=x.dtype, device=x.device)
    y = _ensure_same_shape(x, y)

    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = y.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.squeeze(-1).squeeze(-1).mean(dim=1)  # [B]
