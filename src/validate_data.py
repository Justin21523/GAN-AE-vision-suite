"""
Data/transform/metric validation script.

Purpose:
- Load a dataset from a YAML config
- Fetch one batch
- Save a sample grid to disk
- (Optional) run a tiny "identity-ish" AE forward pass to verify PSNR/SSIM wiring

This is intended as a quick sanity check when setting up a new dataset/config.
"""

from __future__ import annotations

import argparse
import os
import torch
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from torch.utils.data import DataLoader
from src.data.datasets import build_dataset
from src.metrics.image import compute_psnr, compute_ssim
from src.utils.vision import save_image_grid
import torch.nn.functional as F


class IdentityAE(torch.nn.Module):
    """
    A tiny AE-like module that preserves spatial size.

    This is not meant for quality; it is a minimal network to validate:
- the forward pass works
- recon tensors match expected shapes
- PSNR/SSIM code paths run end-to-end
    """

    def __init__(self, channels=1):
        super().__init__()
        self.enc = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dec = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.enc.weight)
        torch.nn.init.zeros_(self.enc.bias)  # type: ignore
        torch.nn.init.zeros_(self.dec.weight)
        torch.nn.init.zeros_(self.dec.bias)  # type: ignore

    def forward(self, x):
        # Keeps HxW the same because padding=1
        return self.dec(F.relu(self.enc(x)))


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_celeba.yaml")
    parser.add_argument(
        "--use_ae", action="store_true", help="Run a quick AE forward for metrics"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build the dataset from config (see `src/data/datasets.py`).
    ds_train = build_dataset(cfg, train=True)
    logger.info(f"Dataset: {cfg['data']['dataset']}, Train size: {len(ds_train)}")

    # sample a small batch
    loader = DataLoader(
        ds_train,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
    )

    # Get one batch and move to device.
    batch = next(iter(loader))
    # batch can be a Tensor, a tuple/list like (images, labels), or a dict
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    elif isinstance(batch, dict):
        # try common keys
        x = batch.get("image", batch.get("images"))
    else:
        x = batch
    x = x.to(device, non_blocking=True)  # type: ignore [B, C, H, W]

    # Save a quick grid for visual inspection.
    os.makedirs(cfg["save"]["out_dir"], exist_ok=True)
    save_image_grid(
        x[:64],
        os.path.join(cfg["save"]["out_dir"], "samples_input.png"),
        nrow=8,
        value_range=(-1, 1),
    )

    xr = x
    if args.use_ae:
        # Run through a tiny network to produce a "reconstruction" tensor.
        ae = IdentityAE(channels=x.shape[1]).to(device)
        xr = ae(x)
        save_image_grid(
            xr[:64],
            os.path.join(cfg["save"]["out_dir"], "samples_recon.png"),
            nrow=8,
            value_range=(-1, 1),
        )

    # Metrics (PSNR/SSIM handle minor shape mismatches internally).
    ps = compute_psnr(x, xr, max_val=1.0).mean().item()
    ss = compute_ssim(x, xr).mean().item()
    logger.info(f"PSNR: {ps:.3f} dB, SSIM: {ss:.4f}")


if __name__ == "__main__":
    main()
