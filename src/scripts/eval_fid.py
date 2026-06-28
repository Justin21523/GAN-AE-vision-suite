"""
Compute FID/KID between real validation images and a directory of generated images.

Usage pattern:
- Train a GAN and export generated samples to a folder (e.g., `gen_dir/`).
- Run this script pointing at the same data config and the generated directory.

This uses `src/metrics/fidkid.FIDKID`, which wraps torchmetrics' FID/KID.
"""

from __future__ import annotations

import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.datasets import build_dataset
from src.data.transforms import build_transforms
from src.metrics.fidkid import FIDKID

from PIL import Image


def main():
    """Entrypoint for FID/KID evaluation against a folder of generated images."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str, required=True, help="same data config used in training"
    )
    p.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="directory of generated images (png/jpg)",
    )
    p.add_argument("--max_samples", type=int, default=10000)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # real
    val_ds = build_dataset(cfg, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)
    metric = FIDKID(device=device)
    if not metric.enabled:
        print("torchmetrics not installed; skip FID/KID.")
        return

    with torch.no_grad():
        for b in val_loader:
            metric.update_real(b[0].to(device))

    # fake (from dir)
    # Note: this assumes generated images are saved as standard RGB files and
    # applies the same resize/normalize convention as training (tanh in [-1,1]).
    to_tensor = build_transforms(cfg, train=False)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths += glob.glob(os.path.join(args.gen_dir, ext))
    paths = paths[: args.max_samples]
    for pth in paths:
        img = to_tensor(Image.open(pth).convert("RGB")).unsqueeze(0).to(device)  # type: ignore
        metric.update_fake(img)

    scores = metric.compute()
    print({k: round(v, 5) for k, v in scores.items()})


if __name__ == "__main__":
    main()
