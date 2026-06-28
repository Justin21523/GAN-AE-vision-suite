"""
Sample images from a saved GAN checkpoint.

This script loads the generator weights from a `ckpt_epoch*.pt` saved by
`src/scripts/train_gan.py` and writes a grid image to disk.
"""

from __future__ import annotations

import argparse
import torch

from src.models.gan.factory import build_generator_from_cfg
from src.models.gan.ema import EMA
from src.utils.checkpoint import load_checkpoint
from src.utils.vision import save_image_grid


def main():
    """Entrypoint for sampling a grid of images."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="samples.png")
    p.add_argument("--ema", action="store_true")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = build_generator_from_cfg(cfg).to(device)
    G.load_state_dict(ckpt["G"])

    if args.ema:
        # EMA sampling: prefer EMA shadow weights saved in the checkpoint.
        shadow = ckpt.get("ema_shadow")
        if isinstance(shadow, dict) and shadow:
            ema = EMA(G, decay=float(cfg.get("ema", {}).get("decay", 0.999)))
            ema.load_shadow(shadow, device=torch.device(device))
            ema.apply_shadow()
        else:
            print("Warning: checkpoint has no ema_shadow; sampling with raw generator weights.")

    torch.manual_seed(args.seed)
    z = torch.randn(args.n, cfg["latent_dim"], device=device)
    with torch.no_grad():
        x = G(z)
    save_image_grid(x, args.out, nrow=8, value_range=(-1, 1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
