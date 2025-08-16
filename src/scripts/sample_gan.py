import argparse, os, torch
from torchvision.utils import save_image
import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gan.generator import DCGANGenerator
from src.models.gan.ema import EMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="samples.png")
    p.add_argument("--ema", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = DCGANGenerator(
        cfg["latent_dim"],
        cfg["img_channels"],
        tuple(cfg["g_channels"]),
        cfg["img_size"],
    ).to(device)
    G.load_state_dict(ckpt["G"])

    if args.ema and cfg.get("ema", {}).get("enabled", False):
        ema = EMA(G, decay=cfg["ema"].get("decay", 0.999))
        ema.register()  # shadow will be overwritten by current weights
        ema.apply_shadow()

    torch.manual_seed(args.seed)
    z = torch.randn(args.n, cfg["latent_dim"], device=device)
    with torch.no_grad():
        x = G(z)
    save_image((x + 1) / 2, args.out, nrow=8)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
