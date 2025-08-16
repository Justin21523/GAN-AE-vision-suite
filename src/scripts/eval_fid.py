import argparse, os, torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.datasets import build_dataset
from src.data.transforms import build_transforms
from src.metrics.fidkid import FIDKID

from PIL import Image
import glob
import torchvision.transforms as T


def main():
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
    to_tensor = T.Compose(
        [
            T.Resize(cfg["data"]["img_size"]),
            T.CenterCrop(cfg["data"]["img_size"]),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
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
