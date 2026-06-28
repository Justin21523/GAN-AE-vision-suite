"""
Create tiny deterministic demo artifacts for screenshots.

This writes only to user-provided output directories (default: `./outputs/demo`).
The generated files are intentionally small and are not meant to represent model
quality; they provide a stable portfolio/demo state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_grid(path: Path, seed: int, n: int = 16, tile: int = 64) -> None:
    rng = np.random.default_rng(int(seed))
    cols = 4
    rows = int(np.ceil(n / cols))
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for idx in range(n):
        base = rng.integers(40, 220, size=3, dtype=np.uint8)
        arr = np.zeros((tile, tile, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:tile, :tile]
        arr[..., 0] = (base[0] + xx * 2) % 255
        arr[..., 1] = (base[1] + yy * 2) % 255
        arr[..., 2] = (base[2] + (xx + yy)) % 255
        cx, cy = tile // 2, tile // 2
        mask = ((xx - cx) ** 2 / (tile * 0.24) ** 2 + (yy - cy) ** 2 / (tile * 0.30) ** 2) < 1
        arr[mask] = (arr[mask] * 0.45 + 35).astype(np.uint8)
        canvas.paste(Image.fromarray(arr), ((idx % cols) * tile, (idx // cols) * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./outputs/demo")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_grid(out / "gan_samples.png", seed=42)
    _write_grid(out / "ae_recon_grid.png", seed=7)

    metrics = [
        {"epoch": 1, "train_loss": 0.4217, "val_loss": 0.3574, "val_psnr": 10.49, "val_ssim": 0.0103},
        {"epoch": 2, "train_loss": 0.3021, "val_loss": 0.2440, "val_psnr": 12.38, "val_ssim": 0.0480},
        {"epoch": 3, "train_loss": 0.2410, "val_loss": 0.1982, "val_psnr": 13.91, "val_ssim": 0.0820},
    ]
    (out / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps({"script": "demo_assets", "created_for": "portfolio walkthrough"}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote demo artifacts to {out}")


if __name__ == "__main__":
    main()
