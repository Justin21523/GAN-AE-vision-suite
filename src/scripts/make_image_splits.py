"""
Create deterministic train/val split lists for an image-folder dataset.

Outputs plain text files compatible with `src.data.datasets.ImageListDataset`:
one image path per line, relative to the dataset root.

Example (fixed 5k val images):
  python -m src.scripts.make_image_splits \\
    --root ./data/anime_faces/images \\
    --train-out ./configs/splits/anime_faces_train.txt \\
    --val-out ./configs/splits/anime_faces_val.txt \\
    --val-count 5000 \\
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _collect_images(root: Path) -> list[Path]:
    exts = ("jpg", "jpeg", "png", "webp", "bmp", "tiff")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(root.rglob(f"*.{ext}"))
        paths.extend(root.rglob(f"*.{ext.upper()}"))
    paths = [p for p in paths if p.is_file()]
    paths.sort()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Image root directory.")
    ap.add_argument("--train-out", type=str, required=True)
    ap.add_argument("--val-out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--val-count",
        type=int,
        default=0,
        help="If >0, place exactly N images into val split.",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="If >0 and --val-count is 0, use this ratio for val split.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    imgs = _collect_images(root)
    if not imgs:
        raise SystemExit(f"No images found under: {root}")

    n = len(imgs)
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n)

    if int(args.val_count) > 0:
        n_val = min(int(args.val_count), n - 1)
    else:
        r = float(args.val_ratio)
        n_val = int(round(n * r)) if r > 0 else 0
        n_val = max(1, min(n_val, n - 1)) if n_val > 0 else 0

    val_idx = set(map(int, perm[:n_val])) if n_val > 0 else set()

    train_lines: list[str] = []
    val_lines: list[str] = []
    for i, p in enumerate(imgs):
        rel = p.relative_to(root).as_posix()
        if i in val_idx:
            val_lines.append(rel)
        else:
            train_lines.append(rel)

    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_out.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"Root: {root}")
    print(f"Train: {train_out} ({len(train_lines)} images)")
    print(f"Val:   {val_out} ({len(val_lines)} images)")


if __name__ == "__main__":
    main()
