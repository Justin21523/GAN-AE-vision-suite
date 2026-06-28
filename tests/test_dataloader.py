"""
Dataloader smoke tests (self-contained).

These tests avoid external dataset downloads by creating a temporary image folder
and verifying that the repo's dataset + transforms + loader glue produces the
expected tensor shapes.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.datasets import build_dataset, get_loader


def _write_dummy_rgb_images(root: Path, n: int = 12, size: int = 20) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (size, size), color=(i * 7 % 255, i * 13 % 255, i * 29 % 255))
        img.save(root / f"img_{i:03d}.png")


def test_imagefolder_loader_shapes(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_dummy_rgb_images(img_dir, n=10, size=19)

    cfg = {
        "data": {
            "dataset": "imagefolder",
            "root": str(img_dir),
            "image_size": 32,
            "batch_size": 4,
            "num_workers": 0,
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        }
    }

    ds = build_dataset(cfg, train=True)
    loader = get_loader(ds, batch_size=4, num_workers=0, train=True)
    x, y = next(iter(loader))

    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)


def test_imagefolder_split_list(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_dummy_rgb_images(img_dir, n=6, size=19)

    # pick a deterministic subset
    subset = ["img_001.png", "img_004.png", "img_005.png"]
    split_file = tmp_path / "train.txt"
    split_file.write_text("\n".join(subset) + "\n", encoding="utf-8")

    cfg = {
        "data": {
            "dataset": "imagefolder",
            "root": str(img_dir),
            "train_list": str(split_file),
            "image_size": 32,
            "batch_size": 4,
            "num_workers": 0,
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        }
    }

    ds = build_dataset(cfg, train=True)
    assert len(ds) == 3
    loader = get_loader(ds, batch_size=2, num_workers=0, train=False)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 32, 32)
    assert y.shape == (2,)
