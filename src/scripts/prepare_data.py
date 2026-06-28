"""
Dataset preparation helper (no downloads).

This repository snapshot intentionally does not auto-download datasets. Instead,
this script helps you:
- validate that your dataset is placed in the expected folder structure
- create a tiny demo image-folder dataset to quickly smoke-test training code

Examples:
  # Validate CelebA layout (expects images under ./data/img_align_celeba/*.jpg)
  python -m src.scripts.prepare_data --dataset celeba --root ./data

  # Validate MNIST raw IDX files (expects train/t10k idx files, optionally .gz)
  python -m src.scripts.prepare_data --dataset mnist --root ./data

  # Create a demo image-folder dataset with random images
  python -m src.scripts.prepare_data --create-demo-imagefolder ./data/demo_images --num-images 32 --img-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _collect_images(root: Path, extensions: Sequence[str] = ("jpg", "jpeg", "png", "webp")) -> list[Path]:
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(root.rglob(f"*.{ext}"))
        paths.extend(root.rglob(f"*.{ext.upper()}"))
    paths.sort()
    return paths


def _resolve_celeba_root(root: Path) -> Path:
    for c in (root / "img_align_celeba", root / "celeba", root):
        if c.exists():
            return c
    return root


def check_celeba(root: Path) -> tuple[bool, str]:
    celeba_root = _resolve_celeba_root(root)
    imgs = _collect_images(celeba_root, extensions=("jpg", "jpeg", "png"))
    if not celeba_root.exists():
        return False, f"CelebA root not found: {celeba_root}"
    if not imgs:
        return (
            False,
            "No images found. Expected something like:\n"
            f"  {root}/img_align_celeba/000001.jpg",
        )
    return True, f"OK: found {len(imgs)} images under {celeba_root}"


def _find_mnist_raw_dir(root: Path) -> Optional[Path]:
    candidates = [
        root / "MNIST" / "raw",
        root / "mnist" / "raw",
        root / "MNIST",
        root / "mnist",
        root / "raw",
        root,
    ]
    for c in candidates:
        if not c.exists():
            continue
        train_images = _find_first_existing(
            [c / "train-images-idx3-ubyte.gz", c / "train-images-idx3-ubyte"]
        )
        train_labels = _find_first_existing(
            [c / "train-labels-idx1-ubyte.gz", c / "train-labels-idx1-ubyte"]
        )
        test_images = _find_first_existing(
            [c / "t10k-images-idx3-ubyte.gz", c / "t10k-images-idx3-ubyte"]
        )
        test_labels = _find_first_existing(
            [c / "t10k-labels-idx1-ubyte.gz", c / "t10k-labels-idx1-ubyte"]
        )
        if train_images and train_labels and test_images and test_labels:
            return c
    return None


def check_mnist(root: Path) -> tuple[bool, str]:
    raw_dir = _find_mnist_raw_dir(root)
    if raw_dir is None:
        return (
            False,
            "MNIST raw IDX files not found.\n"
            "Expected to find:\n"
            f"  {root}/MNIST/raw/train-images-idx3-ubyte(.gz)\n"
            f"  {root}/MNIST/raw/train-labels-idx1-ubyte(.gz)\n"
            f"  {root}/MNIST/raw/t10k-images-idx3-ubyte(.gz)\n"
            f"  {root}/MNIST/raw/t10k-labels-idx1-ubyte(.gz)\n",
        )
    return True, f"OK: found MNIST raw files under {raw_dir}"


def _find_cifar10_dir(root: Path) -> Optional[Path]:
    candidates = [
        root / "cifar-10-batches-py",
        root / "cifar10" / "cifar-10-batches-py",
        root / "CIFAR10" / "cifar-10-batches-py",
        root,
    ]
    for c in candidates:
        if (c / "data_batch_1").exists() and (c / "test_batch").exists():
            return c
    return None


def check_cifar10(root: Path) -> tuple[bool, str]:
    batch_dir = _find_cifar10_dir(root)
    if batch_dir is None:
        return (
            False,
            "CIFAR-10 python batches not found.\n"
            "Expected to find:\n"
            f"  {root}/cifar-10-batches-py/data_batch_1\n"
            f"  {root}/cifar-10-batches-py/test_batch\n",
        )
    return True, f"OK: found CIFAR-10 batches under {batch_dir}"


def check_imagefolder(root: Path) -> tuple[bool, str]:
    if not root.exists():
        return False, f"Image folder not found: {root}"
    imgs = _collect_images(root)
    if not imgs:
        return False, f"No images found under: {root}"
    return True, f"OK: found {len(imgs)} images under {root}"


def create_demo_imagefolder(root: Path, num_images: int, img_size: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(int(num_images)):
        arr = rng.integers(0, 256, size=(int(img_size), int(img_size), 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f"{i:05d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["celeba", "mnist", "cifar10", "imagefolder"],
        help="Dataset type to validate (no downloads).",
    )
    ap.add_argument("--root", type=str, default="./data", help="Dataset root directory.")
    ap.add_argument(
        "--create-demo-imagefolder",
        type=str,
        default=None,
        metavar="PATH",
        help="Create a tiny random image-folder dataset at PATH.",
    )
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=64)
    args = ap.parse_args()

    if args.create_demo_imagefolder:
        out = Path(args.create_demo_imagefolder)
        create_demo_imagefolder(out, num_images=args.num_images, img_size=args.img_size)
        print(f"Created demo imagefolder: {out} ({args.num_images} images)")
        print("You can use it with a config like:")
        print("  data:")
        print("    dataset: imagefolder")
        print(f"    root: {out}")
        return

    if args.dataset is None:
        raise SystemExit("Provide --dataset to validate, or --create-demo-imagefolder to generate a demo dataset.")

    root = Path(args.root)
    checks = {
        "celeba": check_celeba,
        "mnist": check_mnist,
        "cifar10": check_cifar10,
        "imagefolder": check_imagefolder,
    }
    ok, msg = checks[args.dataset](root)
    print(msg)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
