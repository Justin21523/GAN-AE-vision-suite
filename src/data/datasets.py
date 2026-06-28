"""
Dataset utilities (torchvision-free by default).

This repository previously relied on `torchvision.datasets` for MNIST/CIFAR/CelebA.
In some environments, `torchvision` may be missing or incompatible with the
installed `torch` build, so this module implements small loaders that:
- do not import torchvision at module import time
- keep the core training/inference scripts usable

Canonical entrypoints:
- `build_dataset(cfg, train=...)`
- `get_loader(dataset, ...)`
"""

from __future__ import annotations

import gzip
import logging
import pickle
import struct
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_transforms
from src.utils.seed import seed_worker


class BaseDataset(Dataset):
    """Base dataset class with common filesystem helpers."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.logger = logging.getLogger(__name__)

    def _collect_image_paths(
        self, root: Optional[Path] = None, extensions: Optional[Sequence[str]] = None
    ) -> List[Path]:
        """Collect image files under a directory (recursive)."""
        exts = list(extensions or ("jpg", "jpeg", "png", "bmp", "tiff", "webp"))
        base = root or self.root
        paths: List[Path] = []
        for ext in exts:
            paths.extend(base.rglob(f"*.{ext}"))
            paths.extend(base.rglob(f"*.{ext.upper()}"))
        paths.sort()
        return paths


class UnlabeledImageFolder(BaseDataset):
    """Simple image-folder dataset that yields `(image_tensor, 0)`."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform)
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")
        self.image_paths = self._collect_image_paths()
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under: {self.root} (supported: jpg/jpeg/png/bmp/tiff/webp)"
            )
        self.logger.info("Loaded %d images from %s", len(self.image_paths), self.root)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def _read_lines(path: Path) -> List[str]:
    """Read non-empty, non-comment lines from a text file."""
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


class ImageListDataset(BaseDataset):
    """
    Dataset backed by an explicit list of image paths.

    This is useful for reproducible train/val splits without requiring a specific
    directory layout. Each item in the list can be:
    - a path relative to `root`, or
    - an absolute path.
    """

    def __init__(self, root: str, paths: Sequence[str], transform: Optional[Callable] = None):
        super().__init__(root, transform)
        base = Path(root)
        resolved: List[Path] = []
        missing: List[str] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = base / pp
            if not pp.exists():
                missing.append(str(pp))
                continue
            resolved.append(pp)

        if missing:
            preview = "\n".join(missing[:20])
            raise FileNotFoundError(
                f"{len(missing)} paths from split list do not exist. First 20:\n{preview}"
            )

        self.image_paths = resolved
        if not self.image_paths:
            raise FileNotFoundError("Split list produced an empty dataset.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def _open_maybe_gz(path: Path):
    """Open a file that may be gzip-compressed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_idx(path: Path) -> np.ndarray:
    """
    Read an IDX file (optionally .gz) into a NumPy array.

    MNIST files are stored as IDX format:
    - images: magic=2051, shape=(N, H, W)
    - labels: magic=2049, shape=(N,)
    """
    with _open_maybe_gz(path) as f:
        header = f.read(4)
        if len(header) != 4:
            raise ValueError(f"Invalid IDX file (too short): {path}")
        magic = struct.unpack(">I", header)[0]

        if magic == 2051:  # images
            n, rows, cols = struct.unpack(">III", f.read(12))
            data = f.read(n * rows * cols)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)
            return arr
        if magic == 2049:  # labels
            (n,) = struct.unpack(">I", f.read(4))
            data = f.read(n)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(n)
            return arr

        raise ValueError(f"Unsupported IDX magic={magic} in file: {path}")


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _find_mnist_raw_dir(root: Path) -> Optional[Path]:
    """Find a directory that contains MNIST raw files (common torchvision layouts)."""
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
            [
                c / "train-images-idx3-ubyte.gz",
                c / "train-images-idx3-ubyte",
            ]
        )
        train_labels = _find_first_existing(
            [
                c / "train-labels-idx1-ubyte.gz",
                c / "train-labels-idx1-ubyte",
            ]
        )
        test_images = _find_first_existing(
            [
                c / "t10k-images-idx3-ubyte.gz",
                c / "t10k-images-idx3-ubyte",
            ]
        )
        test_labels = _find_first_existing(
            [
                c / "t10k-labels-idx1-ubyte.gz",
                c / "t10k-labels-idx1-ubyte",
            ]
        )
        if train_images and train_labels and test_images and test_labels:
            return c
    return None


class MNISTDataset(Dataset):
    """MNIST dataset loader without `torchvision` (expects raw IDX files on disk)."""

    def __init__(self, root: str, train: bool, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform

        raw_dir = _find_mnist_raw_dir(self.root)
        if raw_dir is None:
            raise FileNotFoundError(
                "MNIST raw files not found.\n"
                f"Expected to find IDX files under one of:\n"
                f"  - {self.root}/MNIST/raw/\n"
                f"  - {self.root}/mnist/raw/\n"
                f"  - {self.root}/raw/\n"
                f"  - {self.root}/\n\n"
                "You can download MNIST manually, or (if torchvision works in your env) run:\n"
                f"  python -c \"from torchvision.datasets import MNIST; MNIST('{self.root}', download=True)\""
            )

        if self.train:
            img_path = _find_first_existing(
                [
                    raw_dir / "train-images-idx3-ubyte.gz",
                    raw_dir / "train-images-idx3-ubyte",
                ]
            )
            lbl_path = _find_first_existing(
                [
                    raw_dir / "train-labels-idx1-ubyte.gz",
                    raw_dir / "train-labels-idx1-ubyte",
                ]
            )
        else:
            img_path = _find_first_existing(
                [
                    raw_dir / "t10k-images-idx3-ubyte.gz",
                    raw_dir / "t10k-images-idx3-ubyte",
                ]
            )
            lbl_path = _find_first_existing(
                [
                    raw_dir / "t10k-labels-idx1-ubyte.gz",
                    raw_dir / "t10k-labels-idx1-ubyte",
                ]
            )

        assert img_path is not None and lbl_path is not None
        self.images = _read_idx(img_path)  # (N, H, W) uint8
        self.labels = _read_idx(lbl_path)  # (N,) uint8

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("MNIST images/labels length mismatch.")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


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


def _load_cifar10_batches(batch_dir: Path, files: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for fname in files:
        path = batch_dir / fname
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="bytes")
        data = obj[b"data"]  # shape (N, 3072)
        labels = obj.get(b"labels") or obj.get(b"fine_labels")
        if labels is None:
            raise ValueError(f"Missing labels in CIFAR batch: {path}")
        x = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        y = np.asarray(labels, dtype=np.int64)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class CIFAR10Dataset(Dataset):
    """CIFAR-10 loader without `torchvision` (expects extracted python batches)."""

    def __init__(self, root: str, train: bool, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform

        batch_dir = _find_cifar10_dir(self.root)
        if batch_dir is None:
            raise FileNotFoundError(
                "CIFAR-10 batch files not found.\n"
                f"Expected to find `cifar-10-batches-py/` under {self.root}.\n\n"
                "Download/extract CIFAR-10 python version, or (if torchvision works) run:\n"
                f"  python -c \"from torchvision.datasets import CIFAR10; CIFAR10('{self.root}', download=True)\""
            )

        if self.train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]
        self.images, self.labels = _load_cifar10_batches(batch_dir, files)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # CIFAR stored as CHW; convert to HWC for PIL.
        chw = self.images[idx]
        hwc = np.transpose(chw, (1, 2, 0))
        img = Image.fromarray(hwc)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def _resolve_celeba_root(root: Path) -> Path:
    """
    Try common CelebA layouts; fall back to the provided root.

    Most users keep:
      <root>/img_align_celeba/*.jpg
    """
    candidates = [
        root / "img_align_celeba",
        root / "celeba",
        root,
    ]
    for c in candidates:
        if c.exists():
            return c
    return root


def build_dataset(cfg: Mapping[str, Any], train: bool = True) -> Dataset:
    """
    Build a dataset from a YAML-style config dict.

    Expected keys (typical):
      cfg["data"]["dataset"]: "mnist" | "cifar10" | "celeba" | "imagefolder"
      cfg["data"]["root"]: dataset root path
    """
    data = cfg.get("data", cfg)
    name = str(data.get("dataset") or data.get("name") or "mnist").lower()
    root = Path(str(data.get("root", "./data")))

    tfm = build_transforms(cfg, train=train)

    # Optional: list-based split for image datasets.
    splits = data.get("splits") or {}
    list_path = data.get("train_list") if train else data.get("val_list")
    if not list_path:
        list_path = splits.get("train") if train else splits.get("val")
    if list_path:
        lp = Path(str(list_path))
        if not lp.is_absolute():
            lp = (Path.cwd() / lp).resolve()
        if not lp.exists():
            raise FileNotFoundError(f"Split list file not found: {lp}")
        items = _read_lines(lp)

        # Resolve dataset root for celeba convenience.
        split_root = root
        if name == "celeba":
            split_root = _resolve_celeba_root(root)
        return ImageListDataset(str(split_root), items, transform=tfm)

    if name in {"imagefolder", "folder", "images"}:
        return UnlabeledImageFolder(str(root), transform=tfm)

    if name == "celeba":
        return UnlabeledImageFolder(str(_resolve_celeba_root(root)), transform=tfm)

    if name == "mnist":
        return MNISTDataset(str(root), train=train, transform=tfm)

    if name in {"cifar10", "cifar-10"}:
        return CIFAR10Dataset(str(root), train=train, transform=tfm)

    raise ValueError(f"Unsupported dataset: {name}")


def get_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults for ML training/eval.

    Notes:
    - `seed_worker()` is used to make NumPy/Python RNG deterministic in workers.
    - Use `pin_memory=True` when training on CUDA for faster host->GPU transfer.
    """
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
