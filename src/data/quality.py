"""
Data quality helpers.

These utilities are used to validate dataset integrity on disk and produce small
reports for debugging and reproducibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from src.data.transforms import Compose


SUPPORTED_IMAGE_EXTS: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")


@dataclass(frozen=True)
class ImageScanResult:
    total_files: int
    ok_files: int
    bad_files: int
    bad_paths: List[str]
    mode_counts: Dict[str, int]
    size_counts: Dict[str, int]  # "WxH" -> count


def collect_image_paths(root: Path, extensions: Sequence[str] = SUPPORTED_IMAGE_EXTS) -> List[Path]:
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(root.rglob(f"*.{ext}"))
        paths.extend(root.rglob(f"*.{ext.upper()}"))
    paths.sort()
    return paths


def scan_images(paths: Iterable[Path], max_bad_paths: int = 50) -> ImageScanResult:
    mode_counts: Dict[str, int] = {}
    size_counts: Dict[str, int] = {}
    bad_paths: List[str] = []
    total = 0
    ok = 0

    for p in paths:
        total += 1
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img2:
                mode_counts[img2.mode] = mode_counts.get(img2.mode, 0) + 1
                w, h = img2.size
                key = f"{w}x{h}"
                size_counts[key] = size_counts.get(key, 0) + 1
            ok += 1
        except Exception:
            if len(bad_paths) < max_bad_paths:
                bad_paths.append(str(p))

    bad = total - ok
    return ImageScanResult(
        total_files=total,
        ok_files=ok,
        bad_files=bad,
        bad_paths=bad_paths,
        mode_counts=mode_counts,
        size_counts=size_counts,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_duplicates_by_hash(paths: Sequence[Path], max_files: Optional[int] = None) -> Dict[str, List[str]]:
    chosen = list(paths[: int(max_files)]) if max_files is not None else list(paths)
    buckets: Dict[str, List[str]] = {}
    for p in chosen:
        try:
            digest = sha256_file(p)
        except Exception:
            continue
        buckets.setdefault(digest, []).append(str(p))
    return {h: ps for h, ps in buckets.items() if len(ps) > 1}


def describe_transforms(tfm: Any) -> List[Dict[str, Any]]:
    """
    Serialize a torchvision-free transform pipeline into a JSON-friendly list.

    This is intentionally best-effort; unknown transforms are represented by
    their class name.
    """
    if isinstance(tfm, Compose):
        out: List[Dict[str, Any]] = []
        for t in tfm.transforms:
            out.extend(describe_transforms(t))
        return out

    name = tfm.__class__.__name__
    d: Dict[str, Any] = {"name": name}
    for k in ("size", "p", "brightness", "contrast", "saturation", "hue"):
        if hasattr(tfm, k):
            d[k] = getattr(tfm, k)
    if hasattr(tfm, "mean") and hasattr(tfm, "std"):
        try:
            d["mean"] = [float(x) for x in tfm.mean.view(-1).tolist()]  # type: ignore[attr-defined]
            d["std"] = [float(x) for x in tfm.std.view(-1).tolist()]  # type: ignore[attr-defined]
        except Exception:
            pass
    return [d]


@torch.no_grad()
def compute_tensor_stats(
    loader: torch.utils.data.DataLoader,
    max_batches: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Compute basic stats over a few batches of an image loader.

    Expects batches like (images, labels) or images tensor.
    """
    dev = device or torch.device("cpu")
    n = 0
    sum_ = None
    sumsq = None
    min_v = None
    max_v = None

    for i, batch in enumerate(loader):
        if i >= int(max_batches):
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(dev, non_blocking=True).float()
        b, c, h, w = x.shape
        flat = x.view(b, c, -1)
        if sum_ is None:
            sum_ = flat.sum(dim=(0, 2))
            sumsq = (flat * flat).sum(dim=(0, 2))
            min_v = flat.amin(dim=(0, 2))
            max_v = flat.amax(dim=(0, 2))
        else:
            sum_ += flat.sum(dim=(0, 2))
            sumsq += (flat * flat).sum(dim=(0, 2))
            min_v = torch.minimum(min_v, flat.amin(dim=(0, 2)))  # type: ignore[arg-type]
            max_v = torch.maximum(max_v, flat.amax(dim=(0, 2)))  # type: ignore[arg-type]
        n += b * h * w

    if sum_ is None or sumsq is None or min_v is None or max_v is None or n == 0:
        return {"count": 0}

    mean = (sum_ / n).cpu()
    var = (sumsq / n - mean * mean).clamp(min=0).cpu()
    std = torch.sqrt(var).cpu()
    return {
        "count": int(n),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "min": [float(v) for v in min_v.cpu().tolist()],
        "max": [float(v) for v in max_v.cpu().tolist()],
    }

