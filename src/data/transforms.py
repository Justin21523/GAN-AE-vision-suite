"""
Transform builders for image datasets (torchvision-free).

Why:
Some environments may not have a compatible `torchvision` build. The core
pipelines in this repository only need a small subset of common transforms
(resize/crop/flip/tensor/normalize), so we provide lightweight equivalents here.

This module provides:
- `build_transforms(cfg, train=...)`: YAML-friendly transform builder.
- `get_transforms(config, is_training=...)`: attribute/dict-friendly builder.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance

from src.utils.vision import make_grid

# Some modules use a typed `DataConfig` object; others pass plain dicts/AddictDict.
# We keep the name for readability while remaining runtime-safe.
DataConfig = Any


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]):
        self.transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    """Resize a PIL image to a square size (or HxW tuple)."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)


class CenterCrop:
    """Center-crop a PIL image to a square size (or HxW tuple)."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        th, tw = self.size
        w, h = img.size
        i = max(0, int(round((h - th) / 2.0)))
        j = max(0, int(round((w - tw) / 2.0)))
        return img.crop((j, i, j + tw, i + th))


class RandomCrop:
    """Random-crop a PIL image to a square size (or HxW tuple)."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        th, tw = self.size
        w, h = img.size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            # If the image is smaller than the crop, fall back to center crop.
            return CenterCrop(self.size)(img)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return img.crop((j, i, j + tw, i + th))


class RandomHorizontalFlip:
    """Randomly flip a PIL image horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    """Randomly flip a PIL image vertically."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class ColorJitter:
    """Minimal color jitter (brightness/contrast/saturation)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.0,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.hue != 0.0:
            # Hue jitter is non-trivial without torchvision; keep it explicit.
            raise NotImplementedError("Hue jitter is not supported in this torchvision-free pipeline.")

        if self.brightness:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.saturation:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        return img


class ToTensor:
    """Convert a PIL image to a float tensor in [0, 1] (CHW)."""

    def __call__(self, img: Any) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            return img
        if not isinstance(img, Image.Image):
            raise TypeError(f"ToTensor expects a PIL.Image or torch.Tensor, got {type(img)}")

        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # HWC uint8 -> CHW float32
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return t


class Normalize:
    """Normalize a tensor image: (x - mean) / std."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Normalize expects a CHW tensor.")
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class Lambda:
    """Apply a simple callable (utility transform)."""

    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def __call__(self, x: Any) -> Any:
        return self.fn(x)


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Get a key from either an attribute object or a dict-like config."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _build_from_spec(spec: Sequence[Mapping[str, Any]]) -> Compose:
    """
    Build a `Compose` from a YAML-friendly transform spec list.

    Each item is a dict like:
      - {name: Resize, size: 32}
      - {name: Normalize, mean: [0.5], std: [0.5]}
    """
    built: List[Callable[[Any], Any]] = []
    for item in spec:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Transform spec missing 'name': {item}")

        if name == "Resize":
            built.append(Resize(int(item["size"])))
        elif name == "CenterCrop":
            built.append(CenterCrop(int(item["size"])))
        elif name == "RandomCrop":
            built.append(RandomCrop(int(item["size"])))
        elif name == "RandomHorizontalFlip":
            built.append(RandomHorizontalFlip(p=float(item.get("p", 0.5))))
        elif name == "RandomVerticalFlip":
            built.append(RandomVerticalFlip(p=float(item.get("p", 0.5))))
        elif name == "ColorJitter":
            built.append(
                ColorJitter(
                    brightness=float(item.get("brightness", 0.2)),
                    contrast=float(item.get("contrast", 0.2)),
                    saturation=float(item.get("saturation", 0.2)),
                    hue=float(item.get("hue", 0.0)),
                )
            )
        elif name == "ToTensor":
            built.append(ToTensor())
        elif name == "Normalize":
            built.append(
                Normalize(
                    mean=tuple(item.get("mean", (0.5,))),
                    std=tuple(item.get("std", (0.5,))),
                )
            )
        else:
            raise ValueError(f"Unsupported transform name '{name}' in spec: {item}")

    return Compose(built)


def build_transforms(cfg: Mapping[str, Any], train: bool = True) -> Compose:
    """
    Build transforms from a repository YAML config (`configs/*.yaml`).

    Args:
        cfg: Full config dict (or an AddictDict) returned by `load_config()`.
        train: If True, uses `data.train_transforms`; otherwise `data.val_transforms`.

    Returns:
        A callable transform pipeline that outputs tensors normalized for models
        that commonly use `tanh` (i.e., values in [-1, 1]) when mean/std are [0.5].
    """
    data = cfg.get("data", cfg)
    list_key = "train_transforms" if train else "val_transforms"

    spec = data.get(list_key)
    if spec:
        return _build_from_spec(spec)

    # Fallback: build a minimal pipeline from common keys.
    image_size = int(data.get("image_size") or data.get("img_size") or 64)
    center_crop = data.get("center_crop")
    mean = data.get("normalize_mean")
    std = data.get("normalize_std")

    # Default normalization maps [0,1] -> [-1,1] which pairs well with Tanh models.
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]

    tfs: List[Callable[[Any], Any]] = []
    if center_crop:
        tfs.append(CenterCrop(int(center_crop)))
    tfs.append(Resize(image_size))
    tfs.append(ToTensor())
    tfs.append(Normalize(tuple(mean), tuple(std)))
    return Compose(tfs)


def get_transforms(config: DataConfig, is_training: bool = True) -> Callable[[Any], Any]:
    """
    Build transforms from an attribute-style config object.

    This is primarily used by newer/experimental configs that look like:
      config.data.image_size, config.data.aug.hflip, ...
    """
    transform_list: List[Callable[[Any], Any]] = []

    image_size = _get(config, "image_size", None)
    if image_size:
        transform_list.append(Resize(int(image_size)))

    if is_training:
        aug = _get(config, "aug", {}) or {}
        if _get(aug, "hflip", False):
            transform_list.append(RandomHorizontalFlip())
        if _get(aug, "vflip", False):
            transform_list.append(RandomVerticalFlip())
        if _get(aug, "random_crop", False):
            crop_size = _get(aug, "crop_size", image_size)
            if crop_size:
                transform_list.append(RandomCrop(int(crop_size)))
        if _get(aug, "color_jitter", False):
            jitter_params = _get(aug, "color_jitter_params", {}) or {}
            transform_list.append(
                ColorJitter(
                    brightness=float(_get(jitter_params, "brightness", 0.2)),
                    contrast=float(_get(jitter_params, "contrast", 0.2)),
                    saturation=float(_get(jitter_params, "saturation", 0.2)),
                    hue=float(_get(jitter_params, "hue", 0.0)),
                )
            )

    transform_list.append(ToTensor())

    # Default: map to [-1, 1] which matches tanh outputs.
    name = str(_get(config, "name", "rgb")).lower()
    if name == "mnist":
        transform_list.append(Normalize((0.5,), (0.5,)))
    else:
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return Compose(transform_list)


def get_denormalize_transform(config: DataConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a denormalize transform for visualization ([-1,1] -> [0,1])."""
    return Lambda(lambda x: torch.clamp((x + 1) / 2, 0, 1))


def create_sample_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Create a grid tensor from a batch for visualization."""
    # Denormalize if needed
    if images.min().item() < 0:
        images = (images + 1) / 2
    return make_grid(images.clamp(0, 1), nrow=int(nrow))

