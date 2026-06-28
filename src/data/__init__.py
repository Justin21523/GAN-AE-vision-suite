"""
Data loading utilities.

Canonical entrypoints (used by scripts in this repo snapshot):
- `src.data.datasets.build_dataset(cfg, train=...)`
- `src.data.datasets.get_loader(dataset, ...)`
- `src.data.transforms.build_transforms(cfg, train=...)`
"""

from .datasets import build_dataset, get_loader
from .transforms import build_transforms

__all__ = ["build_dataset", "get_loader", "build_transforms"]

