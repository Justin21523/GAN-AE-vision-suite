"""
Random seed utilities.

These helpers aim to make training/evaluation runs reproducible by seeding:
- Python's `random`
- NumPy RNG
- PyTorch RNG (CPU and CUDA)

Also includes `seed_worker()` for `torch.utils.data.DataLoader(worker_init_fn=...)`.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python/NumPy/PyTorch RNGs.

    Args:
        seed: Global seed value.
        deterministic: If True, configures cuDNN for deterministic behavior
            (slower but reproducible). If False, enables benchmarking for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism settings (only relevant on CUDA)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def seed_worker(worker_id: int) -> None:
    """
    Seed a DataLoader worker process.

    When you pass a `torch.Generator` with a manual seed to DataLoader, PyTorch
    derives a unique seed for each worker. This hook forwards that derived seed
    into NumPy/Python RNGs so augmentations stay reproducible too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Compatibility alias for newer/experimental modules.
SeedManager = Any
