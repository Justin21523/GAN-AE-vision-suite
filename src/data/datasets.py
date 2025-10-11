import os
from typing import List, Optional, Callable, Tuple, Any, Dict
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from src.utils.config import DataConfig
from src.utils.seed import seed_worker
from .transforms import get_transforms


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.logger = logging.getLogger(__name__)

    def _collect_image_paths(self, extensions: List[str] = None) -> List[Path]:
        """Collect image paths with given extensions."""
        if extensions is None:
            extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

        image_paths = []
        for ext in extensions:
            image_paths.extend(self.root.rglob(f"*.{ext}"))
            image_paths.extend(self.root.rglob(f"*.{ext.upper()}"))

        image_paths.sort()
        return image_paths


class ImageFolderPaired(BaseDataset):
    """Paired image dataset for Pix2Pix (A -> B translation)."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        """
        Initialize paired image dataset.

        Args:
            root: Root directory containing 'trainA', 'trainB', 'testA', 'testB'
            transform: Transform to apply to both images
        """
        super().__init__(root, transform)

        # Load images from both domains
        self.domain_a_dir = self.root / "trainA"
        self.domain_b_dir = self.root / "trainB"

        self.domain_a_paths = self._collect_image_paths()
        self.domain_b_paths = self._collect_image_paths()

        # Ensure matching pairs
        if len(self.domain_a_paths) != len(self.domain_b_paths):
            self.logger.warning(
                f"Unpaired dataset: A has {len(self.domain_a_paths)}, B has {len(self.domain_b_paths)}"
            )

        # Use minimum length
        self.length = min(len(self.domain_a_paths), len(self.domain_b_paths))
        self.logger.info(f"Loaded {self.length} paired images from {root}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_a_path = self.domain_a_paths[idx]
        img_b_path = self.domain_b_paths[idx]

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        if self.transform:
            if hasattr(self.transform, "__call__") and not hasattr(
                self.transform, "transform"
            ):
                # Assume it's a paired transform
                img_a, img_b = self.transform(img_a, img_b)
            else:
                # Apply same transform to both
                img_a = self.transform(img_a)
                img_b = self.transform(img_b)

        return img_a, img_b


class PairedImageDataset(Dataset):
    """Dataset for paired images (e.g., Pix2Pix)."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        """
        Initialize paired image dataset.

        Args:
            root: Root directory containing 'A' and 'B' subdirectories
            transform: Transform to apply to both images
        """
        self.root = root
        self.transform = transform

        # Load images from domain A and B
        self.domain_a_paths = []
        self.domain_b_paths = []

        domain_a_dir = Path(root) / "A"
        domain_b_dir = Path(root) / "B"

        for ext in ["jpg", "jpeg", "png", "bmp"]:
            a_paths = list(domain_a_dir.rglob(f"*.{ext}"))
            b_paths = list(domain_b_dir.rglob(f"*.{ext}"))

            self.domain_a_paths.extend(a_paths)
            self.domain_b_paths.extend(b_paths)

        # Sort and ensure matching pairs
        self.domain_a_paths.sort()
        self.domain_b_paths.sort()

        if len(self.domain_a_paths) != len(self.domain_b_paths):
            logging.warning(
                f"Unpaired dataset: A has {len(self.domain_a_paths)}, B has {len(self.domain_b_paths)} images"
            )

        logging.info(f"Loaded {len(self.domain_a_paths)} paired images from {root}")

    def __len__(self) -> int:
        return min(len(self.domain_a_paths), len(self.domain_b_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_a = Image.open(self.domain_a_paths[idx]).convert("RGB")
        img_b = Image.open(self.domain_b_paths[idx]).convert("RGB")

        if self.transform:
            img_a, img_b = self.transform(img_a, img_b)

        return img_a, img_b


class ImageFolderUnpaired(BaseDataset):
    """Unpaired image dataset for CycleGAN (A and B domains)."""

    def __init__(
        self, root: str, domain: str = "A", transform: Optional[Callable] = None
    ):
        """
        Initialize unpaired image dataset.

        Args:
            root: Root directory containing domain subdirectories
            domain: Domain to load ("A" or "B")
            transform: Transform to apply to images
        """
        super().__init__(root, transform)
        self.domain = domain

        if domain == "A":
            self.domain_dir = self.root / "trainA"
        else:
            self.domain_dir = self.root / "trainB"

        self.image_paths = self._collect_image_paths()
        self.logger.info(
            f"Loaded {len(self.image_paths)} images from {self.domain_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper."""

    def __init__(
        self, root: str, train: bool = True, transform: Optional[Callable] = None
    ):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class MNISTDataset(Dataset):
    """MNIST dataset wrapper."""

    def __init__(
        self, root: str, train: bool = True, transform: Optional[Callable] = None
    ):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def get_dataset(config: DataConfig, is_training: bool = True) -> Dataset:
    """
    Get dataset based on configuration.

    Args:
        config: Data configuration
        is_training: Whether this is for training

    Returns:
        Configured dataset
    """
    from .transforms import get_transforms

    transforms = get_transforms(config, is_training)

    if config.type == "torchvision":
        if config.name.lower() == "mnist":
            return MNISTDataset(config.root, is_training, transforms)
        elif config.name.lower() == "cifar10":
            return CIFAR10Dataset(config.root, is_training, transforms)
        else:
            raise ValueError(f"Unsupported torchvision dataset: {config.name}")

    elif config.type == "imagefolder_paired":
        return ImageFolderPaired(config.root, transforms)

    elif config.type == "imagefolder_unpaired":
        domain = getattr(config, "domain", "A")
        return ImageFolderUnpaired(config.root, domain, transforms)

    else:
        raise ValueError(f"Unsupported dataset type: {config.type}")


def get_dataloader(
    config: DataConfig, is_training: bool = True, seed: int = 42
) -> DataLoader:
    """
    Get data loader based on configuration.

    Args:
        config: Data configuration
        is_training: Whether this is for training
        seed: Random seed for worker initialization

    Returns:
        Configured data loader
    """
    dataset = get_dataset(config, is_training)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
