import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from src.utils.config import DataConfig


class Normalize:
    """Normalize images to [-1, 1] range."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - 0.5) * 2


class Denormalize:
    """Denormalize images from [-1, 1] to [0, 1] range."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img + 1) / 2


class ImageTransform:
    """Config-driven image transforms."""

    def __init__(self, config: DataConfig, is_training: bool = True):
        """
        Initialize image transforms.

        Args:
            config: Data configuration
            is_training: Whether to include training augmentations
        """
        self.config = config
        self.is_training = is_training
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build transform pipeline from configuration."""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(self.config.image_size))

        # Training augmentations
        if self.is_training:
            aug_config = self.config.aug

            # Horizontal flip
            if aug_config.get("hflip", False):
                transform_list.append(transforms.RandomHorizontalFlip())

            # Color jitter
            if aug_config.get("color_jitter", False):
                jitter_params = aug_config.get("color_jitter_params", {})
                brightness = jitter_params.get("brightness", 0.2)
                contrast = jitter_params.get("contrast", 0.2)
                saturation = jitter_params.get("saturation", 0.2)
                hue = jitter_params.get("hue", 0.1)

                transform_list.append(
                    transforms.ColorJitter(brightness, contrast, saturation, hue)
                )

            # Random crop
            if aug_config.get("random_crop", False):
                crop_size = aug_config.get("crop_size", self.config.image_size)
                transform_list.append(transforms.RandomCrop(crop_size))

        # To tensor and normalize
        transform_list.append(transforms.ToTensor())

        # Normalize
        if self.config.name.lower() == "mnist":
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:  # RGB images
            transform_list.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

        return transforms.Compose(transform_list)

    def __call__(self, img):
        """Apply transforms to image."""
        return self.transform(img)


class PairedTransform:
    """Transforms for paired image data (e.g., Pix2Pix)."""

    def __init__(self, config: DataConfig, is_training: bool = True):
        """
        Initialize paired transforms.

        Args:
            config: Data configuration
            is_training: Whether to include training augmentations
        """
        self.config = config
        self.is_training = is_training
        self.transform = self._build_paired_transform()

    def _build_paired_transform(self) -> transforms.Compose:
        """Build paired transform pipeline."""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(self.config.image_size))

        if self.is_training:
            # For paired data, we need to apply the same random transformations to both images
            aug_config = self.config.aug

            if aug_config.get("hflip", False):
                transform_list.append(transforms.RandomHorizontalFlip())

            if aug_config.get("random_crop", False):
                crop_size = aug_config.get("crop_size", self.config.image_size)
                transform_list.append(transforms.RandomCrop(crop_size))

        # To tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    def __call__(self, img_a, img_b):
        """Apply same transforms to both images."""
        return self.transform(img_a), self.transform(img_b)


def get_transforms(config: DataConfig, is_training: bool = True) -> callable:
    """
    Get transforms based on configuration.

    Args:
        config: Data configuration
        is_training: Whether this is for training

    Returns:
        Transform function
    """
    transform_list = []

    # Resize
    if hasattr(config, "image_size") and config.image_size:
        transform_list.append(transforms.Resize(config.image_size))

    # Training augmentations
    if is_training and hasattr(config, "aug") and config.aug:
        aug_config = config.aug

        if aug_config.get("hflip", False):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_config.get("vflip", False):
            transform_list.append(transforms.RandomVerticalFlip())

        if aug_config.get("color_jitter", False):
            jitter_params = aug_config.get("color_jitter_params", {})
            brightness = jitter_params.get("brightness", 0.2)
            contrast = jitter_params.get("contrast", 0.2)
            saturation = jitter_params.get("saturation", 0.2)
            hue = jitter_params.get("hue", 0.1)

            transform_list.append(
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            )

        if aug_config.get("random_crop", False):
            crop_size = aug_config.get("crop_size", config.image_size)
            transform_list.append(transforms.RandomCrop(crop_size))

    # To tensor
    transform_list.append(transforms.ToTensor())

    # Normalize based on dataset
    if config.name.lower() == "mnist":
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:  # RGB images
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


def get_denormalize_transform(config: DataConfig) -> callable:
    """Get denormalize transform for visualization."""
    if config.name.lower() == "mnist":
        return transforms.Compose(
            [
                transforms.Normalize((-1,), (2,)),  # Reverse: (x + 1) / 2
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # Reverse: (x + 1) / 2
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )


def create_sample_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Create a grid of images for visualization."""
    import torchvision.utils as vutils

    # Denormalize if needed
    if images.min() < 0:
        images = (images + 1) / 2

    grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)

    return grid
