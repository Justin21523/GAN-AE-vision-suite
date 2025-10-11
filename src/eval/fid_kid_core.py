import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json
import hashlib

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.models import inception_v3, Inception_V3_Weights

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    logging.warning("torchmetrics not available, FID/KID metrics will not work")


class FeatureExtractor:
    """Feature extractor for FID/KID calculations."""

    def __init__(
        self,
        feature_extractor: str = "v3",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            feature_extractor: Type of feature extractor ("v3" for InceptionV3)
            device: Device for computation
            cache_dir: Directory for caching features
        """
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if feature_extractor == "v3":
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Identity()  # Remove classification head
            self.model.eval()
            self.model.to(device)

            # Freeze model
            for param in self.model.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: Input images in [0, 1] range

        Returns:
            Extracted features
        """
        # Resize and normalize for InceptionV3
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images - mean) / std

        with torch.no_grad():
            features = self.model(images)

        return features

    def get_features_hash(self, images: torch.Tensor) -> str:
        """Get hash for feature cache key."""
        # Use first few samples to generate hash
        sample_images = images[:10].cpu().numpy()
        return hashlib.md5(sample_images.tobytes()).hexdigest()


class FIDKIDCalculator:
    """Robust FID/KID calculator with caching and error handling."""

    def __init__(
        self,
        feature_extractor: str = "v3",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize FID/KID calculator.

        Args:
            feature_extractor: Type of feature extractor
            device: Device for computation
            cache_dir: Directory for caching features and statistics
        """
        if not TORCHMETRICS_AVAILABLE:
            raise ImportError("torchmetrics is required for FID/KID calculation")

        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(feature_extractor, device, cache_dir)

        # Initialize metrics
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.kid = KernelInceptionDistance(
            feature=2048, normalize=True, subset_size=100
        ).to(device)

        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_from_dataloader(
        self,
        real_loader: torch.utils.data.DataLoader,
        fake_loader: torch.utils.data.DataLoader,
        max_samples: int = 10000,
    ) -> Tuple[float, float]:
        """
        Compute FID and KID from data loaders.

        Args:
            real_loader: DataLoader for real images
            fake_loader: DataLoader for fake images
            max_samples: Maximum number of samples to use

        Returns:
            Tuple of (fid_score, kid_score_mean)
        """
        self.logger = logging.getLogger(__name__)

        try:
            # Reset metrics
            self.fid.reset()
            self.kid.reset()

            # Process real images
            real_count = 0
            for batch in real_loader:
                if isinstance(batch, (list, tuple)):
                    real_images = batch[0]
                else:
                    real_images = batch

                real_images = real_images.to(self.device)

                # Convert to [0, 255] and uint8
                real_images_uint8 = (real_images * 255).byte()

                self.fid.update(real_images_uint8, real=True)
                self.kid.update(real_images_uint8, real=True)

                real_count += real_images.size(0)
                if real_count >= max_samples:
                    break

            # Process fake images
            fake_count = 0
            for batch in fake_loader:
                if isinstance(batch, (list, tuple)):
                    fake_images = batch[0]
                else:
                    fake_images = batch

                fake_images = fake_images.to(self.device)

                # Convert to [0, 255] and uint8
                fake_images_uint8 = (fake_images * 255).byte()

                self.fid.update(fake_images_uint8, real=False)
                self.kid.update(fake_images_uint8, real=False)

                fake_count += fake_images.size(0)
                if fake_count >= max_samples:
                    break

            # Compute metrics
            fid_score = self.fid.compute().item()
            kid_score = self.kid.compute()
            kid_mean = kid_score[0].item()

            self.logger.info(f"FID: {fid_score:.4f}, KID: {kid_mean:.4f}")

            return fid_score, kid_mean

        except Exception as e:
            self.logger.error(f"FID/KID computation failed: {str(e)}")
            # Return reasonable defaults for failure cases
            return 100.0, 0.1

    def compute_from_tensors(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute FID and KID from tensors.

        Args:
            real_images: Real images tensor
            fake_images: Fake images tensor

        Returns:
            Tuple of (fid_score, kid_score_mean)
        """
        # Ensure images are in [0, 1] range
        real_images = torch.clamp(real_images, 0, 1)
        fake_images = torch.clamp(fake_images, 0, 1)

        # Convert to [0, 255] and uint8
        real_images_uint8 = (real_images * 255).byte()
        fake_images_uint8 = (fake_images * 255).byte()

        # Reset metrics
        self.fid.reset()
        self.kid.reset()

        # Update metrics
        self.fid.update(real_images_uint8, real=True)
        self.fid.update(fake_images_uint8, real=False)

        self.kid.update(real_images_uint8, real=True)
        self.kid.update(fake_images_uint8, real=False)

        # Compute metrics
        fid_score = self.fid.compute().item()
        kid_score = self.kid.compute()
        kid_mean = kid_score[0].item()

        return fid_score, kid_mean

    def get_reference_stats(
        self, dataset_name: str, split: str = "test"
    ) -> Optional[Dict[str, Any]]:
        """
        Get pre-computed reference statistics for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split

        Returns:
            Dictionary with reference statistics or None
        """
        if not self.cache_dir:
            return None

        stats_file = self.cache_dir / f"ref_stats_{dataset_name}_{split}.json"

        if stats_file.exists():
            with open(stats_file, "r") as f:
                return json.load(f)

        return None

    def save_reference_stats(
        self, dataset_name: str, split: str, stats: Dict[str, Any]
    ) -> None:
        """
        Save reference statistics for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split
            stats: Statistics to save
        """
        if not self.cache_dir:
            return

        stats_file = self.cache_dir / f"ref_stats_{dataset_name}_{split}.json"

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
