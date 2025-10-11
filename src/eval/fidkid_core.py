import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json
import hashlib
from datetime import datetime

try:
    from torchvision.models import inception_v3, Inception_V3_Weights

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class InceptionFeatureExtractor:
    """InceptionV3 feature extractor with caching."""

    def __init__(self, device: str = "cuda", cache_dir: Optional[str] = None):
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logging.getLogger(__name__)

        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for Inception feature extraction"
            )

        # Load InceptionV3
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # Remove classification head
        self.model.eval()
        self.model.to(device)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: Input images in [0, 1] range, shape (N, 3, H, W)

        Returns:
            Extracted features, shape (N, 2048)
        """
        # Resize to 299x299 for InceptionV3
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

    def compute_dataset_stats(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int = 10000,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature statistics for a dataset.

        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to use
            cache_key: Cache key for reusing statistics

        Returns:
            Tuple of (mean, covariance) of features
        """
        # Check cache
        if cache_key and self.cache_dir:
            cache_file = self.cache_dir / f"stats_{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    cached_stats = json.load(f)
                return (np.array(cached_stats["mean"]), np.array(cached_stats["cov"]))

        # Extract features
        features = []
        total_samples = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            batch_features = self.extract_features(images)
            features.append(batch_features.cpu().numpy())

            total_samples += images.size(0)
            if total_samples >= max_samples:
                break

        features = np.concatenate(features, axis=0)

        # Compute statistics
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        # Cache statistics
        if cache_key and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            stats_data = {
                "mean": mu.tolist(),
                "cov": sigma.tolist(),
                "samples": total_samples,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(cache_file, "w") as f:
                json.dump(stats_data, f, indent=2)

        return mu, sigma


class FIDKIDCalculator:
    """FID and KID calculator with reference parity."""

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        feature_extractor: str = "inception",
    ):
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logging.getLogger(__name__)

        if feature_extractor == "inception":
            self.feature_extractor = InceptionFeatureExtractor(device, cache_dir)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    def compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute Fréchet Inception Distance.

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Numerical stability constant

        Returns:
            FID score
        """
        # Center the means
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            self.logger.warning(
                "FID calculation produced singular product; adding eps to diagonal"
            )
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_kid(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        subsets: int = 100,
        subset_size: int = 1000,
    ) -> float:
        """
        Compute Kernel Inception Distance.

        Args:
            features1: Features from first distribution
            features2: Features from second distribution
            subsets: Number of subsets for approximation
            subset_size: Size of each subset

        Returns:
            KID score
        """
        n1, n2 = features1.shape[0], features2.shape[0]

        # Use minimum available samples
        subset_size = min(subset_size, n1, n2)

        kid_scores = []

        for _ in range(subsets):
            # Sample subsets
            idx1 = np.random.choice(n1, subset_size, replace=False)
            idx2 = np.random.choice(n2, subset_size, replace=False)

            sub1 = features1[idx1]
            sub2 = features2[idx2]

            # Compute MMD with polynomial kernel (degree=3)
            K11 = self._polynomial_kernel(sub1, sub1)
            K22 = self._polynomial_kernel(sub2, sub2)
            K12 = self._polynomial_kernel(sub1, sub2)

            mmd = K11.mean() + K22.mean() - 2 * K12.mean()
            kid_scores.append(mmd)

        return float(np.mean(kid_scores))

    def _polynomial_kernel(
        self, X: np.ndarray, Y: np.ndarray, degree: int = 3
    ) -> np.ndarray:
        """Compute polynomial kernel matrix."""
        return (X @ Y.T + 1.0) ** degree

    def compute_metrics(
        self,
        real_loader: torch.utils.data.DataLoader,
        fake_loader: torch.utils.data.DataLoader,
        max_samples: int = 10000,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute FID and KID metrics.

        Args:
            real_loader: DataLoader for real images
            fake_loader: DataLoader for fake images
            max_samples: Maximum number of samples to use
            cache_key: Cache key for reusing statistics

        Returns:
            Dictionary with FID and KID scores
        """
        self.logger.info("Computing FID/KID metrics...")

        # Compute real dataset statistics
        real_cache_key = f"{cache_key}_real" if cache_key else None
        mu_real, sigma_real = self.feature_extractor.compute_dataset_stats(
            real_loader, max_samples, real_cache_key
        )

        # Compute fake dataset statistics and features
        fake_features = []
        total_fake = 0

        for batch in fake_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            features = self.feature_extractor.extract_features(images)
            fake_features.append(features.cpu().numpy())

            total_fake += images.size(0)
            if total_fake >= max_samples:
                break

        fake_features = np.concatenate(fake_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Compute FID
        fid = self.compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

        # Compute KID
        # Get real features for KID computation
        real_features = []
        total_real = 0

        for batch in real_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            features = self.feature_extractor.extract_features(images)
            real_features.append(features.cpu().numpy())

            total_real += images.size(0)
            if total_real >= max_samples:
                break

        real_features = np.concatenate(real_features, axis=0)

        # Ensure we have enough samples for KID
        min_samples = min(len(real_features), len(fake_features))
        if min_samples < 100:
            self.logger.warning(f"Insufficient samples for KID: {min_samples}")
            kid = float("nan")
        else:
            # Use same number of samples from both distributions
            real_subset = real_features[:min_samples]
            fake_subset = fake_features[:min_samples]
            kid = self.compute_kid(real_subset, fake_subset)

        results = {
            "fid": float(fid),
            "kid": float(kid),
            "samples_real": total_real,
            "samples_fake": total_fake,
        }

        self.logger.info(f"FID: {fid:.4f}, KID: {kid:.4f}")
        return results


def get_cache_key(
    dataset_name: str, image_size: int, preprocess_config: Dict[str, Any]
) -> str:
    """Generate cache key for dataset statistics."""
    key_data = {
        "dataset": dataset_name,
        "image_size": image_size,
        "preprocess": preprocess_config,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()
