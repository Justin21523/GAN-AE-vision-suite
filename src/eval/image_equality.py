import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging
import numpy as np


class ImageQualityEvaluator:
    """Batch evaluator for image quality metrics."""

    def __init__(self):
        """Initialize image quality evaluator."""
        self.logger = logging.getLogger(__name__)

    def compute_batch_metrics(
        self, real_loader: DataLoader, fake_loader: DataLoader, max_batches: int = 50
    ) -> Dict[str, float]:
        """
        Compute image quality metrics in batches.

        Args:
            real_loader: DataLoader for real images
            fake_loader: DataLoader for fake images
            max_batches: Maximum number of batches to process

        Returns:
            Dictionary of metrics
        """
        metrics_accum = {"psnr": 0.0, "ssim": 0.0, "mse": 0.0, "mae": 0.0}
        batch_count = 0

        for (real_batch, fake_batch), batch_idx in zip(
            zip(real_loader, fake_loader), range(max_batches)
        ):
            if real_batch is None or fake_batch is None:
                break

            # Extract images from batches
            if isinstance(real_batch, (list, tuple)):
                real_images = real_batch[0]
            else:
                real_images = real_batch

            if isinstance(fake_batch, (list, tuple)):
                fake_images = fake_batch[0]
            else:
                fake_images = fake_batch

            # Ensure same number of images
            min_batch_size = min(real_images.size(0), fake_images.size(0))
            real_images = real_images[:min_batch_size]
            fake_images = fake_images[:min_batch_size]

            # Compute metrics for this batch
            batch_metrics = self._compute_metrics_batch(real_images, fake_images)

            # Accumulate
            for metric, value in batch_metrics.items():
                metrics_accum[metric] += value

            batch_count += 1

        # Average over batches
        if batch_count > 0:
            for metric in metrics_accum:
                metrics_accum[metric] /= batch_count

        self.logger.info(f"Computed image quality metrics over {batch_count} batches")
        return metrics_accum

    def _compute_metrics_batch(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics for a single batch.

        Args:
            real_images: Batch of real images
            fake_images: Batch of fake images

        Returns:
            Dictionary of metrics for the batch
        """
        # Ensure images are in [0, 1] range
        real_images = torch.clamp(real_images, 0, 1)
        fake_images = torch.clamp(fake_images, 0, 1)

        metrics = {}

        # PSNR
        metrics["psnr"] = self._compute_psnr(real_images, fake_images)

        # SSIM (simplified implementation)
        metrics["ssim"] = self._compute_ssim(real_images, fake_images)

        # MSE
        metrics["mse"] = F.mse_loss(real_images, fake_images).item()

        # MAE (L1)
        metrics["mae"] = F.l1_loss(real_images, fake_images).item()

        return metrics

    def _compute_psnr(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """Compute PSNR between real and fake images."""
        mse = F.mse_loss(real, fake)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)).item()

    def _compute_ssim(
        self, real: torch.Tensor, fake: torch.Tensor, window_size: int = 11
    ) -> float:
        """Compute SSIM between real and fake images."""
        # Simple SSIM implementation - for production use a proper SSIM
        try:
            from torchmetrics.functional import structural_similarity_index_measure

            return structural_similarity_index_measure(fake, real).item()
        except ImportError:
            # Fallback to simple correlation-based measure
            real_flat = real.view(real.size(0), -1)
            fake_flat = fake.view(fake.size(0), -1)

            real_mean = real_flat.mean(dim=1, keepdim=True)
            fake_mean = fake_flat.mean(dim=1, keepdim=True)

            real_centered = real_flat - real_mean
            fake_centered = fake_flat - fake_mean

            covariance = (real_centered * fake_centered).mean(dim=1)
            real_std = real_centered.std(dim=1)
            fake_std = fake_centered.std(dim=1)

            ssim_map = (2 * covariance) / (real_std * fake_std + 1e-8)
            return ssim_map.mean().item()

    def compute_from_tensors(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics directly from tensors.

        Args:
            real_images: Real images tensor
            fake_images: Fake images tensor

        Returns:
            Dictionary of metrics
        """
        return self._compute_metrics_batch(real_images, fake_images)
