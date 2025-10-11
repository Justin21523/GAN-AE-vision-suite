import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import json
import hashlib
from datetime import datetime

from src.utils.config import RunConfig
from src.utils.seed import set_seed, SeedManager


class EvalRunner:
    """Unified evaluation runner for FID/KID/PSNR/SSIM metrics."""

    def __init__(self, config: RunConfig, device: str = "cuda"):
        """
        Initialize evaluation runner.

        Args:
            config: Run configuration
            device: Device for evaluation
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Setup cache directory
        self.cache_dir = Path(config.output_dir) / "eval_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = {}
        if "fid" in config.eval.metrics or "kid" in config.eval.metrics:
            from .fid_kid_core import FIDKIDCalculator

            self.fid_kid_calc = FIDKIDCalculator(
                device=device, cache_dir=str(self.cache_dir)
            )
            self.metrics.update({"fid": self.fid_kid_calc, "kid": self.fid_kid_calc})

        if any(
            metric in config.eval.metrics for metric in ["psnr", "ssim", "mse", "mae"]
        ):
            from .image_quality import ImageQualityEvaluator

            self.image_quality_calc = ImageQualityEvaluator()
            self.metrics.update(
                {
                    "psnr": self.image_quality_calc,
                    "ssim": self.image_quality_calc,
                    "mse": self.image_quality_calc,
                    "mae": self.image_quality_calc,
                }
            )

    def evaluate_run(
        self,
        run_id: str,
        checkpoint_path: str,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a run with comprehensive metrics.

        Args:
            run_id: Run identifier
            checkpoint_path: Path to model checkpoint
            num_samples: Number of samples for evaluation
            seed: Random seed for reproducibility

        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating run {run_id} with {num_samples} samples")

        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed, deterministic=True)

        results = {}

        try:
            # Load model and generate samples
            fake_loader = self._create_fake_loader(checkpoint_path, num_samples)

            # Load real data for comparison
            real_loader = self._create_real_loader(num_samples)

            # Compute metrics
            for metric_name in self.config.eval.metrics:
                if metric_name in self.metrics:
                    metric_calc = self.metrics[metric_name]

                    if hasattr(metric_calc, "compute_from_dataloader"):
                        # FID/KID style metrics
                        if metric_name in ["fid", "kid"]:
                            value = metric_calc.compute_from_dataloader(
                                real_loader, fake_loader, num_samples
                            )
                            if metric_name == "fid":
                                results["fid"] = value[0]
                            else:
                                results["kid"] = value[1]

                    elif hasattr(metric_calc, "compute_batch_metrics"):
                        # Image quality metrics
                        batch_results = metric_calc.compute_batch_metrics(
                            real_loader,
                            fake_loader,
                            max_batches=self.config.eval.image.max_batches,
                        )
                        results.update(batch_results)

            # Cache results
            self._cache_results(run_id, checkpoint_path, results, seed)

            self.logger.info(f"Evaluation completed: {results}")

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

        return results

    def _create_fake_loader(self, checkpoint_path: str, num_samples: int) -> DataLoader:
        """Create data loader for generated samples."""
        # Load model based on task type
        if self.config.task_type == "gan":
            from src.models.gan.generator import Generator

            # Calculate output dimension
            if self.config.data.name.lower() == "mnist":
                output_dim = 784
            else:
                output_dim = (
                    self.config.data.channels
                    * self.config.data.image_size
                    * self.config.data.image_size
                )

            generator = Generator(
                latent_dim=self.config.model.z_dim, output_dim=output_dim
            )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "generator_state_dict" in checkpoint:
                generator.load_state_dict(checkpoint["generator_state_dict"])
            else:
                generator.load_state_dict(checkpoint)

            generator.to(self.device)
            generator.eval()

            # Create fake dataset
            class FakeDataset(torch.utils.data.Dataset):
                def __init__(self, generator, num_samples, z_dim, device):
                    self.generator = generator
                    self.num_samples = num_samples
                    self.z_dim = z_dim
                    self.device = device

                def __len__(self):
                    return self.num_samples

                def __getitem__(self, idx):
                    with torch.no_grad():
                        z = torch.randn(1, self.z_dim, device=self.device)
                        fake_img = self.generator(z)

                        # Reshape if needed
                        if fake_img.dim() == 2:  # Flattened
                            channels = self.generator.output_dim // (
                                self.config.data.image_size**2
                            )
                            fake_img = fake_img.view(
                                1,
                                channels,
                                self.config.data.image_size,
                                self.config.data.image_size,
                            )

                        return fake_img.squeeze(0).cpu(), 0

            fake_dataset = FakeDataset(
                generator, num_samples, self.config.model.z_dim, self.device
            )

            return DataLoader(
                fake_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=min(4, self.config.data.num_workers),
            )

        else:
            raise NotImplementedError(
                f"Evaluation for task type {self.config.task_type} not implemented"
            )

    def _create_real_loader(self, num_samples: int) -> DataLoader:
        """Create data loader for real samples."""
        from src.data.datasets import get_dataloader

        # Create a subset of the real dataset
        real_loader = get_dataloader(
            self.config.data,
            is_training=False,
            seed=42,  # Fixed seed for consistent evaluation
        )

        # If we need to limit samples, create a subset
        if num_samples < len(real_loader.dataset):
            from torch.utils.data import Subset

            indices = torch.randperm(len(real_loader.dataset))[:num_samples]
            subset = Subset(real_loader.dataset, indices)

            return DataLoader(
                subset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=min(4, self.config.data.num_workers),
            )

        return real_loader

    def _cache_results(
        self,
        run_id: str,
        checkpoint_path: str,
        results: Dict[str, float],
        seed: Optional[int],
    ) -> None:
        """Cache evaluation results."""
        cache_key = self._generate_cache_key(run_id, checkpoint_path, seed)
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "seed": seed,
            "config_hash": self._get_config_hash(),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _generate_cache_key(
        self, run_id: str, checkpoint_path: str, seed: Optional[int]
    ) -> str:
        """Generate cache key for evaluation results."""
        key_parts = [
            run_id,
            os.path.basename(checkpoint_path),
            str(seed) if seed else "random",
            self._get_config_hash()[:8],
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()

    def _get_config_hash(self) -> str:
        """Get hash of evaluation configuration."""
        config_str = json.dumps(self.config.dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_cached_results(
        self, run_id: str, checkpoint_path: str, seed: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """Get cached evaluation results if available."""
        cache_key = self._generate_cache_key(run_id, checkpoint_path, seed)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Verify config hasn't changed
            if cache_data.get("config_hash") == self._get_config_hash():
                self.logger.info(f"Using cached results for {run_id}")
                return cache_data["results"]

        return None


class EvaluationManager:
    """Manager for coordinating multiple evaluations."""

    def __init__(self, runtime_cfg):
        self.runtime_cfg = runtime_cfg
        self.logger = logging.getLogger(__name__)

    def schedule_evaluation(
        self,
        run_id: str,
        checkpoint_type: str = "best",
        metrics: Optional[List[str]] = None,
        force_recompute: bool = False,
    ) -> str:
        """
        Schedule an evaluation job.

        Args:
            run_id: Run identifier
            checkpoint_type: Checkpoint to evaluate
            metrics: Metrics to compute
            force_recompute: Whether to force recomputation

        Returns:
            Job ID
        """
        from src.jobs.engine import get_job_engine
        from src.jobs.tasks import eval_fidkid_task

        job_engine = get_job_engine()

        job_id = job_engine.submit(
            eval_fidkid_task,
            run_id=run_id,
            checkpoint_type=checkpoint_type,
            metrics=metrics,
            force_recompute=force_recompute,
        )

        self.logger.info(f"Scheduled evaluation job {job_id} for run {run_id}")
        return job_id

    def get_evaluation_status(self, run_id: str) -> Dict[str, Any]:
        """Get evaluation status for a run."""
        # Check for existing results
        results_dir = Path(self.runtime_cfg.metrics_dir) / "reports"
        run_results = []

        for result_file in results_dir.glob(f"*{run_id}*.json"):
            with open(result_file, "r") as f:
                result_data = json.load(f)
                run_results.append(result_data)

        # Sort by timestamp
        run_results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return {
            "run_id": run_id,
            "latest_result": run_results[0] if run_results else None,
            "total_evaluations": len(run_results),
            "results_available": len(run_results) > 0,
        }
