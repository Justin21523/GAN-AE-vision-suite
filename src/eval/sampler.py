import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
import time


class DeterministicSampler:
    """Deterministic sampler for evaluation and grid generation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        self._fixed_noise = {}

    def generate_samples(
        self,
        generator: nn.Module,
        num_samples: int,
        z_dim: int,
        device: str = "cuda",
        seed: Optional[int] = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Generate samples deterministically.

        Args:
            generator: Generator model
            num_samples: Number of samples to generate
            z_dim: Latent dimension
            device: Device for generation
            seed: Random seed (uses instance seed if None)
            batch_size: Batch size for generation

        Returns:
            Generated samples
        """
        if seed is None:
            seed = self.seed

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        generator.eval()
        samples = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)

                # Generate latent noise
                z = torch.randn(current_batch, z_dim, device=device)

                # Generate samples
                batch_samples = generator(z)
                samples.append(batch_samples.cpu())

        generator.train()
        return torch.cat(samples, dim=0)

    def get_fixed_noise(
        self, z_dim: int, num_samples: int, device: str = "cuda", key: str = "default"
    ) -> torch.Tensor:
        """
        Get fixed noise for consistent sampling across evaluations.

        Args:
            z_dim: Latent dimension
            num_samples: Number of samples
            device: Device for noise
            key: Key for caching fixed noise

        Returns:
            Fixed noise tensor
        """
        cache_key = f"{key}_{z_dim}_{num_samples}"

        if cache_key not in self._fixed_noise:
            # Set seed for reproducible noise generation
            original_seed = torch.initial_seed()
            torch.manual_seed(self.seed)

            self._fixed_noise[cache_key] = torch.randn(
                num_samples, z_dim, device=device
            )

            # Restore original seed
            if original_seed is not None:
                torch.manual_seed(original_seed)

        return self._fixed_noise[cache_key]

    def generate_sample_grid(
        self,
        generator: nn.Module,
        grid_size: Tuple[int, int] = (8, 8),
        z_dim: int = 100,
        device: str = "cuda",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a grid of samples for visualization.

        Args:
            generator: Generator model
            grid_size: Grid dimensions (rows, cols)
            z_dim: Latent dimension
            device: Device for generation
            seed: Random seed

        Returns:
            Grid of samples
        """
        rows, cols = grid_size
        num_samples = rows * cols

        if seed is None:
            seed = self.seed

        # Use fixed noise for consistent grids
        fixed_noise = self.get_fixed_noise(
            z_dim, num_samples, device, f"grid_{rows}x{cols}"
        )

        generator.eval()
        with torch.no_grad():
            grid_samples = generator(fixed_noise)
        generator.train()

        return grid_samples


class BudgetedEvaluator:
    """Evaluator with time and compute budget controls."""

    def __init__(self, max_time_s: float = 3600.0, max_samples: int = 50000):
        self.max_time_s = max_time_s
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)

    def evaluate_with_budget(
        self, eval_function: callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation with time and sample budget.

        Args:
            eval_function: Evaluation function to call
            *args: Arguments for evaluation function
            **kwargs: Keyword arguments for evaluation function

        Returns:
            Evaluation results
        """
        start_time = time.time()

        try:
            # Apply sample budget if specified in kwargs
            if "num_samples" in kwargs:
                kwargs["num_samples"] = min(kwargs["num_samples"], self.max_samples)

            # Run evaluation
            results = eval_function(*args, **kwargs)

            # Check time budget
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_time_s:
                self.logger.warning(
                    f"Evaluation exceeded time budget: {elapsed_time:.2f}s > {self.max_time_s}s"
                )
                results["time_exceeded"] = True
            else:
                results["time_exceeded"] = False

            results["evaluation_time_s"] = elapsed_time
            return results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {
                "error": str(e),
                "evaluation_time_s": time.time() - start_time,
                "time_exceeded": False,
            }
