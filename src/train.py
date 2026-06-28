"""
Deprecated training entrypoint (stage-0 bootstrap).

This script currently validates:
- config loading
- logger setup
- deterministic seeding

For actual model training, see the scripts under `src/scripts/`.

Recommended entrypoints:
- GAN: `python -m src.scripts.train_gan --config configs/gan/wgangp_celeba128.yaml`
- AE/VAE: `python -m src.scripts.train_ae --config configs/dataset_mnist.yaml`
"""

import argparse
import warnings

import sys, os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    """CLI entrypoint."""
    warnings.warn(
        "`src/train.py` is deprecated; prefer `python -m src.scripts.train_gan` or "
        "`python -m src.scripts.train_ae`.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(config["logging"]["log_dir"], config["logging"]["level"])
    logger.info("Starting training...")
    logger.info(f"Config: {config}")

    set_seed(config["seed"])
    device = config["device"]
    logger.info(f"Using device: {device}")
    logger.info("Stage 0: Initialization completed.")


if __name__ == "__main__":
    main()
