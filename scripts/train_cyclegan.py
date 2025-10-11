#!/usr/bin/env python3
"""
CycleGAN training script with warehouse integration.
"""

import os
import torch
from pathlib import Path
import logging

# Bootstrap runtime and load config
from src.utils.runtime import bootstrap_runtime
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.img2img.datasets import get_img2img_dataloader
from src.img2img.cyclegan.generators import CycleGANDualGenerator
from src.img2img.cyclegan.discriminators import CycleGANDualDiscriminator
from src.img2img.cyclegan.trainer import CycleGANTrainer


def main():
    """Main training function."""
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Train CycleGAN")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Bootstrap runtime
    cfg, info = bootstrap_runtime()
    logging.info(f"AI Warehouse initialized at: {cfg.cache_root}")

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logger_manager = setup_logger(
        config.run_id, os.path.join(config.output_dir, "logs")
    )
    logger = logging.getLogger(__name__)

    # Set seed
    set_seed(config.seed, config.deterministic)

    # Get data loaders for both domains
    train_loader_A = get_img2img_dataloader(config.data, is_training=True, domain="A")
    train_loader_B = get_img2img_dataloader(config.data, is_training=True, domain="B")

    val_loader_A = get_img2img_dataloader(config.data, is_training=False, domain="A")
    val_loader_B = get_img2img_dataloader(config.data, is_training=False, domain="B")

    # Create models
    generators = CycleGANDualGenerator(
        in_channels=config.data.channels,
        out_channels=config.data.channels,
        features=config.model.features,
        num_residual_blocks=config.model.num_residual_blocks,
    )

    discriminators = CycleGANDualDiscriminator(
        in_channels=config.data.channels,
        features=config.model.features,
        num_layers=config.model.num_layers,
    )

    # Create trainer
    trainer = CycleGANTrainer(
        config=config,
        generators=generators,
        discriminators=discriminators,
        train_loader_A=train_loader_A,
        train_loader_B=train_loader_B,
        val_loader_A=val_loader_A,
        val_loader_B=val_loader_B,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Register run
    from src.registry.index import get_run_registry

    registry = get_run_registry()
    registry.register_run(config.run_id, config.dict(), status="running")

    # Train
    logger.info("Starting CycleGAN training...")
    trainer.train()

    # Update registry
    registry.update_run_status(config.run_id, "completed")

    # Generate final samples
    trainer.sample_images(trainer.global_step)

    # Save final model
    trainer.save_checkpoint("final_model.pth")


if __name__ == "__main__":
    main()
