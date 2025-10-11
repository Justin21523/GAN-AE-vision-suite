#!/usr/bin/env python3
"""
Image-to-Image training script with warehouse integration.
"""

import os
import torch
from pathlib import Path
import logging
import argparse

# Bootstrap runtime and load config
from src.utils.runtime import bootstrap_runtime
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.manifest import create_run_manifest


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Image-to-Image model")
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

    # Create run manifest
    manifest = create_run_manifest(config.run_id, config.output_dir, config.dict())

    # Register run
    from src.registry.index import get_run_registry

    registry = get_run_registry()
    registry.register_run(config.run_id, config.dict(), status="running")

    try:
        # Load data
        from src.data.datasets import get_dataloader

        if config.task == "img2img_paired":
            train_loader = get_dataloader(config.data, is_training=True)
            val_loader = get_dataloader(config.data, is_training=False)

            # Create Pix2Pix model
            from src.models.img2img.pix2pix import (
                Pix2PixGenerator,
                PatchGANDiscriminator,
            )
            from src.training.img2img_trainer import Pix2PixTrainer

            generator = Pix2PixGenerator(
                in_channels=config.model.in_ch,
                out_channels=config.model.out_ch,
                features=config.model.features,
            )

            discriminator = PatchGANDiscriminator(
                in_channels=config.model.in_ch, features=config.model.features
            )

            # Create trainer
            trainer = Pix2PixTrainer(
                config=config,
                generator=generator,
                discriminator=discriminator,
                train_loader=train_loader,
                val_loader=val_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        elif config.task == "img2img_unpaired":
            # For CycleGAN, we need separate loaders for each domain
            from src.data.datasets import get_dataloader

            train_loader_A = get_dataloader(config.data, is_training=True, domain="A")
            train_loader_B = get_dataloader(config.data, is_training=True, domain="B")
            val_loader_A = get_dataloader(config.data, is_training=False, domain="A")
            val_loader_B = get_dataloader(config.data, is_training=False, domain="B")

            # Create CycleGAN model
            from src.models.img2img.cyclegan import (
                CycleGANDualGenerator,
                CycleGANDualDiscriminator,
            )
            from src.training.cyclegan_trainer import CycleGANTrainer

            generators = CycleGANDualGenerator(
                in_channels=config.model.in_ch,
                out_channels=config.model.out_ch,
                features=config.model.features,
                num_residual_blocks=config.model.num_residual_blocks,
            )

            discriminators = CycleGANDualDiscriminator(
                in_channels=config.model.in_ch,
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

        else:
            raise ValueError(f"Unsupported task type: {config.task}")

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        logger.info(f"Starting {config.task} training...")
        trainer.train()

        # Update registry and manifest
        registry.update_run_status(config.run_id, "completed")
        manifest.finalize_manifest("completed")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        registry.update_run_status(config.run_id, "failed")
        manifest.finalize_manifest("failed")
        raise


if __name__ == "__main__":
    main()
