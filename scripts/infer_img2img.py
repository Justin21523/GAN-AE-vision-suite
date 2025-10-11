#!/usr/bin/env python3
"""
Image-to-Image inference script.
"""

import os
import torch
import argparse
from pathlib import Path
import logging

from src.utils.runtime import bootstrap_runtime
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from Image-to-Image model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input-dir", type=str, help="Input directory for source images"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for generated images"
    )
    parser.add_argument(
        "--num-samples", type=int, default=16, help="Number of samples to generate"
    )

    args = parser.parse_args()

    # Bootstrap runtime
    cfg, info = bootstrap_runtime()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set seed
    set_seed(config.seed)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.output_dir) / "inference"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model based on task type
        if config.task == "img2img_paired":
            from src.models.img2img.pix2pix import Pix2PixGenerator

            model = Pix2PixGenerator(
                in_channels=config.model.in_ch,
                out_channels=config.model.out_ch,
                features=config.model.features,
            )

        elif config.task == "img2img_unpaired":
            from src.models.img2img.cyclegan import CycleGANDualGenerator

            model = CycleGANDualGenerator(
                in_channels=config.model.in_ch,
                out_channels=config.model.out_ch,
                features=config.model.features,
                num_residual_blocks=config.model.num_residual_blocks,
            )
        else:
            raise ValueError(f"Unsupported task type: {config.task}")

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "generator_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["generator_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Generate samples
        if args.input_dir:
            # Generate from input images
            from src.data.datasets import ImageFolderPaired, ImageFolderUnpaired
            from src.data.transforms import get_transforms

            transforms = get_transforms(config.data, is_training=False)

            if config.task == "img2img_paired":
                dataset = ImageFolderPaired(args.input_dir, transforms)
            else:
                dataset = ImageFolderUnpaired(args.input_dir, "A", transforms)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False
            )

            for i, batch in enumerate(dataloader):
                if i >= args.num_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    input_img = batch[0]
                else:
                    input_img = batch

                input_img = input_img.to(device)

                with torch.no_grad():
                    if config.task == "img2img_paired":
                        output_img = model(input_img)
                    else:
                        output_img = model.generate_A2B(input_img)

                # Save output
                from src.data.transforms import create_sample_grid

                grid = create_sample_grid(output_img, nrow=1)

                from PIL import Image
                import torchvision.transforms as transforms

                output_pil = transforms.ToPILImage()(grid.cpu())
                output_path = output_dir / f"output_{i:04d}.png"
                output_pil.save(output_path)

                logger.info(f"Saved output: {output_path}")

        else:
            # Generate from random noise
            with torch.no_grad():
                if config.task == "img2img_paired":
                    # For paired tasks, we need input images
                    # Generate random input matching the expected distribution
                    input_imgs = torch.randn(
                        args.num_samples,
                        config.model.in_ch,
                        config.data.image_size,
                        config.data.image_size,
                        device=device,
                    )
                    output_imgs = model(input_imgs)
                else:
                    # For unpaired tasks, generate from random noise
                    input_imgs = torch.randn(
                        args.num_samples,
                        config.model.in_ch,
                        config.data.image_size,
                        config.data.image_size,
                        device=device,
                    )
                    output_imgs = model.generate_A2B(input_imgs)

            # Save grid
            from src.data.transforms import create_sample_grid

            grid = create_sample_grid(output_imgs, nrow=4)

            from src.metrics.image_metrics import GridWriter

            grid_writer = GridWriter(str(output_dir))
            filename = f"generated_samples.png"
            grid_writer.save_image_grid(output_imgs, filename, nrow=4)

            logger.info(f"Samples saved to: {output_dir / filename}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
