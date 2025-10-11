#!/usr/bin/env python3
"""
Export model card for a specific run.
"""

import os
import argparse
import logging
from pathlib import Path

from src.utils.runtime import bootstrap_runtime
from src.reporting.model_card import create_model_card


def main():
    parser = argparse.ArgumentParser(description="Export model card for a run")
    parser.add_argument("run_id", type=str, help="Run identifier")
    parser.add_argument("--output-dir", type=str, help="Output directory", default=None)
    args = parser.parse_args()

    # Bootstrap runtime
    cfg, info = bootstrap_runtime()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(cfg.train_dir, args.run_id)

    # Check if run exists
    if not os.path.exists(output_dir):
        logger.error(f"Run directory not found: {output_dir}")
        return

    # Load manifest
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return

    import json

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Create model card
    card_path = create_model_card(args.run_id, output_dir, manifest)

    logger.info(f"✅ Model card exported: {card_path}")


if __name__ == "__main__":
    main()
