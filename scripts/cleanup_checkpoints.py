#!/usr/bin/env python3
"""
Checkpoint cleanup script with policy-based retention.
"""

import os
import argparse
import logging
from pathlib import Path

from src.utils.runtime import bootstrap_runtime
from src.jobs.tasks import cleanup_checkpoints_task


def main():
    parser = argparse.ArgumentParser(description="Clean up old checkpoints")
    parser.add_argument(
        "--keep-last",
        type=int,
        default=5,
        help="Number of most recent checkpoints to keep",
    )
    parser.add_argument(
        "--keep-best", type=int, default=1, help="Number of best checkpoints to keep"
    )
    parser.add_argument(
        "--min-interval-steps",
        type=int,
        default=1000,
        help="Minimum steps between kept checkpoints",
    )
    args = parser.parse_args()

    # Bootstrap runtime
    cfg, info = bootstrap_runtime()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Run cleanup task
    try:
        result = cleanup_checkpoints_task(
            keep_last=args.keep_last,
            keep_best=args.keep_best,
            min_interval_steps=args.min_interval_steps,
        )

        logger.info(f"Checkpoint cleanup completed: {result}")

    except Exception as e:
        logger.error(f"Checkpoint cleanup failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
