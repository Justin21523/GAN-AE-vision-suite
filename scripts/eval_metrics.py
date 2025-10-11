#!/usr/bin/env python3
"""
Metrics evaluation script for trained models.
"""

import os
import argparse
import logging

# Bootstrap runtime
from src.utils.runtime import bootstrap_runtime
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate model metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="best",
        help="Checkpoint type (latest/best)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid", "kid", "psnr", "ssim"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--force-recompute", action="store_true", help="Force recomputation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples for evaluation",
    )

    args = parser.parse_args()

    # Bootstrap runtime
    cfg, info = bootstrap_runtime()

    # Setup logging
    logger = setup_logger("eval_metrics", os.path.join(cfg.log_dir, "eval"))
    logging.info(f"AI Warehouse: {cfg.cache_root}")

    try:
        from src.service.eval_runner import EvalOrchestrator

        orchestrator = EvalOrchestrator(cfg)

        results = orchestrator.run_evaluation_pipeline(
            run_id=args.run_id,
            checkpoint_type=args.checkpoint_type,
            metrics=args.metrics,
            force_recompute=args.force_recompute,
        )

        logging.info(f"Evaluation results for {args.run_id}:")
        for metric, value in results.items():
            logging.info(f"  {metric}: {value:.4f}")

        # Save results
        output_file = os.path.join(
            cfg.metrics_dir, f"{args.run_id}_{args.checkpoint_type}_eval.json"
        )
        import json

        with open(output_file, "w") as f:
            json.dump(
                {
                    "run_id": args.run_id,
                    "checkpoint_type": args.checkpoint_type,
                    "results": results,
                    "num_samples": args.num_samples,
                },
                f,
                indent=2,
            )

        logging.info(f"Results saved to: {output_file}")

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
