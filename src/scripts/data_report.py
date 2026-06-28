"""
Dataset integrity report (local, no downloads).

This script produces a JSON report plus a few sample grids to help you validate:
- dataset file integrity (bad/unreadable images)
- basic distribution signals (modes, sizes)
- transform pipeline that is actually applied
- tensor stats after transforms (mean/std/min/max)

Example:
  python -m src.scripts.data_report --config configs/dataset_celeba.yaml --out ./outputs/data_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.datasets import build_dataset
from src.data.quality import (
    collect_image_paths,
    describe_transforms,
    find_duplicates_by_hash,
    scan_images,
    compute_tensor_stats,
)
from src.data.transforms import build_transforms
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.vision import save_image_grid


def _resolve_root(cfg: Dict[str, Any]) -> Path:
    data = cfg.get("data", cfg)
    return Path(str(data.get("root", "./data")))


def _maybe_collect_disk_images(cfg: Dict[str, Any]) -> Optional[list[Path]]:
    data = cfg.get("data", cfg)
    name = str(data.get("dataset") or data.get("name") or "").lower()
    root = _resolve_root(cfg)

    if name in {"imagefolder", "folder", "images"}:
        return collect_image_paths(root)
    if name == "celeba":
        for c in (root / "img_align_celeba", root / "celeba", root):
            if c.exists():
                return collect_image_paths(c)
        return collect_image_paths(root)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default=None, help="Output directory.")
    ap.add_argument("--max-scan-files", type=int, default=0, help="0 = scan all image files.")
    ap.add_argument("--hash-duplicates", action="store_true", help="Compute SHA256 duplicates (can be slow).")
    ap.add_argument("--hash-max-files", type=int, default=5000, help="Max files to hash when enabled.")
    ap.add_argument("--sample-batches", type=int, default=4, help="How many batches to sample for stats/grids.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size for sampling.")
    ap.add_argument("--num-workers", type=int, default=0, help="Num workers for sampling loaders.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(str(args.out or cfg.get("save", {}).get("out_dir", "./outputs/data_report")))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir), level=str(cfg.get("logging", {}).get("level", "INFO")))
    logger.info("Config: %s", args.config)
    logger.info("Writing report to: %s", out_dir)

    data = cfg.get("data", cfg)
    dataset_name = str(data.get("dataset") or data.get("name") or "unknown")

    report: Dict[str, Any] = {
        "config": args.config,
        "dataset": dataset_name,
        "data_root": str(_resolve_root(cfg)),
        "train_transforms": describe_transforms(build_transforms(cfg, train=True)),
        "val_transforms": describe_transforms(build_transforms(cfg, train=False)),
    }

    # Disk scan (only for imagefolder/celeba).
    disk_paths = _maybe_collect_disk_images(cfg)
    if disk_paths is not None:
        if args.max_scan_files and len(disk_paths) > args.max_scan_files:
            disk_paths = disk_paths[: int(args.max_scan_files)]
        scan = scan_images(disk_paths)
        report["disk_scan"] = {
            "total_files": scan.total_files,
            "ok_files": scan.ok_files,
            "bad_files": scan.bad_files,
            "bad_paths_preview": scan.bad_paths,
            "mode_counts": scan.mode_counts,
            "size_counts_top": dict(sorted(scan.size_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]),
        }
        if args.hash_duplicates:
            dups = find_duplicates_by_hash(disk_paths, max_files=int(args.hash_max_files))
            report["duplicates"] = {"hash_max_files": int(args.hash_max_files), "groups": dups}

    # Sample from dataset pipeline.
    device = torch.device(str(cfg.get("device", "cpu")) if torch.cuda.is_available() else "cpu")

    def _sample_split(train: bool) -> Dict[str, Any]:
        ds = build_dataset(cfg, train=train)
        bs = int(args.batch_size or data.get("batch_size", 16))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=int(args.num_workers))
        stats = compute_tensor_stats(loader, max_batches=int(args.sample_batches), device=device)
        # Save first batch grid.
        batch0 = next(iter(loader))
        x0 = batch0[0] if isinstance(batch0, (tuple, list)) else batch0
        save_image_grid(
            x0[:64],
            out_dir / ("samples_train.png" if train else "samples_val.png"),
            nrow=8,
            value_range=(-1, 1),
        )
        return {"len": int(len(ds)), "tensor_stats": stats}

    report["train_sample"] = _sample_split(train=True)
    try:
        report["val_sample"] = _sample_split(train=False)
    except Exception as e:
        report["val_sample_error"] = str(e)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote: %s", out_dir / "report.json")


if __name__ == "__main__":
    main()

