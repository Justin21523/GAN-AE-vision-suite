"""
Train an AutoEncoder (AE) or Variational AutoEncoder (VAE) from a YAML config.

This is the canonical AE/VAE training entrypoint in this repository snapshot.
It supports:
- Config-driven dataset + transforms (`configs/dataset_*.yaml`)
- AE and VAE losses (reconstruction, optional beta-KL)
- Optional mixed precision (CUDA only)
- Periodic checkpoint + sample grid saving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict

import torch
from torch.amp.grad_scaler import GradScaler

from src.data.datasets import build_dataset, get_loader
from src.losses.reconstruction import l1_loss, mse_loss
from src.metrics.image import compute_psnr, compute_ssim
from src.scripts.build_model import build_model
from src.utils.checkpoint import checkpoint_payload, load_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.run import JSONLMetricsWriter, build_run_meta, prepare_run_dir, write_config_yaml
from src.utils.vision import save_image_grid


def _extract_images(batch: Any) -> torch.Tensor:
    """Normalize common batch formats to a tensor `x`."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("images", batch.get("pixel_values")))
        if x is None:
            raise ValueError(f"Unexpected batch keys: {batch.keys()}")
        return x
    return batch


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to [0,1] for metric computation/visualization."""
    if x.min().item() < 0:
        x = x * 0.5 + 0.5
    return x.clamp(0, 1)


def _recon_loss_fn(name: str):
    """Pick a reconstruction loss function by name."""
    name = str(name).lower().strip()
    if name in {"l1", "mae"}:
        return l1_loss
    return mse_loss


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional override directory for run artifacts (config/metrics/meta).",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="If set, create a subdir under logging.log_dir (ignored when --run-dir is set).",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint saved by this script to resume from.",
    )
    ap.add_argument(
        "--finetune",
        action="store_true",
        help="Load weights from --resume but reset optimizer/scaler/epoch.",
    )
    args = ap.parse_args()

    if args.config is None and args.resume is None:
        raise SystemExit("Provide either --config (new run) or --resume (resume existing run).")

    cfg_from_file = load_config(args.config) if args.config is not None else None
    ckpt = None
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        if "cfg" not in ckpt:
            raise SystemExit(f"Invalid checkpoint (missing 'cfg'): {args.resume}")
        cfg = ckpt["cfg"]
        if cfg_from_file is not None:
            # Keep model/data from checkpoint; allow overriding training/logging/save knobs.
            if "training" in cfg_from_file:
                cfg.setdefault("training", {}).update(dict(cfg_from_file["training"]))
            if "logging" in cfg_from_file:
                cfg.setdefault("logging", {}).update(dict(cfg_from_file["logging"]))
            if "save" in cfg_from_file:
                cfg.setdefault("save", {}).update(dict(cfg_from_file["save"]))
    else:
        cfg = cfg_from_file
    assert cfg is not None

    tcfg = cfg["training"]
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)

    deterministic = bool(cfg.get("deterministic", True))
    set_seed(int(cfg.get("seed", 42)), deterministic=deterministic)

    device_str = args.device or str(cfg.get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    log_dir = str(cfg.get("logging", {}).get("log_dir", "./logs"))
    level = cfg.get("logging", {}).get("level", "INFO")
    # Run directory for artifacts (config/metrics/meta). Keep as simple local-first.
    if args.run_dir is not None:
        run_dir = prepare_run_dir(args.run_dir, run_name=None, prefix="ae")
    else:
        if ckpt is not None and args.config is None and args.run_name is None:
            run_dir = Path(log_dir)
        else:
            run_dir = prepare_run_dir(log_dir, run_name=args.run_name, prefix="ae")
    run_dir = Path(run_dir)

    logger = setup_logger(str(run_dir), level)
    logger.info("Config: %s", args.config or "<from checkpoint>")
    logger.info("Device: %s", device)

    write_config_yaml(cfg, run_dir / "config_resolved.yaml")
    meta = build_run_meta(cfg, extra={"script": "train_ae"})
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_writer = JSONLMetricsWriter(run_dir / "metrics.jsonl")
    run_started = time.time()

    # Dataset / DataLoader
    train_ds = build_dataset(cfg, train=True)
    val_ds = build_dataset(cfg, train=False)
    train_loader = get_loader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        train=True,
        pin_memory=bool(cfg["data"].get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
        seed=int(cfg.get("seed", 42)),
    )
    val_loader = get_loader(
        val_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        train=False,
        pin_memory=bool(cfg["data"].get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
        seed=int(cfg.get("seed", 42)),
    )

    # Model
    model = build_model(cfg).to(device)
    is_vae = bool(getattr(model, "is_vae", False))
    logger.info("Model: %s (is_vae=%s)", model.__class__.__name__, is_vae)

    # Warm up one forward pass so lazy-built modules allocate parameters in fp32
    # even when AMP is enabled.
    with torch.no_grad():
        img_size = int(cfg["data"]["image_size"])
        img_ch = int(cfg["model"].get("img_channels", 3))
        dummy = torch.zeros(
            1, img_ch, img_size, img_size, device=device, dtype=torch.float32
        )
        _ = model(dummy)  # type: ignore[func-returns-value]

    # Optim / loss
    epochs = int(tcfg.get("epochs", 5))
    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    beta_kl = float(tcfg.get("beta_kl", 1.0))
    rec_fn = _recon_loss_fn(str(tcfg.get("loss", "mse")))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(tcfg.get("amp", False)) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    # Outputs
    ckpt_dir = Path(str(tcfg.get("save_dir", "logs/checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(str(cfg.get("save", {}).get("out_dir", "./outputs/ae")))
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val = float("inf")
    if ckpt is not None:
        logger.info("Loading checkpoint: %s", args.resume)
        model.load_state_dict(ckpt["model"])
        if not args.finetune:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scaler is not None and "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    logger.warning("Failed to restore GradScaler state: %s", e)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            logger.info("Resuming from epoch=%d (best_val=%.6f)", start_epoch, best_val)
        else:
            logger.info("Finetune mode: optimizer/scaler/epoch reset.")

    for epoch in range(start_epoch, epochs + 1):
        metrics_writer.write(
            {
                "event": "epoch_start",
                "epoch": int(epoch),
                "time_s": float(time.time() - run_started),
            }
        )
        model.train()
        total = 0.0
        total_rec = 0.0
        total_kl = 0.0
        n_steps = 0

        for batch in train_loader:
            x = _extract_images(batch).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if is_vae:
                    recon, mu, logvar = model(x)  # type: ignore[misc]
                    rec = rec_fn(recon, x)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = rec + beta_kl * kl
                else:
                    recon = model(x)  # type: ignore[misc]
                    rec = rec_fn(recon, x)
                    kl = torch.tensor(0.0, device=device)
                    loss = rec

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total += float(loss.detach().cpu())
            total_rec += float(rec.detach().cpu())
            total_kl += float(kl.detach().cpu())
            n_steps += 1

        # Validation (loss + quick PSNR/SSIM on [0,1] scale)
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = _extract_images(batch).to(device, non_blocking=True)
                if is_vae:
                    recon, mu, logvar = model(x)  # type: ignore[misc]
                    rec = rec_fn(recon, x)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = rec + beta_kl * kl
                else:
                    recon = model(x)  # type: ignore[misc]
                    loss = rec_fn(recon, x)

                x01 = _to_01(x)
                r01 = _to_01(recon)
                val_loss += float(loss.detach().cpu())
                val_psnr += float(compute_psnr(x01, r01, max_val=1.0).mean().cpu())
                val_ssim += float(compute_ssim(x01, r01).mean().cpu())
                n_val += 1

        train_avg = total / max(n_steps, 1)
        train_rec_avg = total_rec / max(n_steps, 1)
        train_kl_avg = total_kl / max(n_steps, 1)
        val_avg = val_loss / max(n_val, 1)
        psnr_avg = val_psnr / max(n_val, 1)
        ssim_avg = val_ssim / max(n_val, 1)

        logger.info(
            "[epoch %d/%d] train=%.4f (rec=%.4f kl=%.4f)  val=%.4f  psnr=%.2f  ssim=%.4f",
            epoch,
            epochs,
            train_avg,
            train_rec_avg,
            train_kl_avg,
            val_avg,
            psnr_avg,
            ssim_avg,
        )
        metrics_writer.write(
            {
                "event": "epoch_end",
                "epoch": epoch,
                "time_s": float(time.time() - run_started),
                "train_loss": train_avg,
                "train_rec": train_rec_avg,
                "train_kl": train_kl_avg,
                "val_loss": val_avg,
                "val_psnr": psnr_avg,
                "val_ssim": ssim_avg,
            }
        )

        # Save sample grids (input + recon) from the first batch of val split.
        try:
            batch0 = next(iter(val_loader))
            x0 = _extract_images(batch0).to(device)
            if is_vae:
                r0, _, _ = model(x0)  # type: ignore[misc]
            else:
                r0 = model(x0)  # type: ignore[misc]
            save_image_grid(
                x0[:64],
                out_dir / f"epoch{epoch:03d}_input.png",
                nrow=8,
                value_range=(-1, 1),
            )
            save_image_grid(
                r0[:64],
                out_dir / f"epoch{epoch:03d}_recon.png",
                nrow=8,
                value_range=(-1, 1),
            )
        except Exception as e:
            logger.warning("Failed to write sample grids: %s", e)

        # Checkpoints
        ckpt: Dict[str, Any] = checkpoint_payload({
            "epoch": epoch,
            "model_type": str(cfg.get("model", {}).get("type", "ae")),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "best_val": best_val,
        })
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, ckpt_dir / f"ckpt_epoch{epoch}.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ckpt, ckpt_dir / "ckpt_best.pt")
            logger.info(
                "Saved best checkpoint (val=%.4f) -> %s",
                best_val,
                ckpt_dir / "ckpt_best.pt",
            )


if __name__ == "__main__":
    main()
