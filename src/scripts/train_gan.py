"""
GAN training script (DCGAN / WGAN-GP).

High-level flow:
1) Load YAML config (see `configs/gan/*.yaml`)
2) Build dataset + DataLoader
3) Build generator (G) and discriminator/critic (D)
4) Train with either:
   - DCGAN-style hinge losses, or
   - WGAN-GP loss + gradient penalty
5) Periodically save sample grids and checkpoints

Outputs:
- Logs and checkpoints are written under `cfg["training"]["logdir"]`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.checkpoint import checkpoint_payload, load_checkpoint
from src.data.datasets import build_dataset, get_loader
from src.metrics.fidkid import FIDKID
from src.utils.vision import make_grid, save_image_grid
from src.utils.run import JSONLMetricsWriter, build_run_meta, prepare_run_dir, write_config_yaml
from src.utils.diffaugment import diff_augment


from src.models.gan.generator import DCGANGenerator
from src.models.gan.discriminator import DCGANDiscriminator
from src.models.gan.resnet import ResNetDiscriminator, ResNetGenerator
from src.models.gan.ema import EMA
from src.losses.gan import (
    d_bce_loss,
    g_bce_loss,
    d_hinge_loss,
    g_hinge_loss,
    compute_gradient_penalty,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _maybe_tensorboard_writer(run_dir: str):
    enabled = True
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception:
        return None
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)


def _normalize_to_01(x: torch.Tensor, value_range=(-1.0, 1.0)) -> torch.Tensor:
    lo, hi = float(value_range[0]), float(value_range[1])
    y = (x - lo) / (hi - lo)
    return y.clamp(0, 1)


def _save_checkpoint_atomic(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _prune_checkpoints(run_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    p = Path(run_dir)
    ckpts = sorted(p.glob("ckpt_*.pt"), key=lambda x: x.stat().st_mtime)
    # Keep `ckpt_latest.pt` regardless.
    ckpts = [c for c in ckpts if c.name != "ckpt_latest.pt"]
    if len(ckpts) <= keep_last:
        return
    for old in ckpts[: len(ckpts) - keep_last]:
        try:
            old.unlink()
        except Exception:
            pass


def _scale_optimizer_grads(opt: optim.Optimizer, factor: float) -> None:
    if factor == 1.0:
        return
    for group in opt.param_groups:
        for p in group.get("params", []):
            if p is not None and getattr(p, "grad", None) is not None:
                p.grad.mul_(factor)  # type: ignore[union-attr]


def _clip_grad_if_needed(opt: optim.Optimizer, max_norm: float) -> None:
    if max_norm <= 0:
        return
    params = []
    for group in opt.param_groups:
        params.extend([p for p in group.get("params", []) if p is not None and p.grad is not None])
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def make_loader(cfg, train: bool):
    """Build a dataset + DataLoader with fallback defaults."""
    ds = build_dataset(cfg, train=train)
    # Prefer your project's datamodule if available
    try:
        loader = get_loader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            train=train,
            pin_memory=cfg["data"].get("pin_memory", True),
            persistent_workers=cfg["data"].get("persistent_workers", True),
        )
    except Exception:
        loader = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            shuffle=train,
            pin_memory=cfg["data"].get("pin_memory", True),
            persistent_workers=cfg["data"].get("persistent_workers", True),
        )
    return loader


def main():
    """Entrypoint for training a GAN from a YAML config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on generator steps for quick smoke tests.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional override directory for artifacts (config/metrics/samples/ckpt).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="If set, create a subdir under training.logdir (ignored when --run-dir is set).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint saved by this script to resume from.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Load weights from --resume but reset optimizer/scaler/epoch.",
    )
    args = parser.parse_args()

    _require(
        args.config is not None or args.resume is not None,
        "Provide either --config (new run) or --resume (resume existing run).",
    )

    cfg_from_file = load_config(args.config) if args.config is not None else None
    ckpt: Optional[dict] = None

    # If resuming, prefer the config stored in the checkpoint (guarantees model matches).
    # If a --config is also provided, only override training/metrics knobs.
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        if "cfg" not in ckpt:
            raise SystemExit(f"Invalid checkpoint (missing 'cfg'): {args.resume}")
        cfg = ckpt["cfg"]
        if cfg_from_file is not None:
            if "training" in cfg_from_file:
                cfg.setdefault("training", {}).update(dict(cfg_from_file["training"]))
            if "metrics" in cfg_from_file:
                cfg.setdefault("metrics", {}).update(dict(cfg_from_file["metrics"]))
    else:
        cfg = cfg_from_file
    assert cfg is not None

    set_seed(42)
    device = torch.device(
        cfg["training"].get("device", "cuda") if args.device is None else args.device
    )
    if device.type == "cuda":
        tf32 = bool(cfg["training"].get("tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = bool(cfg["training"].get("cudnn_benchmark", True))

    # Run directory:
    # - New runs: optionally create a subdir for isolation.
    # - Resume without overrides: keep writing into the checkpoint's logdir.
    if args.run_dir is not None:
        run_dir = str(prepare_run_dir(args.run_dir, run_name=None, prefix="gan"))
    else:
        base = cfg["training"]["logdir"]
        if ckpt is not None and args.config is None and args.run_name is None:
            # Resume default: keep writing into the checkpoint's logdir.
            run_dir = str(os.path.abspath(base))
        else:
            run_dir = str(prepare_run_dir(base, run_name=args.run_name, prefix="gan"))

    cfg["training"]["logdir"] = str(run_dir)
    os.makedirs(cfg["training"]["logdir"], exist_ok=True)
    logger = setup_logger(cfg["training"]["logdir"])
    logger.info(f"Using device: {device}")

    # Persist resolved config + run metadata.
    write_config_yaml(cfg, Path(cfg["training"]["logdir"]) / "config_resolved.yaml")
    meta = build_run_meta(cfg, extra={"script": "train_gan"})
    with open(Path(cfg["training"]["logdir"]) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    metrics_writer = JSONLMetricsWriter(Path(cfg["training"]["logdir"]) / "metrics.jsonl")
    run_started = time.time()
    tb = _maybe_tensorboard_writer(cfg["training"]["logdir"]) if cfg["training"].get("tensorboard", True) else None
    if tb is not None:
        try:
            tb.add_scalar("meta/start", 1.0, global_step=0)
            tb.flush()
        except Exception:
            pass

    # Data
    train_loader = make_loader(cfg, train=True)
    val_loader = make_loader(cfg, train=False)

    # Models
    mcfg = cfg["model"]
    arch = str(mcfg.get("arch", "dcgan")).lower()
    if arch in {"resnet", "sngan", "biggan"}:
        base = int(mcfg.get("base_channels", 64))
        G = ResNetGenerator(
            latent_dim=mcfg["latent_dim"],
            img_channels=mcfg["img_channels"],
            img_size=mcfg["img_size"],
            base_channels=base,
        ).to(device)
        D = ResNetDiscriminator(
            img_channels=mcfg["img_channels"],
            img_size=mcfg["img_size"],
            base_channels=base,
            sn=bool(mcfg.get("spectral_norm", True)),
        ).to(device)
    else:
        # Generator: z -> image (tanh in [-1, 1])
        G = DCGANGenerator(
            latent_dim=mcfg["latent_dim"],
            img_channels=mcfg["img_channels"],
            channels=tuple(mcfg["g_channels"]),
            img_size=mcfg["img_size"],
        ).to(device)
        # Discriminator/Critic: image -> scalar score/logit
        D = DCGANDiscriminator(
            img_channels=mcfg["img_channels"],
            channels=tuple(mcfg["d_channels"]),
            img_size=mcfg["img_size"],
        ).to(device)

    ema = None
    if mcfg.get("ema", {}).get("enabled", False):
        ema = EMA(G, decay=float(mcfg["ema"].get("decay", 0.999)))
        ema.register()

    # Optims
    betas = tuple(cfg["training"]["betas"])
    optG = optim.Adam(G.parameters(), lr=cfg["training"]["lr_g"], betas=betas)
    optD = optim.Adam(D.parameters(), lr=cfg["training"]["lr_d"], betas=betas)

    use_amp = bool(cfg["training"].get("mixed_precision", True)) and (device.type == "cuda")
    amp_dtype_cfg = str(cfg["training"].get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.float16
    use_scaler = True
    if amp_dtype_cfg in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
        use_scaler = False
    scaler = GradScaler("cuda", enabled=use_amp and use_scaler) if (use_amp and use_scaler) else None

    start_epoch = 1
    gstep = 0
    if ckpt is not None:
        logger.info("Loading checkpoint: %s", args.resume)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])

        if ema is not None:
            shadow = ckpt.get("ema_shadow")
            if isinstance(shadow, dict) and shadow:
                ema.load_shadow(shadow, device=device)
                logger.info("Loaded EMA shadow from checkpoint.")
            else:
                ema.register()

        if not args.finetune:
            if "optG" in ckpt:
                optG.load_state_dict(ckpt["optG"])
            if "optD" in ckpt:
                optD.load_state_dict(ckpt["optD"])
            if scaler is not None and "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    logger.warning("Failed to restore GradScaler state: %s", e)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            gstep = int(ckpt.get("gstep", 0))
            logger.info("Resuming from epoch=%d, step=%d", start_epoch, gstep)
        else:
            logger.info("Finetune mode: optimizer/scaler/epoch reset.")

    # Fixed noise for sampling
    z_fixed = torch.randn(64, mcfg["latent_dim"], device=device)

    # Metrics
    fidkid = None
    if cfg.get("metrics", {}).get("fid_kid", {}).get("enabled", False):
        metric = FIDKID(device=str(device))
        if metric.enabled:
            fidkid = metric
        else:
            logger.warning(
                "FID/KID disabled (missing torchmetrics/torchvision): %s",
                metric.import_error,
            )

    # Training Loop
    epochs = cfg["training"]["epochs"]
    n_critic = int(cfg["training"].get("n_critic", 1))
    lambda_gp = float(cfg["training"].get("lambda_gp", 10.0))
    is_wgan = mcfg["type"].lower() == "wgan-gp"
    sample_every = int(cfg["training"].get("sample_every", 500))
    log_every = int(cfg["training"].get("log_every", 50))
    grad_accum_steps = int(cfg["training"].get("grad_accum_steps", 1))
    save_every_steps = cfg["training"].get("save_every_steps", None)
    save_every_steps = int(save_every_steps) if save_every_steps is not None else None
    keep_last = int(cfg["training"].get("keep_last_checkpoints", 0))
    max_steps = int(args.max_steps) if args.max_steps is not None else None
    stop_early = False
    d_micro = 0
    g_micro = 0
    autocast_device = device.type
    diffaug_policy = str(cfg["training"].get("diffaugment_policy", "")).strip()
    diffaug_p = float(cfg["training"].get("diffaugment_p", 1.0))
    r1_gamma = float(cfg["training"].get("r1_gamma", 0.0))
    r1_every = int(cfg["training"].get("r1_every", 16))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))
    stop_requested = False

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        logger.warning("Received signal %s; will stop after current step.", signum)

    try:
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
    except Exception:
        pass

    for epoch in range(start_epoch, epochs + 1):
        metrics_writer.write(
            {
                "event": "epoch_start",
                "epoch": int(epoch),
                "step": int(gstep),
                "time_s": float(time.time() - run_started),
            }
        )
        G.train()
        D.train()
        for i, batch in enumerate(train_loader, start=1):
            # Most torchvision datasets return (image, label). We only need images.
            real = batch[0].to(device)  # dataset returns (img, _)
            if epoch == 1 and i == 1:
                with torch.no_grad():
                    logger.info(f"[debug] real shape={tuple(real.shape)}")

            bsz = real.size(0)
            # --------------------------
            # Train D / Critic
            # ---------------------------
            for j in range(n_critic):
                if (d_micro % max(1, grad_accum_steps)) == 0:
                    optD.zero_grad(set_to_none=True)
                z = torch.randn(bsz, mcfg["latent_dim"], device=device)
                with torch.autocast(
                    device_type=autocast_device, dtype=amp_dtype, enabled=use_amp
                ):
                    fake = G(z).detach()
                    d_real = diff_augment(real, diffaug_policy, p=diffaug_p) if diffaug_policy else real
                    d_fake = diff_augment(fake, diffaug_policy, p=diffaug_p) if diffaug_policy else fake
                    real_logits = D(d_real)
                    fake_logits = D(d_fake)

                    if is_wgan:
                        # WGAN-GP: maximize D(real) - D(fake) with gradient penalty.
                        with torch.autocast(device_type=autocast_device, enabled=False):
                            gp = compute_gradient_penalty(D, d_real.float(), d_fake.float(), device=str(device))
                        d_loss = (
                            -torch.mean(real_logits)
                            + torch.mean(fake_logits)
                            + lambda_gp * gp
                        )
                    else:
                        # DCGAN-style: hinge loss by default (no sigmoid).
                        d_loss = d_hinge_loss(real_logits, fake_logits)

                # Optional: R1 regularization for hinge GANs (stability at high-res).
                if (not is_wgan) and r1_gamma > 0 and (gstep % max(1, r1_every) == 0):
                    with torch.autocast(device_type=autocast_device, enabled=False):
                        d_real_r1 = d_real.detach().float().requires_grad_(True)
                        real_logits_r1 = D(d_real_r1)
                        grad = torch.autograd.grad(
                            outputs=real_logits_r1.sum(),
                            inputs=d_real_r1,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        r1_pen = 0.5 * r1_gamma * grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()
                    d_loss = d_loss + r1_pen.to(d_loss.dtype)

                loss_d = d_loss / float(max(1, grad_accum_steps))
                if scaler:
                    scaler.scale(loss_d).backward()
                else:
                    loss_d.backward()

                d_micro += 1
                if (d_micro % max(1, grad_accum_steps)) == 0:
                    if scaler:
                        scaler.unscale_(optD)
                        _clip_grad_if_needed(optD, grad_clip_norm)
                        scaler.step(optD)
                        scaler.update()
                    else:
                        _clip_grad_if_needed(optD, grad_clip_norm)
                        optD.step()
            # ---------------------------
            # Train G
            # ---------------------------
            z = torch.randn(bsz, mcfg["latent_dim"], device=device)
            with torch.autocast(
                device_type=autocast_device, dtype=amp_dtype, enabled=use_amp
            ):
                gen = G(z)
                d_gen = diff_augment(gen, diffaug_policy, p=diffaug_p) if diffaug_policy else gen
                gen_logits = D(d_gen)
                if is_wgan:
                    g_loss = -gen_logits.mean()
                else:
                    g_loss = g_hinge_loss(gen_logits)

            if (g_micro % max(1, grad_accum_steps)) == 0:
                optG.zero_grad(set_to_none=True)
            loss_g = g_loss / float(max(1, grad_accum_steps))
            if scaler:
                scaler.scale(loss_g).backward()
            else:
                loss_g.backward()

            did_g_step = False
            g_micro += 1
            if (g_micro % max(1, grad_accum_steps)) == 0:
                if scaler:
                    scaler.unscale_(optG)
                    _clip_grad_if_needed(optG, grad_clip_norm)
                    scaler.step(optG)
                    scaler.update()  # <-- IMPORTANT: update after every step()
                else:
                    _clip_grad_if_needed(optG, grad_clip_norm)
                    optG.step()
                did_g_step = True

            if did_g_step and ema is not None:
                ema.update()

            if did_g_step:
                gstep += 1
            if did_g_step and (gstep % log_every == 0):
                logger.info(
                    f"[epoch {epoch}/{epochs}] step={gstep} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}"
                )
                metrics_writer.write(
                    {
                        "event": "step",
                        "split": "train",
                        "epoch": epoch,
                        "step": gstep,
                        "time_s": float(time.time() - run_started),
                        "d_loss": float(d_loss.detach().cpu()),
                        "g_loss": float(g_loss.detach().cpu()),
                    }
                )
                if tb is not None:
                    try:
                        tb.add_scalar("loss/d", float(d_loss.detach().cpu()), global_step=gstep)
                        tb.add_scalar("loss/g", float(g_loss.detach().cpu()), global_step=gstep)
                        tb.add_scalar("lr/g", float(optG.param_groups[0]["lr"]), global_step=gstep)
                        tb.add_scalar("lr/d", float(optD.param_groups[0]["lr"]), global_step=gstep)
                    except Exception:
                        pass

            if did_g_step and (gstep % sample_every == 0):
                G.eval()
                with torch.no_grad():
                    if ema is not None:
                        ema.apply_shadow()
                    samples = G(z_fixed)
                    if ema is not None:
                        ema.restore()
                grid_path = os.path.join(
                    cfg["training"]["logdir"], f"samples_step{gstep}.png"
                )
                save_image_grid(samples, grid_path, nrow=8, value_range=(-1, 1))
                try:
                    latest = os.path.join(cfg["training"]["logdir"], "samples_latest.png")
                    tmp = latest + ".tmp"
                    save_image_grid(samples, tmp, nrow=8, value_range=(-1, 1))
                    os.replace(tmp, latest)
                except Exception:
                    pass
                if tb is not None:
                    x01 = _normalize_to_01(samples.detach().cpu(), value_range=(-1, 1))
                    tb.add_image("samples/grid", make_grid(x01, nrow=8), global_step=gstep)
                logger.info(f"Saved samples to {grid_path}")
                G.train()

            if did_g_step and save_every_steps is not None and (gstep % save_every_steps == 0):
                ckpt = checkpoint_payload({
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict(),
                    "epoch": epoch,
                    "gstep": gstep,
                    "cfg": cfg,
                })
                if scaler is not None:
                    ckpt["scaler"] = scaler.state_dict()
                if ema is not None and ema.shadow:
                    ckpt["ema_shadow"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
                step_path = os.path.join(cfg["training"]["logdir"], f"ckpt_step{gstep}.pt")
                latest_path = os.path.join(cfg["training"]["logdir"], "ckpt_latest.pt")
                _save_checkpoint_atomic(ckpt, step_path)
                _save_checkpoint_atomic(ckpt, latest_path)
                _prune_checkpoints(cfg["training"]["logdir"], keep_last=keep_last)
                logger.info("Saved step checkpoint: %s", step_path)

            if did_g_step and max_steps is not None and gstep >= max_steps:
                logger.info("Reached --max-steps=%d; stopping early.", max_steps)
                stop_early = True
                break
            if stop_requested:
                stop_early = True
                break

        # If using grad accumulation, avoid dropping the partial micro-batch at epoch boundaries
        # (since grads are not stored in checkpoints).
        accum = max(1, grad_accum_steps)
        d_rem = d_micro % accum
        if d_rem != 0:
            _scale_optimizer_grads(optD, float(accum) / float(d_rem))
            if scaler:
                scaler.unscale_(optD)
                _clip_grad_if_needed(optD, grad_clip_norm)
                scaler.step(optD)
                scaler.update()
            else:
                _clip_grad_if_needed(optD, grad_clip_norm)
                optD.step()
            optD.zero_grad(set_to_none=True)
            d_micro += accum - d_rem
        g_rem = g_micro % accum
        if g_rem != 0:
            _scale_optimizer_grads(optG, float(accum) / float(g_rem))
            if scaler:
                scaler.unscale_(optG)
                _clip_grad_if_needed(optG, grad_clip_norm)
                scaler.step(optG)
                scaler.update()
            else:
                _clip_grad_if_needed(optG, grad_clip_norm)
                optG.step()
            optG.zero_grad(set_to_none=True)
            g_micro += accum - g_rem
            if ema is not None:
                ema.update()
            gstep += 1

        # ---- Validation & FID/KID ----
        if fidkid and (epoch % cfg["metrics"]["fid_kid"]["every_epoch"] == 0):
            logger.info("Accumulating features for FID/KID (val split)...")
            G.eval()
            # collect real features
            max_s = int(cfg["metrics"]["fid_kid"].get("max_samples", 10000))
            with torch.no_grad():
                cnt_real = 0
                for vb in val_loader:
                    if cnt_real >= max_s:
                        break
                    fidkid.update_real(vb[0].to(device))
                    cnt_real += int(vb[0].shape[0])
                cnt = 0
                while cnt < max_s:
                    bs = min(64, max_s - cnt)
                    z = torch.randn(bs, mcfg["latent_dim"], device=device)
                    if ema:
                        ema.apply_shadow()
                    f = G(z)
                    if ema:
                        ema.restore()
                    fidkid.update_fake(f)
                    cnt += bs
            scores = fidkid.compute()
            logger.info(
                f"[epoch {epoch}] FID={scores['fid']:.3f}  KID={scores['kid_mean']:.5f}"
            )
            metrics_writer.write(
                {
                    "event": "epoch_end",
                    "split": "val",
                    "epoch": epoch,
                    "step": gstep,
                    "time_s": float(time.time() - run_started),
                    "fid": float(scores["fid"]),
                    "kid_mean": float(scores["kid_mean"]),
                }
            )
            if tb is not None:
                tb.add_scalar("metrics/fid", float(scores["fid"]), global_step=gstep)
                tb.add_scalar("metrics/kid_mean", float(scores["kid_mean"]), global_step=gstep)
        else:
            metrics_writer.write(
                {
                    "event": "epoch_end",
                    "split": "train",
                    "epoch": int(epoch),
                    "step": int(gstep),
                    "time_s": float(time.time() - run_started),
                }
            )

        # ---- Save ckpt ----
        if epoch % cfg["training"]["save_every"] == 0:
            # Save enough information to re-instantiate the model for sampling later.
            ckpt = checkpoint_payload({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "epoch": epoch,
                "gstep": gstep,
                "cfg": cfg,
            })
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            if ema is not None and ema.shadow:
                ckpt["ema_shadow"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
            path = os.path.join(cfg["training"]["logdir"], f"ckpt_epoch{epoch}.pt")
            latest_path = os.path.join(cfg["training"]["logdir"], "ckpt_latest.pt")
            _save_checkpoint_atomic(ckpt, path)
            _save_checkpoint_atomic(ckpt, latest_path)
            _prune_checkpoints(cfg["training"]["logdir"], keep_last=keep_last)
            logger.info(f"Saved checkpoint: {path}")

        if stop_early:
            break

    if tb is not None:
        try:
            tb.flush()
            tb.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
