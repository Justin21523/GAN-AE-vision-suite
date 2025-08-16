# ) src/scripts/train_gan.py
import os, math, argparse, time
from typing import Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torchvision.utils import save_image
from torchvision import transforms as T

import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.data.datasets import build_dataset, get_loader
from src.data.transforms import build_transforms
from src.metrics.fidkid import FIDKID
from src.data.transforms import build_transforms


from src.models.gan.generator import DCGANGenerator
from src.models.gan.discriminator import DCGANDiscriminator
from src.models.gan.ema import EMA
from src.losses.gan import (
    d_bce_loss,
    g_bce_loss,
    d_hinge_loss,
    g_hinge_loss,
    compute_gradient_penalty,
)


def build_transforms_from_cfg(cfg, train: bool):
    """
    Wrapper to call your existing build_transforms in either style:
      1) build_transforms(cfg, train=True/False)            # config-driven
      2) build_transforms(dataset, image_size, center_crop, mean, std, train=...)
    """
    data = cfg["data"]
    return build_transforms(
        dataset=data.get("name", "celeba"),
        image_size=int(data["img_size"]),
        center_crop=data.get("center_crop", None),
        mean=data.get("mean", None),
        std=data.get("std", None),
    )


def make_loader(cfg, train: bool):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)
    device = torch.device(
        cfg["training"].get("device", "cuda") if args.device is None else args.device
    )

    os.makedirs(cfg["training"]["logdir"], exist_ok=True)
    logger = setup_logger(cfg["training"]["logdir"])
    logger.info(f"Using device: {device}")

    # Data
    train_t = build_dataset(cfg, train=True)
    val_t = build_dataset(cfg, train=False)
    train_loader = make_loader(cfg, train=True)
    val_loader = make_loader(cfg, train=False)

    # Models
    mcfg = cfg["model"]
    G = DCGANGenerator(
        latent_dim=mcfg["latent_dim"],
        img_channels=mcfg["img_channels"],
        channels=tuple(mcfg["g_channels"]),
        img_size=mcfg["img_size"],
    ).to(device)
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

    use_amp = bool(cfg["training"].get("mixed_precision", True)) and (
        device.type == "cuda"
    )
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    # Fixed noise for sampling
    z_fixed = torch.randn(64, mcfg["latent_dim"], device=device)

    # Metrics
    fidkid = (
        FIDKID(device=str(device)) if cfg["metrics"]["fid_kid"]["enabled"] else None
    )

    # Training Loop
    epochs = cfg["training"]["epochs"]
    gstep = 0
    n_critic = int(cfg["training"].get("n_critic", 1))
    lambda_gp = float(cfg["training"].get("lambda_gp", 10.0))
    is_wgan = mcfg["type"].lower() == "wgan-gp"
    sample_every = int(cfg["training"].get("sample_every", 500))

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        for i, batch in enumerate(train_loader, start=1):
            real = batch[0].to(device)  # dataset returns (img, _)
            if epoch == 1 and i == 1:
                with torch.no_grad():
                    dbg = D.feature(real)
                    logger.info(
                        f"[debug] real shape={tuple(real.shape)}  D.feature -> {tuple(dbg.shape)}"
                    )

            bsz = real.size(0)
            # --------------------------
            # Train D / Critic
            # ---------------------------
            for j in range(n_critic):
                z = torch.randn(bsz, mcfg["latent_dim"], device=device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    fake = G(z).detach()
                    real_logits = D(real)
                    fake_logits = D(fake)

                    if is_wgan:
                        gp = compute_gradient_penalty(
                            D, real.data, fake.data, device=str(device)
                        )
                        d_loss = (
                            -torch.mean(real_logits)
                            + torch.mean(fake_logits)
                            + lambda_gp * gp
                        )
                    else:
                        # use hinge loss by default for DCGAN; switch to BCE if preferred
                        d_loss = d_hinge_loss(real_logits, fake_logits)

                optD.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(d_loss).backward()
                    scaler.step(optD)
                    scaler.update()
                else:
                    d_loss.backward()
                    optD.step()
            # ---------------------------
            # Train G
            # ---------------------------
            z = torch.randn(bsz, mcfg["latent_dim"], device=device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                gen = G(z)
                gen_logits = D(gen)
                if is_wgan:
                    g_loss = -gen_logits.mean()
                else:
                    g_loss = g_hinge_loss(gen_logits)

            optG.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(g_loss).backward()
                scaler.step(optG)
                scaler.update()  # <-- IMPORTANT: update after every step()
            else:
                g_loss.backward()
                optG.step()

            if ema is not None:
                ema.update()

            gstep += 1
            if gstep % 50 == 0:
                logger.info(
                    f"[epoch {epoch}/{epochs}] step={gstep} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}"
                )

            if gstep % sample_every == 0:
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
                save_image((samples + 1) / 2, grid_path, nrow=8)
                logger.info(f"Saved samples to {grid_path}")
                G.train()

        # ---- Validation & FID/KID ----
        if fidkid and (epoch % cfg["metrics"]["fid_kid"]["every_epoch"] == 0):
            logger.info("Accumulating features for FID/KID (val split)...")
            G.eval()
            # collect real features
            with torch.no_grad():
                for vb in val_loader:
                    fidkid.update_real(vb[0].to(device))
                max_s = int(cfg["metrics"]["fid_kid"].get("max_samples", 10000))
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

        # ---- Save ckpt ----
        if epoch % cfg["training"]["save_every"] == 0:
            ckpt = {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }
            path = os.path.join(cfg["training"]["logdir"], f"ckpt_epoch{epoch}.pt")
            torch.save(ckpt, path)
            logger.info(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    main()
