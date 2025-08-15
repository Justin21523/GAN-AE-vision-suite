# src/scripts/train_vae.py
import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.datasets import build_dataset, get_loader
from src.metrics.image import compute_psnr, compute_ssim
from src.losses.reconstruction import mse_loss, l1_loss
from src.scripts.build_model import build_model
from src.models.ae import AutoEncoder, ConvAE, VariationalAutoEncoder


def rec_loss_fn(name):
    name = str(name).lower()
    if name == "l1":
        return l1_loss
    return mse_loss  # default


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for (x,) in loader:
        x = x.to(device)
        if hasattr(model, "is_vae") and model.is_vae:
            recon, _, _ = model(x)
        else:
            recon = model(x)
        total_psnr += compute_psnr(recon, x).item()
        total_ssim += compute_ssim(recon, x).item()
        n += 1
    return total_psnr / max(n, 1), total_ssim / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dataset_mnist.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs

    logger = setup_logger(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    logger.info(f"Device: {device}")

    # dataset & loader
    train_dataset = build_dataset(cfg, train=True)
    val_dataset = build_dataset(cfg, train=False)
    train_loader = get_loader(
        train_dataset,
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )
    val_loader = get_loader(
        val_dataset,
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )

    # model
    # MNIST: channels=1；CelebA: channels=3
    if cfg["data"]["dataset"].lower() == "mnist":
        cfg.setdefault("model", {})["img_channels"] = 1
    else:
        cfg.setdefault("model", {})["img_channels"] = cfg["model"].get(
            "img_channels", 3
        )

    model = build_model(cfg).to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    # optim & loss
    tcfg = cfg.setdefault("training", {})
    epochs = int(tcfg.get("epochs", 5))
    lr = float(tcfg.get("lr", 1e-3))
    beta_kl = float(tcfg.get("beta_kl", 1.0))  # for VAE
    rec_name = tcfg.get("loss", "mse")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rec_fn = rec_loss_fn(rec_name)

    best_psnr = -1

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            # batch can be (images, labels) or dict
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                # common keys: 'image', 'pixel_values', ...
                x = batch.get("image", batch.get("pixel_values"))
                if x is None:
                    raise ValueError(f"Unexpected batch keys: {batch.keys()}")
            else:
                x = batch  # already a tensor

            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if getattr(model, "is_vae", False):
                recon, mu, logvar = model(x)
                rec = rec_fn(recon, x)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec + beta_kl * kl
            else:
                recon = model(x)
                loss = rec_fn(recon, x)

            loss.backward()
            optimizer.step()
            running += loss.item()

        # (optional) validation loop uses the same batch handling:
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get("image", batch.get("pixel_values"))
                    if x is None:
                        raise ValueError(f"Unexpected batch keys: {batch.keys()}")
                else:
                    x = batch
                x = x.to(device, non_blocking=True)

                if getattr(model, "is_vae", False):
                    recon, mu, logvar = model(x)
                    rec = rec_fn(recon, x)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss = rec + beta_kl * kl
                else:
                    recon = model(x)
                    val_loss = rec_fn(recon, x)


if __name__ == "__main__":
    main()
    print("Sucess")
