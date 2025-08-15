# scripts/train_ae.py
import torch
from torch.utils.data import DataLoader
import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.datasets import build_dataset, get_loader
from src.models.ae import AutoEncoder
from src.models.ae.conv_ae import ConvAE
from src.models.ae.vae import ConvVAE
from src.losses.reconstruction import mse_loss
from src.metrics.image import compute_ssim, compute_psnr


def main():
    cfg = load_config("configs/dataset_mnist.yaml")
    device = torch.device(cfg["device"])

    # dataset / dataloader
    train_dataset = build_dataset(cfg, train=True)
    train_loader = get_loader(
        train_dataset,
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=True,
        pin_memory=(device.type == "cuda"),  # ✅ 推薦
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )

    # build model
    if cfg["model"]["type"] == "ae":
        model = AutoEncoder(
            img_channels=cfg["model"]["img_channels"],
            latent_dim=cfg["model"]["latent_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            activation=cfg["model"]["activation"],
            input_size=cfg["data"]["image_size"],
        )
    elif cfg["model"]["type"] == "conv_ae":
        model = ConvAE(
            input_channels=cfg["model"]["img_channels"],
            latent_dim=cfg["model"]["latent_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            activation=cfg["model"]["activation"],
            input_size=cfg["data"]["image_size"],
        )
    elif cfg["model"]["type"] == "vae":
        model = ConvVAE(
            input_channels=cfg["model"]["img_channels"],
            latent_dim=cfg["model"]["latent_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            activation=cfg["model"]["activation"],
            input_size=cfg["data"]["image_size"],
        )
    else:
        raise ValueError(f"Unknown model type {cfg['model']['type']}")

    model = model.to(device)  # ✅ 核心修正：整個模型搬到 device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # （可選）加速
    torch.backends.cudnn.benchmark = device.type == "cuda"

    for epoch in range(5):
        model.train()
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)  # ✅ 輸入搬到同一個 device
            recon = model(x)
            loss = mse_loss(recon, x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}] - Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
