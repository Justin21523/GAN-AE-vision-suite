import torch
import torch.nn.functional as F
import sys, os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.image import compute_psnr, compute_ssim


class AETrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def loss_ae(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def loss_vae(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kld_loss, recon_loss, kld_loss

    def train_epoch(self, loader, model_type="ae"):
        self.model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for x, _ in loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()

            if model_type == "ae":
                recon = self.model(x)
                loss = self.loss_ae(recon, x)
            else:
                recon, mu, logvar = self.model(x)
                loss, _, _ = self.loss_vae(recon, x, mu, logvar)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_psnr += compute_psnr(recon, x)
            total_ssim += compute_ssim(recon, x)

        return (
            total_loss / len(loader),
            total_psnr / len(loader),
            total_ssim / len(loader),
        )

    def eval_epoch(self, loader, model_type="ae"):
        self.model.eval()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                if model_type == "ae":
                    recon = self.model(x)
                    loss = self.loss_ae(recon, x)
                else:
                    recon, mu, logvar = self.model(x)
                    loss, _, _ = self.loss_vae(recon, x, mu, logvar)

                total_loss += loss.item()
                total_psnr += compute_psnr(recon, x)
                total_ssim += compute_ssim(recon, x)

        return (
            total_loss / len(loader),
            total_psnr / len(loader),
            total_ssim / len(loader),
        )
