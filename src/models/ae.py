# src/models/ae.py

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, img_size: int = 64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 32x32x32 (if 64x64 in)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x8x8
            nn.ReLU(inplace=True),
        )
        self.enc_proj = nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)

        # Decoder
        self.dec_proj = nn.Linear(latent_dim, 128 * (img_size // 8) * (img_size // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # Cx64x64
            nn.Sigmoid(),
        )
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    def encode(self, x):
        b = x.size(0)
        feat = self.encoder(x)
        feat = feat.view(b, -1)
        z = self.enc_proj(feat)
        return z

    def decode(self, z):
        b = z.size(0)
        feat = self.dec_proj(z).view(b, 128, self.img_size // 8, self.img_size // 8)
        x_rec = self.decoder(feat)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
