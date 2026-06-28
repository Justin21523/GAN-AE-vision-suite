"""
Variational AutoEncoder (VAE) implementations.

This file includes:
- `VariationalAutoEncoder`: a fixed-architecture VAE for a given `input_size`
- `ConvVAE`: a configurable VAE with `hidden_dims` similar to `ConvAE`

VAE basics:
- Encoder outputs (mu, logvar) parameters of a Gaussian q(z|x)
- Sample z using the reparameterization trick
- Decoder maps z back to an image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    """A simple convolutional VAE with fixed channel progression."""

    is_vae: bool = True

    def __init__(self, img_channels=3, latent_dim=128, input_size=64):
        """
        Args:
            img_channels: Input/output image channels.
            latent_dim: Latent vector size.
            input_size: Input spatial size (assumes divisible by 8 due to 3 downsamples).
        """
        super().__init__()
        self.input_size = input_size

        # ===== Encoder =====
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # -> 32 x 32 x 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 64 x 16 x 16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x 8 x 8
            nn.ReLU(True),
            nn.Flatten(),
        )
        enc_out_dim = 128 * (input_size // 8) * (input_size // 8)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # ===== Decoder =====
        self.dec_fc = nn.Linear(latent_dim, enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass returns (reconstruction, mu, logvar)."""
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z = self.dec_fc(z).view(x.size(0), 128, x.size(2) // 8, x.size(3) // 8)
        recon_x = self.dec(z)
        return recon_x, mu, logvar


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder.
    """

    is_vae: bool = True

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list | None = None,
        activation: str = "relu",
        input_size: int = 64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.input_size = int(input_size)
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = int(latent_dim)
        self.act = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)

        # ---- Encoder ----
        enc_layers = []
        in_ch = input_channels
        for h in self.hidden_dims:
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h),
                    self.act,
                )
            )
            in_ch = h
        self.encoder = nn.Sequential(*enc_layers)

        # spatial
        self._feat_hw = self.input_size // (2 ** len(self.hidden_dims))
        feat_dim = self.hidden_dims[-1] * self._feat_hw * self._feat_hw

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(feat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, self.latent_dim)

        # ---- Decoder ----
        self.dec_fc = nn.Sequential(
            nn.Linear(self.latent_dim, feat_dim),
            nn.Unflatten(1, (self.hidden_dims[-1], self._feat_hw, self._feat_hw)),
        )

        dec_layers = []
        rev = self.hidden_dims[::-1]
        for i in range(len(rev) - 1):
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        rev[i],
                        rev[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(rev[i + 1]),
                    self.act,
                )
            )
        dec_layers.append(
            nn.ConvTranspose2d(
                rev[-1],
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        """Sample z via reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass returns (reconstruction, mu, logvar)."""
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        h_dec = self.dec_fc(z)
        out = self.decoder(h_dec)
        return out, mu, logvar
