"""
AutoEncoder implementations (simple convolutional encoder/decoder).

This file contains a small AE that:
- encodes an image with a few Conv2d layers
- flattens to a latent vector via a Linear layer
- decodes back by projecting to the encoded feature shape and using ConvTranspose2d

Implementation detail:
The fully-connected (Linear) layers are built lazily on the first forward pass
so the code can infer the flattened encoder feature size from the input shape.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AutoEncoder(nn.Module):
    """A lightweight convolutional AutoEncoder (AE)."""

    def __init__(
        self,
        img_channels: int,
        latent_dim: int,
        hidden_dims,
        activation: str = "relu",
        input_size: Tuple[int, int] = (28, 28),
    ):
        """
        Args:
            img_channels: Number of channels in the input image (1 or 3).
            latent_dim: Dimensionality of the latent representation.
            hidden_dims: Reserved for future expansion; current blocks are fixed.
            activation: Reserved for future expansion; current blocks use ReLU.
            input_size: Input spatial size hint (currently not used directly).
        """
        super().__init__()
        self.img_channels = int(img_channels)
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Build encoder conv blocks in __init__ so they move with `model.to(device)`.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        # Lazy FC layers (initialized on first forward once we know feature dims).
        self._fc_built = False
        self.flatten = nn.Flatten()

        # Decoder (ConvTranspose2d) is also built lazily after we know (C, H, W).
        self._decoder_conv = None  # will be built after we know shape

    def _init_fc_layers(self, x: torch.Tensor) -> None:
        """
        Lazily build FC + decoder layers using the SAME device/dtype as `x`.

        Why:
        - The encoder output feature size depends on the input resolution.
        - Creating layers on the same device avoids CPU/GPU mismatch errors.
        """
        if self._fc_built:
            return
        dev, dt = x.device, x.dtype

        with torch.no_grad():
            feat = self.encoder(x)  # B, C, H, W
            b, c, h, w = feat.shape
            enc_out_dim = c * h * w

        # Create new layers directly on the target device/dtype (avoid CPU by default).
        self.fc_mu = nn.Linear(
            enc_out_dim, self.latent_dim, bias=True, device=dev, dtype=dt
        )
        self.fc_decode = nn.Linear(
            self.latent_dim, enc_out_dim, bias=True, device=dev, dtype=dt
        )

        # Keep (C, H, W) for decoding from a flat vector back to a feature map.
        self.unflatten = nn.Unflatten(1, (c, h, w))

        # Decoder conv stack (also created on the same device/dtype).
        self._decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c, 32, 4, 2, 1, device=dev, dtype=dt),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, device=dev, dtype=dt),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.img_channels, 3, 1, 1, device=dev, dtype=dt),
            nn.Tanh(),
        )

        self._fc_built = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image into a flattened feature vector (pre-latent)."""
        z = self.encoder(x)
        z = self.flatten(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a flattened feature vector back into an image."""
        z = self.unflatten(z)  # B, C, H, W
        x = self._decoder_conv(z)  # type: ignore
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct `x`."""
        # If lazy layers are not built yet, build them based on `x`'s device/dtype.
        if not self._fc_built:
            self._init_fc_layers(x)

        enc = self.encoder(x)
        enc_flat = self.flatten(enc)
        # Simple AE: treat `fc_mu` as a deterministic projection to latent space.
        z = self.fc_mu(enc_flat)
        dec_flat = self.fc_decode(z)
        recon = self.decode(dec_flat)
        return recon
