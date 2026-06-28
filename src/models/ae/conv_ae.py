"""
Convolutional AutoEncoder (ConvAE).

This model is a "symmetric" encoder/decoder:
- Encoder: strided Conv2d blocks downsample the image.
- Latent: a Linear layer maps the flattened feature map to `latent_dim`.
- Decoder: ConvTranspose2d blocks upsample back to the original resolution.

The output uses `tanh`, so training data is usually normalized to [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    """
    Convolutional Autoencoder with symmetric encoder-decoder architecture.
    Works for variable input sizes (e.g., 28x28 MNIST, 64x64 CelebA).
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list | None = None,
        activation: str = "relu",
        input_size: int = 64,
    ):
        """
        Args:
            input_channels: Input image channels (1 or 3 typically).
            latent_dim: Size of the latent vector.
            hidden_dims: Channel widths per encoder stage; decoder mirrors this.
            activation: "relu" or leaky-relu variant.
            input_size: Input spatial size (must be divisible by 2**len(hidden_dims)).
        """
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

        # Downsampled spatial size after strided conv blocks.
        self._feat_hw = self.input_size // (2 ** len(self.hidden_dims))
        feat_dim = self.hidden_dims[-1] * self._feat_hw * self._feat_hw

        self.enc_fc = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim, self.latent_dim))
        self.dec_fc = nn.Sequential(
            nn.Linear(self.latent_dim, feat_dim),
            nn.Unflatten(1, (self.hidden_dims[-1], self._feat_hw, self._feat_hw)),
        )

        # ---- Decoder (mirror of encoder) ----
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
        # Final stage: map back to image space and squash with tanh.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode -> latent -> decode to reconstruct the input."""
        h = self.encoder(x)
        z = self.enc_fc(h)
        h_dec = self.dec_fc(z)
        out = self.decoder(h_dec)
        return out
