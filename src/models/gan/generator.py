"""
DCGAN-style generator implementation.

The generator maps a latent vector `z` ~ N(0, I) to an image tensor in [-1, 1]
using transposed convolutions and a final `tanh` activation.

This is commonly used as the backbone for DCGAN and can also serve as the
generator for WGAN-GP when paired with a critic that outputs raw scores.
"""

import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
    """
    DCGAN-style weight initialization.

    - Conv/ConvTranspose: N(0, 0.02)
    - BatchNorm weight:   N(1, 0.02), bias: 0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # type: ignore
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)  # type: ignore
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # type: ignore
        nn.init.constant_(m.bias.data, 0.0)  # type: ignore


class DCGANGenerator(nn.Module):
    """Generator that maps z -> RGB image (Tanh in [-1,1])."""

    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        channels=(1024, 512, 256, 128, 64),
        img_size: int = 128,
    ):
        """
        Args:
            latent_dim: Dimensionality of the noise vector z.
            img_channels: Output image channels (1 for grayscale, 3 for RGB).
            channels: Feature map widths for each upsampling stage.
            img_size: Final spatial resolution (e.g., 32/64/128).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Start from 4x4 feature map then upsample x2 → 128x128
        modules = []
        in_c = channels[0]
        modules += [
            nn.ConvTranspose2d(
                latent_dim, in_c, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
        ]
        spatial = 4
        for i in range(len(channels) - 1):
            modules += [
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(True),
            ]
            spatial *= 2

        # Final head: choose stride=2 if we still need one more x2 to hit img_size
        if spatial == img_size:
            modules += [
                nn.Conv2d(
                    channels[-1],
                    img_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.Tanh(),
            ]
        elif spatial * 2 == img_size:
            modules += [
                nn.ConvTranspose2d(
                    channels[-1],
                    img_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Tanh(),
            ]
            spatial *= 2
        else:
            raise AssertionError(
                f"Built size {spatial} cannot match img_size {img_size}. "
                f"Hint: 4 * 2^(len(g_channels)-1) should be either {img_size} or {img_size}/2."
            )

        assert spatial == img_size, f"Built size {spatial} != img_size {img_size}"

        self.net = nn.Sequential(*modules)
        self.apply(weights_init_normal)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Generated images of shape (B, img_channels, img_size, img_size) in [-1, 1].
        """
        # Reshape into a "1x1 feature map" so ConvTranspose2d can upsample it.
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        img = self.net(z)
        return img
