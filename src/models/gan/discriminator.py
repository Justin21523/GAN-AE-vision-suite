"""
DCGAN-style discriminator / WGAN-GP critic implementation.

The discriminator downsamples an input image using strided convolutions and
outputs a single scalar per image:
- For DCGAN (hinge/BCE): interpret as a logit.
- For WGAN-GP: interpret as an unbounded critic score (no sigmoid).
"""

import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
    """Same initialization policy as the generator (DCGAN convention)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # type: ignore
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)  # type: ignore
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # type: ignore
        nn.init.constant_(m.bias.data, 0.0)  # type: ignore


class DCGANDiscriminator(nn.Module):
    """
    Discriminator/Critic:
      - For DCGAN: return logit (use BCEWithLogitsLoss)
      - For WGAN-GP: also ok (don't apply sigmoid), treat as critic score.
    """

    def __init__(
        self,
        img_channels: int = 3,
        channels=(64, 128, 256, 512, 1024),
        img_size: int = 128,
    ):
        """
        Args:
            img_channels: Input image channels (1 or 3 typically).
            channels: Feature widths for each downsampling stage.
            img_size: Input spatial resolution; the network downsamples by 2 each stage.
        """
        super().__init__()
        layers = []
        in_c = img_channels
        spatial = img_size
        # Strided downsampling blocks (each stage halves H/W).
        for i, out_c in enumerate(channels):
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
            spatial //= 2

        self.feature = nn.Sequential(*layers)
        # Do not assume the final spatial size is exactly 4x4.
        # AdaptiveAvgPool2d(1) collapses HxW to 1x1 so the head is resolution-agnostic.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )  # → (B,1,1,1)
        self.apply(weights_init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W) in the same normalization as training data.

        Returns:
            A 1D tensor of shape (B,) containing logits/scores.
        """
        h = self.feature(x)  # (B, C, H', W'), where H'/W' depend on `img_size`.
        out = self.head(h).view(x.size(0))  # (B,)
        return out
