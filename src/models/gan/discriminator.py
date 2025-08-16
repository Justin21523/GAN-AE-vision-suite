import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
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
        super().__init__()
        layers = []
        in_c = img_channels
        spatial = img_size
        # 連續 stride=2 下採樣
        for i, out_c in enumerate(channels):
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
            spatial //= 2

        self.feature = nn.Sequential(*layers)
        # 尾端不再假設輸入一定是 4x4；統一壓成 1x1 再做 1x1 conv
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )  # → (B,1,1,1)
        self.apply(weights_init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)  # (B, C, H', W')，H',W' 可變
        out = self.head(h).view(x.size(0))  # → (B,)
        return out  # logit/score
