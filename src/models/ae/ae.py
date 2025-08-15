# src/models/ae/ae.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


class AutoEncoder(nn.Module):
    def __init__(
        self,
        img_channels: int,
        latent_dim: int,
        hidden_dims,
        activation: str = "relu",
        input_size: Tuple[int, int] = (28, 28),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # ----- build encoder/decoder conv blocks in __init__（這些會隨 model.to(device) 一起移動） -----
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        # lazy FC layers (initialized on first forward)
        self._fc_built = False
        self.flatten = nn.Flatten()

        # decoder conv transpose example（最後再接一個 conv）
        self._decoder_conv = None  # will be built after we know shape

    def _init_fc_layers(self, x: torch.Tensor) -> None:
        """
        Build FC layers lazily on the SAME device/dtype as x.
        """
        if self._fc_built:
            return
        dev, dt = x.device, x.dtype

        with torch.no_grad():
            feat = self.encoder(x)  # B, C, H, W
            b, c, h, w = feat.shape
            enc_out_dim = c * h * w

        # 👉 新建層時直接指定 device / dtype，避免在 CPU
        self.fc_mu = nn.Linear(
            enc_out_dim, self.latent_dim, bias=True, device=dev, dtype=dt
        )
        self.fc_decode = nn.Linear(
            self.latent_dim, enc_out_dim, bias=True, device=dev, dtype=dt
        )

        # 反展平要知道 C,H,W
        self.unflatten = nn.Unflatten(1, (c, h, w))

        # decoder conv（放在同 device）
        self._decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c, 32, 4, 2, 1, device=dev, dtype=dt),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, device=dev, dtype=dt),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1, device=dev, dtype=dt),
        )

        self._fc_built = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.flatten(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.unflatten(z)  # B, C, H, W
        x = self._decoder_conv(z)  # type: ignore
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果還沒建立 lazy 層，先建在 x 的 device/dtype
        if not self._fc_built:
            self._init_fc_layers(x)

        enc = self.encoder(x)
        enc_flat = self.flatten(enc)
        z = self.fc_mu(enc_flat)  # simple AE: just a projection
        dec_flat = self.fc_decode(z)
        recon = self.decode(dec_flat)
        return recon
