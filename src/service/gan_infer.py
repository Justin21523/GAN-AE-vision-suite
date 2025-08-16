# src/service/gan_infer.py
import io
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torchvision.utils import make_grid
from PIL import Image

from src.models.gan.generator import DCGANGenerator


@dataclass
class GenerateParams:
    n: int = 64
    seed: int = 42
    nrow: int = 8
    use_ema_shadow: bool = False  # only works if ckpt contains ema shadow


class GANService:
    """
    Reusable inference core for GAN sampling.
    - Loads a training checkpoint saved by train_gan.py (expects 'cfg' inside).
    - Generates a sample grid and returns PIL.Image (0..255).
    - No globals; you can create multiple instances with different devices/ckpts.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.G: Optional[torch.nn.Module] = None
        self.cfg: Optional[dict] = None
        self.has_ema_shadow: bool = False
        self._ema_shadow: Optional[dict] = None  # if present in ckpt

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.cfg = ckpt["cfg"]["model"]
        self.G = DCGANGenerator(
            latent_dim=self.cfg["latent_dim"],  # type: ignore
            img_channels=self.cfg["img_channels"],  # type: ignore
            channels=tuple(self.cfg["g_channels"]),  # type: ignore
            img_size=self.cfg["img_size"],  # type: ignore
        ).to(self.device)
        self.G.load_state_dict(ckpt["G"])
        self.G.eval()

        # Optional: if your training保存了EMA shadow，這裡一併帶上
        self.has_ema_shadow = "ema_shadow" in ckpt
        if self.has_ema_shadow:
            self._ema_shadow = ckpt["ema_shadow"]  # dict of weights

    def _apply_ema_if_any(self, use_ema: bool):
        if not (use_ema and self.has_ema_shadow):  # nothing to do
            return None
        # Backup current weights and swap to EMA shadow
        backup = {k: p.detach().clone() for k, p in self.G.state_dict().items()}  # type: ignore
        self.G.load_state_dict(self._ema_shadow, strict=False)  # type: ignore
        return backup

    def _restore_backup_if_any(self, backup: Optional[dict]):
        if backup is not None:
            self.G.load_state_dict(backup, strict=False)  # type: ignore

    @torch.no_grad()
    def generate_grid(self, params: GenerateParams) -> Image.Image:
        assert self.G is not None, "Call load_checkpoint() first."
        torch.manual_seed(int(params.seed))
        z = torch.randn(int(params.n), int(self.cfg["latent_dim"]), device=self.device)  # type: ignore
        backup = self._apply_ema_if_any(params.use_ema_shadow)
        x = self.G(z)
        self._restore_backup_if_any(backup)

        # [-1,1] -> [0,1]
        x = (x * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(x, nrow=int(params.nrow))
        # to PIL (0..255)
        nd = (grid * 255).to(torch.uint8).detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(nd)
