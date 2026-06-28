"""
GAN inference helpers.

`GANService` is a small utility that:
- Loads a saved GAN checkpoint (see `src/scripts/train_gan.py`)
- Reconstructs the generator architecture from the checkpoint config
- Generates a grid of samples and returns it as a PIL image

This is used by UI/CLI entrypoints that want a reusable sampling core.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image

from src.models.gan.factory import build_generator_from_cfg
from src.utils.checkpoint import load_checkpoint
from src.utils.vision import make_grid, to_pil_rgb


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
        """Load a training checkpoint and initialize the generator."""
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        self.cfg = ckpt["cfg"]["model"]
        self.G = build_generator_from_cfg(self.cfg).to(self.device)
        self.G.load_state_dict(ckpt["G"])
        self.G.eval()

        # Optional: carry EMA weights if the training checkpoint saved them.
        self.has_ema_shadow = "ema_shadow" in ckpt
        if self.has_ema_shadow:
            self._ema_shadow = ckpt["ema_shadow"]  # dict of weights

    def _apply_ema_if_any(self, use_ema: bool):
        """Temporarily swap generator weights to EMA values (if present)."""
        if not (use_ema and self.has_ema_shadow):  # nothing to do
            return None
        # Backup current weights and swap to EMA shadow
        backup = {k: p.detach().clone() for k, p in self.G.state_dict().items()}  # type: ignore
        self.G.load_state_dict(self._ema_shadow, strict=False)  # type: ignore
        return backup

    def _restore_backup_if_any(self, backup: Optional[dict]):
        """Restore pre-EMA weights after sampling."""
        if backup is not None:
            self.G.load_state_dict(backup, strict=False)  # type: ignore

    @torch.no_grad()
    def generate_grid(self, params: GenerateParams) -> Image.Image:
        """
        Generate a grid of samples and return a PIL image.

        Returns:
            A `PIL.Image` in RGB, with values in [0, 255].
        """
        assert self.G is not None, "Call load_checkpoint() first."
        torch.manual_seed(int(params.seed))
        z = torch.randn(int(params.n), int(self.cfg["latent_dim"]), device=self.device)  # type: ignore
        backup = self._apply_ema_if_any(params.use_ema_shadow)
        x = self.G(z)
        self._restore_backup_if_any(backup)

        # Convert model output (tanh in [-1, 1]) to [0, 1] for saving/visualization.
        x = (x * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(x, nrow=int(params.nrow))
        return to_pil_rgb(grid)
