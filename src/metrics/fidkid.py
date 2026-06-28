"""
FID/KID wrapper built on top of torchmetrics.

This class is used to compute GAN sample quality metrics:
- FID: Frechet Inception Distance
- KID: Kernel Inception Distance

The wrapper is intentionally defensive:
- If torchmetrics isn't available, it becomes a no-op and returns NaNs.
- It converts common GAN output ranges (tanh in [-1,1]) into the formats required
  by torchmetrics (uint8 for FID, float01 for KID).
"""

from typing import Optional, Dict
import torch


class FIDKID:
    """
    Wrapper that uses torchmetrics if available; otherwise becomes no-op.
    Inputs expected in [-1,1] (tanh). Converts to formats required by FID/KID.
    """

    def __init__(self, device: str = "cuda"):
        """
        Create torchmetrics FID/KID instances on `device` if possible.

        Note:
        Importing torchmetrics can indirectly import torchvision (Inception). To
        keep this repository usable in environments where torchvision is missing
        or incompatible with torch, the import is done lazily here.
        """
        self.fid = None
        self.kid = None
        self.enabled = False
        self.import_error: Optional[BaseException] = None

        try:
            from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
            from torchmetrics.image.kid import KernelInceptionDistance  # type: ignore
        except Exception as e:
            self.import_error = e
            return

        try:
            self.fid = FrechetInceptionDistance(feature=2048).to(device)
            self.kid = KernelInceptionDistance(subset_size=1000).to(device)
            self.enabled = True
        except Exception as e:
            self.import_error = e
            self.fid = None
            self.kid = None
            self.enabled = False

    @staticmethod
    def _to_uint8(imgs: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] float tensor to [0,255] uint8 (expected by FID)."""
        x = (imgs * 0.5 + 0.5).clamp(0, 1)
        return (x * 255).to(torch.uint8)

    @staticmethod
    def _to_float01(imgs: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] float tensor to [0,1] float (expected by KID)."""
        return (imgs * 0.5 + 0.5).clamp(0, 1)

    def update_real(self, imgs: torch.Tensor):
        """Accumulate real images."""
        if not self.enabled:
            return
        self.fid.update(self._to_uint8(imgs), real=True)  # type: ignore
        self.kid.update(self._to_float01(imgs), real=True)  # type: ignore

    def update_fake(self, imgs: torch.Tensor):
        """Accumulate fake/generated images."""
        if not self.enabled:
            return
        self.fid.update(self._to_uint8(imgs), real=False)  # type: ignore
        self.kid.update(self._to_float01(imgs), real=False)  # type: ignore

    def compute(self) -> Dict[str, float]:
        """Compute metrics and reset internal state for the next run."""
        if not self.enabled:
            return {"fid": float("nan"), "kid_mean": float("nan")}
        fid = float(self.fid.compute().detach().cpu())  # type: ignore
        kid_mean = float(self.kid.compute()[0].detach().cpu())  # type: ignore
        # reset for next round
        self.fid.reset()  # type: ignore
        self.kid.reset()  # type: ignore
        return {"fid": fid, "kid_mean": kid_mean}
