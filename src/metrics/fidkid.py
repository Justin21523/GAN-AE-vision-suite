from typing import Optional, Dict
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


class FIDKID:
    """
    Wrapper that uses torchmetrics if available; otherwise becomes no-op.
    Inputs expected in [-1,1] (tanh). Converts to formats required by FID/KID.
    """

    def __init__(self, device: str = "cuda"):
        try:
            self.fid = FrechetInceptionDistance(feature=2048).to(device)
            self.kid = KernelInceptionDistance(subset_size=1000).to(device)
            self.enabled = True
        except Exception:
            self.fid = None
            self.kid = None
            self.enabled = False

    @staticmethod
    def _to_uint8(imgs: torch.Tensor) -> torch.Tensor:
        # imgs: [-1,1] float -> [0,255] uint8
        x = (imgs * 0.5 + 0.5).clamp(0, 1)
        return (x * 255).to(torch.uint8)

    @staticmethod
    def _to_float01(imgs: torch.Tensor) -> torch.Tensor:
        return (imgs * 0.5 + 0.5).clamp(0, 1)

    def update_real(self, imgs: torch.Tensor):
        if not self.enabled:
            return
        self.fid.update(self._to_uint8(imgs), real=True)  # type: ignore
        self.kid.update(self._to_float01(imgs), real=True)  # type: ignore

    def update_fake(self, imgs: torch.Tensor):
        if not self.enabled:
            return
        self.fid.update(self._to_uint8(imgs), real=False)  # type: ignore
        self.kid.update(self._to_float01(imgs), real=False)  # type: ignore

    def compute(self) -> Dict[str, float]:
        if not self.enabled:
            return {"fid": float("nan"), "kid_mean": float("nan")}
        fid = float(self.fid.compute().detach().cpu())  # type: ignore
        kid_mean = float(self.kid.compute()[0].detach().cpu())  # type: ignore
        # reset for next round
        self.fid.reset()  # type: ignore
        return {"fid": fid, "kid_mean": kid_mean}
