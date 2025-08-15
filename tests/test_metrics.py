# Edge-case metrics correctness
import torch
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.metrics.image import psnr, ssim


def test_metrics_identity():
    x = torch.rand(4, 3, 32, 32)
    y = x.clone()
    ps = psnr(x, y)
    ss = ssim(x, y)
    # psnr -> inf (we'll treat as very large), ssim ~ 1
    assert torch.isfinite(ps).all()
    assert (ss > 0.99).all()


def test_metrics_zeros_ones():
    x = torch.zeros(2, 1, 16, 16)
    y = torch.ones(2, 1, 16, 16)
    ps = psnr(x, y)
    ss = ssim(x, y)
    assert torch.isfinite(ps).all()
    assert (ss < 0.2).all()
