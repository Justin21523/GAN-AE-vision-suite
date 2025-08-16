# tests/test_dataloader.py
"""
Minimal dataloader smoke tests with MNIST fallback
"""
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.datasets import build_dataset, get_loader
import os


def test_mnist_loader_shapes():
    # 直接載入 MNIST 專用 config
    cfg_path = os.path.join("configs", "dataset_mnist.yaml")
    cfg = load_config(cfg_path)

    # 建立 train / val datasets
    train_dataset = build_dataset(cfg, train=True)
    val_dataset = build_dataset(cfg, train=False)

    # 建立 dataloaders
    train_loader = get_loader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train=True,
    )
    val_loader = get_loader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train=False,
    )

    # 取一個 batch
    x, y = next(iter(train_loader))

    # 驗證輸出 shape
    assert x.shape[0] == cfg["data"]["batch_size"]
    assert x.shape[1:] == (1, cfg["data"]["image_size"], cfg["data"]["image_size"])
    assert y.shape[0] == cfg["data"]["batch_size"]
