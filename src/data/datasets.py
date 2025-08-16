# src/data/datasets
"""
Unified dataset factory for CelebA / MNIST
"""
from typing import Tuple, Optional
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.data.transforms import build_transforms


def _build_transforms(
    name: str,
    image_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    """
    Return a torchvision transform pipeline according to dataset name.
    Always wrap with transforms.Compose to avoid returning a raw list.
    """
    name = name.lower()

    if name in {"mnist", "fashionmnist"}:
        # MNIST is single-channel; normalize with 1-value mean/std
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),  # -> [0,1]
                transforms.Normalize((mean[0],), (std[0],)),  # use first value
            ]
        )

    # default: 3-channel image datasets
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_dataset(cfg: dict, train: bool = True):
    """
    Create a torchvision dataset according to cfg.
    Required cfg keys:
      cfg["data"]["dataset"], cfg["data"]["root"], cfg["data"]["image_size"]
      cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"], cfg["data"]["download"]
    """
    name = cfg["data"]["dataset"].lower()
    root = cfg["data"]["root"]
    download = bool(cfg["data"].get("download", False))
    img_size = int(cfg["data"].get("img_size") or cfg["data"].get("image_size") or 128)

    # mean/std as tuple3 (or tuple1 for MNIST; we'll handle in _build_transforms)
    mean = tuple(cfg["data"].get("normalize_mean", [0.5, 0.5, 0.5]))
    std = tuple(cfg["data"].get("normalize_std", [0.5, 0.5, 0.5]))

    tfm = build_transforms(name, img_size, mean, std)  # type: ignore

    if name == "celeba":
        # If root points to "data", torchvision 會在 data/celeba/ 下找檔案
        split = "train" if train else "valid"  # 或視需求改 "test"
        return datasets.CelebA(
            root=root,
            split=split,  # "train"/"valid"/"test"
            target_type="attr",  # 只要影像可用 "identity" 或 []
            transform=tfm,
            download=download,
        )
    elif name == "mnist":
        return datasets.MNIST(
            root=root,
            train=train,
            transform=tfm,
            download=download,
        )

    elif name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            transform=tfm,
            download=download,
        )

    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_loader(
    ds,
    batch_size: int,
    num_workers: int,
    train: bool,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
):
    """
    Standard DataLoader factory that safely forwards optional PyTorch knobs.
    Only passes args if not None to stay compatible with older torch versions.
    """
    dl_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
    )

    # Only add if provided, to avoid mismatched kwargs on old torch
    if pin_memory is not None:
        dl_kwargs["pin_memory"] = pin_memory

    # persistent_workers requires num_workers > 0 (torch constraint)
    if persistent_workers is not None and num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(**dl_kwargs)  # type: ignore
