# src/data/datasets
"""
Unified dataset factory for CelebA / MNIST
"""
from typing import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


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
    img_size = int(cfg["data"].get("image_size", 28))

    # mean/std as tuple3 (or tuple1 for MNIST; we'll handle in _build_transforms)
    mean = tuple(cfg["data"].get("normalize_mean", [0.5, 0.5, 0.5]))
    std = tuple(cfg["data"].get("normalize_std", [0.5, 0.5, 0.5]))

    tfm = _build_transforms(name, img_size, mean, std)

    if name == "celeba":
        # torchvision CelebA splits are "train"/"valid"/"test"
        split = "train" if train else "test"
        return datasets.CelebA(
            root=root,
            split=split,
            transform=tfm,  # <-- use the composed transform
            target_type="attr",  # or "identity"/"bbox"/"landmarks"
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


def get_loader(ds, batch_size: int, num_workers: int, train: bool):
    """
    Standard DataLoader. Keep default collate_fn so we receive tensors, not lists.
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
