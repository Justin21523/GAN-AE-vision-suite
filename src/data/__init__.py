# src/data/__init__.py
# datasets & transforms live here

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataset(
    name: str,
    root: str,
    split: str | None = "train",
    image_size: int = 224,
    center_crop: int | None = None,
    mean: tuple | None = None,
    std: tuple | None = None,
    to_tensor: bool = True,
    normalize: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    download: bool = False,
    splits: tuple[str, ...] | None = None,
):
    # build transforms
    tfms = []
    if center_crop:
        tfms.append(transforms.CenterCrop(center_crop))
    tfms.append(transforms.Resize(image_size))
    if to_tensor:
        tfms.append(transforms.ToTensor())
    if normalize and mean and std:
        tfms.append(transforms.Normalize(mean, std))
    tfm = transforms.Compose(tfms)

    def make(name, split):
        if name.lower() == "mnist":
            ds = datasets.MNIST(
                root, train=(split == "train"), transform=tfm, download=download
            )
            ch = 1  # 用不到也無妨
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        dl = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )
        return ds, dl

    # 支援一次建多個 split（方案 B）
    if splits is not None:
        out = {}
        for sp in splits:
            ds, dl = make(name, sp)
            out[sp] = {"dataset": ds, "loader": dl}
        return out

    # 單一 split（方案 A）
    return make(name, split)
