# src/data/transforms.py
"""
Build torchvision transforms for train/val
"""
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def build_transforms(
    dataset: str = "celeba",
    image_size: int = 128,
    center_crop: int | None = None,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    train: bool = True,
):
    t = []
    # Resize if needed 先 Resize，確保最短邊 >= image_size
    if image_size is not None:
        t.append(transforms.Resize(image_size, interpolation=Image.Resampling.BICUBIC))  # type: ignore
    t.append(transforms.CenterCrop(image_size))

    if train and center_crop is None:
        # 你也可以在這裡加入 RandomHorizontalFlip / ColorJitter 等
        t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.ToTensor())
    if mean is not None and std is not None:
        # mean/std can be list (RGB) or scalar (grayscale)
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std, (int, float)):
            std = [std]
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
