# src/data/transforms.py
"""
Build torchvision transforms for train/val
"""
from torchvision import transforms


def build_transforms(
    dataset: str, image_size: int, center_crop=None, mean=None, std=None
):
    t = []
    # Optional centered crop (CelebA)
    if center_crop is not None:
        t.append(transforms.CenterCrop(center_crop))
    # Resize if needed
    if image_size is not None:
        t.append(transforms.Resize((image_size, image_size)))
    t.append(transforms.ToTensor())
    if mean is not None and std is not None:
        # mean/std can be list (RGB) or scalar (grayscale)
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std, (int, float)):
            std = [std]
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
