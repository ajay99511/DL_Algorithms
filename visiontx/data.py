"""
Data loading for Project 4: Vision Transformer.

Supports CIFAR-10 (via torchvision) and ImageNette (via HuggingFace datasets).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from visiontx.config import ViTConfig
from shared.seed import fix_all_seeds


# CIFAR-10 normalization statistics (per-channel mean and std)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(config: ViTConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 with train augmentation and a 10% validation split.

    Train augmentation: RandomHorizontalFlip, RandomCrop(32, padding=4), Normalize
    Val/test: Normalize only

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ])

    # Download full train set (50,000 images)
    full_train = datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=train_transform
    )
    # Val set uses eval_transform (no augmentation)
    full_train_eval = datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=eval_transform
    )

    # 10% val split (5000 images) with fixed seed
    fix_all_seeds(config.seed)
    n_total = len(full_train)
    n_val = n_total // 10  # 5000
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def load_imagenette(config: ViTConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load ImageNette (320px) via HuggingFace datasets, resize to 224×224.

    Train augmentation: RandomHorizontalFlip, RandomCrop(224, padding=28), Normalize
    Val/test: Resize + CenterCrop + Normalize

    Returns:
        (train_loader, val_loader, test_loader)
    """
    try:
        from datasets import load_dataset  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` and `Pillow` are required for ImageNette. "
            "Install with: pip install datasets Pillow"
        ) from e

    # ImageNette normalization (ImageNet statistics)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    hf_dataset = load_dataset("frgfm/imagenette", "320px")

    class ImagenetteDataset(Dataset):
        def __init__(self, hf_split, transform):
            self.data = hf_split
            self.transform = transform

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int):
            item = self.data[idx]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = item["label"]
            return self.transform(img), label

    train_ds = ImagenetteDataset(hf_dataset["train"], train_transform)
    val_ds = ImagenetteDataset(hf_dataset["validation"], eval_transform)

    # Use val as test; split train 90/10 for train/val
    fix_all_seeds(config.seed)
    n_total = len(train_ds)
    n_val = n_total // 10
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(ImagenetteDataset(hf_dataset["train"], eval_transform), val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def get_data_loaders(config: ViTConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Dispatch to the correct data loader based on config.dataset.

    Args:
        config: ViTConfig with dataset field set to "cifar10" or "imagenette".

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if config.dataset == "cifar10":
        return load_cifar10(config)
    elif config.dataset == "imagenette":
        return load_imagenette(config)
    else:
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. Choose from: 'cifar10', 'imagenette'"
        )
