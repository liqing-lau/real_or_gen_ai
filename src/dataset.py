import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 224


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def _build_subset(
    src_root: Path,
    dst_root: Path,
    sample_ratio: float,
) -> None:
    """
    从 src_root 下按比例抽样复制到 dst_root，保持 ImageFolder 结构。
    """
    if dst_root.exists():
        # 已经构建过子集，直接复用，避免重复拷贝
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        images = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if not images:
            continue

        k = max(1, int(len(images) * sample_ratio))
        sampled = random.sample(images, k)

        dst_class_dir = dst_root / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for src_path in sampled:
            dst_path = dst_class_dir / src_path.name
            shutil.copy2(src_path, dst_path)


def get_dataloaders(
    train_dir: str = "data/train",
    val_dir: str = "data/val",
    batch_size: int = 32,
    num_workers: int = 4,
    sample_ratio: float | None = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build PyTorch DataLoaders for training and validation.

    The directory structure is assumed to follow torchvision.datasets.ImageFolder:

      train_dir/
        real/
        ai/
      val_dir/
        real/
        ai/

    If sample_ratio is provided (0 < r < 1), a random subset of images
    will be copied into data_sub/train and data_sub/val and used for
    quick experiments.
    """
    train_transform, val_transform = build_transforms()

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
        base = Path(train_dir).parents[1] if len(Path(train_dir).parts) > 1 else Path(".")
        subset_root = base / "data_sub"
        sub_train = subset_root / "train"
        sub_val = subset_root / "val"

        _build_subset(Path(train_dir), sub_train, sample_ratio)
        _build_subset(Path(val_dir), sub_val, sample_ratio)

        train_dir = str(sub_train)
        val_dir = str(sub_val)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.classes


