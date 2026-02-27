import gc
import shutil
from pathlib import Path

import torch
import kagglehub
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_SEED,
    IMAGE_DIMENSION_CIFAR,
    IMAGE_DIMENSION_INTEL,
    IMAGE_DIMENSION_STL,
    IMAGE_NORMALIZATION_MEAN,
    IMAGE_NORMALIZATION_STD,
)
from utils.training import set_seed


def cleanup(experiment_results: dict | None = None, *extra_vars) -> None:
    """Delete experiment artifacts and free device memory (CUDA / MPS / CPU)."""
    if experiment_results is not None:
        del experiment_results
    for var in extra_vars:
        del var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"CUDA  Allocated {allocated:.1f} MB  Reserved {reserved:.1f} MB")
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        allocated = torch.mps.current_allocated_memory() / 1e6
        driver = torch.mps.driver_allocated_memory() / 1e6
        print(f"MPS  Allocated {allocated:.1f} MB  Driver {driver:.1f} MB")
    else:
        print("CPU — No device cache to release")
        gc.collect()


# ── Normalization stats ───────────────────────────────────────────────────────

# RGB: ImageNet mean/std (3-channel)
_RGB_MEAN = IMAGE_NORMALIZATION_MEAN  # [0.485, 0.456, 0.406]
_RGB_STD = IMAGE_NORMALIZATION_STD  # [0.229, 0.224, 0.225]
# Grayscale: luminance-weighted collapse of ImageNet RGB stats
_GRAY_MEAN = [0.299 * _RGB_MEAN[0] + 0.587 * _RGB_MEAN[1] + 0.114 * _RGB_MEAN[2]]
_GRAY_STD = [0.299 * _RGB_STD[0] + 0.587 * _RGB_STD[1] + 0.114 * _RGB_STD[2]]


def _norm(grayscale: bool) -> transforms.Normalize:
    return transforms.Normalize(
        _GRAY_MEAN if grayscale else _RGB_MEAN,
        _GRAY_STD if grayscale else _RGB_STD,
    )


# ── Transform builders ────────────────────────────────────────────────────────


def build_transforms(
    img_dimensions: tuple[int, int],
    grayscale: bool = False,
    crop_padding: int = 0,
    jitter: tuple[float, ...] | None = None,
    grayscale_p: float = 0.0,
    erasing_p: float = 0.0,
    rotation: int = 0,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops = [transforms.Resize(img_dimensions)]
    if crop_padding:
        train_ops.append(transforms.RandomCrop(img_dimensions, padding=crop_padding))
    train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if rotation:
        train_ops.append(transforms.RandomRotation(rotation))
    if jitter:
        train_ops.append(transforms.ColorJitter(*jitter))
    if grayscale_p:
        train_ops.append(transforms.RandomGrayscale(p=grayscale_p))
    if grayscale:
        train_ops.append(transforms.Grayscale(num_output_channels=1))
    train_ops += [transforms.ToTensor(), _norm(grayscale)]
    if erasing_p:
        train_ops.append(
            transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.15), value=0)
        )

    test_ops = [transforms.Resize(img_dimensions)]
    if grayscale:
        test_ops.append(transforms.Grayscale(num_output_channels=1))
    test_ops += [transforms.ToTensor(), _norm(grayscale)]

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_standard_transforms(
    img_dimensions: tuple[int, int], grayscale: bool = False
) -> tuple[transforms.Compose, transforms.Compose]:
    return build_transforms(img_dimensions, grayscale=grayscale, rotation=10)


def get_strong_transforms(
    img_dimensions: tuple[int, int], grayscale: bool = False
) -> tuple[transforms.Compose, transforms.Compose]:
    return build_transforms(
        img_dimensions,
        grayscale=grayscale,
        crop_padding=12,
        jitter=(0.3, 0.3, 0.3, 0.1),
        grayscale_p=0.1,
        erasing_p=0.3,
        rotation=20,
    )


def get_stl10_transforms(
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_STL, grayscale: bool = False
) -> tuple[transforms.Compose, transforms.Compose]:
    return build_transforms(
        img_dimensions,
        grayscale=grayscale,
        crop_padding=8,
        jitter=(0.3, 0.3, 0.2, 0.1),
        grayscale_p=0.05,
        erasing_p=0.2,
    )


def get_cifar_transforms(
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_CIFAR, grayscale: bool = False
) -> tuple[transforms.Compose, transforms.Compose]:
    return build_transforms(
        img_dimensions,
        grayscale=grayscale,
        crop_padding=4,
        jitter=(0.2, 0.2, 0.1),
        erasing_p=0.1,
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────


def class_subset(dataset, num_classes: int) -> Subset:
    """Filter a torchvision dataset to the first `num_classes` classes (alphabetical)."""
    indices = [i for i, lbl in enumerate(dataset.labels) if lbl < num_classes]
    sub = Subset(dataset, indices)
    sub.classes = dataset.classes[:num_classes]
    return sub


# ── Dataset downloads ─────────────────────────────────────────────────────────


def download_stl10(
    root_dir: str = DATA_DIR,
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_STL,
    grayscale: bool = False,
) -> tuple[datasets.STL10, datasets.STL10]:
    """
    Download STL-10. Returns (train_dataset, test_dataset).

    Args:
        grayscale: if True, images are single-channel grayscale; default RGB.
    """
    train_tf, test_tf = get_stl10_transforms(img_dimensions, grayscale=grayscale)
    root = Path(root_dir)
    root.mkdir(exist_ok=True)
    train = datasets.STL10(root=root, split="train", download=True, transform=train_tf)
    test = datasets.STL10(root=root, split="test", download=True, transform=test_tf)
    ch = 1 if grayscale else 3
    print(f"STL-10 ready  train={len(train)}  test={len(test)}  channels={ch}")
    return train, test


def download_cifar10(
    root_dir: str = DATA_DIR,
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_CIFAR,
    grayscale: bool = False,
) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Download CIFAR-10. Returns (train_dataset, test_dataset).

    Args:
        grayscale: if True, images are single-channel grayscale; default RGB.
    """
    train_tf, test_tf = get_cifar_transforms(img_dimensions, grayscale=grayscale)
    root = Path(root_dir)
    root.mkdir(exist_ok=True)
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    ch = 1 if grayscale else 3
    print(f"CIFAR-10 ready  train={len(train)}  test={len(test)}  channels={ch}")
    return train, test


def download_cifar100(
    root_dir: str = DATA_DIR,
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_CIFAR,
    grayscale: bool = False,
) -> tuple[datasets.CIFAR100, datasets.CIFAR100]:
    """
    Download CIFAR-100. Returns (train_dataset, test_dataset).

    Args:
        grayscale: if True, images are single-channel grayscale; default RGB.
    """
    train_tf, test_tf = get_cifar_transforms(img_dimensions, grayscale=grayscale)
    root = Path(root_dir)
    root.mkdir(exist_ok=True)
    train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    ch = 1 if grayscale else 3
    print(f"CIFAR-100 ready  train={len(train)}  test={len(test)}  channels={ch}")
    return train, test


def download_food101(
    root_dir: str = DATA_DIR,
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_STL,
    num_classes: int = 10,
    grayscale: bool = False,
) -> tuple[Subset | datasets.Food101, Subset | datasets.Food101]:
    assert 1 <= num_classes <= 101, "num_classes must be 1–101"
    train_tf, test_tf = get_strong_transforms(img_dimensions, grayscale=grayscale)
    root = Path(root_dir)
    root.mkdir(exist_ok=True)
    train_full = datasets.Food101(
        root=root, split="train", download=True, transform=train_tf
    )
    test_full = datasets.Food101(
        root=root, split="test", download=True, transform=test_tf
    )
    train = class_subset(train_full, num_classes) if num_classes < 101 else train_full
    test = class_subset(test_full, num_classes) if num_classes < 101 else test_full
    ch = 1 if grayscale else 3
    print(
        f"Food-101 ({num_classes} classes) ready  train={len(train)}  test={len(test)}  channels={ch}"
    )
    return train, test


def download_intel(
    root_dir: str = DATA_DIR,
    img_dimensions: tuple[int, int] = IMAGE_DIMENSION_INTEL,
    grayscale: bool = False,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, test_tf = get_standard_transforms(img_dimensions, grayscale=grayscale)
    root = Path(root_dir)
    root.mkdir(exist_ok=True)
    destination = root / "intel-image-classification"
    if not destination.exists():
        print("Downloading Intel dataset via KaggleHub...")
        dataset_path = kagglehub.dataset_download(
            "puneet6060/intel-image-classification"
        )
        shutil.copytree(dataset_path, destination)
        print("Dataset copied.")
    train = datasets.ImageFolder(
        root=destination / "seg_train" / "seg_train", transform=train_tf
    )
    test = datasets.ImageFolder(
        root=destination / "seg_test" / "seg_test", transform=test_tf
    )
    ch = 1 if grayscale else 3
    print(f"Intel ready  train={len(train)}  test={len(test)}  channels={ch}")
    return train, test


# ── DataLoaders ───────────────────────────────────────────────────────────────


def get_data_loaders(
    train_dataset,
    test_dataset,
    batch_size: int = BATCH_SIZE,
    manual_seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader, list[str], int]:
    """
    Wrap train/test datasets into DataLoaders.

    Returns:
        train_dataloader, test_dataloader, class_names, data_channel
    """
    set_seed(manual_seed)
    if hasattr(train_dataset, "classes"):
        class_names = train_dataset.classes
    else:
        class_names = [
            "airplane",
            "bird",
            "car",
            "cat",
            "deer",
            "dog",
            "horse",
            "monkey",
            "ship",
            "truck",
        ]
    data_channel = train_dataset[0][0].shape[0]
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader, class_names, data_channel