import torch

from enum import Enum


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using DEVICE: {device}")
    return device


# ── Training hyperparameters ──────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "HalkaNet"

EPOCHS_MINI: int = 2
EPOCHS: int = 5
EPOCHS_PRO: int = 15
EPOCHS_MAX: int = 60
EPOCHS_PRO_MAX: int = 100

BATCH_SIZE_MINI: int = 64
BATCH_SIZE: int = 64
BATCH_SIZE_PRO: int = 64

LEARNING_RATE: float = 2e-3
WEIGHT_DECAY: float = 5e-4
LABEL_SMOOTHING: float = 0.1

MANUAL_SEED_0: int = 0
MANUAL_SEED_1: int = 1
MANUAL_SEED_42: int = 42
MANUAL_SEED_123: int = 123
MANUAL_SEED_999: int = 999
DEFAULT_SEED: int = MANUAL_SEED_42

EARLY_STOPPING_PATIENCE: int = 7

DATA_DIR: str = "../data"
CHECKPOINT_DIR: str = "../checkpoints"
FIGURE_DIR: str = "../figure"

# ── Image normalization & dimensions ─────────────────────────────────────────

IMAGE_NORMALIZATION_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGE_NORMALIZATION_STD: list[float] = [0.229, 0.224, 0.225]

IMAGE_DIMENSION: tuple[int, int] = (96, 96)
IMAGE_DIMENSION_MCU: tuple[int, int] = (64, 64)
IMAGE_DIMENSION_CIFAR: tuple[int, int] = (32, 32)
IMAGE_DIMENSION_STL: tuple[int, int] = (64, 64)
IMAGE_DIMENSION_FOOD101: tuple[int, int] = (128, 128)
IMAGE_DIMENSION_INTEL: tuple[int, int] = (96, 96)
IMAGE_DIMENSION_150: tuple[int, int] = (150, 150)

INITIAL_CHANNEL: int = 16
RGB_CHANNEL: int = 3
GRAYSCALE_CHANNEL: int = 1
DEFAULT_DROPOUT_RATE: float = 0.1
DEFAULT_GROUP_CHANNEL: int = 8


class BranchType(str, Enum):
    RGB = "rgb"
    FILTER = "filter"


class FilterType(str, Enum):
    LOG = "log"
    GABOR = "gabor"
    LBP = "lbp"
    WAVE = "wave"


DEFAULT_SKIP_EXPANSION: dict[BranchType, float] = {
    BranchType.RGB: 1,
    BranchType.FILTER: 1,
}


DEFAULT_MULTIPLIER = {}