import copy
import math
import random
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DEFAULT_SEED, EARLY_STOPPING_PATIENCE, EPOCHS_PRO
from .metrics import count_parameters


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── EMA ───────────────────────────────────────────────────────────────────────


class EMA:
    """Exponential Moving Average over learnable parameters only.

    BN buffers excluded — call bn_calibrate() after apply_shadow()
    to realign BN stats with EMA weights before evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    1.0 - self.decay
                ) * param.detach() + self.decay * self.shadow[name]

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def bn_calibrate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_batches: int = 30,
) -> None:
    """Recompute BN running stats after apply_shadow() and before model.eval()."""
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            model(images.to(device))


# ── Early Stopping ────────────────────────────────────────────────────────────


class EarlyStopping:
    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ── CutMix ────────────────────────────────────────────────────────────────────


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns (mixed_images, labels_a, labels_b, lam).

    Loss: lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    cut_h = int(H * (1.0 - lam) ** 0.5)
    cut_w = int(W * (1.0 - lam) ** 0.5)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, labels, labels[idx], lam


# ── Training ──────────────────────────────────────────────────────────────────


def model_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = EPOCHS_PRO,
    patience: int = EARLY_STOPPING_PATIENCE,
    save_best: bool = True,
    checkpoint_path: str = f"{CHECKPOINT_DIR}/model.pth",
    use_cutmix: bool = False,
    cutmix_alpha: float = 0.4,
    cutmix_prob: float = 0.3,
    cutmix_start_epoch: int = 0,
    cutmix_end_epoch: int | None = None,
    manual_seed: int = DEFAULT_SEED,
) -> tuple[nn.Module, dict]:
    """Train a model with warmup + cosine LR, EMA, optional CutMix, and early stopping.

    Args:
        model              : model to train (moved to device in-place)
        train_loader       : training DataLoader
        test_loader        : validation DataLoader
        criterion          : loss function
        optimizer          : AdamW or equivalent
        device             : torch.device
        epochs             : maximum training epochs
        patience           : early stopping patience (epochs without improvement)
        save_best          : save best checkpoint to disk
        checkpoint_path    : .pth save path
        use_cutmix         : enable CutMix augmentation
        cutmix_alpha       : Beta distribution alpha for CutMix lam sampling
        cutmix_prob        : per-batch CutMix application probability
        cutmix_start_epoch : epoch index (0-based) to begin CutMix
        cutmix_end_epoch   : epoch index (0-based) to stop CutMix; None = run to end.
                             Disabling CutMix in final epochs lets the model fine-tune
                             on clean labels at near-zero LR → typically +0.5–1% accuracy.
        manual_seed        : RNG seed for reproducibility

    Returns:
        (best_model, history) where history contains per-epoch
        train_loss, train_acc, test_loss, test_acc, lr.
    """
    model.to(device)
    set_seed(manual_seed)

    # if save_best:
    #     os.makedirs(checkpoint_path, exist_ok=True)

    n_params = count_parameters(model)
    n_train = len(train_loader.dataset)
    steps_per_epoch = len(train_loader)

    # Small datasets (<50K): lower peak LR to reduce high-LR variance
    base_lr = 2e-3 if n_train < 50_000 else 3e-3
    for g in optimizer.param_groups:
        g["lr"] = base_lr

    warmup_epochs = max(3, min(8, int(epochs * 0.10)))
    cosine_epochs = max(epochs - warmup_epochs, 1)
    eta_min = 1e-6

    # Effective CutMix end epoch: default to final epoch if not specified
    _cutmix_end = cutmix_end_epoch if cutmix_end_epoch is not None else epochs

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        t = min((epoch - warmup_epochs) / cosine_epochs, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return (eta_min / base_lr) + (1.0 - eta_min / base_lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    preview = [base_lr * lr_schedule(e) for e in range(epochs)]
    key_epochs = sorted(
        set(range(min(4, epochs)))
        | set(range(max(0, epochs - 3), epochs))
        | {warmup_epochs, warmup_epochs // 2}
    )
    print(
        f"\nModel: {n_params:,} params | Dataset: {n_train:,} samples | base_lr: {base_lr:.0e}"
    )
    print(
        f"Warmup: {warmup_epochs} epochs → peak {base_lr:.0e} | Cosine: {cosine_epochs} epochs → {preview[-1]:.1e}"
    )
    print("LR preview:")
    for e in key_epochs:
        if e < epochs:
            print(f"  Epoch {e + 1:>3}: {preview[e]:.2e}")

    ema_decay = float(np.clip(1.0 - 1.0 / (5 * steps_per_epoch), 0.990, 0.9999))
    ema = EMA(model, decay=ema_decay)
    print(f"EMA decay: {ema_decay:.4f}  (≈5-epoch window, {5 * steps_per_epoch} steps)")

    if use_cutmix:
        cutmix_status = (
            f"ON (p={cutmix_prob}, α={cutmix_alpha}, "
            f"ep={cutmix_start_epoch}→{_cutmix_end})"
        )
    else:
        cutmix_status = "OFF"
    print(
        f"\nStarting {epochs} epochs (patience={patience}) | CutMix: {cutmix_status}\n"
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode="max")
    best_test_acc = 0.0
    best_model_state: dict | None = None
    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            apply_cutmix = (
                use_cutmix
                and cutmix_start_epoch <= epoch < _cutmix_end
                and random.random() < cutmix_prob
            )

            if apply_cutmix:
                images, labels_a, labels_b, lam = cutmix_batch(
                    images, labels, cutmix_alpha
                )
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(
                    outputs, labels_b
                )
                with torch.no_grad():
                    preds = outputs.argmax(1)
                    train_acc += (
                        lam * (preds == labels_a).float().mean().item()
                        + (1.0 - lam) * (preds == labels_b).float().mean().item()
                    )
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_acc += (outputs.argmax(1) == labels).float().mean().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            train_loss += loss.item()

        train_loss /= steps_per_epoch
        train_acc /= steps_per_epoch
        scheduler.step()

        # ── Evaluate with EMA weights ──────────────────────────────────────
        ema.apply_shadow(model)
        model.eval()
        test_loss = test_acc = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                test_acc += (outputs.argmax(1) == labels).float().mean().item()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        ema.restore(model)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch + 1:>3}/{epochs} | LR: {current_lr:.5f}")
        print(f"  Train  Acc: {train_acc * 100:.2f}%  Loss: {train_loss:.4f}")
        print(f"  Test   Acc: {test_acc * 100:.2f}%  Loss: {test_loss:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_best:
                ema.apply_shadow(model)
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_acc": test_acc,
                        "test_loss": test_loss,
                        "ema_shadow": ema.shadow,
                    },
                    checkpoint_path,
                )
                ema.restore(model)
            print(f"Best saved (Test Acc: {test_acc * 100:.2f}%)")

        if early_stopping(test_acc, epoch + 1):
            print(
                f"\nEarly stop @ epoch {epoch + 1} | "
                f"Best: {best_test_acc * 100:.2f}% @ epoch {early_stopping.best_epoch}"
            )
            break

        print("-" * 35)

    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (Test Acc: {best_test_acc * 100:.2f}%)")

    return model, history


# ── Test-Time Augmentation ────────────────────────────────────────────────────


def evaluate_tta(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    pad: int = 8,
) -> float:
    """10-view TTA over a DataLoader: 5 reflect-padded crops × 2 (+ hflip).

    Scale-preserving — no resize, so fixed-frequency filters see identical
    spatial frequencies as during training.

    Returns top-1 accuracy as a float in [0, 1].
    """
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in test_loader:
            B, C, H, W = images.shape
            p2 = pad * 2
            padded = F.pad(images, [pad] * 4, mode="reflect")

            crops = [
                padded[:, :, pad : pad + H, pad : pad + W],
                padded[:, :, 0:H, 0:W],
                padded[:, :, 0:H, p2 : p2 + W],
                padded[:, :, p2 : p2 + H, 0:W],
                padded[:, :, p2 : p2 + H, p2 : p2 + W],
            ]
            views = [v for crop in crops for v in (crop, torch.flip(crop, dims=[-1]))]

            flat = torch.stack(views, dim=1).view(B * 10, C, H, W).to(device)
            probs = torch.softmax(model(flat), dim=1).view(B, 10, -1).mean(dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    preds = torch.cat(all_probs).argmax(dim=1)
    labels = torch.cat(all_labels)
    return (preds == labels).float().mean().item()


def predict_tta(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    pad: int = 8,
) -> torch.Tensor:
    """Single-image TTA — same 10 views as evaluate_tta().

    Args:
        image_tensor: (1, C, H, W), already normalised.

    Returns:
        (1, num_classes) averaged softmax probabilities.
    """
    model.eval()
    _, C, H, W = image_tensor.shape
    p2 = pad * 2
    padded = F.pad(image_tensor, [pad] * 4, mode="reflect")

    crops = [
        padded[:, :, pad : pad + H, pad : pad + W],
        padded[:, :, 0:H, 0:W],
        padded[:, :, 0:H, p2 : p2 + W],
        padded[:, :, p2 : p2 + H, 0:W],
        padded[:, :, p2 : p2 + H, p2 : p2 + W],
    ]
    views = [v for crop in crops for v in (crop, torch.flip(crop, dims=[-1]))]

    flat = torch.cat(views, dim=0).to(device)
    with torch.no_grad():
        return torch.softmax(model(flat), dim=1).mean(dim=0, keepdim=True)
