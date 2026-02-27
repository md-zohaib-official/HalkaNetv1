import torch
from torch import nn
from torchinfo import summary
import time
import numpy as np

from config import IMAGE_DIMENSION, IMAGE_NORMALIZATION_MEAN, IMAGE_NORMALIZATION_STD


def print_model_summary(model: nn.Module, img_dim=IMAGE_DIMENSION) -> None:
    print(
        summary(
            model,
            input_size=(1, 3, img_dim[0], img_dim[1]),
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
            ],
        )
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] == 3:
        for i in range(3):
            img[i] = img[i] * IMAGE_NORMALIZATION_STD[i] + IMAGE_NORMALIZATION_MEAN[i]
    elif img.shape[0] == 1:
        mean_avg = sum(IMAGE_NORMALIZATION_MEAN) / 3
        std_avg = sum(IMAGE_NORMALIZATION_STD) / 3
        img[0] = img[0] * std_avg + mean_avg
    else:
        raise ValueError(f"Expected 1 or 3 channel but got {img.shape[0]}.")
    return img


# ── CPU Info ──────────────────────────────────────────────────────────────────


def get_cpu_info() -> dict[str, str]:
    """
    Returns CPU brand, physical core count, and logical core count.
    Requires: pip install py-cpuinfo psutil
    Falls back gracefully if either package is missing.
    """
    info = {}

    try:
        import cpuinfo

        raw = cpuinfo.get_cpu_info()
        info["brand"] = raw.get("brand_raw", "Unknown CPU")
        info["arch"] = raw.get("arch", "Unknown arch")
        info["hz"] = raw.get("hz_advertised_friendly", "Unknown Hz")
    except ImportError:
        info["brand"] = "Unknown (pip install py-cpuinfo)"
        info["arch"] = "Unknown"
        info["hz"] = "Unknown"

    try:
        import psutil

        info["cores_physical"] = str(psutil.cpu_count(logical=False))
        info["cores_logical"] = str(psutil.cpu_count(logical=True))
    except ImportError:
        info["cores_physical"] = "Unknown (pip install psutil)"
        info["cores_logical"] = "Unknown"

    return info


def print_cpu_info() -> dict[str, str]:
    """Print and return CPU info. Call once at the top of your benchmark cell."""
    info = get_cpu_info()
    print("┌─ CPU Information " + "─" * 36)
    print(f"│  Brand    : {info['brand']}")
    print(f"│  Arch     : {info['arch']}")
    print(f"│  Clock    : {info['hz']}")
    print(
        f"│  Cores    : {info['cores_physical']} physical / {info['cores_logical']} logical"
    )
    print("└" + "─" * 54)
    return info


# ── Top-K Accuracy (batch-level) ──────────────────────────────────────────────


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Compute top-K accuracy for a single batch.

    Args:
        outputs : raw logits (B, num_classes) — do NOT softmax before passing
        targets : ground truth class indices (B,)
        k       : K for top-K. Use k=5 for CIFAR-100, Food-101 (101 classes).
                  Automatically clamped to num_classes if k > num_classes.

    Returns:
        float: top-K accuracy in range [0, 1]

    Usage:
        outputs = model(images)
        top5 = top_k_accuracy(outputs, labels, k=5)
    """
    with torch.no_grad():
        k_clamped = min(k, outputs.size(1))
        _, top_k_preds = outputs.topk(k_clamped, dim=1, largest=True, sorted=True)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()


# ── Full Test-Set Evaluation (Top-1 + Top-K) ──────────────────────────────────


def evaluate_topk(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate model on full test set, returning Top-1 and Top-K accuracy.
    Always pass test_loader (not train_loader).

    Usage:
        results = evaluate_topk(model, test_loader, device, k=5)
        print(f"Top-1: {results['top1']*100:.2f}%  Top-5: {results['top5']*100:.2f}%")
    """
    model.eval()
    top1_correct = 0
    topk_correct = 0
    n_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            preds = outputs.argmax(dim=1)
            top1_correct += preds.eq(labels).sum().item()

            k_clamped = min(k, outputs.size(1))
            _, topk_preds = outputs.topk(k_clamped, dim=1, largest=True, sorted=True)
            topk_correct += (
                topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
                .any(dim=1)
                .sum()
                .item()
            )
            n_total += labels.size(0)

    top1 = top1_correct / n_total
    topk = topk_correct / n_total

    print(f"  Top-1 Accuracy : {top1 * 100:.2f}%")
    print(f"  Top-{k_clamped} Accuracy : {topk * 100:.2f}%")
    print(f"  Samples        : {n_total:,}")

    return {"top1": top1, f"top{k_clamped}": topk, "n_samples": n_total}


# ── CPU Inference Benchmark ────────────────────────────────────────────────────


def benchmark_cpu_inference(
    model: nn.Module,
    img_dim: tuple[int, int],
    img_channels: int = 3,
    n_warmup: int = 50,
    n_runs: int = 500,
    batch_size: int = 1,
) -> dict[str, float]:
    r"""
    Measure CPU inference latency and automatically select the correct
    statistic for publication (mean±std vs median±IQR).

    WHY TWO STATISTICS?
    ───────────────────
    On a general-purpose OS, background tasks (disk I/O, network, other
    processes) randomly pause your program mid-inference, creating timing
    spikes. For fast models these spikes can be 10–30× the real inference
    time, making mean±std misleading.

    The function checks this automatically:

      Stability ratio = std / mean

      ratio < 0.3  → timings are clean → mean ± std  (use this in paper)
      ratio ≥ 0.3  → OS jitter present → median ± IQR (use this in paper)

    IQR (Interquartile Range) = Q75 − Q25
      This covers the middle 50% of runs, completely ignoring the spikes.
      It is unaffected by how extreme the outliers are.

    WHY n_runs=500 (increased from 200)?
      More runs = better median/IQR estimates. 500 runs is still fast
      (< 30 seconds for models under 100ms) and gives stable statistics.

    Args:
        model        : trained model (CPU copy used internally)
        img_dim      : (H, W) e.g. (32, 32) or (96, 96)
        img_channels : 3 for RGB
        n_warmup     : warm-up runs discarded before timing (increased to 50)
        n_runs       : timed runs (increased to 500)
        batch_size   : 1 for latency, >1 for throughput

    Returns:
        dict with mean_ms, std_ms, median_ms, iqr_ms, q25_ms, q75_ms,
               min_ms, max_ms, throughput_fps, recommended_stat

    Usage:
        stats = benchmark_cpu_inference(model, img_dim=(32, 32))
        stats = benchmark_cpu_inference(model, img_dim=(96, 96))
    """
    # CPU copy — never move original model off GPU
    if next(model.parameters()).device.type == "cpu":
        cpu_model = model
    else:
        import copy

        cpu_model = copy.deepcopy(model).cpu()

    cpu_model.eval()
    dummy = torch.randn(batch_size, img_channels, img_dim[0], img_dim[1])

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"[benchmark_cpu_inference] Warming up ({n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = cpu_model(dummy)

    # ── Timed runs ────────────────────────────────────────────────────────────
    print(
        f"[benchmark_cpu_inference] Timing ({n_runs} runs, batch_size={batch_size})..."
    )
    times_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = cpu_model(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)

    # ── Statistics ────────────────────────────────────────────────────────────
    mean_ms = float(arr.mean())
    std_ms = float(arr.std())
    median_ms = float(np.median(arr))
    q25_ms = float(np.percentile(arr, 25))
    q75_ms = float(np.percentile(arr, 75))
    iqr_ms = q75_ms - q25_ms
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    fps = 1000.0 / median_ms * batch_size  # fps from median (more honest)

    # ── Stability check: decide which stat to report ───────────────────────
    # std/mean > 0.3 means OS jitter is significant → use median ± IQR
    stability_ratio = std_ms / mean_ms if mean_ms > 0 else 0.0
    use_median = stability_ratio >= 0.3
    recommended_stat = "median ± IQR" if use_median else "mean ± std"

    if use_median:
        paper_line = f"{median_ms:.2f} ± {iqr_ms:.2f} ms  [median ± IQR, n={n_runs}]"
        jitter_note = (
            f"  ⚠  Jitter detected (std/mean = {stability_ratio:.2f} ≥ 0.30)\n"
            f"     OS background tasks are adding random spikes.\n"
            f"     median ± IQR is the honest statistic for publication."
        )
    else:
        paper_line = f"{mean_ms:.2f} ± {std_ms:.2f} ms  [mean ± std, n={n_runs}]"
        jitter_note = (
            f"     Clean signal (std/mean = {stability_ratio:.2f} < 0.30)\n"
            f"     mean ± std is appropriate for publication."
        )

    # ── CPU info (shown once per benchmark call) ───────────────────────────
    cpu_info = get_cpu_info()

    # ── Print ──────────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  CPU Inference Benchmark  (batch={batch_size})")
    print(f"  Input  : {img_channels}×{img_dim[0]}×{img_dim[1]}")
    print(f"  Device : {cpu_info['brand']}")
    print(
        f"  Cores  : {cpu_info['cores_physical']} physical / {cpu_info['cores_logical']} logical"
    )
    print(sep)
    print(f"  Mean latency   : {mean_ms:.3f} ms  ± {std_ms:.3f} ms (std)")
    print(f"  Median latency : {median_ms:.3f} ms")
    print(f"  IQR  (Q25–Q75) : {q25_ms:.3f} – {q75_ms:.3f} ms  (IQR = {iqr_ms:.3f} ms)")
    print(f"  Min  / Max     : {min_ms:.3f} / {max_ms:.3f} ms")
    print(f"  Throughput     : {fps:.1f} FPS  (from median)")
    print(
        f"  Stability      : std/mean = {stability_ratio:.2f}  → use {recommended_stat}"
    )
    print(sep)
    print(jitter_note)
    print(sep)
    print(f"    Paper format: {paper_line}")
    print(
        f"    Paper CPU   : {cpu_info['brand']} ({cpu_info['cores_physical']} cores)"
    )
    print(f"{sep}\n")

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "iqr_ms": iqr_ms,
        "q25_ms": q25_ms,
        "q75_ms": q75_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "throughput_fps": fps,
        "recommended_stat": recommended_stat,
        "cpu_brand": cpu_info["brand"],
    }


# ── Full Paper-Ready Model Report ─────────────────────────────────────────────


def print_paper_report(
    model: nn.Module,
    img_dim: tuple[int, int],
    num_classes: int,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    img_channels: int = 3,
    topk: int = 5,
    n_warmup: int = 50,
    n_runs: int = 500,
) -> None:
    """
    Single call — prints everything needed for a research paper:
      1. CPU info (brand, cores)
      2. torchinfo summary (params, MACs, model size)
      3. Top-1 + Top-K accuracy on test set
      4. CPU latency benchmark with automatic mean/median selection

    Usage:
        print_paper_report(
            model, img_dim=(32, 32), num_classes=10,
            test_loader=test_loader, device=device,
        )
        # CIFAR-100 / Food-101:
        print_paper_report(
            model, img_dim=(32, 32), num_classes=100,
            test_loader=test_loader, device=device, topk=5,
        )
    """
    print("=" * 55)
    print("  HALKANET — PAPER REPORT")
    print(
        f"  Input: {img_channels}×{img_dim[0]}×{img_dim[1]}  |  Classes: {num_classes}"
    )
    print("=" * 55)

    print("\n[0/3] System:")
    print_cpu_info()

    print("\n[1/3] Architecture Summary (torchinfo):")
    print_model_summary(model, img_dim=img_dim)

    print(f"\n[2/3] Test Set Accuracy (Top-1 + Top-{topk}):")
    evaluate_topk(model, test_loader, device, k=topk)

    print("\n[3/3] CPU Inference Latency:")
    benchmark_cpu_inference(
        model,
        img_dim=img_dim,
        img_channels=img_channels,
        n_warmup=n_warmup,
        n_runs=n_runs,
        batch_size=1,
    )
