import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.models as tvm

from torchinfo import summary as ti_summary
from .training import model_train
from .metrics import benchmark_cpu_inference, evaluate_topk
from config import (
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEFAULT_MODEL_NAME,
    EPOCHS,
    CHECKPOINT_DIR,
)

# ── Colour palette ────────────────────────────────────────────────────────────

_ACCENT = "#1565c0"
_BASELINE = "#757575"
_LR_CURVE = "#546e7a"
_TEXT_TITLE = "#000000"
_TEXT_DIM = "#424242"
_EDGE_OURS = "#000000"

# ── Plot style ────────────────────────────────────────────────────────────────

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#bbbbbb",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": _TEXT_TITLE,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 1.8,
    "font.family": "DejaVu Sans",
}


# ── Comparison model registry ─────────────────────────────────────────────────


def get_comparison_models(num_classes: int) -> dict[str, nn.Module]:
    return {
        "ShuffleNetV2-0.5×": tvm.shufflenet_v2_x0_5(
            weights=None, num_classes=num_classes
        ),
        "MobileNetV2-0.35×": tvm.mobilenet_v2(
            weights=None, width_mult=0.35, num_classes=num_classes
        ),
        "SqueezeNet-1.1": tvm.squeezenet1_1(weights=None, num_classes=num_classes),
        "MobileNetV3-Small": tvm.mobilenet_v3_small(
            weights=None, num_classes=num_classes
        ),
    }


# ── Metric collector ──────────────────────────────────────────────────────────


def _collect_metrics(
    name: str,
    model: nn.Module,
    history: dict,
    test_loader,
    device: torch.device,
    img_dim: tuple[int, int],
) -> dict:
    acc = evaluate_topk(model, test_loader, device, k=5)
    lat = benchmark_cpu_inference(model, img_dim=img_dim, n_warmup=50, n_runs=500)
    info = ti_summary(model, input_size=(1, 3, img_dim[0], img_dim[1]), verbose=0)

    topk_key = next(k for k in acc if k.startswith("top") and k != "top1")

    return {
        "name": name,
        "top1": acc["top1"] * 100,
        "top5": acc[topk_key] * 100,
        "params": info.total_params,
        "macs_M": info.total_mult_adds / 1e6,
        "size_MB": info.total_param_bytes / 1e6,
        "latency_ms": lat["median_ms"],
        "iqr_ms": lat["iqr_ms"],
        "fps": lat["throughput_fps"],
        "history": history,
    }


# ── Main runner ───────────────────────────────────────────────────────────────


def run_experiment(
    halkanet: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    num_classes: int,
    img_dim: tuple[int, int] = (64, 64),
    epochs: int = EPOCHS,
    patience: int = 30,
    use_cutmix: bool = True,
    cutmix_alpha: float = 0.4,
    cutmix_prob: float = 0.3,
    cutmix_start_epoch: int = 20,
    cutmix_end_epoch: int | None = None,
    comparison_epochs: int = 100,
    compare: bool = False,
    checkpoint_dir: str = CHECKPOINT_DIR,
    label_smoothing: float = LABEL_SMOOTHING,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    save_best: bool = True,
) -> dict[str, dict]:
    """Train HalkaNet and optionally all comparison models, collect metrics.

    HalkaNet  : CutMix ON (start→end epoch), patience as specified.
    Baselines : CutMix OFF, same LR / WD / criterion / patience.

    Returns dict[model_name → metrics_dict].
    Pass to print_comparison_table() and plot_training_curves().
    """
    if save_best:
        os.makedirs(checkpoint_dir, exist_ok=True)

    results = {}
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print("\n" + "=" * 55)
    print(f"  Training {DEFAULT_MODEL_NAME}")
    print("=" * 55)

    halkanet_opt = optim.AdamW(halkanet.parameters(), lr=lr, weight_decay=weight_decay)
    trained_halkanet, h_history = model_train(
        model=halkanet,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=halkanet_opt,
        device=device,
        epochs=epochs,
        patience=patience,
        use_cutmix=use_cutmix,
        cutmix_alpha=cutmix_alpha,
        cutmix_prob=cutmix_prob,
        cutmix_start_epoch=cutmix_start_epoch,
        cutmix_end_epoch=cutmix_end_epoch,
        checkpoint_path=f"{checkpoint_dir}/halkanet.pth",
        save_best=save_best,
    )

    results[DEFAULT_MODEL_NAME] = _collect_metrics(
        DEFAULT_MODEL_NAME,
        trained_halkanet,
        h_history,
        test_loader,
        device,
        img_dim,
    )

    if compare:
        for name, model in get_comparison_models(num_classes).items():
            print("\n" + "=" * 55)
            print(f"  Training: {name}")
            print("=" * 55)

            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            trained_model, b_history = model_train(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=opt,
                device=device,
                epochs=comparison_epochs,
                patience=patience,
                use_cutmix=False,
                save_best=save_best,
                checkpoint_path=f"{checkpoint_dir}/{name.replace(' ', '_').replace('×', 'x')}.pth",
            )

            results[name] = _collect_metrics(
                name,
                trained_model,
                b_history,
                test_loader,
                device,
                img_dim,
            )

    return results


# ── Comparison table ──────────────────────────────────────────────────────────


def print_comparison_table(results: dict[str, dict]) -> None:
    sep = "─" * 100
    print(f"\n{sep}")
    print(
        f"  {'Model':<22} {'Top-1':>6} {'Top-5':>6} {'Params':>10} "
        f"{'MACs(M)':>9} {'Size(MB)':>9} {'Lat(ms)':>9} {'FPS':>7}"
    )
    print(sep)

    for name, r in results.items():
        marker = "  ◄" if DEFAULT_MODEL_NAME in name else ""
        print(
            f"  {name:<22} {r['top1']:>5.2f}% {r['top5']:>5.2f}% "
            f"{r['params']:>10,} {r['macs_M']:>9.1f} "
            f"{r['size_MB']:>9.2f} {r['latency_ms']:>7.2f} "
            f"{r['fps']:>7.1f}{marker}"
        )

    print(sep)

    h = results.get(DEFAULT_MODEL_NAME)
    if h:
        print(
            f"\n  {DEFAULT_MODEL_NAME} — {h['top1']:.2f}% Top-1 | "
            f"{h['params']:,} params | "
            f"{h['latency_ms']:.2f} ± {h['iqr_ms']:.2f} ms | "
            f"{h['fps']:.1f} FPS\n"
        )


# ── Training curves + comparison chart ───────────────────────────────────────


def _is_ours(name: str) -> bool:
    return DEFAULT_MODEL_NAME in name


def plot_training_curves(
    results: dict[str, dict],
    save_path: str | None = None,
    dataset_name: str = " ",
    title: str = "HalkaNet - Training & Comparison:",
) -> None:
    """Plot per-model training panels and a Top-1 comparison bar chart.

    Args:
        results      : output from run_experiment()
        save_path    : optional save path; both .pdf and .png are always written.
        dataset_name : label shown in the comparison bar chart title.
    """
    has_comparison = len(results) > 1
    n_models = len(results)
    n_rows = n_models + (1 if has_comparison else 0)

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(16, n_rows * 3.8), layout="constrained")
        fig.suptitle(
            f"{title} {dataset_name}",
            fontsize=15,
            fontweight="bold",
            color=_TEXT_TITLE,
        )
        gs = gridspec.GridSpec(n_rows, 3, figure=fig)
        model_colors = [_ACCENT if _is_ours(n) else _BASELINE for n in results]

        for row_idx, ((name, r), c) in enumerate(zip(results.items(), model_colors)):
            h = r["history"]
            epochs_ran = len(h["train_acc"])
            ep = range(1, epochs_ran + 1)
            best_acc = max(h["test_acc"]) * 100

            ax_acc = fig.add_subplot(gs[row_idx, 0])
            ax_acc.plot(
                ep,
                [v * 100 for v in h["train_acc"]],
                label="Train",
                color=c,
                alpha=0.55,
                linestyle="--",
            )
            ax_acc.plot(ep, [v * 100 for v in h["test_acc"]], label="Test", color=c)
            ax_acc.axhline(best_acc, color=c, alpha=0.25, linewidth=1, linestyle=":")
            ax_acc.annotate(
                f"{best_acc:.2f}%",
                xy=(epochs_ran, best_acc),
                xytext=(-4, 5),
                textcoords="offset points",
                fontsize=7.5,
                color=c,
                ha="right",
            )
            ax_acc.set_title(f"{name}  —  Accuracy", fontsize=9, pad=6)
            ax_acc.set_xlabel("Epoch", fontsize=8)
            ax_acc.set_ylabel("Accuracy (%)", fontsize=8)
            ax_acc.legend(fontsize=7.5, framealpha=0.6)
            ax_acc.tick_params(labelsize=7)

            ax_loss = fig.add_subplot(gs[row_idx, 1])
            ax_loss.plot(
                ep, h["train_loss"], label="Train", color=c, alpha=0.55, linestyle="--"
            )
            ax_loss.plot(ep, h["test_loss"], label="Test", color=c)
            ax_loss.set_title(f"{name}  —  Loss", fontsize=9, pad=6)
            ax_loss.set_xlabel("Epoch", fontsize=8)
            ax_loss.set_ylabel("Loss", fontsize=8)
            ax_loss.legend(fontsize=7.5, framealpha=0.6)
            ax_loss.tick_params(labelsize=7)

            ax_lr = fig.add_subplot(gs[row_idx, 2])
            ax_lr.plot(ep, h["lr"], color=_LR_CURVE, linewidth=1.4)
            ax_lr.fill_between(ep, h["lr"], alpha=0.08, color=_LR_CURVE)
            ax_lr.set_title(f"{name}  —  LR Schedule", fontsize=9, pad=6)
            ax_lr.set_xlabel("Epoch", fontsize=8)
            ax_lr.set_ylabel("Learning Rate", fontsize=8)
            ax_lr.set_yscale("log")
            ax_lr.tick_params(labelsize=7)

        if has_comparison:
            ax_bar = fig.add_subplot(gs[n_models, :])
            names = list(results.keys())
            top1_vals = [r["top1"] for r in results.values()]

            bars = ax_bar.barh(
                names,
                top1_vals,
                color=[_ACCENT if _is_ours(n) else _BASELINE for n in names],
                edgecolor=[_EDGE_OURS if _is_ours(n) else "none" for n in names],
                linewidth=0.8,
                height=0.55,
            )

            ax_bar.set_xlim(max(0, min(top1_vals) - 6), min(100, max(top1_vals) + 6))
            ax_bar.set_title(
                f"Top-1 Accuracy Comparison  ({dataset_name} · from scratch)",
                fontsize=10,
                pad=8,
            )
            ax_bar.set_xlabel("Top-1 Accuracy (%)", fontsize=9)
            ax_bar.tick_params(axis="y", labelsize=8.5)
            ax_bar.tick_params(axis="x", labelsize=8)
            ax_bar.invert_yaxis()

            for bar, val, name in zip(bars, top1_vals, names):
                ax_bar.text(
                    val + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold" if _is_ours(name) else "normal",
                    color=_ACCENT if _is_ours(name) else _TEXT_DIM,
                )

        if save_path:
            base = save_path.rsplit(".", 1)[0]
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
            fig.savefig(
                f"{base}.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            fig.savefig(
                f"{base}.png",
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            print(f"Saved → {base}.pdf / .png")

        plt.show()
