import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from .metrics import unnormalize


def plot_confusion_matrix(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    title: str = "Confusion Matrix",
    threshold: float = 0.10,
) -> None:
    model.eval()
    all_preds = []
    all_labels = []

    print("Generating predictions for Confusion Matrix...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = np.trace(cm) / np.sum(cm) * 100
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    n = len(class_names)
    fig_size = max(8, n * 0.7)  
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{title}\nTest Accuracy: {accuracy:.2f}%", fontsize=14, fontweight="bold"
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    plt.show()

    print(f"\nMost Confused Pairs (Errors > {threshold * 100:.2f}%):")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_normalized[i][j] > threshold:
                print(
                    f"  - True '{class_names[i]}' predicted as "
                    f"'{class_names[j]}': {cm_normalized[i][j] * 100:.1f}%"
                )


def show_img(
    X: torch.Tensor,
    y: torch.Tensor | int,
    y_pred: torch.Tensor | int | None = None,
    classes_str: list[str] = None,
) -> None:
    if isinstance(y, torch.Tensor):
        y = y.item()

    if y_pred is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.item()

    try:
        true_class = classes_str[y]
    except IndexError:
        print("True class index out of range.")
        return None

    pred_class = None
    if y_pred is not None:
        try:
            pred_class = classes_str[y_pred]
        except IndexError:
            print("Prediction class index out of range.")
            return None

    X = X.cpu().detach()
    img = unnormalize(X.clone())

    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np.clip(0, 1)

    title = f"True: {true_class}"
    if pred_class is not None:
        color = "green" if y == y_pred else "red"
        title += f"\nPred: {pred_class}"
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.title(title, color=color, fontweight="bold")
    else:
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def rgb_to_grayscale(img: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
    if img.dim() == 3:
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

        if keep_channels:
            gray = gray.unsqueeze(0).repeat(3, 1, 1)
        else:
            gray = gray.unsqueeze(0)

    elif img.dim() == 4:
        gray = (
            0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        )

        if keep_channels:
            gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            gray = gray.unsqueeze(1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.dim()}D")

    return gray


def show_img_grayscale(
    X: torch.Tensor,
    y: torch.Tensor | int,
    y_pred: torch.Tensor | int | None = None,
    classes_str: list[str] = None,
) -> None:
    if isinstance(y, torch.Tensor):
        y = y.item()

    if y_pred is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.item()

    try:
        true_class = classes_str[y]
    except IndexError:
        print("True class index out of range.")
        return None

    pred_class = None
    if y_pred is not None:
        try:
            pred_class = classes_str[y_pred]
        except IndexError:
            print("Prediction class index out of range.")
            return None

    X = X.cpu().detach()
    img = unnormalize(X.clone())

    if img.shape[0] == 1:
        img_np = img.squeeze(0).numpy()
    else:
        img_np = img[0].numpy()

    img_np = img_np.clip(0, 1)

    title = f"True: {true_class}"
    if pred_class is not None:
        color = "green" if y == y_pred else "red"
        title += f"\nPred: {pred_class}"
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        plt.title(title, color=color, fontweight="bold")
    else:
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def apply_haar_wavelet(img: torch.Tensor) -> dict[str, torch.Tensor]:
    gray = rgb_to_grayscale(img, keep_channels=False)

    device = gray.device
    dtype = gray.dtype

    ll = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device) / 2.0
    lh = torch.tensor([[1, 1], [-1, -1]], dtype=dtype, device=device) / 2.0
    hl = torch.tensor([[1, -1], [1, -1]], dtype=dtype, device=device) / 2.0
    hh = torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device) / 2.0

    kernels = torch.stack([ll, lh, hl, hh]).unsqueeze(1)

    gray = gray.unsqueeze(0)

    out = torch.nn.functional.conv2d(gray, kernels, padding=1)

    H, W = img.shape[1:]
    out = out[:, :, :H, :W]

    return {
        "LL": out[:, 0:1],
        "LH": out[:, 1:2],
        "HL": out[:, 2:3],
        "HH": out[:, 3:4],
    }


def show_wavelet_subbands(
    x: torch.Tensor,
    y: int | torch.Tensor,
    classes_str: list[str],
) -> None:
    if isinstance(y, torch.Tensor):
        y = y.item()

    wavelet_dict = apply_haar_wavelet(x)

    title_prefix = f"Class: {classes_str[y]}"

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    for ax, (name, band) in zip(axes, wavelet_dict.items()):
        band = band.squeeze().cpu().numpy()
        band = (band - band.min()) / (band.max() - band.min() + 1e-6)

        ax.imshow(band, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title_prefix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def apply_sobel_for_vis(img: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_grayscale(img, keep_channels=False)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=gray.device,
    )

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=gray.device,
    )

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    gray = gray.unsqueeze(0)

    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)

    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

    return grad_mag.squeeze(0)


def apply_gabor(
    img: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 3.0,
    lambd: float = 5.0,
    gamma: float = 0.5,
    psi: float = 0.0,
    orientations: tuple = (0, 45, 90, 135),
) -> dict[str, torch.Tensor]:
    gray = rgb_to_grayscale(img, keep_channels=False)
    device = gray.device
    dtype = gray.dtype

    half = kernel_size // 2
    y_g, x_g = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )

    kernels = []
    for theta_deg in orientations:
        theta = theta_deg * math.pi / 180
        x_theta = x_g * math.cos(theta) + y_g * math.sin(theta)
        y_theta = -x_g * math.sin(theta) + y_g * math.cos(theta)
        gb = torch.exp(
            -(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)
        ) * torch.cos(2 * math.pi * x_theta / lambd + psi)
        kernels.append(gb.to(dtype))

    weight = torch.stack(kernels).unsqueeze(1)
    gray = gray.unsqueeze(0)
    out = torch.nn.functional.conv2d(gray, weight, padding=half)

    return {f"{deg}°": out[:, i : i + 1] for i, deg in enumerate(orientations)}


def show_gabor_responses(
    x: torch.Tensor,
    y: int | torch.Tensor,
    classes_str: list[str],
    orientations: tuple = (0, 45, 90, 135),
    kernel_size: int = 7,
    sigma: float = 3.0,
    lambd: float = 5.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> None:
    if isinstance(y, torch.Tensor):
        y = y.item()

    img_rgb = unnormalize(x.clone().cpu().detach())
    img_np = img_rgb.permute(1, 2, 0).numpy().clip(0, 1)

    gabor_dict = apply_gabor(
        x.cpu().detach(),
        orientations=orientations,
        kernel_size=kernel_size,
        sigma=sigma,
        lambd=lambd,
        gamma=gamma,
        psi=psi,
    )

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, (name, response) in zip(axes[1:], gabor_dict.items()):
        resp = response.squeeze().cpu().numpy()
        resp = (resp - resp.min()) / (resp.max() - resp.min() + 1e-6)
        ax.imshow(resp, cmap="gray")
        ax.set_title(f"Gabor {name}")
        ax.axis("off")

    fig.suptitle(f"Class: {classes_str[y]}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def apply_log_for_vis(img: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    img: (3, H, W) RGB tensor, normalized (same as other vis funcs)
    Returns:
        dict with y3, y5 and combined LoG responses as 1×1×H×W tensors.
    """
    gray = rgb_to_grayscale(img, keep_channels=False)  # (1, H, W)
    gray = gray.unsqueeze(0)  # (1, 1, H, W)

    device = gray.device
    dtype = gray.dtype

    k3 = (
        torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=dtype,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,3,3)

    k5 = (
        torch.tensor(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=dtype,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,5,5)

    y3 = torch.nn.functional.conv2d(gray, k3, padding=1)  # (1,1,H,W)
    y5 = torch.nn.functional.conv2d(gray, k5, padding=2)  # (1,1,H,W)
    y = y3 + y5  # combined LoG

    return {
        "LoG 3×3": y3,
        "LoG 5×5": y5,
        "LoG sum": y,
    }


def show_log_responses(
    x: torch.Tensor,
    y: int | torch.Tensor,
    classes_str: list[str],
) -> None:
    """
    x: (3, H, W) RGB tensor from dataset (normalized)
    y: class index or tensor
    """
    if isinstance(y, torch.Tensor):
        y = y.item()

    # Original RGB (unnormalized) for reference
    img_rgb = unnormalize(x.clone().cpu().detach())
    img_np = img_rgb.permute(1, 2, 0).numpy().clip(0, 1)

    log_dict = apply_log_for_vis(x.cpu().detach())

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    # Panel 0 — original RGB
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panels 1–3 — LoG 3×3, LoG 5×5, LoG sum
    for ax, (name, resp) in zip(axes[1:], log_dict.items()):
        arr = resp.squeeze().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        ax.imshow(arr, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"LoG Responses — Class: {classes_str[y]}", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def apply_lbp_for_vis(img: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply the exact 4-channel compressed LBPFilter for visualization.

    Uses identical opposite-pair kernels from LBPFilter:
        Ch0: N0-N4  top minus bottom         → vertical edge direction
        Ch1: N2-N6  right minus left         → horizontal edge direction
        Ch2: N1-N5  top-right minus btm-left → diagonal \\ direction
        Ch3: N3-N7  btm-right minus top-left → diagonal / direction

    img: (3, H, W) RGB tensor, normalized
    Returns: dict of 4 named response maps, each (1, 1, H, W)
    """
    gray = rgb_to_grayscale(img, keep_channels=False)  # (1, H, W)
    gray = gray.unsqueeze(0)  # (1, 1, H, W)
    device = gray.device
    dtype = gray.dtype

    # Exact same pairs as LBPFilter.__init__
    pairs = [
        ((0, 1), (2, 1), "Ch0: top−bottom (↕)"),
        ((1, 2), (1, 0), "Ch1: right−left (↔)"),
        ((0, 2), (2, 0), "Ch2: diag \\"),
        ((2, 2), (0, 0), "Ch3: diag /"),
    ]

    kernels = []
    names = []
    for (r1, c1), (r2, c2), name in pairs:
        k = torch.zeros(3, 3, dtype=dtype, device=device)
        k[r1, c1] = 1.0
        k[r2, c2] = -1.0
        kernels.append(k)
        names.append(name)

    weight = torch.stack(kernels).unsqueeze(1)  # (4, 1, 3, 3)
    out = torch.nn.functional.conv2d(gray, weight, padding=1)
    # out: (1, 4, H, W)

    return {name: out[:, i : i + 1] for i, name in enumerate(names)}


def show_lbp_responses(
    x: torch.Tensor,
    y: int | torch.Tensor,
    classes_str: list[str],
) -> None:
    """Visualize all 4 compressed LBP directional responses for a single image.

    Layout:
        Row 0: [Original RGB] [Grayscale input] [Ch0 ↕] [Ch1 ↔]
        Row 1: [empty]        [empty]           [Ch2 \\] [Ch3 /]

    RdBu_r colormap on all 4 channels (signed responses):
        Red  → first neighbor direction is brighter  (e.g. top > bottom)
        Blue → opposite direction is brighter        (e.g. bottom > top)
        White → symmetric or flat region

    Usage:
        images, labels = next(iter(train_dataloader))
        show_lbp_responses(x=images[5], y=labels[5], classes_str=class_names)
    """
    if isinstance(y, torch.Tensor):
        y = y.item()

    img_rgb = unnormalize(x.clone().cpu().detach())
    img_np = img_rgb.permute(1, 2, 0).numpy().clip(0, 1)

    lbp_dict = apply_lbp_for_vis(x.cpu().detach())

    # Reserve right margin for colorbar manually — avoids tight_layout conflict
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)

    # Row 0, col 0: original RGB
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    # Row 0, col 1: grayscale (actual LBP input)
    gray_np = rgb_to_grayscale(x.cpu().detach()).squeeze().numpy()
    gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min() + 1e-6)
    axes[0, 1].imshow(gray_np, cmap="gray")
    axes[0, 1].set_title("Grayscale input", fontweight="bold")
    axes[0, 1].axis("off")

    # Hide unused row 1 cols 0-1
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")

    # 4 LBP channels — compute global vmax first for shared colorbar
    all_arrs = [lbp_dict[n].squeeze().cpu().numpy() for n in lbp_dict]
    global_vmax = max(max(abs(a.max()), abs(a.min())) for a in all_arrs) + 1e-6

    # [Ch0, Ch1] in row 0 cols 2-3, [Ch2, Ch3] in row 1 cols 2-3
    positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    for (row, col), (name, arr) in zip(positions, zip(lbp_dict.keys(), all_arrs)):
        axes[row, col].imshow(arr, cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax)
        axes[row, col].set_title(name, fontsize=9)
        axes[row, col].axis("off")

    # Colorbar in manually reserved right margin — no axis stealing
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-global_vmax, vmax=global_vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("neighbor − opposite", fontsize=9)
    cbar.set_ticks([-global_vmax, 0, global_vmax])
    cbar.set_ticklabels(["opposite\nbrighter", "symmetric", "first\nbrighter"])

    fig.suptitle(
        f"LBP Responses (4-ch compressed) — Class: {classes_str[y]}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.show()


def apply_dog_for_vis(
    img: torch.Tensor,
    sigma1: float = 1.0,
    sigma2: float = 2.5,
    kernel_size: int = 7,
) -> dict[str, torch.Tensor]:
    """Apply the exact DoGFilter kernel to a single image for visualization.

    Uses identical kernel construction to DoGFilter._build_kernel:
    two normalized Gaussians (sigma1, sigma2) subtracted → bandpass.

    img: (3, H, W) RGB tensor, normalized
    Returns dict with:
        "G1 (fine)"   — fine Gaussian (sigma1) response
        "G2 (coarse)" — coarse Gaussian (sigma2) response
        "DoG"         — G1 - G2 (the actual filter output)
    """
    gray = rgb_to_grayscale(img, keep_channels=False)  # (1, H, W)
    gray = gray.unsqueeze(0)  # (1, 1, H, W)
    device = gray.device
    dtype = gray.dtype

    # Exact same kernel build as DoGFilter._build_kernel
    half = kernel_size // 2
    y_g, x_g = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )
    r2 = x_g**2 + y_g**2

    g1 = torch.exp(-r2 / (2 * sigma1**2))
    g2 = torch.exp(-r2 / (2 * sigma2**2))
    g1 = g1 / g1.sum()
    g2 = g2 / g2.sum()
    dog = g1 - g2  # sum ≈ 0

    pad = kernel_size // 2

    resp_g1 = torch.nn.functional.conv2d(
        gray, g1.to(dtype).unsqueeze(0).unsqueeze(0), padding=pad
    )
    resp_g2 = torch.nn.functional.conv2d(
        gray, g2.to(dtype).unsqueeze(0).unsqueeze(0), padding=pad
    )
    resp_dog = torch.nn.functional.conv2d(
        gray, dog.to(dtype).unsqueeze(0).unsqueeze(0), padding=pad
    )

    return {
        f"G1 σ={sigma1}": resp_g1,  # blurred fine
        f"G2 σ={sigma2}": resp_g2,  # blurred coarse
        "DoG (G1−G2)": resp_dog,  # bandpass result
    }


def show_dog_responses(
    x: torch.Tensor,
    y: int | torch.Tensor,
    classes_str: list[str],
    sigma1: float = 1.0,
    sigma2: float = 2.5,
    kernel_size: int = 7,
) -> None:
    """Visualize DoGFilter responses: G1, G2, and the final DoG for a single image.

    Layout: [Original] [Grayscale] [G1 fine blur] [G2 coarse blur] [DoG = G1−G2]

    G1 and G2 use grayscale (they are blurred images, always positive).
    DoG uses RdBu_r (diverging) — sign is meaningful:
        Red  → bright structure on dark background (bright blob, highlight)
        Blue → dark structure on bright background (pupil, nostril, dark fur)
        White → flat region / background

    Usage:
        images, labels = next(iter(train_dataloader))
        show_dog_responses(x=images[5], y=labels[5], classes_str=class_names)
    """
    if isinstance(y, torch.Tensor):
        y = y.item()

    img_rgb = unnormalize(x.clone().cpu().detach())
    img_np = img_rgb.permute(1, 2, 0).numpy().clip(0, 1)

    dog_dict = apply_dog_for_vis(
        x.cpu().detach(),
        sigma1=sigma1,
        sigma2=sigma2,
        kernel_size=kernel_size,
    )

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    # Panel 0: original RGB
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    # Panel 1: grayscale (what DoG actually receives)
    gray_np = rgb_to_grayscale(x.cpu().detach()).squeeze().numpy()
    gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min() + 1e-6)
    axes[1].imshow(gray_np, cmap="gray")
    axes[1].set_title("Grayscale input", fontweight="bold")
    axes[1].axis("off")

    # Panels 2-3: G1 and G2 (blurred images — always positive, use gray)
    labels_list = list(dog_dict.keys())
    for i, name in enumerate(labels_list[:2]):
        arr = dog_dict[name].squeeze().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        axes[i + 2].imshow(arr, cmap="gray")
        axes[i + 2].set_title(name, fontsize=9)
        axes[i + 2].axis("off")

    # Panel 4: DoG — signed, use diverging colormap
    dog_arr = dog_dict["DoG (G1−G2)"].squeeze().cpu().numpy()
    vmax = max(abs(dog_arr.max()), abs(dog_arr.min())) + 1e-6
    axes[4].imshow(dog_arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[4].set_title("DoG (G1−G2)", fontsize=9, fontweight="bold")
    axes[4].axis("off")

    fig.suptitle(
        f"DoG Responses — Class: {classes_str[y]}  "
        f"[σ1={sigma1}, σ2={sigma2}, k={kernel_size}]",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ── t-SNE / UMAP Feature-Space Visualisation ──────────────────────────────────


def _find_penultimate_layer_name(model: torch.nn.Module) -> str:
    """
    Walk the module tree and return the name of the leaf module sitting
    immediately before the final nn.Linear (classifier head).
    Raises ValueError if auto-detection fails — pass feature_layer= explicitly.
    """
    modules = list(model.named_modules())
    last_linear_idx = None
    for i, (name, mod) in enumerate(modules):
        if isinstance(mod, torch.nn.Linear):
            last_linear_idx = i
    if last_linear_idx is None or last_linear_idx == 0:
        raise ValueError(
            "Auto-detection failed: no nn.Linear found. "
            "Pass feature_layer='<layer_name>' explicitly.\n"
            "Run:  for n, _ in model.named_modules(): print(n)"
        )
    for i in range(last_linear_idx - 1, 0, -1):
        name, mod = modules[i]
        if len(list(mod.children())) == 0:  # leaf module (no children)
            return name
    raise ValueError(
        "Auto-detection failed: could not find a leaf layer before the classifier. "
        "Pass feature_layer='<layer_name>' explicitly."
    )


def plot_feature_space(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    method: str = "tsne",
    n_samples: int = 2000,
    feature_layer: str | None = None,
    perplexity: int = 30,
    random_state: int = 42,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Visualise learned feature representations using t-SNE or UMAP.

    ╔══════════════════════════════════════════════════════════════╗
    ║  ALWAYS pass test_loader — NEVER train_loader.           ║
    ║  Passing train_loader leaks the training distribution into   ║
    ║  the visualisation, making clusters look artificially tight  ║
    ║  and misrepresenting the model's generalisation ability.     ║
    ╚══════════════════════════════════════════════════════════════╝

    Args:
        model:          Trained model (set to eval() internally).
        test_loader:    DataLoader built from the TEST split ONLY.
                        This must use the clean test_tf (no augmentation),
                        which is already guaranteed by _build_transforms().
        class_names:    List of class label strings (e.g. from get_data_loaders).
        device:         torch.device for inference.
        method:         "tsne" (sklearn, always available) or "umap"
                        (requires: pip install umap-learn).
        n_samples:      Max test samples to embed. 2000 is fast and clean.
                        Do not exceed 5000 for t-SNE (O(n²) memory).
        feature_layer:  Named module to hook for features. None = auto-detect
                        (last leaf layer before the final Linear head).
                        To inspect available names:
                            for n, _ in model.named_modules(): print(n)
        perplexity:     t-SNE perplexity. Must be < n_samples. Ignored for UMAP.
        random_state:   Seed for reproducible embeddings.
        title:          Override the plot title.
        save_path:      If given, saves figure here (e.g. "results/tsne_cifar10.png").

    Usage:
        # Basic (t-SNE, auto layer):
        plot_feature_space(model, test_loader, class_names, device)

        # UMAP:
        plot_feature_space(model, test_loader, class_names, device, method="umap")

        # Pin a specific layer + save:
        plot_feature_space(
            model, test_loader, class_names, device,
            feature_layer="backbone.gap",
            save_path="results/tsne_stl10.png",
        )
    """
    # ── 1. Resolve which layer to hook ────────────────────────────────────────
    layer_name = feature_layer or _find_penultimate_layer_name(model)
    _features_cache: list[torch.Tensor] = []
    hook_handle = None

    def _hook_fn(module, inp, out):
        # Flatten spatial dims (H×W) if present — works for both conv and linear outputs
        _features_cache.append(out.detach().cpu().flatten(start_dim=1))

    for name, mod in model.named_modules():
        if name == layer_name:
            hook_handle = mod.register_forward_hook(_hook_fn)
            break

    if hook_handle is None:
        raise ValueError(
            f"Layer '{layer_name}' not found in the model.\n"
            f"Run: for n, _ in model.named_modules(): print(n)"
        )

    # ── 2. Feature extraction — TEST SET ONLY ─────────────────────────────────
    # model.eval() ensures:
    #   - BatchNorm uses running stats (not batch stats)
    #   - Dropout is disabled
    #   - No training-time randomness contaminates features
    model.eval()
    all_labels: list[int] = []
    n_collected = 0

    print(f"[plot_feature_space] Hooking layer: '{layer_name}'")
    print(f"[plot_feature_space] Collecting up to {n_samples} TEST samples...")
    print(
        "[plot_feature_space] Using test_loader (no augmentation — clean features)"
    )

    with torch.no_grad():  # no gradients = no leakage of grad info, lower memory
        for images, labels in test_loader:
            remaining = n_samples - n_collected
            if remaining <= 0:
                break
            # Trim the last batch to not exceed n_samples
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]
            model(images.to(device))  # triggers hook
            all_labels.extend(labels.tolist())
            n_collected += images.size(0)

    hook_handle.remove()  # always clean up hooks

    if not _features_cache:
        raise RuntimeError("No features collected — test_loader appears to be empty.")

    features = torch.cat(_features_cache, dim=0).numpy()  # (N, D)
    labels_np = np.array(all_labels)
    print(
        f"[plot_feature_space] Collected {features.shape[0]} samples | feature dim = {features.shape[1]}"
    )

    # ── 3. Dimensionality reduction ───────────────────────────────────────────
    method_clean = method.lower().strip()

    if method_clean == "tsne":
        from sklearn.manifold import TSNE

        perp = min(perplexity, features.shape[0] - 1)
        print(f"[plot_feature_space] Running t-SNE (perplexity={perp})...")
        embedding = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        ).fit_transform(features)
        method_label = f"t-SNE (perplexity={perp})"

    elif method_clean == "umap":
        try:
            import umap as _umap
        except ImportError:
            raise ImportError(
                "umap-learn is not installed.\n"
                "Fix: pip install umap-learn\n"
                "Or use method='tsne' instead."
            )
        print("[plot_feature_space] Running UMAP...")
        embedding = _umap.UMAP(
            n_components=2,
            n_neighbors=min(15, features.shape[0] - 1),
            min_dist=0.1,
            random_state=random_state,
            n_jobs=1,
        ).fit_transform(features)
        method_label = "UMAP (n_neighbors=15, min_dist=0.1)"

    else:
        raise ValueError(f"method must be 'tsne' or 'umap', got '{method}'.")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    n_classes = len(class_names)
    # tab10 for ≤10 classes, tab20 for 11–20; beyond 20 colours wrap (rare in practice)
    cmap = plt.colormaps["tab10" if n_classes <= 10 else "tab20"].resampled(n_classes)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels_np == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.65,
            color=cmap(cls_idx),
            label=cls_name,
            linewidths=0,
        )

    ax.legend(
        title="Classes",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        markerscale=2.5,
        framealpha=0.9,
    )
    ax.set_xlabel(f"{method_label} — Dim 1", fontsize=11)
    ax.set_ylabel(f"{method_label} — Dim 2", fontsize=11)
    ax.set_title(
        title
        or (
            f"HalkaNet Feature Space — {method_label}\n"
            f"Test set only  |  layer: '{layer_name}'  |  n={features.shape[0]}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_feature_space] Figure saved -> '{save_path}'")

    plt.show()
    print("[plot_feature_space] Done.")
