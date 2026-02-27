"""
filters.py
==========
Fixed (non-trainable) hand-crafted convolutional filter banks for HalkaNet.

All filters operate on single-channel grayscale input ``(B, 1, H, W)`` and
produce ``(B, 1, H, W)`` output unless stated otherwise.  Kernels are
registered as buffers so they move with ``model.to(device)`` and are saved /
loaded via ``state_dict``, but are excluded from ``model.parameters()``.

Filter inventory
----------------
HaarWaveletFilter      – Single-subband 2-D Haar wavelet (LL / LH / HL / HH)
GaborFilter            – Single-orientation Gabor bandpass filter
LoGFilter              – Laplacian-of-Gaussian edge detector (3×3 or 5×5)
SobelGradMagnitude     – Sobel gradient magnitude (L1 MCU-safe / L2 Euclidean)
OrientedGradientFilter – Signed directional gradient pair (simplified LBP axis)
DoGFilter              – Difference-of-Gaussians bandpass blob detector

Naming updates (original → standard)
-------------------------------------
HaarWaveletFilter      unchanged  (standard signal-processing name)
GaborFilterSingle      → GaborFilter
LoGFilter5x5 / 3x3    → LoGFilter(kernel_size=5|3)
SobelMagnitudeFilterMCU→ SobelGradMagnitude
LBPSingleChannel       → OrientedGradientFilter
DoGFilter              unchanged  (standard name)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Return a normalised 2-D Gaussian kernel of shape ``(size, size)``."""
    half = size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def _to_conv_weight(kernel: torch.Tensor) -> torch.Tensor:
    """Reshape a 2-D kernel ``(H, W)`` to a conv2d weight ``(1, 1, H, W)``."""
    return kernel.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Haar Wavelet Filter
# ---------------------------------------------------------------------------


class HaarWaveletFilter(nn.Module):
    """Single-subband 2-D Haar wavelet decomposition filter.

    Applies the 2×2 Haar analysis filter bank and returns exactly one
    frequency subband, keeping spatial dimensions unchanged via ``padding=1``.

    Subbands
    --------
    ``LL``  Low × Low   — smoothed / coarse structure (approximation)
    ``LH``  Low × High  — horizontal edges
    ``HL``  High × Low  — vertical edges
    ``HH``  High × High — diagonal corners / fine texture

    Args:
        subband: One of ``"LL"``, ``"LH"``, ``"HL"``, ``"HH"`` (case-insensitive).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` single subband response
    """

    _KERNELS: dict[str, list[float]] = {
        "LL": [1.0, 1.0, 1.0, 1.0],
        "LH": [1.0, 1.0, -1.0, -1.0],
        "HL": [1.0, -1.0, 1.0, -1.0],
        "HH": [1.0, -1.0, -1.0, 1.0],
    }

    def __init__(self, subband: str = "HH") -> None:
        super().__init__()
        key = subband.upper()
        if key not in self._KERNELS:
            raise ValueError(
                f"subband must be one of {list(self._KERNELS)}, got {subband!r}"
            )
        self.subband = key
        raw = torch.tensor(self._KERNELS[key], dtype=torch.float32) / 2.0
        # shape: (1, 1, 2, 2)
        self.register_buffer("weight", raw.view(1, 1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 1, (
            f"HaarWaveletFilter expects 1-channel grayscale, got {x.shape[1]} channels"
        )
        b, _, h, w = x.shape
        out = F.conv2d(x, self.weight, padding=1)
        # crop back to original spatial size
        return out[:, :, :h, :w]

    def extra_repr(self) -> str:
        return f"subband={self.subband!r}"


# ---------------------------------------------------------------------------
# Gabor Filter
# ---------------------------------------------------------------------------


class GaborFilter(nn.Module):
    """Single-orientation Gabor bandpass filter.

    A Gabor filter is a Gaussian-windowed sinusoidal plane wave — the standard
    model of simple-cell receptive fields in the mammalian visual cortex.  It
    is selective for both *spatial frequency* (controlled by ``lambd``) and
    *orientation* (controlled by ``orientation``).

    Args:
        kernel_size:  Filter kernel spatial size (default 5).
        sigma:        Gaussian envelope width in pixels (default 1.8).
        lambd:        Sinusoidal wavelength in pixels (default 3.5).
        gamma:        Spatial aspect ratio — elongation of the Gaussian
                      envelope (default 0.5).
        psi:          Phase offset of the sinusoid in radians (default 0.0).
        orientation:  Filter orientation in **degrees** (default 45.0).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` filter response
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.8,
        lambd: float = 3.5,
        gamma: float = 0.5,
        psi: float = 0.0,
        orientation: float = 45.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer(
            "weight",
            self._build_kernel(kernel_size, sigma, lambd, gamma, psi, orientation),
        )

    @staticmethod
    def _build_kernel(
        k: int,
        sigma: float,
        lambd: float,
        gamma: float,
        psi: float,
        orientation: float,
    ) -> torch.Tensor:
        half = k // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij",
        )
        theta = orientation * math.pi / 180.0
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        kernel = torch.exp(
            -(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)
        ) * torch.cos(2 * math.pi * x_theta / lambd + psi)
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

    def extra_repr(self) -> str:
        k = self.weight.shape[-1]
        return f"kernel_size={k}"


# ---------------------------------------------------------------------------
# Laplacian-of-Gaussian (LoG) Filter
# ---------------------------------------------------------------------------


class LoGFilter(nn.Module):
    """Laplacian-of-Gaussian (LoG) edge and blob detector.

    The LoG filter (Marr & Hildreth 1980) combines Gaussian smoothing with
    the Laplacian second-derivative operator.  Zero-crossings of the response
    mark edge locations; the sign encodes bright-on-dark vs dark-on-bright.

    Two fixed discrete approximations are provided via ``kernel_size``:

    * **3×3** — discrete Laplacian (``[0,1,0 / 1,-4,1 / 0,1,0]``).
      Minimal footprint, maximum speed; suitable for fine-edge detection.
    * **5×5** — 5-tap LoG approximation
      (``[0,0,-1,0,0 / 0,-1,-2,-1,0 / -1,-2,16,-2,-1 / ...]``).
      Larger receptive field; captures broader blob structure.

    Args:
        kernel_size: ``3`` or ``5`` (default 5).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` LoG response (signed — do **not** apply AbsActivation)
    """

    _K3 = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    _K5 = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, -2.0, -1.0, 0.0],
            [-1.0, -2.0, 16.0, -2.0, -1.0],
            [0.0, -1.0, -2.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size == 3:
            kernel = self._K3
        elif kernel_size == 5:
            kernel = self._K5
        else:
            raise ValueError(f"kernel_size must be 3 or 5, got {kernel_size}")
        self.kernel_size = kernel_size
        self.register_buffer("weight", _to_conv_weight(kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


# ---------------------------------------------------------------------------
# Sobel Gradient Magnitude
# ---------------------------------------------------------------------------


class SobelGradMagnitude(nn.Module):
    """Sobel gradient magnitude — edge strength map.

    Computes per-pixel gradient magnitude from horizontal (Gx) and vertical
    (Gy) Sobel responses.  Two combination modes:

    ``"l1"`` (default, **MCU-safe**)::

        |Gx| + |Gy|   — taxicab / Manhattan approximation.
        Uses only Abs + Add; no Pow/Sqrt → compatible with TFLite Micro.

    ``"l2"`` (**GPU/CPU only**)::

        sqrt(Gx² + Gy²)   — true Euclidean gradient magnitude.
        Rotationally invariant but requires Pow + Sqrt (not MCU-safe).

    Output is always **non-negative**.  Do **not** apply
    :class:`~components.AbsActivation` after this filter; replace the
    downstream ``BatchNorm2d(2)`` with ``BatchNorm2d(1)``.

    Args:
        mode: ``"l1"`` (MCU-safe) or ``"l2"`` (Euclidean).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` non-negative gradient magnitude
    """

    def __init__(self, mode: str = "l1") -> None:
        super().__init__()
        if mode not in ("l1", "l2"):
            raise ValueError(f"mode must be 'l1' or 'l2', got {mode!r}")
        self.mode = mode
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        # shape: (2, 1, 3, 3) — two output channels (Gx, Gy)
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx, gy = F.conv2d(x, self.kernel, padding=1).chunk(2, dim=1)
        if self.mode == "l2":
            return torch.sqrt(gx**2 + gy**2 + 1e-6)
        return gx.abs() + gy.abs()  # l1: MCU-safe

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"


# ---------------------------------------------------------------------------
# Oriented Gradient Filter  (formerly LBPSingleChannel)
# ---------------------------------------------------------------------------


class OrientedGradientFilter(nn.Module):
    """Signed oriented gradient filter — one directional axis at a time.

    Computes the **signed difference** between two diametrically opposite
    neighbours of the centre pixel using a fixed 3×3 kernel with ``+1`` and
    ``−1`` at the selected pair positions.  This is a directional first-order
    finite difference, equivalent to a single oriented Prewitt/Sobel component.

    Note: despite the original name ``LBPSingleChannel``, this is **not**
    classic Local Binary Pattern encoding (which uses binary thresholding +
    bit packing).  It is an *oriented pairwise gradient* — a simpler, signed,
    MCU-friendly alternative that captures the same directional edge polarity.

    The four available axes cover the principal edge directions:

    =========  ====  ==========================
    ``axis``   Pair  Direction captured
    =========  ====  ==========================
    ``"v"``    N–S   Vertical edge (top − bottom)
    ``"h"``    E–W   Horizontal edge (right − left)
    ``"d1"``   NE–SW Diagonal NE–SW
    ``"d2"``   NW–SE Diagonal NW–SE
    =========  ====  ==========================

    Args:
        axis: One of ``"v"``, ``"h"``, ``"d1"``, ``"d2"`` (default ``"d2"``).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` signed gradient response
    """

    # (row1, col1, row2, col2) — +1 at (r1,c1), -1 at (r2,c2)
    _AXES: dict[str, tuple[int, int, int, int]] = {
        "v": (0, 1, 2, 1),  # top  − bottom   (N − S)
        "h": (1, 2, 1, 0),  # right − left     (E − W)
        "d1": (0, 2, 2, 0),  # NE   − SW
        "d2": (2, 2, 0, 0),  # NW   − SE  (formerly channel_idx=3)
    }

    # backward-compat mapping from the old integer channel_idx API
    _IDX_TO_AXIS: dict[int, str] = {0: "v", 1: "h", 2: "d1", 3: "d2"}

    def __init__(self, axis: str | int = "d2") -> None:
        super().__init__()
        if isinstance(axis, int):
            axis = self._IDX_TO_AXIS[axis]
        if axis not in self._AXES:
            raise ValueError(f"axis must be one of {list(self._AXES)}, got {axis!r}")
        self.axis = axis
        r1, c1, r2, c2 = self._AXES[axis]
        k = torch.zeros(3, 3, dtype=torch.float32)
        k[r1, c1] = 1.0
        k[r2, c2] = -1.0
        self.register_buffer("weight", _to_conv_weight(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)

    def extra_repr(self) -> str:
        return f"axis={self.axis!r}"


# ---------------------------------------------------------------------------
# Difference of Gaussians (DoG) Filter
# ---------------------------------------------------------------------------


class DoGFilter(nn.Module):
    """Difference-of-Gaussians (DoG) bandpass blob detector.

    Approximates the Laplacian-of-Gaussian by subtracting a wider Gaussian
    from a narrower one.  Isolates image structure at the spatial-frequency
    band defined by ``(sigma1, sigma2)``:

    * **Positive** response → bright structure surrounded by darker region
      at that scale (e.g. highlight blobs, rounded corners).
    * **Negative** response → dark structure surrounded by brighter region
      (e.g. eye pupils, nostrils, pores).

    Keep **both signs** — unlike Sobel/LoG outputs, sign here encodes
    polarity (bright-on-dark vs dark-on-bright), which is structurally
    meaningful.  Do **not** apply :class:`~components.AbsActivation`.

    Args:
        sigma1:      Std-dev of the narrow (inner) Gaussian (default 1.0).
        sigma2:      Std-dev of the wide (outer) Gaussian (default 2.5).
        kernel_size: Filter spatial size; should be ≥ ``2 * ceil(3*sigma2) + 1``
                     to capture the outer Gaussian tail (default 7).

    Input:  ``(B, 1, H, W)`` grayscale
    Output: ``(B, 1, H, W)`` signed DoG response
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.5,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.kernel_size = kernel_size
        self.register_buffer(
            "weight",
            self._build_kernel(sigma1, sigma2, kernel_size),
        )

    @staticmethod
    def _build_kernel(sigma1: float, sigma2: float, k: int) -> torch.Tensor:
        g1 = _gaussian_kernel(k, sigma1)
        g2 = _gaussian_kernel(k, sigma2)
        return _to_conv_weight(g1 - g2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

    def extra_repr(self) -> str:
        return f"sigma1={self.sigma1}, sigma2={self.sigma2}, kernel_size={self.kernel_size}"
