"""
architectures.py
================
HalkaNet model definition — lightweight CNN for MCU / edge-device inference.

Architecture overview
---------------------
HalkaNet processes an input image through three parallel branches, merges
them via per-branch SE attention and an MBConv fusion block, then classifies
through a single wide inverted-bottleneck tail.

Spatial flow (default 64×64 RGB input)::

    Input (B, 3, 64, 64)
        │
        ├─ [RGB branch]
        │     1×1 Conv+BN+ReLU
        │     DepthwiseSeparableConv(stride=2)   → 32×32
        │     ResidualDenseGrowthLayer
        │     MaxPool(2)                          → 16×16
        │
        ├─ Grayscale projection (fixed ITU-R BT.601 weights, non-trainable)
        │     [optional AvgPool(2) if downsample_filter_groups=True]
        │         │
        │         ├─ [Coarse filter bank]  DoG + LoG(5) + Gabor(135°) [+ LBP/Haar]
        │         │     1×1 Proj+BN+ReLU
        │         │     DepthwiseSeparableConv(stride=2)   → 32×32
        │         │     DenseGrowthLayer
        │         │     AvgPool(2)                          → 16×16
        │         │
        │         └─ [Fine filter bank]   Sobel + LoG(3) + Gabor(0°) [+ LBP/Haar]
        │               1×1 Proj+BN+ReLU
        │               DepthwiseSeparableConv(stride=2)   → 32×32
        │               DenseGrowthLayer
        │               AvgPool(2)                          → 16×16
        │
        └─ BranchSE  (independent SE on each branch)
              Concat([rgb, coarse, fine], dim=1)
              MBConvSEBlock + AvgPool(2)           → 8×8
              MBConvProjection (wide tail)
              AdaptiveAvgPool(1)
              Dropout → Linear → logits

Filter naming updates (original → current)
-------------------------------------------
GaborFilterSingle      → GaborFilter
LoGFilter5x5 / 3x3    → LoGFilter(kernel_size=5|3)
SobelMagnitudeFilterMCU→ SobelGradMagnitude
LBPSingleChannel       → OrientedGradientFilter

Component naming updates
-------------------------
SkipConnectionConcatenation         → DenseGrowthLayer
ResidualSkipConnectionConcatenation → ResidualDenseGrowthLayer
DepthwiseConv                       → DepthwiseSeparableConv
FusionBlock                         → MBConvSEBlock
WideExpandBlock                     → MBConvProjection
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .components import (
    BranchSE,
    DenseGrowthLayer,
    DepthwiseSeparableConv,
    MBConvProjection,
    MBConvSEBlock,
    ResidualDenseGrowthLayer,
)
from .filters import (
    DoGFilter,
    GaborFilter,
    HaarWaveletFilter,
    LoGFilter,
    OrientedGradientFilter,
    SobelGradMagnitude,
)
from config import (
    BranchType,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_GROUP_CHANNEL,
    DEFAULT_SKIP_EXPANSION,
    FilterType,
    INITIAL_CHANNEL,
    RGB_CHANNEL,
)

# Filters that are optional (added on top of the always-present base filters)
_OPTIONAL_FILTERS: frozenset[FilterType] = frozenset(
    {
        FilterType.LOG,
        FilterType.GABOR,
        FilterType.LBP,
        FilterType.WAVE,
    }
)

_VALID_FILTER_NAMES: frozenset[str] = frozenset(FilterType)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round8(x: float) -> int:
    """Round *x* to the nearest multiple of 8, minimum 8."""
    return max(8, int(round(x / 8)) * 8)


def _coarse_filter_specs(
    active: frozenset[str],
) -> list[tuple[nn.Module, bool]]:
    """Build the coarse-branch filter list.

    Returns a list of ``(filter_module, use_abs)`` pairs.  ``use_abs=True``
    means the filter output should be passed through
    :class:`~components.AbsActivation` before stacking.

    Always-present:
        - DoG(σ₁=0.3, σ₂=1.5, k=5) — bandpass blob structure (signed, no abs)

    Optional (controlled by *active*):
        - LoG(k=5)           — broad edge / blob (abs after)
        - GaborFilter(135°)  — diagonal texture (abs after)
        - OrientedGradientFilter("h") — horizontal gradient (abs after)
        - HaarWaveletFilter("HH")    — diagonal detail texture (abs after)
    """
    specs: list[tuple[nn.Module, bool]] = [
        (DoGFilter(sigma1=0.3, sigma2=1.5, kernel_size=5), False),
    ]
    if FilterType.LOG in active:
        specs.append((LoGFilter(kernel_size=5), True))
    if FilterType.GABOR in active:
        specs.append((GaborFilter(orientation=135.0), True))
    if FilterType.LBP in active:
        specs.append((OrientedGradientFilter(axis="h"), True))
    if FilterType.WAVE in active:
        specs.append((HaarWaveletFilter("HH"), True))
    return specs


def _fine_filter_specs(
    active: frozenset[str],
) -> list[tuple[nn.Module, bool]]:
    """Build the fine-branch filter list.

    Always-present:
        - SobelGradMagnitude("l1") — overall edge strength (non-negative, no abs)

    Optional (controlled by *active*):
        - LoGFilter(k=3)            — fine edge detail (abs after)
        - GaborFilter(0°, k=3)      — horizontal texture (abs after)
        - OrientedGradientFilter("d2") — NW-SE diagonal gradient (abs after)
        - HaarWaveletFilter("LH")      — horizontal-edge Haar subband (abs after)
    """
    specs: list[tuple[nn.Module, bool]] = [
        (SobelGradMagnitude(mode="l1"), False),
    ]
    if FilterType.LOG in active:
        specs.append((LoGFilter(kernel_size=3), True))
    if FilterType.GABOR in active:
        specs.append(
            (
                GaborFilter(
                    orientation=0.0,
                    kernel_size=3,
                    sigma=1.2,
                    lambd=2.5,
                    gamma=0.5,
                    psi=0.0,
                ),
                True,
            )
        )
    if FilterType.LBP in active:
        specs.append((OrientedGradientFilter(axis="d2"), True))
    if FilterType.WAVE in active:
        specs.append((HaarWaveletFilter("LH"), True))
    return specs


# ---------------------------------------------------------------------------
# HalkaNet
# ---------------------------------------------------------------------------


class HalkaNet(nn.Module):
    """HalkaNet v2 — Wide-tail lightweight CNN for edge / MCU deployment.

    Three-branch architecture that fuses learned RGB features with
    fixed hand-crafted filter responses (Coarse + Fine branches) before
    classifying via a compact inverted-bottleneck tail.

    The sequential MBConv depth-stack of HalkaNet v1 is replaced by a single
    :class:`~components.MBConvProjection` (WideExpandBlock) whose intermediate
    width is computed to match the parameter budget of the stacked version::

        mid_ch  = _round8(2 * fusion_out - fusion_out / tail_depth)

    Args:
        num_classes:
            Number of output classes.
        img_channels:
            Input image channels. Use 3 for RGB, 1 for grayscale.
        rgb_stem_channels:
            Base channel width for the RGB branch stem.
        filter_channels:
            Base channel width for each filter branch.
        skip_expansion:
            Per-branch growth multiplier for the dense skip layer.
            Dict keys: ``BranchType.RGB``, ``BranchType.FILTER``.
        dropout_rate:
            Dropout probability before the final linear classifier.
        downsample_filter_groups:
            If ``True`` and input spatial size > 32, the grayscale map is
            downsampled (AvgPool×2) before entering the filter banks,
            saving FLOPs on MCU targets.
        filters:
            List of optional :class:`FilterType` values to activate.
            ``None`` uses the defaults defined in ``config.py``.
        tail_depth:
            Controls the target output width of the tail block.  The actual
            number of tail passes is always 1 (wide single-pass MBConv).
        tail_grow_ratio:
            Per-step channel increment as a fraction of ``fused_ch``
            (must be in ``(0, 1)``).

    Introspection attributes (set after ``__init__``)::

        model.fused_ch       — channels entering MBConvSEBlock
        model.grow_ch        — per-step channel delta (same formula as v1)
        model.fusion_out_ch  — channels leaving MBConvSEBlock
        model.wide_mid_ch    — MBConvProjection intermediate width
        model.final_ch       — tail output / FC input channels
        model.tail_schedule  — [fusion_out_ch, final_ch]
    """

    _VALID_FILTERS: frozenset[str] = _VALID_FILTER_NAMES

    def __init__(
        self,
        num_classes: int,
        img_channels: int = RGB_CHANNEL,
        rgb_stem_channels: int = INITIAL_CHANNEL,
        filter_channels: int = DEFAULT_GROUP_CHANNEL,
        skip_expansion: dict[BranchType, float] | None = None,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        downsample_filter_groups: bool = False,
        filters: list[FilterType] | None = None,
        tail_depth: int = 1,
        tail_grow_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        # --- Validate arguments -------------------------------------------
        skip: dict[BranchType, float] = {
            **DEFAULT_SKIP_EXPANSION,
            **(skip_expansion or {}),
        }
        rgb_skip = skip[BranchType.RGB]
        filt_skip = skip[BranchType.FILTER]
        if rgb_skip <= 0 or filt_skip <= 0:
            raise ValueError(f"skip_expansion values must be > 0, got {skip}")
        if tail_depth < 1:
            raise ValueError(f"tail_depth must be >= 1, got {tail_depth}")
        if not (0 < tail_grow_ratio < 1.0):
            raise ValueError(
                f"tail_grow_ratio must be in (0, 1), got {tail_grow_ratio}"
            )

        # --- Filter resolution --------------------------------------------
        if filters is None:
            filters = []
        active = frozenset(filters)
        unknown = active - self._VALID_FILTERS
        if unknown:
            raise ValueError(
                f"Unknown filter names: {sorted(unknown)}. "
                f"Valid: {sorted(self._VALID_FILTERS)}"
            )
        self.active_filters: frozenset[str] = frozenset(str(f) for f in active)
        self.downsample_filter_groups = downsample_filter_groups
        self.tail_depth = tail_depth
        self.tail_grow_ratio = tail_grow_ratio

        # --- Channel widths -----------------------------------------------
        rgb_ch = rgb_stem_channels
        filt_ch = filter_channels
        self.rgb_out_ch = math.floor(rgb_ch * (1 + rgb_skip))
        self.coarse_out_ch = math.floor(filt_ch * (1 + filt_skip))
        self.fine_out_ch = math.floor(filt_ch * (1 + filt_skip))

        # --- Grayscale projection (non-trainable, ITU-R BT.601) ----------
        if img_channels == 3:
            self.gray_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.gray_conv.weight.copy_(
                    torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                )
            self.gray_conv.weight.requires_grad_(False)
        else:
            self.gray_conv = None
        self.gray_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- Filter banks -------------------------------------------------
        coarse_specs = _coarse_filter_specs(active)
        fine_specs = _fine_filter_specs(active)
        n_coarse = len(coarse_specs)
        n_fine = len(fine_specs)

        self.coarse_filter_modules = nn.ModuleList(m for m, _ in coarse_specs)
        self.coarse_use_abs = [use_abs for _, use_abs in coarse_specs]
        self.coarse_filter_bn = nn.BatchNorm2d(n_coarse)

        self.fine_filter_modules = nn.ModuleList(m for m, _ in fine_specs)
        self.fine_use_abs = [use_abs for _, use_abs in fine_specs]
        self.fine_filter_bn = nn.BatchNorm2d(n_fine)

        # --- RGB branch ---------------------------------------------------
        self.rgb_stage = nn.Sequential(
            nn.Conv2d(img_channels, rgb_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(rgb_ch, rgb_ch, kernel_size=3, stride=2),
            ResidualDenseGrowthLayer(rgb_ch, multiplier=rgb_skip),
            nn.MaxPool2d(2),
        )

        # --- Coarse branch ------------------------------------------------
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(n_coarse, filt_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(filt_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(filt_ch, filt_ch, kernel_size=3, stride=2),
        )
        self.coarse_skip = DenseGrowthLayer(filt_ch, multiplier=filt_skip)
        self.coarse_pool = nn.AvgPool2d(2)

        # --- Fine branch --------------------------------------------------
        self.fine_proj = nn.Sequential(
            nn.Conv2d(n_fine, filt_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(filt_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(filt_ch, filt_ch, kernel_size=3, stride=2),
        )
        self.fine_skip = DenseGrowthLayer(filt_ch, multiplier=filt_skip)
        self.fine_pool = nn.AvgPool2d(2)

        # --- Branch SE + Fusion -------------------------------------------
        self.branch_se = BranchSE(
            self.rgb_out_ch,
            self.coarse_out_ch,
            self.fine_out_ch,
            reduction=3,
            min_squeeze=3,
        )

        fused_ch = self.rgb_out_ch + self.coarse_out_ch + self.fine_out_ch
        self.fused_ch = fused_ch

        grow_ch = max(8, _round8(fused_ch * tail_grow_ratio))
        self.grow_ch = int(grow_ch)

        fusion_expand = 0 if tail_depth == 1 else max(0, _round8(grow_ch * tail_depth))
        fusion_out_ch = fused_ch + fusion_expand
        self.fusion_out_ch = int(fusion_out_ch)

        self.fusion = nn.Sequential(
            MBConvSEBlock(fused_ch, fusion_out_ch, se_reduction=2, activation=nn.ReLU),
            nn.AvgPool2d(2),
        )

        # --- Wide tail (MBConvProjection) ---------------------------------
        final_ch = fused_ch + tail_depth * grow_ch
        wide_mid_ch = _round8(2 * fusion_out_ch - fusion_out_ch / tail_depth)
        self.final_ch = int(final_ch)
        self.wide_mid_ch = int(wide_mid_ch)
        self.tail_schedule: list[int] = [self.fusion_out_ch, self.final_ch]

        self.tail = MBConvProjection(
            in_channels=fusion_out_ch,
            mid_channels=wide_mid_ch,
            out_channels=final_ch,
            activation=nn.ReLU,
        )

        # --- Classifier head ----------------------------------------------
        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_drop = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(final_ch, num_classes)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _apply_filter_bank(
        self,
        x_gray: torch.Tensor,
        modules: nn.ModuleList,
        bn: nn.BatchNorm2d,
        use_abs_flags: list[bool],
    ) -> torch.Tensor:
        """Run each filter, optionally abs, then stack and BN."""
        outputs = [
            filt(x_gray).abs() if use_abs else filt(x_gray)
            for filt, use_abs in zip(modules, use_abs_flags)
        ]
        return bn(torch.cat(outputs, dim=1))

    def _coarse_branch(self, x_grayf: torch.Tensor, do_pool: bool) -> torch.Tensor:
        x = self._apply_filter_bank(
            x_grayf,
            self.coarse_filter_modules,
            self.coarse_filter_bn,
            self.coarse_use_abs,
        )
        x = self.coarse_skip(self.coarse_proj(x))
        if do_pool:
            x = self.coarse_pool(x)
        return x

    def _fine_branch(self, x_grayf: torch.Tensor, do_pool: bool) -> torch.Tensor:
        x = self._apply_filter_bank(
            x_grayf,
            self.fine_filter_modules,
            self.fine_filter_bn,
            self.fine_use_abs,
        )
        x = self.fine_skip(self.fine_proj(x))
        if do_pool:
            x = self.fine_pool(x)
        return x

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Grayscale projection -----------------------------------------
        x_gray = self.gray_conv(x) if self.gray_conv is not None else x

        # Optional early spatial downsampling for MCU targets
        do_gray_ds = self.downsample_filter_groups and x_gray.shape[-1] > 32
        x_grayf = self.gray_downsample(x_gray) if do_gray_ds else x_gray
        do_pool = not do_gray_ds  # pool inside branch if not already done

        # --- Three branches -----------------------------------------------
        x_rgb = self.rgb_stage(x)
        x_coarse = self._coarse_branch(x_grayf, do_pool)
        x_fine = self._fine_branch(x_grayf, do_pool)

        # --- Per-branch SE attention --------------------------------------
        x_rgb, x_coarse, x_fine = self.branch_se([x_rgb, x_coarse, x_fine])

        # --- Fuse ---------------------------------------------------------
        x = torch.cat([x_rgb, x_coarse, x_fine], dim=1)
        del x_rgb, x_coarse, x_fine  # free intermediate tensors

        # --- Tail + classify ----------------------------------------------
        x = self.tail(self.fusion(x))
        x = self.classifier_drop(self.classifier_pool(x).flatten(1))
        return self.classifier(x)
