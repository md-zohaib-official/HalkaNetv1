"""
architectures_ablation_gray.py
==============================
HalkaNetGrayBranches — Ablation variant of HalkaNet where the Coarse and Fine
filter branches are replaced with plain grayscale branches (no hand-crafted
filters).  Three branches are preserved so the fusion topology, channel
arithmetic, BranchSE, and tail are structurally identical to full HalkaNet.

Purpose
-------
Isolates the contribution of the *filter banks specifically* (DoG, Sobel,
Gabor, LoG, OGF, Haar) from the contribution of simply having *more branches*
and *more input channels*.  Comparing this model against:

  * HalkaNet          → measures filter bank value
  * HalkaNetRGBOnly   → measures multi-branch topology value

Architecture overview
---------------------
Spatial flow (default 64×64 RGB input)::

    Input (B, 3, 64, 64)
        │
        ├─ [RGB branch]  (unchanged from HalkaNet)
        │     1×1 Conv+BN+ReLU
        │     DepthwiseSeparableConv(stride=2)   → 32×32
        │     ResidualDenseGrowthLayer
        │     MaxPool(2)                          → 16×16
        │
        ├─ Grayscale projection (fixed ITU-R BT.601, non-trainable)
        │     [optional AvgPool(2) if downsample_gray_branches=True]
        │         │
        │         ├─ [Gray-A branch]
        │         │     1×1 Conv+BN+ReLU          (1 → filt_ch)
        │         │     DepthwiseSeparableConv(stride=2)   → 32×32
        │         │     DenseGrowthLayer
        │         │     AvgPool(2)                          → 16×16
        │         │
        │         └─ [Gray-B branch]
        │               1×1 Conv+BN+ReLU          (1 → filt_ch)
        │               DepthwiseSeparableConv(stride=2)   → 32×32
        │               DenseGrowthLayer
        │               AvgPool(2)                          → 16×16
        │
        └─ BranchSE  (independent SE on each branch)
              Concat([rgb, gray_a, gray_b], dim=1)
              MBConvSEBlock + AvgPool(2)           → 8×8
              MBConvProjection (wide tail)
              AdaptiveAvgPool(1)
              Dropout → Linear → logits

Differences from full HalkaNet
-------------------------------
- coarse_filter_modules / fine_filter_modules removed entirely
- coarse_filter_bn / fine_filter_bn removed
- coarse_proj / fine_proj now take 1-channel grayscale directly (no filter stack)
- Everything else — channel widths, SE, fusion, tail, head — is verbatim.

Differences from HalkaNetRGBOnly
---------------------------------
- Three branches preserved (RGB + Gray-A + Gray-B) instead of one.
- fused_ch = rgb_out_ch + gray_a_out_ch + gray_b_out_ch  (matches HalkaNet shape).
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
from config import (
    BranchType,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_GROUP_CHANNEL,
    DEFAULT_SKIP_EXPANSION,
    INITIAL_CHANNEL,
    RGB_CHANNEL,
)


def _round8(x: float) -> int:
    """Round *x* to the nearest multiple of 8, minimum 8."""
    return max(8, int(round(x / 8)) * 8)


class HalkaNetGrayBranches(nn.Module):
    """HalkaNet ablation — three branches, filter banks replaced with plain grayscale.

    The Coarse and Fine branches no longer apply any hand-crafted filter
    (DoG / Sobel / Gabor / LoG / OGF / Haar).  Instead, both branches receive
    the raw grayscale map directly and learn purely from data.  The branch
    topology, channel widths, SE attention, fusion block, wide tail, and
    classifier are all structurally identical to full HalkaNet.

    Args:
        num_classes:
            Number of output classes.
        img_channels:
            Input image channels (default 3 for RGB).
        rgb_stem_channels:
            Base channel width for the RGB branch stem.
        filter_channels:
            Base channel width for each grayscale branch  (mirrors HalkaNet's
            filter_channels so channel counts stay comparable).
        skip_expansion:
            Per-branch growth multiplier for dense skip layers.
            Dict keys: BranchType.RGB, BranchType.FILTER.
        dropout_rate:
            Dropout probability before the final linear classifier.
        downsample_gray_branches:
            If True and input spatial size > 32, AvgPool the grayscale map
            before both gray branches (mirrors HalkaNet downsample_filter_groups).
        tail_depth:
            Controls target output width of the tail block.
        tail_grow_ratio:
            Per-step channel increment as a fraction of fused_ch (0 < x < 1).

    Introspection attributes::

        model.fused_ch       — channels entering MBConvSEBlock
        model.grow_ch        — per-step channel delta
        model.fusion_out_ch  — channels leaving MBConvSEBlock
        model.wide_mid_ch    — MBConvProjection intermediate width
        model.final_ch       — tail output / FC input channels
        model.tail_schedule  — [fusion_out_ch, final_ch]
    """

    def __init__(
        self,
        num_classes: int,
        img_channels: int = RGB_CHANNEL,
        rgb_stem_channels: int = INITIAL_CHANNEL,
        filter_channels: int = DEFAULT_GROUP_CHANNEL,
        skip_expansion: dict[BranchType, float] | None = None,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        downsample_gray_branches: bool = False,
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

        self.downsample_gray_branches = downsample_gray_branches
        self.tail_depth = tail_depth
        self.tail_grow_ratio = tail_grow_ratio

        # --- Channel widths (identical to HalkaNet) -----------------------
        rgb_ch = rgb_stem_channels
        filt_ch = filter_channels
        self.rgb_out_ch = math.floor(rgb_ch * (1 + rgb_skip))
        self.gray_a_out_ch = math.floor(filt_ch * (1 + filt_skip))
        self.gray_b_out_ch = math.floor(filt_ch * (1 + filt_skip))

        # --- Grayscale projection (non-trainable, ITU-R BT.601) ----------
        # Shared by both gray branches — same as HalkaNet.
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

        # --- RGB branch (verbatim from HalkaNet) --------------------------
        self.rgb_stage = nn.Sequential(
            nn.Conv2d(img_channels, rgb_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(rgb_ch, rgb_ch, kernel_size=3, stride=2),
            ResidualDenseGrowthLayer(rgb_ch, multiplier=rgb_skip),
            nn.MaxPool2d(2),
        )

        # --- Gray-A branch  (replaces Coarse — no filter bank) -----------
        # Input is raw 1-channel grayscale instead of a stacked filter map.
        # The 1×1 proj now reads from 1 channel (not n_coarse).
        self.gray_a_proj = nn.Sequential(
            nn.Conv2d(1, filt_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(filt_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(filt_ch, filt_ch, kernel_size=3, stride=2),
        )
        self.gray_a_skip = DenseGrowthLayer(filt_ch, multiplier=filt_skip)
        self.gray_a_pool = nn.AvgPool2d(2)

        # --- Gray-B branch  (replaces Fine — no filter bank) -------------
        self.gray_b_proj = nn.Sequential(
            nn.Conv2d(1, filt_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(filt_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(filt_ch, filt_ch, kernel_size=3, stride=2),
        )
        self.gray_b_skip = DenseGrowthLayer(filt_ch, multiplier=filt_skip)
        self.gray_b_pool = nn.AvgPool2d(2)

        # --- BranchSE (three independent SE blocks) ----------------------
        self.branch_se = BranchSE(
            self.rgb_out_ch,
            self.gray_a_out_ch,
            self.gray_b_out_ch,
            reduction=3,
            min_squeeze=3,
        )

        # --- Channel budget (identical formulae to HalkaNet) -------------
        fused_ch = self.rgb_out_ch + self.gray_a_out_ch + self.gray_b_out_ch
        self.fused_ch = fused_ch

        grow_ch = max(8, _round8(fused_ch * tail_grow_ratio))
        self.grow_ch = int(grow_ch)

        fusion_expand = 0 if tail_depth == 1 else max(0, _round8(grow_ch * tail_depth))
        fusion_out_ch = fused_ch + fusion_expand
        self.fusion_out_ch = int(fusion_out_ch)

        # --- MBConvSEBlock fusion + spatial downsampling -----------------
        self.fusion = nn.Sequential(
            MBConvSEBlock(fused_ch, fusion_out_ch, se_reduction=2, activation=nn.ReLU),
            nn.AvgPool2d(2),
        )

        # --- Wide tail (MBConvProjection) --------------------------------
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

        # --- Classifier head (verbatim from HalkaNet) --------------------
        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_drop = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(final_ch, num_classes)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _gray_a_branch(self, x_gray: torch.Tensor, do_pool: bool) -> torch.Tensor:
        """Gray-A branch: plain grayscale → proj → DenseGrowthLayer → pool."""
        x = self.gray_a_skip(self.gray_a_proj(x_gray))
        if do_pool:
            x = self.gray_a_pool(x)
        return x

    def _gray_b_branch(self, x_gray: torch.Tensor, do_pool: bool) -> torch.Tensor:
        """Gray-B branch: plain grayscale → proj → DenseGrowthLayer → pool."""
        x = self.gray_b_skip(self.gray_b_proj(x_gray))
        if do_pool:
            x = self.gray_b_pool(x)
        return x

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Grayscale projection -----------------------------------------
        x_gray = self.gray_conv(x) if self.gray_conv is not None else x

        # Optional early spatial downsampling (mirrors HalkaNet behaviour)
        do_gray_ds = self.downsample_gray_branches and x_gray.shape[-1] > 32
        x_grayf = self.gray_downsample(x_gray) if do_gray_ds else x_gray
        do_pool = not do_gray_ds

        # --- Three branches -----------------------------------------------
        x_rgb = self.rgb_stage(x)
        x_gray_a = self._gray_a_branch(x_grayf, do_pool)
        x_gray_b = self._gray_b_branch(x_grayf, do_pool)

        # --- Per-branch SE attention --------------------------------------
        x_rgb, x_gray_a, x_gray_b = self.branch_se([x_rgb, x_gray_a, x_gray_b])

        # --- Fuse ---------------------------------------------------------
        x = torch.cat([x_rgb, x_gray_a, x_gray_b], dim=1)
        del x_rgb, x_gray_a, x_gray_b

        # --- Tail + classify ----------------------------------------------
        x = self.tail(self.fusion(x))
        x = self.classifier_drop(self.classifier_pool(x).flatten(1))
        return self.classifier(x)
