"""
architectures_ablation.py
=========================
HalkaNetRGBOnly — Ablation variant of HalkaNet with filter branches removed.

Used to isolate the contribution of the hand-crafted filter banks (Coarse +
Fine branches) in the full HalkaNet model.  Everything else — the RGB stem,
BranchSE, MBConvSEBlock fusion, MBConvProjection wide tail, and classifier
head — is kept structurally identical.

Architecture overview
---------------------
Spatial flow (default 64×64 RGB input)::

    Input (B, 3, 64, 64)
        │
        └─ [RGB branch]
              1×1 Conv+BN+ReLU
              DepthwiseSeparableConv(stride=2)   → 32×32
              ResidualDenseGrowthLayer
              MaxPool(2)                          → 16×16
              │
              BranchSE  (single branch)
              MBConvSEBlock + AvgPool(2)          → 8×8
              MBConvProjection (wide tail)
              AdaptiveAvgPool(1)
              Dropout → Linear → logits

Differences from full HalkaNet
-------------------------------
- No gray_conv, gray_downsample, downsample_filter_groups logic
- No coarse_filter_modules / fine_filter_modules
- No coarse_proj / fine_proj / coarse_skip / fine_skip / coarse_pool / fine_pool
- BranchSE receives only [x_rgb]  (single-element list)
- fused_ch = rgb_out_ch  (no coarse/fine contribution)
- All channel-width formulae for grow_ch / fusion_out_ch / wide_mid_ch /
  final_ch are preserved verbatim so the tail budget scales the same way
  relative to fused_ch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .components import (
    BranchSE,
    DepthwiseSeparableConv,
    MBConvProjection,
    MBConvSEBlock,
    ResidualDenseGrowthLayer,
)
from config import (
    BranchType,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_SKIP_EXPANSION,
    INITIAL_CHANNEL,
    RGB_CHANNEL,
)


def _round8(x: float) -> int:
    """Round *x* to the nearest multiple of 8, minimum 8."""
    return max(8, int(round(x / 8)) * 8)


class HalkaNetRGBOnly(nn.Module):
    """HalkaNet ablation — RGB branch only, no hand-crafted filter banks.

    Structurally identical to HalkaNet except the Coarse and Fine filter
    branches are entirely removed.  The RGB stem, SE attention, MBConvSEBlock
    fusion, MBConvProjection wide tail, and classifier head are unchanged,
    allowing a clean ablation of the filter branch contribution.

    Args:
        num_classes:
            Number of output classes.
        img_channels:
            Input image channels (default 3 for RGB).
        rgb_stem_channels:
            Base channel width for the RGB branch stem.
        skip_expansion:
            Per-branch growth multiplier for the dense skip layer.
            Only BranchType.RGB is used here.
        dropout_rate:
            Dropout probability before the final linear classifier.
        tail_depth:
            Controls target output width of the tail block.
        tail_grow_ratio:
            Per-step channel increment as a fraction of fused_ch (0 < x < 1).

    Introspection attributes::

        model.fused_ch       — channels entering MBConvSEBlock (= rgb_out_ch)
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
        skip_expansion: dict[BranchType, float] | None = None,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
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
        if rgb_skip <= 0:
            raise ValueError(f"skip_expansion[RGB] must be > 0, got {rgb_skip}")
        if tail_depth < 1:
            raise ValueError(f"tail_depth must be >= 1, got {tail_depth}")
        if not (0 < tail_grow_ratio < 1.0):
            raise ValueError(
                f"tail_grow_ratio must be in (0, 1), got {tail_grow_ratio}"
            )

        self.tail_depth = tail_depth
        self.tail_grow_ratio = tail_grow_ratio

        # --- Channel widths -----------------------------------------------
        rgb_ch = rgb_stem_channels
        self.rgb_out_ch = math.floor(rgb_ch * (1 + rgb_skip))

        # --- RGB branch ---------------------------------------------------
        self.rgb_stage = nn.Sequential(
            nn.Conv2d(img_channels, rgb_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(rgb_ch, rgb_ch, kernel_size=3, stride=2),
            ResidualDenseGrowthLayer(rgb_ch, multiplier=rgb_skip),
            nn.MaxPool2d(2),
        )

        # --- Single-branch SE attention -----------------------------------
        # BranchSE is kept identical — it now wraps only the RGB branch.
        self.branch_se = BranchSE(
            self.rgb_out_ch,
            reduction=3,
            min_squeeze=3,
        )

        # --- Channel budget (identical formulae to HalkaNet) --------------
        fused_ch = self.rgb_out_ch  # only one branch now
        self.fused_ch = fused_ch

        grow_ch = max(8, _round8(fused_ch * tail_grow_ratio))
        self.grow_ch = int(grow_ch)

        fusion_expand = 0 if tail_depth == 1 else max(0, _round8(grow_ch * tail_depth))
        fusion_out_ch = fused_ch + fusion_expand
        self.fusion_out_ch = int(fusion_out_ch)

        # --- MBConvSEBlock fusion + spatial downsampling ------------------
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
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- RGB branch ---------------------------------------------------
        x_rgb = self.rgb_stage(x)

        # --- Single-branch SE attention -----------------------------------
        # BranchSE.forward expects a list; unpack the single-element result.
        (x_rgb,) = self.branch_se([x_rgb])

        # --- Fuse (single branch — no concat needed) ----------------------
        # x_rgb is already the full fused tensor.

        # --- Tail + classify ----------------------------------------------
        x = self.tail(self.fusion(x_rgb))
        x = self.classifier_drop(self.classifier_pool(x).flatten(1))
        return self.classifier(x)
