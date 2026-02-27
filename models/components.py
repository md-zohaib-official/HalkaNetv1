"""
components.py
=============
Reusable building blocks for HalkaNet.

Naming conventions (updated from original):
  ScaledSigmoid                       → ScaledSigmoid          (no standard equivalent)
  AbsActivation                       → AbsActivation           (folded-ReLU / |x| activation)
  SEBlock                             → SEBlock                 (Squeeze-and-Excitation, SENet)
  BranchSE                            → BranchSE                (multi-branch SE wrapper)
  SkipConnectionConcatenation         → DenseGrowthLayer        (DenseNet dense-layer)
  ResidualSkipConnectionConcatenation → ResidualDenseGrowthLayer (RDB micro-step)
  DepthwiseConv                       → DepthwiseSeparableConv  (MobileNetV1 DSConv)
  FusionBlock                         → MBConvSEBlock           (MBConv + SE, EfficientNet-style)
  WideExpandBlock                     → MBConvProjection        (single-pass inverted bottleneck)
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round8(x: float) -> int:
    """Round *x* to the nearest multiple of 8, minimum 8."""
    return max(8, int(round(x / 8)) * 8)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class ScaledSigmoid(nn.Module):
    """Scaled sigmoid gate: ``scale * sigmoid(scale * x)``.

    Steepens the standard sigmoid curve, producing sharper gating with
    a controllable saturation plateau.  Not equivalent to hard-sigmoid.

    Args:
        scale: Slope / plateau multiplier (default 1.5).
    """

    __constants__ = ["scale"]

    def __init__(self, scale: float = 1.5) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sigmoid(self.scale * x)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class AbsActivation(nn.Module):
    """Element-wise absolute value activation: ``|x|``.

    Used after signed filter responses (e.g. Sobel, LoG) where only
    magnitude — not polarity — carries useful information for the branch.
    Equivalent to a *folded ReLU* that preserves negative activations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x.abs()


# ---------------------------------------------------------------------------
# Channel Attention
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block (SENet, Hu et al. 2018).

    Globally pools spatial information, learns per-channel recalibration
    weights via a two-layer FC bottleneck, and rescales the input feature map.

    Args:
        in_channels: Number of input (and output) channels.
        reduction:   Bottleneck reduction ratio (default 4).
        scale:       Optional fixed multiplier applied after sigmoid gate
                     (set to 1 to use a plain sigmoid, i.e. standard SE).
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        squeezed = max(1, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeezed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeezed, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self._scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # squeeze: (B, C, 1, 1) → (B, C)
        w = self.pool(x).view(b, c)
        # excitation: (B, C) → (B, C, 1, 1)
        w = self.fc(w).view(b, c, 1, 1)
        if self._scale != 1.0:
            w = w * self._scale
        return x * w


class BranchSE(nn.Module):
    """Independent SE attention applied to each branch tensor in a list.

    Each branch gets its own :class:`SEBlock` sized to its channel count,
    avoiding cross-branch information mixing before the fusion concat.

    Args:
        branch_channels: Channel count for each branch, in order.
        reduction:       SE reduction ratio (default 4).
        min_squeeze:     Floor for the squeezed dimension (default 4).
    """

    def __init__(
        self,
        *branch_channels: int,
        reduction: int = 4,
        min_squeeze: int = 4,
    ) -> None:
        super().__init__()
        self.se_blocks = nn.ModuleList(
            SEBlock(
                ch,
                reduction=max(1, min(reduction, ch // min_squeeze)),
            )
            for ch in branch_channels
        )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [se(x) for se, x in zip(self.se_blocks, xs)]


# ---------------------------------------------------------------------------
# Dense Growth Layers  (DenseNet-style skip connections)
# ---------------------------------------------------------------------------


class DenseGrowthLayer(nn.Module):
    """Single DenseNet growth step (Huang et al. 2017).

    Computes a new feature map from the input via a small convolution and
    concatenates it with the original input along the channel axis:

        out = BN+ReLU( cat([x,  conv(x)] ) )

    The number of new channels is ``round(in_channels * multiplier)``,
    playing the role of DenseNet's *growth rate k*.

    Args:
        in_channels:  Input channel count.
        multiplier:   Growth fraction; new channels = round(in * multiplier).
        kernel_size:  Spatial kernel for the growth convolution (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        multiplier: float = 1.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        assert multiplier > 0, "multiplier must be > 0"
        growth = int(round(in_channels * multiplier))
        self.in_channels = in_channels
        self.growth = growth
        self.out_channels = in_channels + growth

        self.conv = nn.Conv2d(
            in_channels,
            growth,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn_act(torch.cat([x, self.conv(x)], dim=1))

    def extra_repr(self) -> str:
        return f"in={self.in_channels}, growth={self.growth}, out={self.out_channels}"


class ResidualDenseGrowthLayer(nn.Module):
    """Residual Dense Growth Layer — RDB micro-step (Wang et al. ESRGAN 2018).

    Extends :class:`DenseGrowthLayer` by adding a residual projection of the
    input *into* the growth feature map before the concat:

        extra  = conv(x) + proj(x)        # residual pre-addition
        out    = BN+ReLU( cat([x, extra]) )

    When ``growth == in_channels`` the projection is a 1×1 identity-compatible
    conv; otherwise a 1×1 pointwise aligns the channel count.  This lets
    gradient flow through both the dense path and the shortcut simultaneously.

    Args:
        in_channels:  Input channel count.
        multiplier:   Growth fraction; growth channels = round(in * multiplier).
        kernel_size:  Spatial kernel for the growth convolution (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        multiplier: float = 1.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        assert multiplier > 0, "multiplier must be > 0"
        growth = int(round(in_channels * multiplier))
        self.in_channels = in_channels
        self.growth = growth
        self.out_channels = in_channels + growth

        self.conv = nn.Conv2d(
            in_channels,
            growth,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        # 1×1 projection aligns x → growth channels for the residual addition
        self.proj = (
            None
            if growth == in_channels
            else nn.Conv2d(in_channels, growth, kernel_size=1, bias=False)
        )
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = self.conv(x)
        extra = extra + (self.proj(x) if self.proj is not None else x)
        return self.bn_act(torch.cat([x, extra], dim=1))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, growth={self.growth}, "
            f"out={self.out_channels}, proj={self.proj is not None}"
        )


# ---------------------------------------------------------------------------
# Depthwise Separable Convolution  (MobileNetV1, Howard et al. 2017)
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution (DSConv) from MobileNetV1.

    Factorises a standard conv into:
      1. Depthwise  3×3 conv (``groups=in_channels``) + BN + Act
      2. Pointwise  1×1 conv + BN + Act

    Reduces computation by roughly ``1/out_ch + 1/k²`` vs a full conv while
    maintaining a comparable receptive field.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels (pointwise output).
        kernel_size:  DW kernel size (default 3).
        stride:       DW conv stride, controls spatial downsampling (default 1).
        activation:   Activation class applied after each BN (default ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            activation(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# MBConv-SE Block  (EfficientNet / MobileNetV3-style)
# ---------------------------------------------------------------------------


class MBConvSEBlock(nn.Module):
    """MBConv with Squeeze-and-Excitation (EfficientNet, Tan & Le 2019).

    Implements a Mobile Inverted Bottleneck Convolution fused with an SE
    channel-attention gate.  Two residual modes are selected automatically:

    * **Add** (ResNet-style): when ``out_channels == in_channels``.
      ``out = act(x + conv_path(x))``

    * **Concat** (DenseNet-style): when ``out_channels > in_channels``.
      ``out = act( cat([x, conv_path(x)]) )``
      where ``conv_path`` projects to the *extra* ``out - in`` channels only.

    Structure::

        x → [1×1 expand] → [3×3 DW] → [1×1 project] → SE → (+/cat) x → Act

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count; must be ≥ in_channels.
        se_reduction: SE bottleneck reduction ratio (default 4).
        activation:   Activation class (default SiLU, as in EfficientNet).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_reduction: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        assert out_channels >= in_channels, (
            f"MBConvSEBlock requires out_channels >= in_channels, "
            f"got {in_channels} → {out_channels}"
        )
        expand_ch = out_channels - in_channels
        self._use_concat = expand_ch > 0

        if not self._use_concat:
            # standard inverted bottleneck; projects back to in_channels
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True),
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
            )
        else:
            # concat path: project to the extra channels only
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True),
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True),
                nn.Conv2d(in_channels, expand_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(expand_ch),
            )

        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.act = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_concat:
            out = self.se(torch.cat([x, self.conv(x)], dim=1))
        else:
            out = self.se(x + self.conv(x))
        return self.act(out)


# ---------------------------------------------------------------------------
# MBConv Projection  (single-pass inverted bottleneck, no residual)
# ---------------------------------------------------------------------------


class MBConvProjection(nn.Module):
    """Single-pass inverted bottleneck without a residual shortcut.

    Used as HalkaNet's *wide tail* block: maps ``in_channels → out_channels``
    (where ``out_channels > in_channels``) in one MBConv pass.  Because channel
    dimensions change, a residual shortcut is intentionally omitted.

    The intermediate width ``mid_channels`` is computed by the caller as::

        mid_ch = _round8(2 * fusion_out - fusion_out / tail_depth)

    which widens the single depthwise pass to compensate for the merged depth,
    keeping the total parameter budget identical to a stacked MBConv chain.

    Structure::

        x → [1×1 expand → mid_ch] → [3×3 DW] → [1×1 project → out_ch] → BN+Act

    Args:
        in_channels:   Channels entering the block (= fusion output channels).
        mid_channels:  Intermediate expansion width.
        out_channels:  Final output channels (= classifier input).
        activation:    Activation class (default ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        assert out_channels >= in_channels, (
            f"MBConvProjection requires out_channels >= in_channels, "
            f"got {in_channels} → {out_channels}"
        )
        self.block = nn.Sequential(
            # expand
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            # depthwise
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            # project
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
