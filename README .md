# HalkaNet — Tri-Branch Lightweight CNN with Fixed-Filter Inductive Bias

<p align="center">
  <img src="assets/architecture.png" alt="HalkaNet Architecture" width="860"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/params-88K--400K-green" alt="Params"/>
  <img src="https://img.shields.io/badge/platform-MCU%20%7C%20Edge%20%7C%20CPU-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/status-research--paper-yellow" alt="Status"/>
</p>

> **"Halka"** (हल्का / ہلکا) means *lightweight* in Hindi/Urdu — reflecting the core design goal of this architecture.

---

## Overview

**HalkaNet** is a lightweight convolutional neural network designed for accurate image classification on resource-constrained devices — microcontrollers (MCUs), edge SoCs, and embedded CPUs — **without using any pretrained weights**.

The core idea: instead of making a CNN learn edge detectors, blob detectors, and texture descriptors from scratch, HalkaNet **hard-wires these as fixed, non-trainable filter banks** and reserves all learnable parameters for high-level discrimination. This gives the model a strong *inductive bias* from epoch 0, enabling strong accuracy at sub-100K parameter budgets.

---

## Architecture

HalkaNet processes input through **three parallel branches** that are fused via per-branch SE attention and an inverted-bottleneck tail.

```
Input RGB (B × 3 × H × W)
    │
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
① RGB Branch (learned)               Gray Projection (fixed, ITU-R BT.601)
   1×1 Stem → DSConv(stride=2)           0.299·R + 0.587·G + 0.114·B
   → ResidualDenseGrowth                      │
   → MaxPool(2)                     [Optional AvgPool(2) for MCU targets]
   → H/4 × W/4                                │
                                     ┌─────────┴──────────┐
                                     ▼                     ▼
                             ② Coarse Branch        ③ Fine Branch
                              (fixed filters)         (fixed filters)
                               DoG  + optional          Sobel + optional
                               LoG / Gabor /            LoG / Gabor /
                               LBP  / Haar              LBP / Haar
                               1×1 Proj → DSConv        1×1 Proj → DSConv
                               → DenseGrowth            → DenseGrowth
                               → AvgPool(2)             → AvgPool(2)
                               → H/4 × W/4              → H/4 × W/4
    │                                │                        │
    └────────────────────────────────┴────────────────────────┘
                          BranchSE (independent SE per branch, r=3)
                                         │
                               Channel Concatenation
                          C_fused = C_rgb + C_coarse + C_fine
                                         │
                            MBConvSEBlock (Fusion, SE r=2)
                            + AvgPool(2)  →  H/8 × W/8
                                         │
                            MBConvProjection (Wide Tail)
                          1×1 Expand → DW 3×3 → 1×1 Project
                                         │
                       AdaptiveAvgPool(1) → Dropout → Linear
                                         │
                                  Logits (B × K)
```

### Core Building Blocks

| Component | Role | Inspiration |
|---|---|---|
| `DepthwiseSeparableConv` | DW 3×3 + PW 1×1, BN·ReLU ×2 | MobileNetV1 |
| `DenseGrowthLayer` | `out = BN·ReLU( cat[x, conv(x)] )` | DenseNet |
| `ResidualDenseGrowthLayer` | `extra = conv(x) + proj(x)` → dense concat | RDB / ESRGAN |
| `MBConvSEBlock` | MBConv + SE, concat-mode when out > in | EfficientNet |
| `MBConvProjection` | Wide single-pass inverted bottleneck, no residual | — |
| `BranchSE` | Independent SE per branch before fusion | SENet |

### Fixed Filter Banks

| Filter | Branch | Always Present | Description |
|---|---|---|---|
| `DoGFilter(σ=(0.3,1.5), k=5)` | Coarse | ✅ | Signed blob/band structure |
| `SobelGradMagnitude(L1)` | Fine | ✅ | Edge gradient magnitude |
| `LoGFilter(k=5)` | Coarse | Optional | Broad edge/blob detection |
| `LoGFilter(k=3)` | Fine | Optional | Fine edge detail |
| `GaborFilter(135°)` | Coarse | Optional | Diagonal texture |
| `GaborFilter(0°, k=3)` | Fine | Optional | Horizontal texture |
| `OrientedGradientFilter(H)` | Coarse | Optional | Horizontal gradient |
| `OrientedGradientFilter(D2)` | Fine | Optional | NW-SE diagonal gradient |
| `HaarWaveletFilter(HH)` | Coarse | Optional | Diagonal Haar subband |
| `HaarWaveletFilter(LH)` | Fine | Optional | Horizontal-edge Haar subband |

> Optional filters are activated via the `filters=[...]` argument. All optional filter outputs pass through `|·|` (abs) rectification before stacking.

---

## Installation

```bash
# Clone
git clone https://github.com/md-zohaib-official/HalkaNetv1.git
cd HalkaNetv1

# Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** Python ≥ 3.9, PyTorch ≥ 2.0, torchvision, torchinfo

---

## Quick Start

### Lightweight config — MCU / Edge target (~88K params)

```python
import torch
from halkanet import HalkaNet
from config import BranchType, FilterType

model = HalkaNet(
    num_classes=6,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.0},
    filter_channels=18,
    filters=[FilterType.LOG, FilterType.LBP, FilterType.WAVE],
    tail_depth=1,
)

x = torch.randn(1, 3, 96, 96)
logits = model(x)          # (1, 6)

from torchinfo import summary
summary(model, input_size=(1, 3, 96, 96))
# Total params: 88,354  |  MACs: 46.9 M  |  Size: 0.30 MB
```

### Balanced config — Desktop / Server target (~160K params)

```python
model = HalkaNet(
    num_classes=100,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.0},
    rgb_stem_channels=16,
    filter_channels=20,
    filters=[FilterType.LOG, FilterType.GABOR, FilterType.LBP, FilterType.WAVE],
    tail_depth=2,
)
# Suitable for CIFAR-100, STL-10, or similar multi-class benchmarks
```

---

## Benchmark Results

### MCU / Edge Deployment (Intel Image Classification, 96×96, 6 classes)
> Trained from scratch for 100 epochs. No pretrained weights.
> CPU Benchmark: Intel Core i5-10300H @ 2.50GHz, 4 physical / 8 logical cores, batch=1, n=500 runs.

| Model | Top-1 | Top-5 | Params | MACs (M) | Size (MB) | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|
| **HalkaNet** | **91.83%** | **99.73%** | **88,354** | **46.9** | **0.30** | **6.93 ± 0.61** | **144.2** |
| ShuffleNetV2-0.5× | 87.67% | 98.93% | 347,942 | 7.3 | 1.39 | 6.15 | 162.5 |
| MobileNetV2-0.35× | 87.83% | 99.60% | 403,814 | 10.7 | 1.62 | 6.50 | 153.9 |
| SqueezeNet-1.1 | 88.70% | 99.77% | 725,574 | 43.2 | 2.90 | 3.01 | 331.7 |
| MobileNetV3-Small | 88.67% | 99.33% | 1,524,006 | 11.1 | 6.10 | 4.92 | 203.3 |

> HalkaNet achieves **+3.13–4.16 pp higher Top-1** than all baselines while using **4×–17× fewer parameters**.

### Filter Inductive Bias — Accuracy Saturation Study
> Scaling params 4.4× (88K → 392K) yields only +0.10 pp accuracy gain — confirming accuracy is **filter-bound, not parameter-bound**.

| Config | Params | MACs (M) | Top-1 | Top-5 | Latency (ms) |
|---|---|---|---|---|---|
| HalkaNet (88K, 3 filter types) | 88,354 | 46.9 | 91.83% | 99.73% | 6.93 ± 0.61 |
| HalkaNet (392K, 4 filter types) | 392,473 | 98.6 | 91.93% | 99.87% | 8.18 ± 1.96 |

---

## Filter Design Philosophy

HalkaNet uses **heterogeneous filter grouping** — each branch is built from filters drawn from *orthogonal functional classes* rather than grouping similar filters together. This is a deliberate design decision:

- **Homogeneous filter groups** (e.g., multiple Gabor orientations in one branch) produce responses within the *same feature subspace* — high inter-filter correlation, low representational rank, diminishing returns.
- **Heterogeneous filter groups** (e.g., DoG + LBP + Haar together) produce *complementary, near-orthogonal projections* — maximising the rank of the joint response matrix and improving class-conditional separability.

In practice: DoG (blob structure) + Sobel (gradient magnitude) + Haar (subband texture) together encode *shape, boundary, and surface simultaneously* in a single forward pass with zero learned parameters. This pre-conditions the feature space before any learned layer sees it.

---

## Channel Width Equations

```
C_rgb_out    = ⌊ C_rgb × (1 + s_rgb) ⌋
C_coarse_out = ⌊ C_f   × (1 + s_f)   ⌋      (= C_fine_out)

C_fused      = C_rgb_out + C_coarse_out + C_fine_out
g            = max(8, round8( C_fused × r_grow ))
C_final      = C_fused + tail_depth × g
C_mid        = round8( 2 × C_fused_out − C_fused_out / tail_depth )
```

`s_rgb`, `s_f` = skip_expansion ratios &nbsp;|&nbsp; `r_grow` = tail_grow_ratio &nbsp;|&nbsp; `round8` = nearest multiple of 8

---

## Project Structure

```
HalkaNetv1/
├── halkanet/
│   ├── __init__.py
│   ├── architectures.py     # HalkaNet model definition
│   ├── components.py        # DSConv, DenseGrowth, SE, MBConv blocks
│   └── filters.py           # Fixed filter bank implementations
├── config.py                # Enums, default hyperparameters
├── train.py                 # Training script
├── evaluate.py              # Evaluation & benchmark utilities
├── assets/
│   └── architecture.png     # Architecture diagram
├── requirements.txt
└── README.md
```

---

## Citation

If you use HalkaNet in your research or build upon it, please cite:

```bibtex
@misc{zohaib2025halkanet,
  title  = {HalkaNet: A Tri-Branch Lightweight CNN with Fixed-Filter
            Inductive Bias for Accurate Edge-Device Image Classification},
  author = {Md Zohaib},
  year   = {2025},
  url    = {https://github.com/md-zohaib-official/HalkaNetv1}
}
```

---

## About

This project is the **M.Tech Final Semester Research Project** of **Md Zohaib**,
Department of Computer Science and Engineering,
**SRM Institute of Science and Technology (SRMIST)**,
Kattankulathur, Chennai — 603203, Tamil Nadu, India.

---

<p align="center">
  Made with ❤️ at <b>SRM Institute of Science and Technology</b>, Chennai<br/>
  <i>M.Tech Final Semester Project — 2025</i>
</p>
