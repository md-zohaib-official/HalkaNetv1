# HalkaNet — Tri-Branch Lightweight CNN with Fixed-Filter Inductive Bias

<p align="center">
  <img src="assets/architecture.png" alt="HalkaNet Architecture" width="860"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/params-45K--160K-green" alt="Params"/>
  <img src="https://img.shields.io/badge/platform-MCU%20%7C%20Edge%20%7C%20CPU-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/status-M.Tech%20Research-yellow" alt="Status"/>
</p>

> **"Halka"** (हल्का / ہلکا) means *lightweight* in Hindi/Urdu — the core design philosophy of this architecture.

---

## Overview

**HalkaNet** is a lightweight convolutional neural network built for accurate image classification on resource-constrained devices — microcontrollers (MCUs), edge SoCs, and embedded CPUs — **without any pretrained weights**.

The core idea: instead of making a CNN discover edge detectors, blob detectors, and texture descriptors from scratch, HalkaNet **hard-wires these as fixed, non-trainable filter banks** and reserves every learnable parameter for high-level discrimination. This *inductive bias* from epoch 0 enables surprisingly high accuracy at sub-100K — even sub-50K — parameter budgets.

---

## Architecture

HalkaNet processes input through **three parallel branches** fused via per-branch SE attention and a wide inverted-bottleneck tail.

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
| `DoGFilter(σ=(0.3, 1.5), k=5)` | Coarse | ✅ | Signed blob / band structure |
| `SobelGradMagnitude(L1)` | Fine | ✅ | Edge gradient magnitude |
| `LoGFilter(k=5)` | Coarse | Optional | Broad edge / blob detection |
| `LoGFilter(k=3)` | Fine | Optional | Fine edge detail |
| `GaborFilter(135°)` | Coarse | Optional | Diagonal texture |
| `GaborFilter(0°, k=3)` | Fine | Optional | Horizontal texture |
| `OrientedGradientFilter(H)` | Coarse | Optional | Horizontal gradient |
| `OrientedGradientFilter(D2)` | Fine | Optional | NW–SE diagonal gradient |
| `HaarWaveletFilter(HH)` | Coarse | Optional | Diagonal Haar subband |
| `HaarWaveletFilter(LH)` | Fine | Optional | Horizontal-edge Haar subband |

> Optional filters are enabled via `filters=[...]`. All optional outputs pass through `|·|` (abs) rectification before channel-stacking.

---

## Installation

```bash
git clone https://github.com/md-zohaib-official/HalkaNetv1.git
cd HalkaNetv1

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Core dependencies:** Python ≥ 3.9, PyTorch ≥ 2.0, torchvision, torchinfo

---

## Quick Start

### MCU / Edge target — ultra-lightweight (~45K–88K params)

```python
import torch
from halkanet import HalkaNet
from config import BranchType, FilterType

# ~88K params — Intel Image / 96×96
model = HalkaNet(
    num_classes=6,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.0},
    filter_channels=18,
    filters=[FilterType.LOG, FilterType.LBP, FilterType.WAVE],
    tail_depth=1,
)

x = torch.randn(1, 3, 96, 96)
logits = model(x)   # (1, 6)

from torchinfo import summary
summary(model, input_size=(1, 3, 96, 96))
# Total params: 88,354 | MACs: 46.9 M | Size: 0.30 MB
```

### Desktop / multi-class target (~160K params)

```python
# ~160K params — CIFAR-100 / STL-10
model = HalkaNet(
    num_classes=100,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.0},
    rgb_stem_channels=16,
    filter_channels=20,
    filters=[FilterType.LOG, FilterType.GABOR, FilterType.LBP, FilterType.WAVE],
    tail_depth=2,
)
```

---

## Benchmark Results

> All models trained **from scratch** (no pretrained weights) under identical conditions.
> CPU: **Intel Core i5-10300H @ 2.50 GHz** — 4 physical / 8 logical cores, batch=1, n=500 runs.

### MCU / Edge — Intel Image Classification (96×96, 6 classes, 100 epochs)

| Model | Top-1 | Top-5 | Params | MACs (M) | Size (MB) | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|
| **HalkaNet** | **91.83%** | **99.73%** | **88,354** | **46.9** | **0.30** | **6.93 ± 0.61** | **144.2** |
| ShuffleNetV2-0.5× | 87.67% | 98.93% | 347,942 | 7.3 | 1.39 | 6.15 | 162.5 |
| MobileNetV2-0.35× | 87.83% | 99.60% | 403,814 | 10.7 | 1.62 | 6.50 | 153.9 |
| SqueezeNet-1.1 | 88.70% | 99.77% | 725,574 | 43.2 | 2.90 | 3.01 | 331.7 |
| MobileNetV3-Small | 88.67% | 99.33% | 1,524,006 | 11.1 | 6.10 | 4.92 | 203.3 |

> HalkaNet achieves **+3.13–4.16 pp** higher Top-1 than all baselines using **4×–17× fewer parameters**.

### High-Throughput — STL-10 (96×96, 10 classes, 100 epochs)

| Model | Top-1 | Top-5 | Params | MACs (M) | Size (MB) | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|
| **HalkaNet** | **76.14%** | **96.20%** | **88,354** | **46.9** | **0.30** | **6.93 ± 0.61** | **144.2** |
| ShuffleNetV2-0.5× | 71.60% | 95.98% | 347,942 | 7.3 | 1.39 | 6.15 | 162.5 |
| MobileNetV2-0.35× | 72.52% | 95.38% | 403,814 | 10.7 | 1.62 | 6.50 | 153.9 |
| SqueezeNet-1.1 | 72.90% | 95.22% | 725,574 | 43.2 | 2.90 | 3.01 | 331.7 |
| MobileNetV3-Small | 74.18% | 95.98% | 1,524,006 | 11.1 | 6.10 | 4.92 | 203.3 |

### Extreme Lightweight — CIFAR-10 (32×32, 10 classes)

| Model | Top-1 | Params | Notes |
|---|---|---|---|
| **HalkaNet (< 50K)** | **~87%** | **< 50,000** | From scratch, no pretrained weights |

> Under 50K parameters with ~87% accuracy on CIFAR-10 — demonstrating HalkaNet's efficiency floor.

### MCU Hardware Deployment (Edge Impulse)

> Quantised and deployed via Edge Impulse on real embedded hardware.

| Device | Latency | RAM | Flash |
|---|---|---|---|
| Cortex-M4 @ 80 MHz | — | < 256 KB | < 1 MB |
| Cortex-M7 @ 216 MHz | — | < 256 KB | < 1 MB |
| ESP32 @ 240 MHz | — | < 320 KB | < 4 MB |

> Exact latency numbers and Edge Impulse deployment screenshots are available in the `assets/` folder.

---

## Filter Design Philosophy

HalkaNet uses **heterogeneous filter grouping** — each branch combines filters from orthogonal functional classes, not similar ones. This is intentional:

- **Homogeneous grouping** (e.g., multiple Gabor orientations together) produces responses in the *same feature subspace* — high inter-filter correlation, low representational rank, diminishing returns for classification.
- **Heterogeneous grouping** (e.g., DoG + LBP + Haar together) produces *complementary, near-orthogonal projections* — maximising the rank of the joint response matrix and improving class-conditional separability.

Empirically: mixing filter types across branches consistently yields **+1–2% accuracy** over same-type grouping under identical training budgets, confirming that filter diversity is the dominant accuracy driver — not parameter count.

---

## Channel Width Equations

```
C_rgb_out    = ⌊ C_rgb × (1 + s_rgb) ⌋
C_coarse_out = ⌊ C_f   × (1 + s_f)   ⌋     (= C_fine_out)

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
  Made at <b>SRM Institute of Science and Technology</b>, Chennai<br/>
  <i>M.Tech Final Semester Project — 2025</i>
</p>
