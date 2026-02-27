# HalkaNet — Tri-Branch Lightweight CNN with Fixed-Filter Inductive Bias

<p align="center">
  <img src="assets/architecture.png" alt="HalkaNet Architecture" width="860"/>
</p>

<p align="center">
  <a href="https://github.com/md-zohaib-official/HalkaNetv1/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/params-88K--400K-green" alt="Params"/>
  <img src="https://img.shields.io/badge/platform-MCU%20%7C%20Edge%20%7C%20CPU-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/status-research--paper-yellow" alt="Status"/>
</p>

> **"Halka"** (हल्का / ہلکا) means *lightweight* in Hindi/Urdu — reflecting the core design goal of this architecture.

---

## Overview

**HalkaNet** is a lightweight convolutional neural network designed for accurate image classification on resource-constrained devices — microcontrollers (MCUs), edge SoCs, and embedded CPUs — without using pretrained weights.

The key idea is simple: instead of making a CNN learn edge detectors, blob detectors, and texture descriptors from scratch (wasting capacity), HalkaNet **hard-wires these as fixed, non-trainable filter banks** and reserves all learnable parameters for high-level discrimination. This gives the model a strong *inductive bias* from epoch 0.

### Key Results

| Model | Top-1 (Intel) | Params | MACs | Size | Latency |
|---|---|---|---|---|---|
| **HalkaNet** | **91.83%** | **88,354** | **46.9 M** | **0.30 MB** | **6.93 ms** |
| ShuffleNetV2-0.5× | 87.67% | 347,942 | 7.3 M | 1.39 MB | 6.15 ms |
| MobileNetV2-0.35× | 87.83% | 403,814 | 10.7 M | 1.62 MB | 6.50 ms |
| SqueezeNet-1.1 | 88.70% | 725,574 | 43.2 M | 2.90 MB | 3.01 ms |
| MobileNetV3-Small | 88.67% | 1,524,006 | 11.1 M | 6.10 MB | 4.92 ms |

> HalkaNet achieves **+3.13–4.16 pp higher Top-1 accuracy** than all baselines while using **4–17× fewer parameters**. All models trained from scratch under identical conditions.

---

## Architecture

HalkaNet processes input through **three parallel branches** that are fused via SE attention and an inverted-bottleneck tail:

```
Input RGB (B × 3 × H × W)
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
① RGB Branch (learned)              Gray Projection (fixed, ITU-R BT.601)
   1×1 Stem → DSConv(s=2)               0.299R + 0.587G + 0.114B
   → ResidualDenseGrowth                     │
   → MaxPool(2)                    [Optional AvgPool(2) for MCU]
   → H/4 × W/4                              │
                                    ┌────────┴─────────┐
                                    ▼                   ▼
                            ② Coarse Branch      ③ Fine Branch
                             (fixed filters)      (fixed filters)
                              DoG + optional        Sobel + optional
                              LoG/Gabor/LBP/Haar    LoG/Gabor/LBP/Haar
                              1×1 Proj → DSConv     1×1 Proj → DSConv
                              → DenseGrowth          → DenseGrowth
                              → AvgPool(2)           → AvgPool(2)
                              → H/4 × W/4            → H/4 × W/4
    │                               │                       │
    └───────────────────────────────┴───────────────────────┘
                              BranchSE (independent SE per branch)
                                        │
                              Channel Concatenation
                              C_fused = C_rgb + C_coarse + C_fine
                                        │
                              MBConvSEBlock (Fusion)
                              + AvgPool(2) → H/8 × W/8
                                        │
                              MBConvProjection (Wide Tail)
                              1×1 Expand → DW 3×3 → 1×1 Project
                                        │
                              AdaptiveAvgPool(1) → Dropout → Linear
                                        │
                                   Logits (B × K)
```

### Core Components

| Component | Description | Reference |
|---|---|---|
| `DepthwiseSeparableConv` | DW 3×3 + PW 1×1, BN·ReLU ×2 | MobileNetV1 |
| `DenseGrowthLayer` | `out = BN·ReLU(cat[x, conv(x)])` | DenseNet |
| `ResidualDenseGrowthLayer` | `extra = conv(x) + proj(x)`, then dense concat | RDB (ESRGAN) |
| `MBConvSEBlock` | MBConv + SE, concat-mode when out > in | EfficientNet |
| `MBConvProjection` | Wide single-pass inverted bottleneck, no residual | Custom |
| `BranchSE` | Independent SE per branch before fusion | SENet |
| `DoGFilter` | Difference of Gaussians, signed (no abs) | Classic CV |
| `SobelGradMagnitude` | L1 gradient magnitude | Classic CV |
| `LoGFilter` | Laplacian of Gaussian, k=3 or k=5 | Classic CV |
| `GaborFilter` | Oriented sinusoidal wavelet | Gabor 1946 |
| `HaarWaveletFilter` | Orthogonal subband (LH, HH, etc.) | Haar 1909 |
| `OrientedGradientFilter` | Directional finite-difference gradient | Classic CV |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/md-zohaib-official/HalkaNetv1.git
cd HalkaNetv1

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, torchvision, torchinfo

---

## Quick Start

```python
import torch
from halkanet import HalkaNet
from config import BranchType, FilterType

# Lightweight config (~88K params, MCU-ready)
model = HalkaNet(
    num_classes=6,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.0},
    filter_channels=18,
    filters=[FilterType.LOG, FilterType.LBP, FilterType.WAVE],
    tail_depth=1,
)

# Inference
x = torch.randn(1, 3, 96, 96)
logits = model(x)          # (1, 6)
print(logits.shape)

# Check parameter count
from torchinfo import summary
summary(model, input_size=(1, 3, 96, 96))
```

### Larger config (~392K params, higher accuracy)

```python
model = HalkaNet(
    num_classes=6,
    skip_expansion={BranchType.RGB: 1.5, BranchType.FILTER: 1.5},
    rgb_stem_channels=16,
    filter_channels=20,
    filters=[FilterType.LOG, FilterType.LBP, FilterType.WAVE, FilterType.GABOR],
    tail_depth=2,
)
```

---

## Filter Bank Design Philosophy

HalkaNet uses **heterogeneous filter grouping** — each branch mixes filters from orthogonal functional classes rather than grouping similar filters together. This is deliberate:

- **Same-type filters** in one branch produce responses in the *same feature subspace* — their joint representation has low rank and is redundant.
- **Different-type filters** in one branch produce *complementary, near-orthogonal projections* of the image, maximising the rank of the joint response space and improving class-conditional separability.

Example: DoG (blob structure) + LBP (texture ordering) + Sobel (gradient magnitude) together encode **what an object looks like**, **how its surface feels**, and **where its boundaries are** — simultaneously, in a single forward pass with zero learned parameters.

### Filter Reference

| Filter | Type | Branch | Always Present | Purpose |
|---|---|---|---|---|
| `DoGFilter(σ=(0.3,1.5), k=5)` | Bandpass blob | Coarse | ✅ | Signed blob/band structure |
| `SobelGradMagnitude(L1)` | Gradient magnitude | Fine | ✅ | Edge strength |
| `LoGFilter(k=5)` | Laplacian of Gaussian | Coarse | Optional | Broad edges/blobs |
| `LoGFilter(k=3)` | Laplacian of Gaussian | Fine | Optional | Fine edge detail |
| `GaborFilter(135°)` | Oriented wavelet | Coarse | Optional | Diagonal texture |
| `GaborFilter(0°, k=3)` | Oriented wavelet | Fine | Optional | Horizontal texture |
| `OrientedGradientFilter(H)` | Directional gradient | Coarse | Optional | Horizontal gradient |
| `OrientedGradientFilter(D2)` | Directional gradient | Fine | Optional | NW-SE gradient |
| `HaarWaveletFilter(HH)` | Wavelet subband | Coarse | Optional | Diagonal detail |
| `HaarWaveletFilter(LH)` | Wavelet subband | Fine | Optional | Horizontal-edge subband |

---

## Benchmark Results

### Intel Image Classification (96×96, 6 classes, 100 epochs, from scratch)

| Model | Top-1 | Top-5 | Params | MACs (M) | Size (MB) | Lat (ms) | FPS |
|---|---|---|---|---|---|---|---|
| **HalkaNet (88K)** | **91.83%** | **99.73%** | **88,354** | **46.9** | **0.30** | **6.93** | **144.2** |
| ShuffleNetV2-0.5× | 87.67% | 98.93% | 347,942 | 7.3 | 1.39 | 6.15 | 162.5 |
| MobileNetV2-0.35× | 87.83% | 99.60% | 403,814 | 10.7 | 1.62 | 6.50 | 153.9 |
| SqueezeNet-1.1 | 88.70% | 99.77% | 725,574 | 43.2 | 2.90 | 3.01 | 331.7 |
| MobileNetV3-Small | 88.67% | 99.33% | 1,524,006 | 11.1 | 6.10 | 4.92 | 203.3 |

> CPU: Intel Core i5-10300H @ 2.50GHz (4 physical / 8 logical cores), batch=1, n=500 runs.
> Latency reported as **mean ± std**: **6.93 ± 0.61 ms**

### Intel Image Classification (150×150, 6 classes, 100 epochs, from scratch)

| Model | Top-1 | Top-5 | Params | MACs (M) | Size (MB) | Lat (ms) | FPS |
|---|---|---|---|---|---|---|---|
| **HalkaNet (392K)** | **91.93%** | **99.87%** | **392,473** | **98.6** | **1.57** | **7.87** | **127.1** |

> Latency: **8.18 ± 1.96 ms** [mean ± std, n=500]

---

## Channel Width Equations

```
C_rgb_out    = floor( C_rgb  × (1 + s_rgb)  )
C_coarse_out = floor( C_f    × (1 + s_f)    )
C_fine_out   = floor( C_f    × (1 + s_f)    )

C_fused      = C_rgb_out + C_coarse_out + C_fine_out
g            = max(8, round8( C_fused × r_grow ))
C_final      = C_fused + tail_depth × g
C_mid        = round8( 2 × C_fused_out − C_fused_out / tail_depth )
```

Where `s_rgb`, `s_f` = skip_expansion ratios, `r_grow` = tail_grow_ratio, `round8` = round to nearest multiple of 8.

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
├── LICENSE
└── README.md
```

---

## Citation

If you use HalkaNet in your research, please cite:

```bibtex
@article{zohaib2025halkanet,
  title   = {HalkaNet: A Tri-Branch Lightweight CNN with Fixed-Filter Inductive
             Bias for Accurate Edge-Device Image Classification},
  author  = {Md Zohaib},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/md-zohaib-official/HalkaNetv1}
}
```

---

## About

This project is the M.Tech final semester research project of **Md Zohaib**, Department of Computer Science and Engineering, **SRM Institute of Science and Technology (SRMIST)**, Kattankulathur, Chennai — 603203, Tamil Nadu, India.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2025 Md Zohaib
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to the conditions in the LICENSE file.
```

---

<p align="center">
  Made with ❤️ at <b>SRM Institute of Science and Technology</b>, Chennai<br/>
  <i>M.Tech Final Semester Project — 2025</i>
</p>
