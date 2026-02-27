"""ONNX export utilities — EI-compatible, with optional int8 and accuracy check."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import shape_inference
from onnxruntime.quantization import QuantType, quantize_dynamic

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class ExportResult:
    onnx_path: str
    int8_path: Optional[str]
    fp32_size_kb: float
    int8_size_kb: Optional[float]
    fp32_accuracy: Optional[float]  # % over eval samples
    int8_accuracy: Optional[float]
    eval_samples: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "── Export Result ─────────────────────────────",
            f"  float32 : {self.onnx_path:<40} ({self.fp32_size_kb:.1f} KB)",
        ]
        if self.int8_path:
            ratio = self.fp32_size_kb / self.int8_size_kb
            lines.append(
                f"  int8    : {self.int8_path:<40} ({self.int8_size_kb:.1f} KB, {ratio:.1f}x smaller)"
            )
        if self.fp32_accuracy is not None:
            lines.append(f"\n  Accuracy over {self.eval_samples} samples:")
            lines.append(f"    float32 : {self.fp32_accuracy:.2f}%")
            if self.int8_accuracy is not None:
                drop = self.fp32_accuracy - self.int8_accuracy
                lines.append(
                    f"    int8    : {self.int8_accuracy:.2f}%  (Δ {drop:+.2f} pp)"
                )
        for note in self.notes:
            lines.append(f"  ⚠  {note}")
        lines.append("──────────────────────────────────────────────")
        return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _verify_bn_stats(model: nn.Module) -> None:
    """Assert the first BatchNorm2d has non-trivial running stats."""
    bn = next((m for m in model.modules() if isinstance(m, nn.BatchNorm2d)), None)
    if bn is None:
        return  # no BN layers — skip
    assert not torch.all(bn.running_mean == 0), (
        "BatchNorm running_mean is all zeros. "
        "Did you call model.eval() and load weights before exporting?"
    )


def _run_accuracy(
    sessions: dict[str, ort.InferenceSession],
    test_loader,
    input_name: str,
    max_samples: int,
) -> tuple[dict[str, float], int]:
    """Run all ORT sessions against test_loader, return {name: acc%} and total."""
    correct = {k: 0 for k in sessions}
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs_np = imgs.numpy()
            labels_np = labels.numpy()
            for name, sess in sessions.items():
                preds = sess.run(None, {input_name: imgs_np})[0].argmax(axis=1)
                correct[name] += (preds == labels_np).sum()
            total += len(labels_np)
            if total >= max_samples:
                break

    return {k: v / total * 100 for k, v in correct.items()}, total


# ── Public API ─────────────────────────────────────────────────────────────────


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path = "model.onnx",
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    opset_version: int = 11,
    input_name: str = "input",
    output_name: str = "output",
    quantize: bool = True,
    test_loader=None,
    max_eval_samples: int = 500,
    accuracy_guard: float = 20.0,
) -> ExportResult:
    """
    Export a PyTorch model to an EI-compatible ONNX file.

    Parameters
    ----------
    model            : Must already be in eval() mode with weights loaded.
    output_path      : Destination .onnx file path (float32).
    input_shape      : Export dummy input shape — (batch, C, H, W).
    opset_version    : EI supports opset <= 12. Default 11 is safest.
    input_name       : ONNX input node name.
    output_name      : ONNX output node name.
    quantize         : Also produce a dynamic int8 .onnx alongside the float32.
    test_loader      : Optional DataLoader for accuracy check after export.
    max_eval_samples : Cap on samples used for the accuracy check.
    accuracy_guard   : Raise if float32 accuracy falls below this threshold.

    Returns
    -------
    ExportResult dataclass with paths, sizes, and accuracy.
    """
    output_path = Path(output_path)
    int8_path = output_path.with_stem(output_path.stem + "_int8") if quantize else None
    _shaped_tmp = output_path.with_stem(output_path.stem + "_shaped_tmp")

    # ── Guard: model must be in eval mode ─────────────────────────────────────
    assert not model.training, (
        "model.train() is active — call model.eval() before exporting."
    )
    _verify_bn_stats(model)

    # ── 1. ONNX export ────────────────────────────────────────────────────────
    dummy = torch.zeros(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,  # legacy TorchScript exporter — required for EI
    )
    print(f"✓ ONNX export → {output_path}  (opset {opset_version})")

    result_kwargs: dict = {
        "onnx_path": str(output_path),
        "int8_path": None,
        "fp32_size_kb": os.path.getsize(output_path) / 1024,
        "int8_size_kb": None,
        "fp32_accuracy": None,
        "int8_accuracy": None,
        "notes": [],
    }

    # ── 2. Shape inference + dynamic quantization ─────────────────────────────
    if quantize:
        proto = shape_inference.infer_shapes(onnx.load(str(output_path)))
        onnx.save(proto, str(_shaped_tmp))
        quantize_dynamic(str(_shaped_tmp), str(int8_path), weight_type=QuantType.QInt8)
        _shaped_tmp.unlink()  # remove intermediate file
        print(f"✓ INT8 export  → {int8_path}")

        result_kwargs["int8_path"] = str(int8_path)
        result_kwargs["int8_size_kb"] = os.path.getsize(int8_path) / 1024

    # ── 3. Accuracy check ─────────────────────────────────────────────────────
    if test_loader is not None:
        sessions = {"float32": ort.InferenceSession(str(output_path))}
        if quantize:
            sessions["int8"] = ort.InferenceSession(str(int8_path))

        accs, total = _run_accuracy(sessions, test_loader, input_name, max_eval_samples)
        result_kwargs["fp32_accuracy"] = accs["float32"]
        result_kwargs["int8_accuracy"] = accs.get("int8")
        result_kwargs["eval_samples"] = total

        if accs["float32"] < accuracy_guard:
            raise RuntimeError(
                f"float32 accuracy {accs['float32']:.1f}% is below guard ({accuracy_guard}%). "
                "Do not upload — check weights, eval mode, or preprocessing."
            )

    return ExportResult(**result_kwargs)
