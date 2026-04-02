# regenerate_quant_models.py
#
# Requirements:
#   pip install onnxruntime onnx onnxruntime-tools
#
# Usage:
#   python regenerate_quant_models.py \
#       --input  /path/to/onnx/model.onnx \
#       --outdir /path/to/onnx/
# regenerate_quant_models.py

import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference

# Gather  — 量化 embedding 表，从 46MB FP32 → 11.5MB INT8，是体积减半的关键
# MatMul/Gemm — attention + FFN 权重
# Add 明确不在列表里 — 避免生成 QLinearAdd，ORT 旧版 CPU EP 无此 kernel
SAFE_OPS = ["MatMul", "Gemm", "Gather"]


def preprocess(model_input: Path, out: Path) -> None:
    print(f"[preprocess] {model_input} → {out}")
    shape_inference.quant_pre_process(
        input_model_path=str(model_input),
        output_model_path=str(out),
        skip_optimization=True,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        verbose=0,
    )


def make_avx2(preprocessed: Path, out: Path) -> None:
    print(f"[avx2] {preprocessed} → {out}")
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(out),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=SAFE_OPS,
        per_channel=False,
        reduce_range=True,
    )
    print(f"[avx2] done — {out.stat().st_size / 1e6:.1f} MB")


def make_avx512(preprocessed: Path, out: Path) -> None:
    print(f"[avx512] {preprocessed} → {out}")
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=SAFE_OPS,
        per_channel=False,
        reduce_range=False,
    )
    print(f"[avx512] done — {out.stat().st_size / 1e6:.1f} MB")


def make_arm64(preprocessed: Path, out: Path) -> None:
    print(f"[arm64] {preprocessed} → {out}")
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=SAFE_OPS,
        per_channel=False,
        reduce_range=False,
    )
    print(f"[arm64] done — {out.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="FP32 model.onnx")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target", default="all",
                        choices=["all", "avx2", "avx512", "arm64"])
    args = parser.parse_args()

    src = Path(args.input)
    assert "quint8" not in src.name and "qint8" not in src.name, \
        f"Input must be FP32 model.onnx, got: {src.name}"

    dst = Path(args.outdir)
    dst.mkdir(parents=True, exist_ok=True)

    preprocessed = dst / "model_preprocessed.onnx"
    preprocess(src, preprocessed)

    if args.target in ("all", "avx2"):
        make_avx2(preprocessed,   dst / "model_quint8_avx2.onnx")
    if args.target in ("all", "avx512"):
        make_avx512(preprocessed, dst / "model_qint8_avx512.onnx")
    if args.target in ("all", "arm64"):
        make_arm64(preprocessed,  dst / "model_qint8_arm64.onnx")

    preprocessed.unlink()
    print("all done")


if __name__ == "__main__":
    main()
