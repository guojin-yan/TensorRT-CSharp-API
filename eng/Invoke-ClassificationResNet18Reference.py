#!/usr/bin/env python3
"""Generate independent ResNet18 references from the exact C# input tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
EXPECTED_ONNX_SHA256 = "ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903"
EXPECTED_LABELS_SHA256 = "ee4fa67b1dd46919ef87529b70e25bdd1ec9ebd350de3ad426e6640eb70320a1"
EXPECTED_INPUT_SHA256 = "18a5b601971e67521f895f9b0b89c3c0ba7820a9d0ff09d1981943947a692fa9"
PREPROCESS_CANONICAL = (
    "resizeMode=shorter-side-center-crop;resizeShorterSide=256;tensorLayout=NCHW;"
    "colorOrder=RGB;scale=0.003921569;mean=0.485,0.456,0.406;"
    "std=0.229,0.224,0.225;interpolation=bilinear-half-pixel;crop=center"
)
OUTPUT_CONTRACT_CANONICAL = "tensorName=logits;shape=1,1000;dataType=float32;valueKind=probabilities"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input-tensor", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{role} SHA256 mismatch: expected {expected}, actual {actual}: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def csharp_softmax(logits: Any) -> Any:
    import numpy as np

    flattened = np.asarray(logits, dtype=np.float32).reshape(-1)
    maximum = np.max(flattened).astype(np.float32)
    exponentials = np.exp(flattened - maximum).astype(np.float32)
    total = sum(float(value) for value in exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Softmax normalization failed")
    return np.asarray([np.float32(float(value) / total) for value in exponentials], dtype=np.float32)


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    expected_input_sha256 = args.expected_input_sha256.strip().lower()
    if len(expected_input_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_input_sha256):
        raise ValueError("--expected-input-sha256 must be a 64-character lowercase hexadecimal SHA256")

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    from torchvision.models import resnet18

    require_sha256(args.weights, EXPECTED_WEIGHTS_SHA256, "weights")
    require_sha256(args.onnx, EXPECTED_ONNX_SHA256, "ONNX")
    require_sha256(args.labels, EXPECTED_LABELS_SHA256, "labels")
    require_sha256(args.input_tensor, expected_input_sha256, "C# input tensor")

    labels = args.labels.read_text(encoding="utf-8").splitlines()
    if len(labels) != 1000 or any(not label for label in labels):
        raise ValueError("The labels file must contain exactly 1000 non-empty lines")

    input_values = np.fromfile(args.input_tensor, dtype="<f4")
    if input_values.size != 1 * 3 * 224 * 224 or not np.isfinite(input_values).all():
        raise ValueError("The C# input tensor must contain 150528 finite little-endian float32 values")
    input_tensor = input_values.reshape(1, 3, 224, 224)

    graph = onnx.load(str(args.onnx))
    onnx.checker.check_model(graph)
    if [value.name for value in graph.graph.input] != ["images"]:
        raise ValueError("Expected the ONNX input name to be images")
    if [value.name for value in graph.graph.output] != ["logits"]:
        raise ValueError("Expected the ONNX output name to be logits")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_logits = np.asarray(session.run(["logits"], {"images": input_tensor})[0], dtype=np.float32)
    if ort_logits.shape != (1, 1000) or not np.isfinite(ort_logits).all():
        raise ValueError(f"Unexpected ONNX Runtime output: {ort_logits.shape}")

    model = resnet18(weights=None)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.inference_mode():
        pytorch_logits = model(torch.from_numpy(input_tensor.copy())).detach().cpu().numpy().astype(np.float32)
    pytorch_ort_absolute = np.abs(pytorch_logits - ort_logits)
    pytorch_ort_maximum_absolute_error = float(np.max(pytorch_ort_absolute))
    pytorch_ort_mean_absolute_error = float(np.mean(pytorch_ort_absolute))
    if int(np.argmax(pytorch_logits)) != int(np.argmax(ort_logits)):
        raise ValueError("PyTorch and ONNX Runtime argmax differ")

    probabilities = csharp_softmax(ort_logits)
    probability_sum = float(np.sum(probabilities, dtype=np.float64))
    top_indices = sorted(range(probabilities.size), key=lambda index: (-float(probabilities[index]), index))[: args.top_k]

    preprocess_contract_sha256 = sha256_bytes(PREPROCESS_CANONICAL.encode("utf-8"))
    output_tensor_contract_sha256 = sha256_bytes(OUTPUT_CONTRACT_CANONICAL.encode("utf-8"))
    task_semantics_canonical = (
        f"task=classification;scoreTransform=softmax;topK={args.top_k};"
        "argmaxRule=max-score-then-lowest-index;"
        f"labelsSha256={EXPECTED_LABELS_SHA256}"
    )
    task_semantics_sha256 = sha256_bytes(task_semantics_canonical.encode("utf-8"))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_reference_path = args.output_directory / "logits.onnxruntime.reference.json"
    task_reference_path = args.output_directory / "classification.onnxruntime.reference.json"
    tampered_reference_path = args.output_directory / "classification.onnxruntime.tampered.reference.json"
    report_path = args.output_directory / "resnet18-reference-report.json"

    source_classification = "independent-onnxruntime-cpu-execution-provider-from-exact-csharp-image-tensor"
    write_json(
        raw_reference_path,
        {
            "schemaVersion": 1,
            "tensorName": "logits",
            "shape": [1, 1000],
            "values": [float(value) for value in ort_logits.reshape(-1)],
            "sourceClassification": source_classification,
        },
    )
    task_reference = {
        "schemaVersion": 1,
        "tensorName": "logits",
        "shape": [1, 1000],
        "values": [float(value) for value in probabilities],
        "valueKind": "probabilities",
        "modelSha256": EXPECTED_ONNX_SHA256,
        "inputTensorSha256": expected_input_sha256,
        "preprocessContractSha256": preprocess_contract_sha256,
        "outputTensorContractSha256": output_tensor_contract_sha256,
        "labelsSha256": EXPECTED_LABELS_SHA256,
        "taskSemanticsSha256": task_semantics_sha256,
        "sourceClassification": source_classification,
    }
    write_json(task_reference_path, task_reference)
    tampered_reference = dict(task_reference)
    tampered_reference["values"] = list(task_reference["values"])
    tampered_reference["values"][0] = float(tampered_reference["values"][0]) + 0.125
    tampered_reference["sourceClassification"] = source_classification + "-controlled-negative-index-0-plus-0.125"
    write_json(tampered_reference_path, tampered_reference)

    write_json(
        report_path,
        {
            "schemaVersion": 1,
            "recordKind": "classification-resnet18-independent-reference",
            "toolchain": {
                "torchVersion": torch.__version__,
                "torchvisionVersion": torchvision.__version__,
                "onnxVersion": onnx.__version__,
                "onnxRuntimeVersion": ort.__version__,
                "providers": session.get_providers(),
            },
            "assets": {
                "weightsSha256": EXPECTED_WEIGHTS_SHA256,
                "onnxSha256": EXPECTED_ONNX_SHA256,
                "labelsSha256": EXPECTED_LABELS_SHA256,
                "inputTensorSha256": expected_input_sha256,
            },
            "contracts": {
                "preprocessCanonical": PREPROCESS_CANONICAL,
                "preprocessContractSha256": preprocess_contract_sha256,
                "outputTensorContractCanonical": OUTPUT_CONTRACT_CANONICAL,
                "outputTensorContractSha256": output_tensor_contract_sha256,
                "taskSemanticsCanonical": task_semantics_canonical,
                "taskSemanticsSha256": task_semantics_sha256,
            },
            "pytorchOnnxRuntimeComparison": {
                "comparedElementCount": int(ort_logits.size),
                "maximumAbsoluteError": pytorch_ort_maximum_absolute_error,
                "meanAbsoluteError": pytorch_ort_mean_absolute_error,
                "sameArgmax": True,
                "passed": pytorch_ort_maximum_absolute_error <= 1.0e-4,
            },
            "reference": {
                "rawLogitsSha256": sha256_bytes(ort_logits.astype("<f4", copy=False).tobytes()),
                "probabilitiesSha256": sha256_bytes(probabilities.astype("<f4", copy=False).tobytes()),
                "probabilitySum": probability_sum,
                "rawReferencePath": str(raw_reference_path.resolve()),
                "rawReferenceSha256": sha256(raw_reference_path),
                "taskReferencePath": str(task_reference_path.resolve()),
                "taskReferenceSha256": sha256(task_reference_path),
                "tamperedReferencePath": str(tampered_reference_path.resolve()),
                "tamperedReferenceSha256": sha256(tampered_reference_path),
                "topK": [
                    {"rank": rank + 1, "classIndex": index, "className": labels[index], "score": float(probabilities[index])}
                    for rank, index in enumerate(top_indices)
                ],
            },
            "boundary": {
                "independentCpuReference": True,
                "ownerReviewedGolden": False,
                "publicRedistributionApproved": False,
                "packageConsumerRuntimeProof": False,
                "publicPackageProof": False,
                "postPublishProof": False,
                "uploadsAssets": False,
                "performsPublish": False,
            },
        },
    )

    if pytorch_ort_maximum_absolute_error > 1.0e-4:
        raise ValueError(f"PyTorch/ONNX Runtime maximum absolute error is too large: {pytorch_ort_maximum_absolute_error}")
    if abs(probability_sum - 1.0) > 1.0e-5:
        raise ValueError(f"Probability sum is invalid: {probability_sum}")

    print(f"ReferenceReport={report_path.resolve()}")
    print(f"PyTorchOrtMaximumAbsoluteError={pytorch_ort_maximum_absolute_error}")
    print(f"ProbabilitySum={probability_sum}")
    print(f"Top1={top_indices[0]}:{labels[top_indices[0]]}:{float(probabilities[top_indices[0]])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
