#!/usr/bin/env python3
"""Export and validate the official torchvision LRASPP semantic model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any


VOC_LABELS = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def tensor_shape(value_info: Any) -> list[int | str]:
    shape: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        shape.append(int(dimension.dim_value) if dimension.HasField("dim_value") else dimension.dim_param)
    return shape


def write_tensor_reference(path: Path, values: Any, source: str) -> None:
    write_json(
        path,
        {
            "schemaVersion": 1,
            "tensorName": "semantic",
            "shape": [1, 21, 320, 320],
            "values": [float(value) for value in values.reshape(-1)],
            "sourceClassification": source,
        },
    )


def class_histogram(class_indices: Any) -> list[dict[str, Any]]:
    import numpy as np

    counts = np.bincount(class_indices.reshape(-1), minlength=len(VOC_LABELS))
    return [
        {
            "classId": class_id,
            "className": VOC_LABELS[class_id],
            "pixelCount": int(counts[class_id]),
        }
        for class_id in range(len(VOC_LABELS))
    ]


def write_class_index_artifact(path: Path, logits: Any) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    class_indices = np.argmax(logits[0], axis=0).astype("<i4")
    class_indices.tofile(path)
    return class_indices, class_histogram(class_indices)


def load_model(weights_path: Path) -> Any:
    import torch
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def create_python_input(image_path: Path, ppm_path: Path) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image.save(ppm_path, format="PPM")
    resized = image.resize((320, 320), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    standard_deviation = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    values = (values - mean) / standard_deviation
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=np.float32)


def model_logits(model: Any, tensor: Any) -> Any:
    import torch

    with torch.no_grad():
        output = model(torch.from_numpy(tensor))["out"]
    return output.detach().cpu().numpy()


def export_onnx(model: Any, destination: Path) -> None:
    import torch

    class OutputWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, images: Any) -> Any:
            return self.wrapped(images)["out"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    torch.onnx.export(
        OutputWrapper(model).eval(),
        dummy,
        str(destination),
        input_names=["images"],
        output_names=["semantic"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def compare_artifact(actual_manifest_path: Path, reference_indices: Any, output_directory: Path) -> dict[str, Any]:
    import numpy as np

    with actual_manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if manifest.get("schemaVersion") != "yolovision-semantic-map-artifacts.v1":
        raise ValueError("Actual artifact manifest has an unexpected schemaVersion.")
    artifact = manifest.get("classIndexArtifact", {})
    actual_path = Path(str(artifact.get("path", ""))).resolve()
    if not actual_path.is_file():
        raise FileNotFoundError(actual_path)
    if sha256(actual_path) != artifact.get("sha256"):
        raise ValueError("Actual semantic class-index artifact SHA256 does not match its manifest.")
    actual = np.fromfile(actual_path, dtype="<i4")
    expected = reference_indices.reshape(-1)
    if actual.size != expected.size:
        mismatch_count = max(actual.size, expected.size)
        first_mismatch_index = 0
    else:
        mismatches = np.flatnonzero(actual != expected)
        mismatch_count = int(mismatches.size)
        first_mismatch_index = int(mismatches[0]) if mismatches.size else -1
    comparison = {
        "schemaVersion": 1,
        "recordKind": "yolovision-semantic-class-index-comparison",
        "referenceElementCount": int(expected.size),
        "actualElementCount": int(actual.size),
        "mismatchCount": mismatch_count,
        "firstMismatchIndex": first_mismatch_index,
        "referenceClassIndexSha256": sha256(output_directory / "semantic-class-index-onnxruntime.i32.bin"),
        "actualClassIndexSha256": sha256(actual_path),
        "completed": True,
        "passed": mismatch_count == 0,
        "boundary": (
            "Full-resolution argmax comparison only; not package-consumer, post-publish, "
            "redistribution, or release proof."
        ),
    }
    comparison_path = output_directory / "semantic-class-index-comparison.json"
    write_json(comparison_path, comparison)
    comparison["path"] = str(comparison_path)
    return comparison


def compare_output_report(actual_path: Path, reference_histogram: list[dict[str, Any]]) -> dict[str, Any]:
    with actual_path.open("r", encoding="utf-8-sig") as stream:
        report = json.load(stream)
    if report.get("schemaVersion") != "yolovision-output.v1" or report.get("task") != "sem":
        raise ValueError("Actual output must be a yolovision-output.v1 semantic report.")
    outputs = report.get("outputs", [])
    if len(outputs) != 1 or outputs[0].get("shape") != [1, 21, 320, 320]:
        raise ValueError("Actual output does not retain semantic:[1,21,320,320].")
    predictions = [item for item in report.get("predictions", []) if item.get("task") == "sem"]
    if len(predictions) != 1:
        raise ValueError("Actual output must contain exactly one semantic prediction summary.")
    actual_histogram = predictions[0].get("classHistogram", [])
    return {
        "schemaVersion": 1,
        "recordKind": "yolovision-semantic-output-summary-comparison",
        "actualOutputPath": str(actual_path),
        "actualOutputSha256": sha256(actual_path),
        "histogramMatches": actual_histogram == reference_histogram,
        "completed": True,
        "passed": actual_histogram == reference_histogram,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--csharp-tensor", type=Path)
    parser.add_argument("--actual-output", type=Path)
    parser.add_argument("--actual-artifact-manifest", type=Path)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    for path in (args.weights, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import PIL
    import torch
    import torchvision
    from PIL import Image

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    ppm_path = output_directory / "dog.ppm"
    labels_path = output_directory / "voc-semantic.names"
    labels_path.write_text("\n".join(VOC_LABELS) + "\n", encoding="utf-8", newline="\n")
    python_input = create_python_input(args.image.resolve(), ppm_path)
    python_input_path = output_directory / "input-python-imagenet-1x3x320x320.fp32.bin"
    python_input.tofile(python_input_path)

    model = load_model(args.weights.resolve())
    if args.export_onnx:
        export_onnx(model, args.onnx.resolve())
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    graph = onnx.load(str(args.onnx.resolve()))
    onnx.checker.check_model(graph)
    inputs = [(item.name, tensor_shape(item)) for item in graph.graph.input]
    outputs = [(item.name, tensor_shape(item)) for item in graph.graph.output]
    if inputs != [("images", [1, 3, 320, 320])] or outputs != [("semantic", [1, 21, 320, 320])]:
        raise ValueError(f"Unexpected ONNX contract: inputs={inputs}, outputs={outputs}")

    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    python_ort_logits = session.run(["semantic"], {"images": python_input})[0].astype(np.float32)
    python_torch_logits = model_logits(model, python_input).astype(np.float32)
    torch_ort_maximum_absolute_error = float(np.max(np.abs(python_torch_logits - python_ort_logits)))
    if torch_ort_maximum_absolute_error > 1e-4:
        raise ValueError(f"PyTorch/ONNX Runtime mismatch: {torch_ort_maximum_absolute_error}")

    python_output_path = output_directory / "semantic-onnxruntime-python-input.fp32.bin"
    python_ort_logits.tofile(python_output_path)
    python_reference_path = output_directory / "semantic-python-input.reference.json"
    write_tensor_reference(
        python_reference_path,
        python_ort_logits,
        "official-torchvision-lraspp-onnxruntime-cpu-python-imagenet-input",
    )

    runtime_input = python_input
    runtime_logits = python_ort_logits
    runtime_source = "official-torchvision-lraspp-onnxruntime-cpu-python-imagenet-input"
    csharp_input_comparison: dict[str, Any] | None = None
    if args.csharp_tensor is not None:
        csharp_path = args.csharp_tensor.resolve()
        csharp_input = np.fromfile(csharp_path, dtype=np.float32)
        if csharp_input.size != python_input.size:
            raise ValueError("C# input tensor element count does not match [1,3,320,320].")
        csharp_input = csharp_input.reshape(python_input.shape)
        if not np.all(np.isfinite(csharp_input)):
            raise ValueError("C# input tensor contains NaN or Infinity.")
        runtime_input = csharp_input
        runtime_logits = session.run(["semantic"], {"images": csharp_input})[0].astype(np.float32)
        runtime_source = "official-torchvision-lraspp-onnxruntime-cpu-csharp-imagenet-input"
        difference = np.abs(csharp_input - python_input)
        csharp_input_comparison = {
            "path": str(csharp_path),
            "sha256": sha256(csharp_path),
            "mismatchCount": int(np.count_nonzero(difference)),
            "maximumAbsoluteError": float(np.max(difference)),
            "meanAbsoluteError": float(np.mean(difference)),
        }

    runtime_input_path = output_directory / "input-runtime-1x3x320x320.fp32.bin"
    runtime_input.tofile(runtime_input_path)
    runtime_output_path = output_directory / "semantic-onnxruntime-runtime-input.fp32.bin"
    runtime_logits.tofile(runtime_output_path)
    runtime_reference_path = output_directory / "semantic.reference.json"
    write_tensor_reference(runtime_reference_path, runtime_logits, runtime_source)
    tampered_logits = runtime_logits.copy()
    tampered_logits.reshape(-1)[0] += np.float32(10.0)
    tampered_reference_path = output_directory / "semantic.tampered.reference.json"
    write_tensor_reference(tampered_reference_path, tampered_logits, "controlled-negative-single-value-mutation")

    class_index_path = output_directory / "semantic-class-index-onnxruntime.i32.bin"
    class_indices, histogram = write_class_index_artifact(class_index_path, runtime_logits)
    dominant = max(histogram, key=lambda item: int(item["pixelCount"]))
    comparison: dict[str, Any] | None = None
    if args.actual_artifact_manifest is not None:
        comparison = compare_artifact(args.actual_artifact_manifest.resolve(), class_indices, output_directory)
    output_summary_comparison: dict[str, Any] | None = None
    if args.actual_output is not None:
        output_summary_comparison = compare_output_report(args.actual_output.resolve(), histogram)

    report = {
        "schemaVersion": 1,
        "recordKind": "yolovision-torchvision-lraspp-independent-reference",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "pillow": PIL.__version__,
        },
        "source": {
            "model": "LRASPP MobileNetV3 Large",
            "torchvisionTag": "v0.25.0",
            "torchvisionCommit": "8ac84ee75afb1c327902156b5336f56ad63b7e2f",
            "weightsUrl": "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
            "inputCommit": "c7895df70c7767403e36f82786d6b611b7984557",
            "license": "BSD-3-Clause",
            "redistributionApprovedForRepository": False,
        },
        "contracts": {
            "inputs": [{"name": name, "shape": shape} for name, shape in inputs],
            "outputs": [{"name": name, "shape": shape, "valueKind": "class-logits"} for name, shape in outputs],
            "preprocess": {
                "resize": "stretch-320x320",
                "colorOrder": "RGB",
                "layout": "NCHW",
                "scale": 1.0 / 255.0,
                "mean": [0.485, 0.456, 0.406],
                "standardDeviation": [0.229, 0.224, 0.225],
            },
            "postprocess": {
                "operation": "argmax-over-class-dimension",
                "classCount": len(VOC_LABELS),
                "classIndexDataType": "int32-little-endian",
                "classIndexLayout": "row-major-hw",
            },
        },
        "hashes": {
            "weightsSha256": sha256(args.weights.resolve()),
            "sourceImageSha256": sha256(args.image.resolve()),
            "ppmSha256": sha256(ppm_path),
            "labelsSha256": sha256(labels_path),
            "onnxSha256": sha256(args.onnx.resolve()),
            "pythonInputSha256": sha256(python_input_path),
            "runtimeInputSha256": sha256(runtime_input_path),
            "runtimeOutputSha256": sha256(runtime_output_path),
            "runtimeReferenceSha256": sha256(runtime_reference_path),
            "tamperedReferenceSha256": sha256(tampered_reference_path),
            "classIndexSha256": sha256(class_index_path),
        },
        "validation": {
            "pytorchOnnxRuntimeMaximumAbsoluteError": torch_ort_maximum_absolute_error,
            "logitMinimum": float(np.min(runtime_logits)),
            "logitMaximum": float(np.max(runtime_logits)),
            "allLogitsFinite": bool(np.all(np.isfinite(runtime_logits))),
            "dominantClass": dominant,
            "classHistogram": histogram,
            "csharpInputComparison": csharp_input_comparison,
            "actualArtifactComparison": comparison,
            "actualOutputSummaryComparison": output_summary_comparison,
        },
        "proofBoundary": {
            "independentCpuReferenceOnly": True,
            "tensorRtRuntimeExecuted": False,
            "performsPublish": False,
            "uploadsAssets": False,
        },
    }
    report_path = output_directory / "semantic-reference.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    print(f"ReferenceReport={report_path}")
    if comparison is not None and not comparison["passed"]:
        return 1
    if output_summary_comparison is not None and not output_summary_comparison["passed"]:
        return 1
    if not math.isfinite(torch_ort_maximum_absolute_error):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
