#!/usr/bin/env python3
"""Create reproducible YOLOv8 classification tensors and independent references."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        else:
            result.append(dimension.dim_param)
    return result


def top_k(values: np.ndarray, names: list[str], count: int = 5) -> list[dict[str, object]]:
    indices = np.argsort(-values.reshape(-1))[:count]
    return [
        {
            "index": int(index),
            "label": names[int(index)],
            "score": float(values.reshape(-1)[int(index)]),
        }
        for index in indices
    ]


def write_reference(path: Path, values: np.ndarray, source: str) -> None:
    document = {
        "schemaVersion": 1,
        "tensorName": "output0",
        "shape": [1, 1000],
        "values": [float(value) for value in values.reshape(-1)],
        "sourceClassification": source,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def export_onnx(model: YOLO, destination: Path) -> None:
    exported = Path(
        model.export(
            format="onnx",
            imgsz=224,
            opset=17,
            simplify=True,
            dynamic=False,
            batch=1,
            device="cpu",
        )
    ).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported != destination.resolve():
        shutil.copy2(exported, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--imagenet-yaml", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--csharp-tensor", type=Path)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    for path in (args.weights, args.imagenet_yaml, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    yaml_document = yaml.safe_load(args.imagenet_yaml.read_text(encoding="utf-8"))
    labels = list(yaml_document["map"].values())
    model_labels = [model.names[index] for index in range(len(model.names))]
    if labels != model_labels or len(labels) != 1000:
        raise ValueError("ImageNet.yaml map order does not match the 1000 model.names entries exactly.")

    labels_path = output_directory / "imagenet-yolov8n-cls.names"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")

    bgr = cv2.imread(str(args.image.resolve()), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    ppm_path = output_directory / (args.image.stem + ".ppm")
    pil_image.save(ppm_path, format="PPM")
    input_tensor = model.model.transforms(pil_image).unsqueeze(0).numpy().astype(np.float32)
    if list(input_tensor.shape) != [1, 3, 224, 224]:
        raise ValueError(f"Unexpected transformed input shape: {list(input_tensor.shape)}")
    tensor_path = output_directory / "input-ultralytics-1x3x224x224.fp32.bin"
    input_tensor.tofile(tensor_path)

    if args.export_onnx:
        export_onnx(model, args.onnx.resolve())
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    graph = onnx.load(str(args.onnx.resolve()))
    onnx.checker.check_model(graph)
    inputs = [(item.name, tensor_shape(item)) for item in graph.graph.input]
    outputs = [(item.name, tensor_shape(item)) for item in graph.graph.output]
    if inputs != [("images", [1, 3, 224, 224])] or outputs != [("output0", [1, 1000])]:
        raise ValueError(f"Unexpected ONNX contract: inputs={inputs}, outputs={outputs}")
    if graph.graph.node[-1].op_type != "Softmax":
        raise ValueError("The official classification export must end with Softmax.")

    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    ort_output = session.run(["output0"], {"images": input_tensor})[0].astype(np.float32)
    with torch.no_grad():
        torch_output = model.model(torch.from_numpy(input_tensor))
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    pytorch_output = torch_output.detach().cpu().numpy().astype(np.float32)
    pytorch_maximum_absolute_error = float(np.max(np.abs(ort_output - pytorch_output)))
    if pytorch_maximum_absolute_error > 1e-5:
        raise ValueError(
            "PyTorch/ONNX Runtime classification output mismatch: "
            f"maximum absolute error {pytorch_maximum_absolute_error}"
        )
    if abs(float(np.sum(ort_output)) - 1.0) > 1e-5:
        raise ValueError("ONNX Runtime output is not a normalized probability vector.")

    ort_output_path = output_directory / "output0-onnxruntime.fp32.bin"
    ort_output.tofile(ort_output_path)
    reference_path = output_directory / "output0.reference.json"
    write_reference(reference_path, ort_output, "official-yolov8n-cls-onnxruntime-cpu")
    tampered = ort_output.copy()
    tampered.reshape(-1)[0] += np.float32(0.125)
    tampered_reference_path = output_directory / "output0.tampered.reference.json"
    write_reference(tampered_reference_path, tampered, "controlled-single-value-tamper-negative")

    csharp_comparison: dict[str, object] | None = None
    if args.csharp_tensor is not None:
        csharp_input = np.fromfile(args.csharp_tensor.resolve(), dtype=np.float32)
        if csharp_input.size != input_tensor.size:
            raise ValueError("C# input tensor element count does not match the Ultralytics tensor.")
        csharp_input = csharp_input.reshape(input_tensor.shape)
        csharp_output = session.run(["output0"], {"images": csharp_input})[0].astype(np.float32)
        csharp_output_path = output_directory / "output0-csharp-input-onnxruntime.fp32.bin"
        csharp_output.tofile(csharp_output_path)
        csharp_reference_path = output_directory / "output0-csharp-input.reference.json"
        write_reference(
            csharp_reference_path,
            csharp_output,
            "official-yolov8n-cls-onnxruntime-cpu-csharp-input",
        )
        input_difference = np.abs(csharp_input - input_tensor)
        output_difference = np.abs(csharp_output - ort_output)
        csharp_comparison = {
            "tensorSha256": sha256(args.csharp_tensor.resolve()),
            "outputTensorSha256": sha256(csharp_output_path),
            "referenceSha256": sha256(csharp_reference_path),
            "referenceFileName": csharp_reference_path.name,
            "inputMismatchCount": int(np.count_nonzero(input_difference)),
            "inputMaximumAbsoluteError": float(np.max(input_difference)),
            "inputMeanAbsoluteError": float(np.mean(input_difference)),
            "outputMaximumAbsoluteError": float(np.max(output_difference)),
            "sameTop5IndicesAndOrder": [item["index"] for item in top_k(csharp_output, labels)]
            == [item["index"] for item in top_k(ort_output, labels)],
            "top5": top_k(csharp_output, labels),
        }

    report = {
        "schemaVersion": 1,
        "recordKind": "yolovision-yolov8n-cls-independent-reference",
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "contracts": {
            "inputs": [{"name": name, "shape": shape} for name, shape in inputs],
            "outputs": [{"name": name, "shape": shape, "valueKind": "probabilities"} for name, shape in outputs],
            "lastOnnxNode": graph.graph.node[-1].op_type,
            "preprocess": {
                "resize": "shorter-side-to-224",
                "crop": "center-224x224",
                "colorOrder": "RGB",
                "layout": "NCHW",
                "scale": "1/255",
                "mean": [0.0, 0.0, 0.0],
                "standardDeviation": [1.0, 1.0, 1.0],
            },
        },
        "hashes": {
            "weightsSha256": sha256(args.weights.resolve()),
            "imagenetYamlSha256": sha256(args.imagenet_yaml.resolve()),
            "sourceImageSha256": sha256(args.image.resolve()),
            "ppmSha256": sha256(ppm_path),
            "labelsSha256": sha256(labels_path),
            "onnxSha256": sha256(args.onnx.resolve()),
            "inputTensorSha256": sha256(tensor_path),
            "outputTensorSha256": sha256(ort_output_path),
            "referenceSha256": sha256(reference_path),
            "tamperedReferenceSha256": sha256(tampered_reference_path),
        },
        "validation": {
            "classCount": len(labels),
            "labelsMatchModelNamesExactly": labels == model_labels,
            "probabilitySum": float(np.sum(ort_output)),
            "pytorchOnnxRuntimeMaximumAbsoluteError": pytorch_maximum_absolute_error,
            "top5": top_k(ort_output, labels),
            "csharpCenterCrop": csharp_comparison,
        },
        "proofBoundary": {
            "independentCpuReferenceOnly": True,
            "tensorRtRuntimeExecuted": False,
            "performsPublish": False,
            "uploadsAssets": False,
        },
    }
    report_path = output_directory / "classification-reference.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"ReferenceReport={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
