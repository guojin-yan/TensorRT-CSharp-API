#!/usr/bin/env python3
"""Create raw and postprocess references for the official YOLOv8n detection model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def tensor_shape(value_info: Any) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        result.append(int(dimension.dim_value) if dimension.HasField("dim_value") else dimension.dim_param)
    return result


def box_iou(left: list[float], right: list[float]) -> float:
    def corners(box: list[float]) -> tuple[float, float, float, float]:
        x, y, width, height = box
        return x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0

    left_x1, left_y1, left_x2, left_y2 = corners(left)
    right_x1, right_y1, right_x2, right_y2 = corners(right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def write_tensor_reference(path: Path, values: Any, source: str) -> None:
    write_json(
        path,
        {
            "schemaVersion": 1,
            "tensorName": "output0",
            "shape": [1, 84, 8400],
            "values": [float(value) for value in values.reshape(-1)],
            "sourceClassification": source,
        },
    )


def export_onnx(model: Any, destination: Path) -> None:
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
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


def create_input_tensor(image_path: Path) -> tuple[Any, tuple[int, int], Path]:
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics.data.augment import LetterBox

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    height, width = bgr.shape[:2]
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32)
    transformed = letterbox(image=bgr)
    tensor = transformed[:, :, ::-1].transpose((2, 0, 1))[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / np.float32(255.0)
    ppm_path = image_path.parent / (image_path.stem + ".ppm")
    Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
    return tensor, (height, width), ppm_path


def model_output_tensor(output: Any) -> Any:
    import torch

    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected PyTorch output type: {type(output)!r}")
    return output.detach().cpu().numpy()


def actual_box_in_source_space(prediction: dict[str, Any], report: dict[str, Any]) -> list[float]:
    box = prediction["box"]
    input_report = report["input"]
    letterbox = input_report["letterbox"]
    image = input_report["image"]
    scale_x = float(letterbox["scaleX"])
    scale_y = float(letterbox["scaleY"])
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("YoloVision letterbox scales must be positive.")
    width = float(image["width"])
    height = float(image["height"])
    center_x = (float(box["x"]) - float(letterbox["padX"])) / scale_x
    center_y = (float(box["y"]) - float(letterbox["padY"])) / scale_y
    result = [
        min(max(center_x, 0.0), width),
        min(max(center_y, 0.0), height),
        float(box["width"]) / scale_x,
        float(box["height"]) / scale_y,
    ]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("YoloVision detection box contains a non-finite source-space value.")
    return result


def compare_actual(
    actual_path: Path,
    reference: dict[str, Any],
    reference_path: Path,
    maximum_score_error: float,
    minimum_box_iou: float,
) -> tuple[dict[str, Any], Path]:
    with actual_path.open("r", encoding="utf-8-sig") as stream:
        actual_report: dict[str, Any] = json.load(stream)
    if actual_report.get("schemaVersion") != "yolovision-output.v1" or actual_report.get("task") != "det":
        raise ValueError("Actual output must be a yolovision-output.v1 detection report.")
    boundary = actual_report.get("boundary", {})
    if boundary.get("isRuntimeProof") is not False:
        raise ValueError("Actual output report must retain its readonly proof boundary.")
    if actual_report.get("labels", {}).get("classCount") != 80:
        raise ValueError("Actual output must declare exactly 80 COCO labels.")
    output = actual_report.get("outputs", [])
    if len(output) != 1 or output[0].get("shape") != [1, 84, 8400] or output[0].get("role") != "boxes":
        raise ValueError("Actual output must retain the official output0 [1,84,8400] detection contract.")

    actual_predictions = [item for item in actual_report.get("predictions", []) if item.get("task") == "det"]
    unmatched = set(range(len(actual_predictions)))
    comparisons: list[dict[str, Any]] = []
    for expected in reference["predictions"]:
        candidates = [
            index
            for index in unmatched
            if int(actual_predictions[index].get("classId", -1)) == int(expected["classId"])
        ]
        if not candidates:
            comparisons.append(
                {
                    "referenceIndex": expected["index"],
                    "classId": expected["classId"],
                    "className": expected["className"],
                    "matched": False,
                    "passed": False,
                    "diagnostic": "No unmatched YoloVision detection has the same classId.",
                }
            )
            continue
        source_boxes = {
            index: actual_box_in_source_space(actual_predictions[index], actual_report) for index in candidates
        }
        actual_index = max(candidates, key=lambda index: box_iou(expected["sourceBox"], source_boxes[index]))
        unmatched.remove(actual_index)
        actual = actual_predictions[actual_index]
        actual_box = source_boxes[actual_index]
        overlap = box_iou(expected["sourceBox"], actual_box)
        score_error = abs(float(expected["score"]) - float(actual["score"]))
        passed = overlap >= minimum_box_iou and score_error <= maximum_score_error
        comparisons.append(
            {
                "referenceIndex": expected["index"],
                "actualIndex": actual_index,
                "classId": expected["classId"],
                "className": expected["className"],
                "matched": True,
                "referenceBox": expected["sourceBox"],
                "actualBox": actual_box,
                "boxIoU": overlap,
                "scoreAbsoluteError": score_error,
                "passed": passed,
            }
        )
    passed = len(comparisons) == len(actual_predictions) and not unmatched and all(
        item["passed"] for item in comparisons
    )
    comparison = {
        "schemaVersion": 1,
        "recordKind": "yolovision-detection-independent-reference-comparison",
        "referencePath": str(reference_path),
        "referenceSha256": sha256(reference_path),
        "actualOutputPath": str(actual_path),
        "actualOutputSha256": sha256(actual_path),
        "thresholds": {
            "maximumScoreAbsoluteError": maximum_score_error,
            "minimumBoxIoU": minimum_box_iou,
        },
        "referencePredictionCount": len(reference["predictions"]),
        "actualPredictionCount": len(actual_predictions),
        "unmatchedActualIndices": sorted(unmatched),
        "comparisons": comparisons,
        "completed": True,
        "passed": passed,
        "boundary": (
            "Independent detection postprocess comparison; not Owner acceptance, package-consumer proof, "
            "post-publish proof, redistribution approval, or release proof."
        ),
    }
    comparison_path = reference_path.parent / "yolovision-independent-comparison.json"
    write_json(comparison_path, comparison)
    return comparison, comparison_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--coco-yaml", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--csharp-tensor", type=Path)
    parser.add_argument("--actual-output", type=Path)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--maximum-score-error", type=float, default=0.01)
    parser.add_argument("--minimum-box-iou", type=float, default=0.995)
    args = parser.parse_args()
    for path in (args.weights, args.coco_yaml, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("confidence and IoU threshold must be in [0, 1]")
    if args.max_detections <= 0 or args.maximum_score_error < 0.0 or not 0.0 <= args.minimum_box_iou <= 1.0:
        parser.error("invalid comparison thresholds")

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_directory / ".ultralytics-config"))
    model = YOLO(str(args.weights.resolve()))
    yaml_document = yaml.safe_load(args.coco_yaml.read_text(encoding="utf-8"))
    names = yaml_document["names"]
    labels = [str(names[index]) for index in range(len(names))] if isinstance(names, dict) else list(names)
    model_labels = [str(model.names[index]) for index in range(len(model.names))]
    if labels != model_labels or len(labels) != 80:
        raise ValueError("coco.yaml names do not match all 80 embedded model names exactly.")
    labels_path = output_directory / "coco.names"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")

    input_tensor, original_shape, ppm_path = create_input_tensor(args.image.resolve())
    tensor_path = output_directory / "input-ultralytics-letterbox-1x3x640x640.fp32.bin"
    input_tensor.tofile(tensor_path)
    if args.export_onnx:
        export_onnx(model, args.onnx.resolve())
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    graph = onnx.load(str(args.onnx.resolve()))
    onnx.checker.check_model(graph)
    inputs = [(item.name, tensor_shape(item)) for item in graph.graph.input]
    outputs = [(item.name, tensor_shape(item)) for item in graph.graph.output]
    if inputs != [("images", [1, 3, 640, 640])] or outputs != [("output0", [1, 84, 8400])]:
        raise ValueError(f"Unexpected ONNX contract: inputs={inputs}, outputs={outputs}")

    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    ort_output = session.run(["output0"], {"images": input_tensor})[0].astype(np.float32)
    with torch.no_grad():
        pytorch_output = model_output_tensor(model.model(torch.from_numpy(input_tensor))).astype(np.float32)
    pytorch_maximum_absolute_error = float(np.max(np.abs(ort_output - pytorch_output)))
    if pytorch_maximum_absolute_error > 5e-3:
        raise ValueError(f"PyTorch/ONNX Runtime output mismatch: maximum absolute error {pytorch_maximum_absolute_error}")

    canonical_output_path = output_directory / "output0-onnxruntime-ultralytics-input.fp32.bin"
    ort_output.tofile(canonical_output_path)
    runtime_reference_output = ort_output
    runtime_reference_source = "official-yolov8n-det-onnxruntime-cpu-ultralytics-letterbox-input"
    csharp_input_report: dict[str, Any] | None = None
    if args.csharp_tensor:
        csharp_values = np.fromfile(args.csharp_tensor.resolve(), dtype=np.float32)
        if csharp_values.size != input_tensor.size:
            raise ValueError("C# tensor element count does not match [1,3,640,640].")
        csharp_tensor = csharp_values.reshape(input_tensor.shape)
        csharp_output = session.run(["output0"], {"images": csharp_tensor})[0].astype(np.float32)
        runtime_reference_output = csharp_output
        runtime_reference_source = "official-yolov8n-det-onnxruntime-cpu-csharp-letterbox-input"
        csharp_input_report = {
            "path": str(args.csharp_tensor.resolve()),
            "sha256": sha256(args.csharp_tensor.resolve()),
            "maximumAbsoluteError": float(np.max(np.abs(csharp_tensor - input_tensor))),
            "meanAbsoluteError": float(np.mean(np.abs(csharp_tensor - input_tensor))),
            "outputMaximumAbsoluteError": float(np.max(np.abs(csharp_output - ort_output))),
        }

    ort_output_path = output_directory / "output0-onnxruntime-runtime-input.fp32.bin"
    runtime_reference_output.tofile(ort_output_path)
    reference_path = output_directory / "output0.reference.json"
    write_tensor_reference(reference_path, runtime_reference_output, runtime_reference_source)
    tampered = runtime_reference_output.copy()
    tampered.reshape(-1)[0] += np.float32(125.0)
    tampered_reference_path = output_directory / "output0.tampered.reference.json"
    write_tensor_reference(tampered_reference_path, tampered, "controlled-negative-single-value-mutation")

    source_bgr = cv2.imread(str(args.image.resolve()), cv2.IMREAD_COLOR)
    prediction = model.predict(
        source=source_bgr,
        imgsz=640,
        conf=args.confidence,
        iou=args.iou_threshold,
        max_det=args.max_detections,
        agnostic_nms=False,
        rect=False,
        device="cpu",
        verbose=True,
    )[0]
    canonical_predictions = []
    for index in range(len(prediction.boxes)):
        class_id = int(prediction.boxes.cls[index].item())
        canonical_predictions.append(
            {
                "index": index,
                "classId": class_id,
                "className": labels[class_id],
                "score": float(prediction.boxes.conf[index].item()),
                "sourceBox": [float(value) for value in prediction.boxes.xywh[index].tolist()],
            }
        )
    predictions = canonical_predictions
    reference_source = "independent-ultralytics-pytorch-cpu-canonical-image-pipeline"
    if args.csharp_tensor:
        from ultralytics.utils.nms import non_max_suppression

        with torch.no_grad():
            csharp_pytorch_output = model_output_tensor(model.model(torch.from_numpy(csharp_tensor))).astype(np.float32)
        retained = non_max_suppression(
            torch.from_numpy(csharp_pytorch_output),
            conf_thres=args.confidence,
            iou_thres=args.iou_threshold,
            agnostic=False,
            max_det=args.max_detections,
            nc=80,
        )[0].detach().cpu().numpy()
        source_height, source_width = original_shape
        scale = min(640.0 / source_width, 640.0 / source_height)
        resized_width = round(source_width * scale)
        resized_height = round(source_height * scale)
        pad_x = (640 - resized_width) // 2
        pad_y = (640 - resized_height) // 2
        predictions = []
        for index, row in enumerate(retained):
            left, top, right, bottom, score, class_value = [float(value) for value in row]
            class_id = int(class_value)
            center_x = min(max(((left + right) / 2.0 - pad_x) / scale, 0.0), float(source_width))
            center_y = min(max(((top + bottom) / 2.0 - pad_y) / scale, 0.0), float(source_height))
            predictions.append(
                {
                    "index": index,
                    "classId": class_id,
                    "className": labels[class_id],
                    "score": score,
                    "sourceBox": [
                        center_x,
                        center_y,
                        (right - left) / scale,
                        (bottom - top) / scale,
                    ],
                }
            )
        reference_source = "independent-ultralytics-pytorch-cpu-csharp-letterbox-tensor"
    annotated_path = output_directory / "ultralytics-pytorch-reference.jpg"
    if not cv2.imwrite(str(annotated_path), prediction.plot()):
        raise ValueError(f"Unable to write annotated image: {annotated_path}")
    canonical_reference = {
        "schemaVersion": 1,
        "recordKind": "yolovision-canonical-ultralytics-detection-reference",
        "sourceClassification": "independent-ultralytics-pytorch-cpu-canonical-image-pipeline",
        "predictionCount": len(canonical_predictions),
        "predictions": canonical_predictions,
        "boundary": "Canonical image-pipeline comparison only; not release or redistribution proof.",
    }
    canonical_reference_path = output_directory / "ultralytics-canonical-image-reference.json"
    write_json(canonical_reference_path, canonical_reference)
    postprocess_reference = {
        "schemaVersion": 1,
        "recordKind": "yolovision-independent-detection-postprocess-reference",
        "sourceClassification": reference_source,
        "model": {"path": str(args.weights.resolve()), "sha256": sha256(args.weights.resolve())},
        "input": {
            "imagePath": str(args.image.resolve()),
            "imageSha256": sha256(args.image.resolve()),
            "originalShape": list(original_shape),
            "imageSize": 640,
            "confidenceThreshold": args.confidence,
            "iouThreshold": args.iou_threshold,
            "maxDetections": args.max_detections,
            "agnosticNms": False,
            "rect": False,
        },
        "runtime": {
            "framework": "Ultralytics PyTorch",
            "ultralyticsVersion": ultralytics.__version__,
            "torchVersion": torch.__version__,
            "onnxVersion": onnx.__version__,
            "onnxRuntimeVersion": ort.__version__,
            "pythonVersion": platform.python_version(),
            "device": "cpu",
        },
        "predictionCount": len(predictions),
        "predictions": predictions,
        "annotatedImage": {"path": str(annotated_path), "sha256": sha256(annotated_path)},
        "boundary": (
            "Independent framework/runtime postprocess reference; not Owner acceptance, redistribution approval, "
            "package-consumer proof, post-publish proof, or release proof."
        ),
    }
    postprocess_reference_path = output_directory / "ultralytics-pytorch-reference.json"
    write_json(postprocess_reference_path, postprocess_reference)
    report = {
        "schemaVersion": 1,
        "recordKind": "yolovision-yolov8n-detection-independent-reference-report",
        "weightsSha256": sha256(args.weights.resolve()),
        "onnxSha256": sha256(args.onnx.resolve()),
        "labelsSha256": sha256(labels_path),
        "imageSha256": sha256(args.image.resolve()),
        "ppmSha256": sha256(ppm_path),
        "ultralyticsInputTensorSha256": sha256(tensor_path),
        "runtimeInputTensorSha256": (
            sha256(args.csharp_tensor.resolve()) if args.csharp_tensor else sha256(tensor_path)
        ),
        "canonicalOutputTensorSha256": sha256(canonical_output_path),
        "outputTensorSha256": sha256(ort_output_path),
        "rawReferenceSha256": sha256(reference_path),
        "tamperedReferenceSha256": sha256(tampered_reference_path),
        "postprocessReferenceSha256": sha256(postprocess_reference_path),
        "canonicalImageReferenceSha256": sha256(canonical_reference_path),
        "inputShape": [1, 3, 640, 640],
        "outputShape": [1, 84, 8400],
        "classCount": 80,
        "hasObjectness": False,
        "layout": "channels-first",
        "pytorchOnnxRuntimeMaximumAbsoluteError": pytorch_maximum_absolute_error,
        "csharpTensorComparison": csharp_input_report,
        "predictionCount": len(predictions),
        "predictions": predictions,
        "canonicalImagePipelinePredictionCount": len(canonical_predictions),
        "canonicalImagePipelinePredictions": canonical_predictions,
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxRuntime": ort.__version__,
        },
    }
    report_path = output_directory / "independent-reference-report.json"
    write_json(report_path, report)
    print(f"ReferenceReport={report_path}")
    print(f"RawReference={reference_path}")
    print(f"PostprocessReference={postprocess_reference_path}")
    print(f"Predictions={len(predictions)}")
    print(f"PyTorchOnnxRuntimeMaximumAbsoluteError={pytorch_maximum_absolute_error:.9g}")
    if args.actual_output:
        comparison, comparison_path = compare_actual(
            args.actual_output.resolve(),
            postprocess_reference,
            postprocess_reference_path,
            args.maximum_score_error,
            args.minimum_box_iou,
        )
        print(f"Comparison={comparison_path}")
        print(f"ComparisonPassed={comparison['passed']}")
        for item in comparison["comparisons"]:
            print(
                f"ComparisonPrediction={item.get('className')} Passed={item['passed']} "
                f"BoxIoU={item.get('boxIoU', 0.0):.6f} ScoreError={item.get('scoreAbsoluteError', 0.0):.6f}"
            )
        return 0 if comparison["passed"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
