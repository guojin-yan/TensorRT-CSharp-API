#!/usr/bin/env python3
"""Generate independent YOLOv8 OBB references and compare YoloVision output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any], compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        if compact:
            json.dump(value, stream, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        else:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def generate_raw_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import numpy as np
    import onnxruntime as ort

    model_path = Path(args.onnx_model).resolve()
    tensor_path = Path(args.input_tensor).resolve()
    values = np.fromfile(tensor_path, dtype=np.float32)
    if values.size != math.prod(args.input_shape):
        raise ValueError("Input tensor element count does not match --input-shape.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"ONNX Runtime providers are not CPU-only: {session.get_providers()}.")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The pinned OBB contract requires exactly one input and one output.")
    if session.get_inputs()[0].name != args.input_name or session.get_outputs()[0].name != args.output_name:
        raise ValueError("ONNX tensor names do not match the declared OBB contract.")

    output = session.run(
        [args.output_name],
        {args.input_name: values.reshape(tuple(args.input_shape))},
    )[0].astype(np.float32, copy=False)
    if list(output.shape) != args.output_shape:
        raise ValueError(f"ONNX output shape {list(output.shape)} does not match {args.output_shape}.")
    if not np.all(np.isfinite(output)):
        raise ValueError("ONNX Runtime output contains non-finite values.")

    reference = {
        "schemaVersion": 1,
        "tensorName": args.output_name,
        "sourceClassification": "independent-onnxruntime-cpu-execution-provider",
        "shape": [int(value) for value in output.shape],
        "values": output.reshape(-1).tolist(),
    }
    output_directory = Path(args.output_directory).resolve()
    reference_path = output_directory / f"{args.output_name}.reference.json"
    write_json(reference_path, reference, compact=True)
    metadata = {
        "schemaVersion": 1,
        "recordKind": "yolovision-obb-onnxruntime-reference-metadata",
        "sourceClassification": "independent-onnxruntime-cpu-execution-provider",
        "modelPath": str(model_path),
        "modelSha256": sha256_file(model_path),
        "inputTensorPath": str(tensor_path),
        "inputTensorSha256": sha256_file(tensor_path),
        "inputName": args.input_name,
        "inputShape": args.input_shape,
        "outputName": args.output_name,
        "outputShape": [int(value) for value in output.shape],
        "outputElementCount": int(output.size),
        "referencePath": str(reference_path),
        "referenceSha256": sha256_file(reference_path),
        "runtime": {
            "framework": "ONNX Runtime",
            "version": ort.__version__,
            "providers": session.get_providers(),
            "pythonVersion": platform.python_version(),
            "platform": platform.platform(),
        },
        "boundary": "Independent CPU raw-output reference; not redistribution, package-consumer, post-publish, or release proof.",
    }
    metadata_path = output_directory / "onnxruntime-reference-metadata.json"
    write_json(metadata_path, metadata)
    return metadata, metadata_path


def generate_ultralytics_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights_path = Path(args.weights).resolve()
    image_path = Path(args.image).resolve()
    result = YOLO(str(weights_path)).predict(
        source=str(image_path),
        imgsz=args.image_size,
        conf=args.confidence,
        iou=args.iou_threshold,
        max_det=args.max_detections,
        agnostic_nms=False,
        rect=False,
        device="cpu",
        verbose=True,
    )[0]
    if result.obb is None:
        raise RuntimeError("Ultralytics did not return OBB predictions.")

    predictions: list[dict[str, Any]] = []
    for index in range(len(result.obb)):
        class_id = int(result.obb.cls[index].item())
        predictions.append(
            {
                "index": index,
                "classId": class_id,
                "className": str(result.names[class_id]),
                "score": float(result.obb.conf[index].item()),
                "sourceXywhr": [float(value) for value in result.obb.xywhr[index].tolist()],
            }
        )

    output_directory = Path(args.output_directory).resolve()
    annotated_path = output_directory / "ultralytics-pytorch-reference.png"
    if not cv2.imwrite(str(annotated_path), result.plot()):
        raise RuntimeError(f"Unable to write annotated OBB reference '{annotated_path}'.")
    reference = {
        "schemaVersion": 1,
        "recordKind": "yolovision-independent-obb-postprocess-reference",
        "sourceClassification": "independent-ultralytics-pytorch-cpu-reference",
        "model": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        "input": {
            "imagePath": str(image_path),
            "imageSha256": sha256_file(image_path),
            "originalShape": [int(value) for value in result.orig_shape],
            "imageSize": args.image_size,
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
            "pythonVersion": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
        },
        "predictionCount": len(predictions),
        "predictions": predictions,
        "annotatedImage": {"path": str(annotated_path), "sha256": sha256_file(annotated_path)},
        "boundary": "Independent framework/runtime OBB reference; not redistribution, package-consumer, post-publish, or release proof.",
    }
    reference_path = output_directory / "ultralytics-pytorch-reference.json"
    write_json(reference_path, reference)
    return reference, reference_path


def actual_box_in_source_space(prediction: dict[str, Any], report: dict[str, Any]) -> list[float]:
    letterbox = report["input"]["letterbox"]
    scale_x = float(letterbox["scaleX"])
    scale_y = float(letterbox["scaleY"])
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("Actual report letterbox scales must be positive.")
    return [
        (float(prediction["center"]["x"]) - float(letterbox["padX"])) / scale_x,
        (float(prediction["center"]["y"]) - float(letterbox["padY"])) / scale_y,
        float(prediction["size"]["width"]) / scale_x,
        float(prediction["size"]["height"]) / scale_y,
        float(prediction["angle"]),
    ]


def rotated_iou(first: list[float], second: list[float]) -> float:
    import cv2

    first_rect = ((first[0], first[1]), (first[2], first[3]), math.degrees(first[4]))
    second_rect = ((second[0], second[1]), (second[2], second[3]), math.degrees(second[4]))
    _, points = cv2.rotatedRectangleIntersection(first_rect, second_rect)
    intersection = 0.0 if points is None else abs(float(cv2.contourArea(points)))
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def periodic_angle_error(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def compare_actual(
    args: argparse.Namespace,
    reference: dict[str, Any],
    reference_path: Path,
) -> tuple[dict[str, Any], Path]:
    actual_path = Path(args.actual_output).resolve()
    with actual_path.open("r", encoding="utf-8-sig") as stream:
        actual_report: dict[str, Any] = json.load(stream)
    if actual_report.get("schemaVersion") != "yolovision-output.v1" or actual_report.get("task") != "obb":
        raise ValueError("Actual output must be a yolovision-output.v1 OBB report.")
    if actual_report.get("boundary", {}).get("isRuntimeProof") is not False:
        raise ValueError("Actual output report must retain its readonly proof boundary.")

    actual_predictions = list(actual_report.get("predictions", []))
    unmatched = set(range(len(actual_predictions)))
    comparisons: list[dict[str, Any]] = []
    for expected in reference["predictions"]:
        expected_box = [float(value) for value in expected["sourceXywhr"]]
        candidates = [
            index for index in unmatched
            if int(actual_predictions[index].get("classId", -1)) == int(expected["classId"])
        ]
        if not candidates:
            comparisons.append({
                "referenceIndex": expected["index"],
                "classId": expected["classId"],
                "matched": False,
                "passed": False,
                "diagnostic": "No unmatched YoloVision prediction has the same classId.",
            })
            continue

        actual_boxes = {
            index: actual_box_in_source_space(actual_predictions[index], actual_report)
            for index in candidates
        }
        actual_index = max(candidates, key=lambda index: rotated_iou(expected_box, actual_boxes[index]))
        unmatched.remove(actual_index)
        actual = actual_predictions[actual_index]
        actual_box = actual_boxes[actual_index]
        overlap = rotated_iou(expected_box, actual_box)
        coordinate_errors = [abs(expected_box[index] - actual_box[index]) for index in range(4)]
        angle_error = periodic_angle_error(expected_box[4], actual_box[4])
        score_error = abs(float(expected["score"]) - float(actual["score"]))
        passed = (
            overlap >= args.minimum_rotated_iou
            and max(coordinate_errors) <= args.maximum_coordinate_error
            and angle_error <= args.maximum_angle_error
            and score_error <= args.maximum_score_error
        )
        comparisons.append({
            "referenceIndex": expected["index"],
            "actualIndex": actual_index,
            "classId": expected["classId"],
            "className": expected["className"],
            "matched": True,
            "referenceXywhr": expected_box,
            "actualXywhr": actual_box,
            "rotatedIoU": overlap,
            "coordinateAbsoluteErrors": coordinate_errors,
            "maximumCoordinateAbsoluteError": max(coordinate_errors),
            "anglePeriodicAbsoluteErrorRadians": angle_error,
            "scoreAbsoluteError": score_error,
            "passed": passed,
        })

    passed = (
        len(comparisons) == len(actual_predictions)
        and not unmatched
        and all(item["passed"] for item in comparisons)
    )
    comparison = {
        "schemaVersion": 1,
        "recordKind": "yolovision-obb-independent-reference-comparison",
        "referencePath": str(reference_path),
        "referenceSha256": sha256_file(reference_path),
        "actualOutputPath": str(actual_path),
        "actualOutputSha256": sha256_file(actual_path),
        "thresholds": {
            "minimumRotatedIoU": args.minimum_rotated_iou,
            "maximumCoordinateError": args.maximum_coordinate_error,
            "maximumAngleErrorRadians": args.maximum_angle_error,
            "maximumScoreError": args.maximum_score_error,
        },
        "referencePredictionCount": len(reference["predictions"]),
        "actualPredictionCount": len(actual_predictions),
        "unmatchedActualIndices": sorted(unmatched),
        "comparisons": comparisons,
        "completed": True,
        "passed": passed,
        "boundary": "Independent OBB postprocess comparison; not redistribution, package-consumer, post-publish, or release proof.",
    }
    comparison_path = Path(args.output_directory).resolve() / "yolovision-independent-comparison.json"
    write_json(comparison_path, comparison)
    return comparison, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--input-tensor", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--actual-output")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 1024, 1024])
    parser.add_argument("--output-shape", type=int, nargs="+", default=[1, 20, 21504])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--minimum-rotated-iou", type=float, default=0.98)
    parser.add_argument("--maximum-coordinate-error", type=float, default=5.0)
    parser.add_argument("--maximum-angle-error", type=float, default=0.02)
    parser.add_argument("--maximum-score-error", type=float, default=0.03)
    args = parser.parse_args()
    for name in ("confidence", "iou_threshold", "minimum_rotated_iou"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.image_size <= 0 or args.max_detections <= 0:
        parser.error("image size and maximum detections must be positive")
    if min(args.maximum_coordinate_error, args.maximum_angle_error, args.maximum_score_error) < 0.0:
        parser.error("comparison tolerances must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    output_directory = Path(args.output_directory).resolve()
    config_directory = output_directory / ".ultralytics-config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))
    raw_metadata, raw_metadata_path = generate_raw_reference(args)
    reference, reference_path = generate_ultralytics_reference(args)
    print(f"RawReference={raw_metadata['referencePath']}")
    print(f"RawReferenceSha256={raw_metadata['referenceSha256']}")
    print(f"RawOutputElements={raw_metadata['outputElementCount']}")
    print(f"RawMetadata={raw_metadata_path}")
    print(f"PostprocessReference={reference_path}")
    print(f"Predictions={reference['predictionCount']}")
    if not args.actual_output:
        return 0
    comparison, comparison_path = compare_actual(args, reference, reference_path)
    overlaps = [item.get("rotatedIoU", 0.0) for item in comparison["comparisons"]]
    angle_errors = [item.get("anglePeriodicAbsoluteErrorRadians", 0.0) for item in comparison["comparisons"]]
    print(f"Comparison={comparison_path}")
    print(f"ComparisonPassed={comparison['passed']}")
    print(f"MinimumRotatedIoU={min(overlaps, default=0.0):.6f}")
    print(f"MaximumAngleErrorRadians={max(angle_errors, default=0.0):.6f}")
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
