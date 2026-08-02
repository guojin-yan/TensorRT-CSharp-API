#!/usr/bin/env python3
"""Generate independent YOLOv8 pose references and compare YoloVision output."""

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


def box_iou(left: list[float], right: list[float]) -> float:
    left_x1 = left[0] - left[2] / 2.0
    left_y1 = left[1] - left[3] / 2.0
    left_x2 = left[0] + left[2] / 2.0
    left_y2 = left[1] + left[3] / 2.0
    right_x1 = right[0] - right[2] / 2.0
    right_y1 = right[1] - right[3] / 2.0
    right_x2 = right[0] + right[2] / 2.0
    right_y2 = right[1] + right[3] / 2.0
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1 = min(max(box[0] - box[2] / 2.0, 0.0), float(width))
    y1 = min(max(box[1] - box[3] / 2.0, 0.0), float(height))
    x2 = min(max(box[0] + box[2] / 2.0, 0.0), float(width))
    y2 = min(max(box[1] + box[3] / 2.0, 0.0), float(height))
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def generate_raw_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import numpy as np
    import onnxruntime as ort

    model_path = Path(args.onnx_model).resolve()
    tensor_path = Path(args.input_tensor).resolve()
    values = np.fromfile(tensor_path, dtype=np.float32)
    expected_elements = math.prod(args.input_shape)
    if values.size != expected_elements:
        raise ValueError(
            f"Input tensor contains {values.size} values; expected {expected_elements}."
        )

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    if session.get_providers() != providers:
        raise RuntimeError(f"ONNX Runtime providers are not CPU-only: {session.get_providers()}.")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The pinned pose contract requires exactly one input and one output.")
    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    if input_metadata.name != args.input_name or output_metadata.name != args.output_name:
        raise ValueError("ONNX tensor names do not match the declared pose contract.")

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
    reference_path = Path(args.output_directory).resolve() / f"{args.output_name}.reference.json"
    write_json(reference_path, reference, compact=True)

    metadata = {
        "schemaVersion": 1,
        "recordKind": "yolovision-pose-onnxruntime-reference-metadata",
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
        "boundary": (
            "Independent CPU raw-output reference; not Owner acceptance, redistribution approval, "
            "package-consumer proof, post-publish proof, or release proof."
        ),
    }
    metadata_path = Path(args.output_directory).resolve() / "onnxruntime-reference-metadata.json"
    write_json(metadata_path, metadata)
    return metadata, metadata_path


def generate_ultralytics_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
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
    if result.keypoints is None or len(result.boxes) != len(result.keypoints.data):
        raise RuntimeError("Ultralytics did not return one keypoint row for every retained box.")

    predictions: list[dict[str, Any]] = []
    keypoint_values = result.keypoints.data.detach().cpu().numpy()
    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        predictions.append(
            {
                "index": index,
                "classId": class_id,
                "className": str(result.names[class_id]),
                "score": float(result.boxes.conf[index].item()),
                "sourceBox": [float(value) for value in result.boxes.xywh[index].tolist()],
                "keypoints": [
                    {
                        "index": keypoint_index,
                        "x": float(row[0]),
                        "y": float(row[1]),
                        "score": float(row[2]),
                    }
                    for keypoint_index, row in enumerate(keypoint_values[index])
                ],
            }
        )

    reference = {
        "schemaVersion": 1,
        "recordKind": "yolovision-independent-pose-postprocess-reference",
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
        "keypointCount": args.keypoint_count,
        "predictions": predictions,
        "boundary": (
            "Independent framework/runtime pose postprocess reference; not Owner acceptance, "
            "redistribution approval, package-consumer proof, post-publish proof, or release proof."
        ),
    }
    reference_path = Path(args.output_directory).resolve() / "ultralytics-pytorch-reference.json"
    write_json(reference_path, reference)
    return reference, reference_path


def actual_prediction_in_source_space(
    prediction: dict[str, Any],
    report: dict[str, Any],
) -> tuple[list[float], list[dict[str, float]]]:
    image = report["input"]["image"]
    letterbox = report["input"]["letterbox"]
    width = int(image["width"])
    height = int(image["height"])
    scale_x = float(letterbox["scaleX"])
    scale_y = float(letterbox["scaleY"])
    pad_x = float(letterbox["padX"])
    pad_y = float(letterbox["padY"])
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("Actual report letterbox scales must be positive.")

    box = prediction["box"]
    source_box = clip_box(
        [
            (float(box["x"]) - pad_x) / scale_x,
            (float(box["y"]) - pad_y) / scale_y,
            float(box["width"]) / scale_x,
            float(box["height"]) / scale_y,
        ],
        width,
        height,
    )
    keypoints = [
        {
            "x": min(max((float(item["x"]) - pad_x) / scale_x, 0.0), float(width)),
            "y": min(max((float(item["y"]) - pad_y) / scale_y, 0.0), float(height)),
            "score": float(item["score"]),
        }
        for item in prediction["keypoints"]
    ]
    return source_box, keypoints


def compare_actual(
    args: argparse.Namespace,
    reference: dict[str, Any],
    reference_path: Path,
) -> tuple[dict[str, Any], Path]:
    actual_path = Path(args.actual_output).resolve()
    with actual_path.open("r", encoding="utf-8-sig") as stream:
        actual_report: dict[str, Any] = json.load(stream)
    if actual_report.get("schemaVersion") != "yolovision-output.v1" or actual_report.get("task") != "pose":
        raise ValueError("Actual output must be a yolovision-output.v1 pose report.")
    boundary = actual_report.get("boundary", {})
    if boundary.get("isRuntimeProof") is not False:
        raise ValueError("Actual output report must retain its readonly proof boundary.")

    actual_predictions = [
        prediction for prediction in actual_report.get("predictions", []) if prediction.get("task") == "pose"
    ]
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
                    "matched": False,
                    "passed": False,
                    "diagnostic": "No unmatched YoloVision pose has the same classId.",
                }
            )
            continue

        source_values = {
            index: actual_prediction_in_source_space(actual_predictions[index], actual_report)
            for index in candidates
        }
        actual_index = max(
            candidates,
            key=lambda index: box_iou(expected["sourceBox"], source_values[index][0]),
        )
        unmatched.remove(actual_index)
        actual_box, actual_keypoints = source_values[actual_index]
        expected_keypoints = expected["keypoints"]
        if len(actual_keypoints) != args.keypoint_count or len(expected_keypoints) != args.keypoint_count:
            raise ValueError("Pose keypoint count does not match the declared contract.")

        keypoint_score_errors = [
            abs(float(left["score"]) - float(right["score"]))
            for left, right in zip(expected_keypoints, actual_keypoints)
        ]
        visible_coordinate_errors = [
            math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
            for left, right in zip(expected_keypoints, actual_keypoints)
            if float(left["score"]) >= args.minimum_keypoint_score
        ]
        overlap = box_iou(expected["sourceBox"], actual_box)
        score_error = abs(float(expected["score"]) - float(actual_predictions[actual_index]["score"]))
        maximum_keypoint_coordinate_error = max(visible_coordinate_errors, default=0.0)
        maximum_keypoint_score_error = max(keypoint_score_errors, default=0.0)
        passed = (
            overlap >= args.minimum_box_iou
            and score_error <= args.maximum_score_error
            and maximum_keypoint_coordinate_error <= args.maximum_keypoint_coordinate_error
            and maximum_keypoint_score_error <= args.maximum_keypoint_score_error
        )
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
                "visibleKeypointCount": len(visible_coordinate_errors),
                "maximumKeypointCoordinateError": maximum_keypoint_coordinate_error,
                "maximumKeypointScoreError": maximum_keypoint_score_error,
                "passed": passed,
            }
        )

    passed = (
        len(comparisons) == len(actual_predictions)
        and not unmatched
        and all(item["passed"] for item in comparisons)
    )
    comparison = {
        "schemaVersion": 1,
        "recordKind": "yolovision-pose-independent-reference-comparison",
        "referencePath": str(reference_path),
        "referenceSha256": sha256_file(reference_path),
        "actualOutputPath": str(actual_path),
        "actualOutputSha256": sha256_file(actual_path),
        "thresholds": {
            "minimumBoxIoU": args.minimum_box_iou,
            "maximumScoreError": args.maximum_score_error,
            "minimumKeypointScore": args.minimum_keypoint_score,
            "maximumKeypointCoordinateError": args.maximum_keypoint_coordinate_error,
            "maximumKeypointScoreError": args.maximum_keypoint_score_error,
        },
        "referencePredictionCount": len(reference["predictions"]),
        "actualPredictionCount": len(actual_predictions),
        "unmatchedActualIndices": sorted(unmatched),
        "comparisons": comparisons,
        "completed": True,
        "passed": passed,
        "boundary": (
            "Independent source-tree postprocess comparison; not Owner acceptance, package-consumer "
            "proof, post-publish proof, redistribution approval, or release proof."
        ),
    }
    comparison_path = Path(args.output_directory).resolve() / "yolovision-independent-comparison.json"
    write_json(comparison_path, comparison)
    return comparison, comparison_path


def parse_shape(value: str) -> list[int]:
    values = [int(token) for token in value.lower().replace("x", ",").split(",") if token]
    if not values or any(dimension <= 0 for dimension in values):
        raise argparse.ArgumentTypeError("shape must contain positive dimensions")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-tensor", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--actual-output")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--input-shape", type=parse_shape, default=parse_shape("1x3x640x640"))
    parser.add_argument("--output-shape", type=parse_shape, default=parse_shape("1x56x8400"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--keypoint-count", type=int, default=17)
    parser.add_argument("--minimum-box-iou", type=float, default=0.98)
    parser.add_argument("--maximum-score-error", type=float, default=0.03)
    parser.add_argument("--minimum-keypoint-score", type=float, default=0.25)
    parser.add_argument("--maximum-keypoint-coordinate-error", type=float, default=5.0)
    parser.add_argument("--maximum-keypoint-score-error", type=float, default=0.03)
    args = parser.parse_args()
    for name in ("confidence", "iou_threshold", "minimum_box_iou", "minimum_keypoint_score"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.image_size <= 0 or args.max_detections <= 0 or args.keypoint_count <= 0:
        parser.error("image size, maximum detections, and keypoint count must be positive")
    for name in (
        "maximum_score_error",
        "maximum_keypoint_coordinate_error",
        "maximum_keypoint_score_error",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    config_directory = Path(args.output_directory).resolve() / ".ultralytics-config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))
    raw_metadata, raw_metadata_path = generate_raw_reference(args)
    pose_reference, pose_reference_path = generate_ultralytics_reference(args)
    print(f"RawReference={raw_metadata['referencePath']}")
    print(f"RawReferenceSha256={raw_metadata['referenceSha256']}")
    print(f"RawReferenceMetadata={raw_metadata_path}")
    print(f"PoseReference={pose_reference_path}")
    print(f"PoseReferenceSha256={sha256_file(pose_reference_path)}")
    print(f"PosePredictions={pose_reference['predictionCount']}")
    if not args.actual_output:
        return 0

    comparison, comparison_path = compare_actual(args, pose_reference, pose_reference_path)
    print(f"Comparison={comparison_path}")
    print(f"ComparisonPassed={comparison['passed']}")
    for item in comparison["comparisons"]:
        print(
            "ComparisonPrediction="
            f"{item.get('referenceIndex')} Passed={item['passed']} "
            f"BoxIoU={item.get('boxIoU', 0.0):.6f} "
            f"KeypointError={item.get('maximumKeypointCoordinateError', 0.0):.6f}"
        )
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
