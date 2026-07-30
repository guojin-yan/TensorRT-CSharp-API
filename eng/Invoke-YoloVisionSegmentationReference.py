#!/usr/bin/env python3
"""Generate an independent Ultralytics/PyTorch segmentation reference and compare YoloVision masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def sanitize(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized or "class"


def box_iou(left: list[float], right: list[float]) -> float:
    def corners(box: list[float]) -> tuple[float, float, float, float]:
        x, y, width, height = box
        return x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0

    left_x1, left_y1, left_x2, left_y2 = corners(left)
    right_x1, right_y1, right_x2, right_y2 = corners(right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    union += max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union -= intersection
    return intersection / union if union > 0.0 else 0.0


def box_values(box: dict[str, Any]) -> list[float]:
    values = [float(box[name]) for name in ("x", "y", "width", "height")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Box coordinates must be finite.")
    return values


def read_thresholded_mask(
    artifact: dict[str, Any],
    shape: tuple[int, ...],
    expected_role: str | None = None,
) -> Any:
    import numpy as np

    if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Thresholded mask shape is invalid: {shape}.")
    if artifact.get("dataType") != "uint8-0-or-1":
        raise ValueError("Thresholded mask dataType must be uint8-0-or-1.")
    if expected_role is not None and artifact.get("role") != expected_role:
        raise ValueError(f"Thresholded mask role must be {expected_role}.")

    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Thresholded mask path is missing.")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"Thresholded mask file does not exist: '{path}'.")

    expected_elements = math.prod(shape)
    declared_elements = int(artifact.get("elementCount", expected_elements))
    declared_bytes = int(artifact.get("byteLength", -1))
    if declared_elements != expected_elements:
        raise ValueError("Thresholded mask elementCount does not match shape.")
    if declared_bytes != expected_elements or path.stat().st_size != expected_elements:
        raise ValueError("Thresholded mask byteLength does not match its uint8 shape or file size.")

    declared_sha256 = artifact.get("sha256")
    if not isinstance(declared_sha256, str) or sha256_file(path) != declared_sha256.lower():
        raise ValueError("Thresholded mask SHA256 does not match the manifest.")

    values = np.fromfile(path, dtype=np.uint8)
    if values.size != expected_elements:
        raise ValueError("Thresholded mask value count does not match shape.")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("Thresholded mask values must be exactly 0 or 1.")
    active_pixel_count = int(np.count_nonzero(values))
    if "activePixelCount" in artifact and int(artifact["activePixelCount"]) != active_pixel_count:
        raise ValueError("Thresholded mask activePixelCount does not match the file.")
    return values


def generate_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    output_directory = Path(args.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise RuntimeError(f"OpenCV could not decode input image '{image_path}'.")
    result = YOLO(str(model_path)).predict(
        source=source_image,
        imgsz=args.image_size,
        conf=args.confidence,
        iou=args.iou_threshold,
        max_det=args.max_detections,
        agnostic_nms=args.agnostic_nms,
        retina_masks=True,
        rect=False,
        device="cpu",
        verbose=True,
    )[0]
    if result.masks is None or len(result.boxes) != len(result.masks.data):
        raise RuntimeError("Ultralytics did not return one mask for every retained box.")

    masks = result.masks.data.detach().cpu().numpy()
    predictions: list[dict[str, Any]] = []
    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        class_name = str(result.names[class_id])
        thresholded = (masks[index] >= args.mask_threshold).astype(np.uint8)
        file_name = f"prediction-{index:02d}-class-{class_id}-{sanitize(class_name)}-source-thresholded.u8.bin"
        file_path = output_directory / file_name
        thresholded.tofile(file_path)
        predictions.append(
            {
                "index": index,
                "classId": class_id,
                "className": class_name,
                "score": float(result.boxes.conf[index].item()),
                "sourceBox": [float(value) for value in result.boxes.xywh[index].tolist()],
                "maskShape": [int(value) for value in thresholded.shape],
                "maskThreshold": args.mask_threshold,
                "activePixelCount": int(np.count_nonzero(thresholded)),
                "thresholdedMask": {
                    "fileName": file_name,
                    "path": str(file_path),
                    "byteLength": int(file_path.stat().st_size),
                    "sha256": sha256_file(file_path),
                    "dataType": "uint8-0-or-1",
                },
            }
        )

    annotated_path = output_directory / "ultralytics-pytorch-reference.png"
    if not cv2.imwrite(str(annotated_path), result.plot()):
        raise RuntimeError(f"Unable to write annotated reference image '{annotated_path}'.")

    reference = {
        "schemaVersion": 1,
        "recordKind": "yolovision-independent-segmentation-postprocess-reference",
        "sourceClassification": "independent-ultralytics-pytorch-cpu-reference",
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "input": {
            "imagePath": str(image_path),
            "imageSha256": sha256_file(image_path),
            "originalShape": [int(value) for value in result.orig_shape],
            "imageSize": args.image_size,
            "confidenceThreshold": args.confidence,
            "iouThreshold": args.iou_threshold,
            "maxDetections": args.max_detections,
            "agnosticNms": args.agnostic_nms,
            "retinaMasks": True,
            "rect": False,
            "maskThreshold": args.mask_threshold,
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
        "annotatedImage": {
            "path": str(annotated_path),
            "sha256": sha256_file(annotated_path),
        },
        "boundary": (
            "Independent framework/runtime postprocess reference candidate; not Owner acceptance, "
            "redistribution approval, package-consumer proof, post-publish proof, or release proof."
        ),
    }
    reference_path = output_directory / "ultralytics-pytorch-reference.json"
    write_json(reference_path, reference)
    return reference, reference_path


def compare_actual(
    args: argparse.Namespace,
    reference: dict[str, Any],
    reference_path: Path,
) -> tuple[dict[str, Any], Path]:
    import numpy as np

    manifest_path = Path(args.actual_manifest).resolve()
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if manifest.get("schemaVersion") != "yolovision-segmentation-mask-artifacts.v1":
        raise ValueError("Actual mask manifest has an unsupported schemaVersion.")
    actual_predictions = list(manifest.get("predictions", []))
    if int(manifest.get("predictionCount", -1)) != len(actual_predictions):
        raise ValueError("Actual mask manifest predictionCount does not match predictions.")
    if manifest.get("spatialTransformApplied") is not True:
        raise ValueError("Actual mask manifest must contain source-image spatial transforms.")
    boundary = manifest.get("boundary", {})
    if any(
        boundary.get(name) is not False
        for name in ("isRuntimeProof", "isPackageConsumerProof", "isPostPublishProof")
    ):
        raise ValueError("Actual mask manifest proof boundary must remain non-promotable.")
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
                    "diagnostic": "No unmatched YoloVision prediction has the same classId.",
                }
            )
            continue

        expected_box = [float(value) for value in expected["sourceBox"]]
        actual_index = max(
            candidates,
            key=lambda index: box_iou(expected_box, box_values(actual_predictions[index]["sourceBox"])),
        )
        unmatched.remove(actual_index)
        actual = actual_predictions[actual_index]
        actual_box = box_values(actual["sourceBox"])
        coordinate_errors = [abs(left - right) for left, right in zip(expected_box, actual_box)]
        expected_score = float(expected["score"])
        actual_score = float(actual["score"])
        if not math.isfinite(expected_score) or not math.isfinite(actual_score):
            raise ValueError("Prediction scores must be finite.")
        score_error = abs(expected_score - actual_score)
        box_overlap = box_iou(expected_box, actual_box)

        expected_shape = tuple(int(value) for value in expected["maskShape"])
        actual_artifact = actual.get("sourceThresholded")
        if not actual_artifact:
            raise ValueError("Actual prediction is missing sourceThresholded artifacts.")
        actual_shape = tuple(int(value) for value in actual_artifact["shape"])
        expected_mask = read_thresholded_mask(expected["thresholdedMask"], expected_shape)
        actual_mask = read_thresholded_mask(
            actual_artifact,
            actual_shape,
            expected_role="source-image-thresholded",
        )
        shape_matches = expected_shape == actual_shape
        size_matches = expected_mask.size == actual_mask.size
        if shape_matches and size_matches:
            expected_active = expected_mask != 0
            actual_active = actual_mask != 0
            intersection = int(np.count_nonzero(expected_active & actual_active))
            union = int(np.count_nonzero(expected_active | actual_active))
            mask_iou = intersection / union if union else 1.0
            mismatch_count = int(np.count_nonzero(expected_active != actual_active))
        else:
            intersection = 0
            union = 0
            mask_iou = 0.0
            mismatch_count = max(expected_mask.size, actual_mask.size)

        passed = (
            shape_matches
            and size_matches
            and max(coordinate_errors) <= args.maximum_box_coordinate_error
            and score_error <= args.maximum_score_error
            and box_overlap >= args.minimum_box_iou
            and mask_iou >= args.minimum_mask_iou
        )
        comparisons.append(
            {
                "referenceIndex": expected["index"],
                "actualIndex": actual_index,
                "classId": expected["classId"],
                "className": expected["className"],
                "matched": True,
                "referenceBox": expected_box,
                "actualBox": actual_box,
                "boxCoordinateAbsoluteErrors": coordinate_errors,
                "maximumBoxCoordinateAbsoluteError": max(coordinate_errors),
                "boxIoU": box_overlap,
                "scoreAbsoluteError": score_error,
                "referenceMaskShape": list(expected_shape),
                "actualMaskShape": list(actual_shape),
                "maskIntersectionPixelCount": intersection,
                "maskUnionPixelCount": union,
                "maskMismatchPixelCount": mismatch_count,
                "maskIoU": mask_iou,
                "passed": passed,
            }
        )

    passed = len(comparisons) == len(actual_predictions) and not unmatched and all(
        comparison["passed"] for comparison in comparisons
    )
    comparison = {
        "schemaVersion": 1,
        "recordKind": "yolovision-segmentation-independent-reference-comparison",
        "referencePath": str(reference_path),
        "referenceSha256": sha256_file(reference_path),
        "actualManifestPath": str(manifest_path),
        "actualManifestSha256": sha256_file(manifest_path),
        "thresholds": {
            "maximumBoxCoordinateAbsoluteError": args.maximum_box_coordinate_error,
            "maximumScoreAbsoluteError": args.maximum_score_error,
            "minimumBoxIoU": args.minimum_box_iou,
            "minimumMaskIoU": args.minimum_mask_iou,
        },
        "referencePredictionCount": len(reference["predictions"]),
        "actualPredictionCount": len(actual_predictions),
        "unmatchedActualIndices": sorted(unmatched),
        "comparisons": comparisons,
        "completed": True,
        "passed": passed,
        "boundary": (
            "Independent postprocess comparison for source-tree runtime evidence; not Owner acceptance, "
            "package-consumer proof, post-publish proof, redistribution approval, or release proof."
        ),
    }
    comparison_path = Path(args.output_directory).resolve() / "yolovision-independent-comparison.json"
    write_json(comparison_path, comparison)
    return comparison, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--actual-manifest")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-box-coordinate-error", type=float, default=1.0)
    parser.add_argument("--maximum-score-error", type=float, default=0.01)
    parser.add_argument("--minimum-box-iou", type=float, default=0.995)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.99)
    args = parser.parse_args()
    for name in ("confidence", "iou_threshold", "mask_threshold", "minimum_box_iou", "minimum_mask_iou"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.image_size <= 0 or args.max_detections <= 0:
        parser.error("image size and maximum detections must be positive")
    if args.maximum_box_coordinate_error < 0.0 or args.maximum_score_error < 0.0:
        parser.error("maximum coordinate and score errors must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    config_directory = Path(args.output_directory).resolve() / ".ultralytics-config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))
    reference, reference_path = generate_reference(args)
    print(f"Reference={reference_path}")
    print(f"ReferenceSha256={sha256_file(reference_path)}")
    print(f"Predictions={len(reference['predictions'])}")
    if not args.actual_manifest:
        return 0
    comparison, comparison_path = compare_actual(args, reference, reference_path)
    print(f"Comparison={comparison_path}")
    print(f"ComparisonPassed={comparison['passed']}")
    for item in comparison["comparisons"]:
        print(
            "ComparisonPrediction="
            f"{item.get('className')} Passed={item['passed']} "
            f"BoxIoU={item.get('boxIoU', 0.0):.6f} "
            f"MaskIoU={item.get('maskIoU', 0.0):.6f}"
        )
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
