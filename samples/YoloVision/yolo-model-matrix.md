# YoloVision Model Matrix

`samples/YoloVision/yolo-model-matrix.json` is the machine-readable support matrix for the unified YOLO sample. It covers YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv26, detection-only YOLOX, and custom models across `det`, `cls`, `seg`, `obb`, `pose`, and `sem`.

This matrix is not `real-model-runtime` proof. Real promotion still requires owner-provided ONNX, labels, input assets, license notes, SHA256 values, a TensorRtExec/OnnxToEngine build report, a `YoloVision Passed=True` runner log, and a sample-run-evidence validator result.

## Families

| Family | Tasks | Status |
| --- | --- | --- |
| YOLOv5 | det, cls, seg | Managed postprocess ready; requires owner assets. |
| YOLOv6 | det | Managed postprocess ready; requires owner assets. |
| YOLOv7 | det, pose | Pose requires auxiliary metadata. |
| YOLOv8 | det, cls, seg, obb, pose | Official YOLOv8n detection, classification, segmentation, embedded pose, and embedded-angle OBB source-tree proofs. |
| YOLOv9 | det, seg | Planned runtime proof; managed decode surface available. |
| YOLOv10 | det | Managed `[1,N,6]` xyxy/score/classId decoder ready; owner assets and runtime proof still required. |
| YOLOv11 | det, cls, seg, obb, pose | Planned runtime proof; matrix-ready. |
| YOLOv26 | det, cls, seg, obb, pose, sem | Future-family planning until concrete owner assets exist. |
| custom | det, cls, seg, obb, pose, sem | Metadata-driven custom path. |
| yolox | det | Official raw grid/stride decode; non-detection tasks are an explicit unsupported boundary. |

## Required Evidence

- Model source URL, license, export command, and SHA256.
- Labels source, license, class count, and SHA256.
- Input image or preprocessed tensor source, license, and SHA256.
- TensorRtExec or OnnxToEngine build-only report.
- YoloVision runner log and sample-run-evidence record.
