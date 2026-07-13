# YoloVision Detection 教程

本文讲 det 任务的接入路径：从 YOLO-family ONNX、labels 和输入图片开始，到 TensorRtExec build-only report，再到 YoloVision runner 和 sample-run-evidence。本文不提供伪造模型 hash，也不把示例命令写成真实 proof。

## 准备资产

owner 需要准备：

- YOLO detection ONNX，记录来源、export 命令、license。
- labels 文件，行数要和 class count 一致。
- input image，记录来源和授权。
- preprocess 说明：resize、letterbox、normalize、RGB/BGR。
- output metadata：layout、class count、objectness 是否存在。

## 推荐目录

```text
models/
  yolo-det.onnx
  labels.txt
  input.jpg
  yolo-det-build-report.json
  yolo-det-sidecar.json
  yolo-det-sample-run-evidence.json
```

## 构建 engine

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolo-det.onnx `
  --saveEngine .\models\yolo-det.engine `
  --fp16 `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --buildOnly `
  --exportProfile .\models\yolo-det-build-report.json
```

该 report 是 `build-only`。它可以帮助诊断模型转换，但不证明检测框正确。

## 运行 YoloVision

运行前要确认 profile：

- family：例如 v5/v8/v11/custom。
- task：det。
- layout：常见 `[1,84,8400]` 或 `[1,8400,84]`。
- objectness：不同家族可能有或没有。
- NMS：class-aware 或 class-agnostic。

```powershell
dotnet run --project .\samples\YoloVision -- `
  --engine .\models\yolo-det.engine `
  --labels .\models\labels.txt `
  --image .\models\input.jpg `
  --family v8 `
  --task det `
  --output-layout channels-first `
  --class-count 80 `
  --confidence-threshold 0.25 `
  --iou-threshold 0.45
```

## Evidence 回填

真实 evidence 至少需要：

- model/labels/input SHA256。
- TensorRtExec build-only report SHA256。
- YoloVision runner log SHA256。
- stdout/stderr summary。
- detection count、top detections 摘要。
- sample-run-evidence record。

sample-run-evidence 最多晋级 `real-model-runtime`，不能声明 `package-consumer-runtime`。

## 常见误区

- 只看到 engine 文件，不代表检测输出正确。
- layout 猜错会导致框坐标或 class score 错位。
- sidecar-only 不等于 runtime proof。
- `blocked-by-cuda-driver` 不是 API 缺失。
- `TrtexecAlignmentStatus=parse-only` 参数仍需要单独提升。

## 下一步

检测任务跑通后，再扩展到 seg、pose、obb 等任务。不要复用检测 metadata 去解释多输出模型；每个任务都应有独立 tensor role 和 sample evidence。
