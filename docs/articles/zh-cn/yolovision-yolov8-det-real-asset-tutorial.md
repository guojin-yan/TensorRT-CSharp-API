# YOLOv8 Detection 真实资产候选教程

本文是 `samples/YoloVision` 的 YOLOv8 detection 候选执行包说明。它提供模型获取、ONNX 导出、TensorRtExec build-only、YoloVision run command 和 proof checklist，但当前仍是 owner-action-required 模板，不是 runtime proof。

候选模板：

```text
samples/assets/yolovision-yolov8-det-candidate.template.json
```

## 资产准备

Owner 需要准备：

- YOLOv8 detection 模型来源 URL。
- 模型许可证与 license evidence。
- 导出的 ONNX 文件和 SHA256。
- labels 文件、来源、许可证、行数和 SHA256。
- 可再分发测试图片、许可证和 SHA256。
- 预处理后的 float32 tensor 和 SHA256。

建议导出命令示例：

```powershell
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
```

## 构建 engine

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8-det.onnx `
  --saveEngine .\models\yolov8-det.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --exportReport .\models\yolov8-det-build-report.json `
  --evidenceSidecar .\models\yolov8-det-evidence.sidecar.json
```

这一步是 build-only，不是 real-model-runtime proof，也不是 package-consumer-runtime proof。

## 运行 YoloVision

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8-det.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8-det-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45
```

## Evidence checklist

真实记录至少需要：

- owner 提供的 YoloVision 成功标记日志行：`YoloVision Passed=True`
- `Profile Family=v8 Task=Detection`
- `InputSource=external`
- `Postprocess Task=Detection;Detections=...`
- 检测行：`Detection Class=... Score=... BoxCxCyWh=...`
- model / labels / image / preprocessed tensor / run log 的 SHA256
- stdoutSummary / stderrSummary
- host OS、GPU、driver、CUDA、TensorRT、cuDNN

## Proof boundary

该教程和模板只能帮助形成 `real-model-runtime` 候选。`package-consumer-runtime` 必须来自公开包源、仓库外 clean consumer、runtime package key、exitCode=0、nativeAssetsCopied=true、日志 hash 和 strict validator。
