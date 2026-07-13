# YOLOv8 Segmentation 真实资产候选教程

本文是 `samples/YoloVision` 的 YOLOv8 segmentation 候选执行包说明。它强调 mask prototype、多输出 metadata 和 owner proof checklist。当前文件是 `owner-action-required` 教程与候选模板说明，不是 runtime proof。

候选模板：

```text
samples/assets/yolovision-yolov8-seg-candidate.template.json
```

## 资产准备

Owner 需要准备：

- YOLOv8 segmentation 模型来源 URL。
- 模型许可证与 license evidence。
- ONNX 导出命令、opset、input shape 和 SHA256。
- labels 文件、许可证和 SHA256。
- 可再分发测试图片和预处理 tensor。
- 输出 tensor 名称、shape、role map。
- mask coefficient count、prototype tensor shape、crop/scale rule。

建议导出命令示例：

```powershell
yolo export model=yolov8n-seg.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
```

## 构建 engine

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8-seg.onnx `
  --saveEngine .\models\yolov8-seg.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --exportReport .\models\yolov8-seg-build-report.json `
  --evidenceSidecar .\models\yolov8-seg-evidence.sidecar.json
```

该报告只能作为 build-only evidence。

## 运行 YoloVision

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8-seg.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task seg `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --aux-layout boxes-first `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45
```

## Evidence checklist

真实记录至少需要：

- owner 提供的 YoloVision 成功标记日志行：`YoloVision Passed=True`
- `Profile Family=v8 Task=Segmentation`
- `Output=... Outputs=...`
- `Postprocess Task=Segmentation;Detections=...;Segmentations=...`
- mask prototype tensor metadata
- model / labels / image / preprocessed tensor / run log 的 SHA256
- stdoutSummary / stderrSummary
- host OS、GPU、driver、CUDA、TensorRT、cuDNN

## Proof boundary

Segmentation 教程、模板、build report、sidecar 和 README 都不能替代 proof。`real-model-runtime` 需要真实 YoloVision run log 和 validator；`package-consumer-runtime` 需要 clean external consumer proof。
