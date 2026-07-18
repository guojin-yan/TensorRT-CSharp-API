# YoloVision YOLOv8n Semantic Segmentation Map 输出指南

本文面向需要把语义分割模型纳入 `YoloVision` 样例和后续公众号/博客案例的开发者。目标是讲清楚 semantic segmentation 的模型获取、ONNX 导出、TensorRtExec build-only 记录、YoloVision 运行命令、输出 metadata 和 proof 边界。

## 场景

语义分割与 detection/instance segmentation 不同：输出通常是每个像素的类别 logits 或类别索引图，而不是 box、mask coefficients 或 prototype。`YoloVision` 的 `sem` 任务把这个输出归一到 `SemanticMap`，并要求 owner 明确 `classCount`、`semanticMapShape`、`classMapLayout`、palette 和 void class 策略。

## 模型与资产

建议 owner 准备：

- `models/yolov8n-sem.pt` 或兼容 YOLOv8 semantic segmentation 的权重。
- `models/semantic-classes.names`，每行一个类别名。
- `models/semantic-palette.json`，记录类别到 RGB 颜色的映射。
- 一张可公开授权的测试图片。
- 预处理后的 `models/yolov8n-sem-fp32.bin`。
- 所有资产的 SHA256、来源 URL、许可证和 owner review 记录。

## 导出 ONNX

示例命令：

```powershell
yolo export model=.\models\yolov8n-sem.pt format=onnx opset=12 dynamic=True simplify=True imgsz=640
```

导出后记录：

- ONNX 文件 SHA256。
- 原始权重 SHA256。
- export stdout/stderr 摘要。
- opset、dynamic、imgsz 和 simplify 参数。

## 构建 Engine 记录

`TensorRtExec` 可用于生成 build-only 报告：

```powershell
dotnet run --project .\applications\TensorRtExec -- --onnx .\models\yolov8n-sem.onnx --saveEngine .\models\yolov8n-sem.plan --minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:4x3x640x640 --fp16 --buildOnly --exportReport .\models\yolov8n-sem-build-report.json
```

这个报告只说明构建意图、shape profile、precision intent 和输出 artifact 路径。它不是 runtime proof，也不是 package-consumer-runtime proof。

## YoloVision 离线 Preflight

语义分割的 class map、palette 和输出 shape 需要 owner 确认；先生成离线预检报告：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-sem.onnx --labels .\models\semantic-classes.names --input-data .\models\yolov8n-sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count owner-required --semantic-map-shape owner-required --class-map-layout NCHW-logits-or-NHW-class-index-owner-confirmed --palette .\models\semantic-palette.json --preflight --preflight-report .\models\yolov8n-sem-preflight.json
```

该报告只允许 `yolovision-preflight.v1`/`proofClassification=precheck`，并要求所有 execution 与 promotion flag 为 `false`；它不能替代真实 semantic map 输出和 owner review。

## 运行 YoloVision

示例命令：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-sem.onnx --labels .\models\semantic-classes.names --input-data .\models\yolov8n-sem-fp32.bin --input-shape 1x3x640x640 --family custom --task sem --semantic-output semantic --class-count owner-required --semantic-map-shape owner-required --class-map-layout NCHW-logits-or-NHW-class-index-owner-confirmed --palette .\models\semantic-palette.json
```

真实 owner 回填日志中至少应出现以下 expected evidence lines；在日志、hash、host metadata 和 owner review 完成前，这些行只是待采集证据，不是当前 runtime proof：

```text
Profile Family=custom Task=sem
SemanticMap Classes=...
YoloVision Passed=<owner-confirmed-true>
```

## 输出 Metadata

owner 必须确认：

- `classCount`：类别数量。
- `semanticOutputRole`：语义图输出 tensor 的角色名。
- `semanticMapShape`：例如 `1xCxHxW` 或 `1xHxW`。
- `classMapLayout`：例如 `NCHW-logits` 或 `NHW-class-index`。
- `paletteSha256`：palette 文件 SHA256。
- `voidClassPolicy`：忽略类别、背景类别或无效像素处理方式。

这些字段已经进入 `samples/assets/yolovision-article-case-pack.json` 的 `yolov8n-sem` case，可作为文章撰写和 owner backfill 的统一来源。

## Proof Boundary

本文、`samples/assets/yolovision-article-case-pack.json`、TensorRtExec build-only report、YoloVision matrix、sidecar-only report、dry-run、local feed、ProjectReference 和 direct `.nupkg` 都不是 runtime proof。

只有在 owner 提供真实模型、labels、palette、输入图片、预处理 tensor、运行日志、输出 JSON、SHA256、stdout/stderr 摘要，并通过对应 validator 后，样例证据才可以晋级 `real-model-runtime`。它仍然不能替代 `package-consumer-runtime` proof。

## Owner Backfill Checklist

- 填写模型来源、许可证和 SHA256。
- 填写 labels、palette、输入图片和预处理 tensor SHA256。
- 保存 TensorRtExec build-only report 和 engine SHA256。
- 保存 YoloVision run log，并确认 `YoloVision Passed=True`。
- 保存 output JSON 和 semantic map 摘要。
- 用 owner proof input/template 或 sample-run-evidence validator 做严格验证。
