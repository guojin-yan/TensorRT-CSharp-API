# YoloVision YOLOv8n Classification Labels 与 Top-K 指南

## 文章定位

本文面向希望用 `samples/YoloVision` 跑 YOLOv8n-cls 或类似 classification 模型的开发者。Classification 没有 box、NMS 和 mask，表面上比 detection 简单，但它对 labels 顺序、softmax 约定、Top-K 输出和输入尺寸非常敏感。

本文给出从模型来源、ONNX 导出、TensorRtExec build-only、YoloVision 运行到输出 JSON 的完整文章结构。它用于真实资产回填和公众号/博客发布，不是 runtime proof。

## 适用场景

当你需要验证一个分类模型是否可以通过 TensorRtSharp4.0 的 runtime package 被 C# 项目消费时，可以从 YOLOv8n-cls 开始。典型输入 shape 是 `1x3x224x224`，labels 通常来自 ImageNet 或自定义分类集。

这篇文章特别适合解释“为什么 labels 文件也是 proof 的一部分”。如果 labels 顺序错了，即使 logits 数值正确，最终类别名也会错。

## 模型与许可证

owner 需要记录：

- `yolov8n-cls.pt` 来源、许可证和 SHA256。
- ONNX 导出命令、opset、输入尺寸和 SHA256。
- labels 文件来源、许可证、class count 和 SHA256。
- 输入图片来源、许可证和 SHA256。
- 是否在模型图中已经包含 softmax。

如果模型输出是 logits，YoloVision 或上层应用要明确是否执行 softmax；如果模型输出已经是 probability，就不能重复 softmax 后再解释 Top-K。

## 导出 ONNX

```powershell
yolo export model=.\models\yolov8n-cls.pt format=onnx opset=12 dynamic=True simplify=True imgsz=224
```

导出后记录：

- input tensor name，例如 `images`。
- input shape，例如 `1x3x224x224`。
- output tensor name，例如 `logits`。
- class count。
- output score type：`logits` 或 `probabilities`。
- labels 文件顺序。

## TensorRtExec Build-Only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n-cls.onnx `
  --saveEngine .\models\yolov8n-cls.plan `
  --minShapes images:1x3x224x224 `
  --optShapes images:1x3x224x224 `
  --maxShapes images:8x3x224x224 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\yolov8n-cls-build-report.json
```

这一步验证 ONNX parser、profile 和 engine serialization。它不能证明分类结果可信，因为它不包含真实输入运行、labels 对齐和 Top-K 输出 review。

## YoloVision 离线 Preflight

真实运行前先生成配置预检报告：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-cls.onnx --labels .\models\imagenet.names --input-data .\models\yolov8n-cls-fp32.bin --input-shape 1x3x224x224 --family v8 --task cls --classification-output logits --preflight --preflight-report .\models\yolov8n-cls-preflight.json
```

报告的 schema 必须是 `yolovision-preflight.v1`，分类必须是 `precheck`；它只检查资产、profile 和分类输出配置，不是 labels/top-k 的 runtime proof。

## YoloVision 运行

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8n-cls.onnx `
  --labels .\models\imagenet.names `
  --input-data .\models\yolov8n-cls-fp32.bin `
  --input-shape 1x3x224x224 `
  --family v8 `
  --task cls `
  --classification-output logits
```

真实日志至少应包含：

- `Profile Family=v8 Task=cls`
- `InputSource=external`
- `Classification Class=... Score=...`
- `Postprocess Task=cls`
- owner 提供的 YoloVision 成功标记日志行：`YoloVision Passed=True`

如果日志只来自 synthetic tensor，它只能证明分类 decoder 能执行，不能证明真实图片分类正确。

## Top-K 输出解释

建议输出 JSON 包含：

- `classCount`
- `labelsSha256`
- `topK`
- `classScoreField`
- `softmaxApplied`
- `topPredictions[index].classId`
- `topPredictions[index].className`
- `topPredictions[index].score`
- `inputPreprocess`

文章中建议展示 Top-5，并说明每个 score 的来源。如果输出是 logits，score 是否经过 softmax 必须写清楚；如果输出是 probabilities，Top-K 排序可以直接使用概率。

## 常见问题

如果分类结果完全不对，优先检查输入尺寸、center crop、resize、RGB/BGR、均值方差、labels 顺序和 softmax。分类模型的错误经常来自预处理差异，而不是 TensorRT engine 本身。

如果 Top-K 类别名为空，检查 labels 文件行数和 class count 是否一致。如果 score 看起来都很接近，检查是否把 logits 当 probability 展示。

## Proof Boundary

本文、Top-K 表格、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、template、dry-run、build-only、sidecar-only report、screenshot、local feed、ProjectReference、direct `.nupkg` 都不是 runtime proof。

真实 classification runtime proof 必须包含 owner 提供的模型、labels、输入图片、preprocessed tensor、run log、SHA256、host metadata、许可证说明和 owner review。`package-consumer-runtime` proof 仍需要外部 clean consumer 从公开包来源运行。

## Owner Backfill Checklist

- 保存模型、ONNX、labels、图片、preprocessed tensor 和 run log SHA256。
- 记录 labels 行数、class count、Top-K 和 softmax 策略。
- 保存 TensorRtExec build-only report。
- 保存 YoloVision 运行命令和 owner 提供的成功标记日志行。
- 提供 Top-K 输出 JSON，至少包含 class id、class name 和 score。
- 由 owner review 分类结果是否符合输入图片语义。
