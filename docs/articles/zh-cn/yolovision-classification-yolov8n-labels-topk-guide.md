# YoloVision YOLOv8n Classification Labels 与 Top-K 指南

## 适用读者

本文适合需要验证图像分类模型、labels 顺序和 Top-K 结果的部署工程师，也适合需要把 `samples/YoloVision` 输出接入 C# 业务系统或文章案例的维护者。

## 解决问题

本文解决：

1. YOLOv8n-cls 模型、labels、输入图片和预处理 tensor 如何形成可复核资产。
2. logits、probability、softmax 和 Top-K 的边界如何明确记录。
3. 分类输出 JSON、SVG、运行日志和 SHA256 如何关联。
4. 如何区分 build-only、分类 runtime 候选和 package-consumer-runtime proof。

## 背景与场景

分类没有 box、NMS 和 mask，但 labels 行序、类别数、输入预处理和 score 语义一旦漂移，结果会稳定地“错得很像真的”。因此分类案例需要把模型输出向量、labels 和输入图像作为一个整体验证，而不是只截一张 Top-5 截图。

## 文章定位

本文面向希望用 `samples/YoloVision` 跑 YOLOv8n-cls 或类似 classification 模型的开发者。Classification 没有 box、NMS 和 mask，表面上比 detection 简单，但它对 labels 顺序、softmax 约定、Top-K 输出和输入尺寸非常敏感。

本文给出从模型来源、ONNX 导出、TensorRtExec build-only、YoloVision 运行到输出 JSON/SVG 的完整文章结构。它用于真实资产回填和公众号/博客发布，不是 runtime proof。

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

## 可复用资产目录与完整验证

建议为 YOLOv8n-cls 建立独立的 E 盘 case workspace，把模型、分类 labels、输入图、预处理 tensor、engine、报告和日志分开保存，避免模型和临时包落到系统盘：

E:\TensorRtSharpAssets\cases\yolov8n-cls\models
E:\TensorRtSharpAssets\cases\yolov8n-cls\labels
E:\TensorRtSharpAssets\cases\yolov8n-cls\images
E:\TensorRtSharpAssets\cases\yolov8n-cls\tensors
E:\TensorRtSharpAssets\cases\yolov8n-cls\engines
E:\TensorRtSharpAssets\cases\yolov8n-cls\reports
E:\TensorRtSharpAssets\cases\yolov8n-cls\logs

从 `samples/assets/yolovision-yolov8-cls-candidate.template.json` 开始回填 `model.sourceUrl`、`model.downloadUrl`、`model.license`、`model.sha256`、`labels.sourceUrl`、`labels.sha256`、`labels.classCount`、`input.imageSha256`、`input.preprocessedTensorSha256`，以及 `outputMetadata.classificationOutput`、`outputMetadata.outputShape`、`outputMetadata.classCount`、`outputMetadata.labelsPath`、`outputMetadata.topK`、`outputMetadata.classScoreField` 和 `outputMetadata.postprocessMetadata.activation`。`classScoreField` 要明确是 logits 还是 probability，不能把两者混用。

模型、labels、原图、预处理 tensor、engine、build report、preflight report、output JSON、overlay SVG 和 run log 分别计算 SHA256：

```powershell
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-cls\models\yolov8n-cls.onnx
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-cls\labels\imagenet.names
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-cls\images\cat.ppm
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-cls\tensors\cat-fp32.bin
```

先只做预处理，确认 224 输入、RGB、NCHW、归一化和 tensor hash：

```powershell
dotnet run --project .\samples\YoloVision -- --preprocess-only --image E:\TensorRtSharpAssets\cases\yolov8n-cls\images\cat.ppm --preprocessed-output E:\TensorRtSharpAssets\cases\yolov8n-cls\tensors\cat-fp32.bin --input-shape 1x3x224x224 --tensor-layout NCHW --color-order RGB --resize letterbox
```

这条命令只适用于 owner 已确认导出契约使用 letterbox 的模型。标准 classification pipeline 可能要求先按短边 resize，再做 center crop；YoloVision 当前通用 CLI 不会隐式补做 center crop。遇到这种模型时，应使用 owner-approved preprocess pipeline 生成同一 `cat-fp32.bin`，记录 resize/crop/normalize 命令与工具版本，再计算 tensor SHA256。未对齐 center crop 的可执行输入不能作为分类正确性 proof。

分类运行保留显式 output role、Top-K 和输出产物：

```powershell
dotnet run --project .\samples\YoloVision -- --model E:\TensorRtSharpAssets\cases\yolov8n-cls\models\yolov8n-cls.onnx --labels E:\TensorRtSharpAssets\cases\yolov8n-cls\labels\imagenet.names --input-data E:\TensorRtSharpAssets\cases\yolov8n-cls\tensors\cat-fp32.bin --input-shape 1x3x224x224 --family v8 --task cls --classification-output logits --top-k 5 --output-json E:\TensorRtSharpAssets\cases\yolov8n-cls\reports\yolov8n-cls-output.json --visualization-svg E:\TensorRtSharpAssets\cases\yolov8n-cls\reports\yolov8n-cls-output.svg
```

当前 `yolovision-output.v1` 的分类记录包含 `postprocess.topK`，每个 prediction 至少包含 `task=cls`、`classId`、`className` 和 `score`。输出 JSON 还应关联 `classCount`、`labelsSha256`、`modelSha256`、`imageSha256`、`preprocessedTensorSha256`、score 语义和 run log hash。程序不会替 owner 推断模型是否已经执行 softmax；`softmaxApplied`、label locale 和 score precision 必须在 candidate metadata 或 owner 记录中明确。

建议按以下顺序验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
```

只有 owner 回填真实模型、labels、输入 hash、class count、Top-K、score/softmax 语义、`YoloVision Passed=True`、stdout/stderr summary、run log hash 并通过 validator 后，才能形成 `real-model-runtime` 候选。TensorRtExec build-only、preflight、Top-K 截图、分类 SVG、local feed 和 direct `.nupkg` 仍不是 `package-consumer-runtime` proof。

## 代码与文件入口

- `samples/YoloVision/YoloSampleRunner.cs`：分类 logits/top-k 解码。
- `samples/YoloVision/YoloVisionOutputReport.cs`：`task=cls`、`classId`、`className`、`score` 和 `postprocess.topK` 输出。
- `samples/YoloVision/YoloVisionVisualizationWriter.cs`：Top-K 分类 SVG。
- `samples/YoloVision/yolovision-task-output-contract.json`：classification output role 与必填 metadata。
- `samples/YoloVision/Program.cs`：`--task cls`、`--classification-output`、`--top-k`、输出参数入口。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：分类 candidate、labels 和证据字段验证。

## 图示建议

建议准备：

1. Netron 中分类输入/output tensor 和 class vector shape。
2. labels 前几行、class count 与模型输出维度对照表。
3. 原图、预处理输入和 Top-5 SVG/JSON 对照。
4. logits 与 probability/softmax 语义说明。
5. 输出 JSON、run log 和 SHA256 关联示意图。

截图和 Top-K 表格只用于人工理解，不能替代真实输入、结构化输出、run log、hash 和 validator。

## 常见问题

如果分类结果完全不对，优先检查输入尺寸、center crop、resize、RGB/BGR、均值方差、labels 顺序和 softmax。分类模型的错误经常来自预处理差异，而不是 TensorRT engine 本身。

如果 Top-K 类别名为空，检查 labels 文件行数和 class count 是否一致。如果 score 看起来都很接近，检查是否把 logits 当 probability 展示。

如果输出只有一个类别，检查 `--top-k`、输出 tensor role 和 class vector shape；如果 SVG 与 JSON 的类别顺序不一致，优先以 JSON 为准并检查 labels 文件编码和行尾。

## 边界说明（Proof Boundary）

本文、Top-K 表格、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、template、dry-run、build-only、sidecar-only report、screenshot、local feed、ProjectReference、direct `.nupkg` 都不是 runtime proof。

真实 classification runtime proof 必须包含 owner 提供的模型、labels、输入图片、preprocessed tensor、run log、SHA256、host metadata、许可证说明和 owner review。`package-consumer-runtime` proof 仍需要外部 clean consumer 从公开包来源运行。

## Owner Backfill Checklist

- 保存模型、ONNX、labels、图片、preprocessed tensor 和 run log SHA256。
- 记录 labels 行数、class count、Top-K 和 softmax 策略。
- 保存 TensorRtExec build-only report。
- 保存 YoloVision 运行命令和 owner 提供的成功标记日志行。
- 提供 Top-K 输出 JSON，至少包含 class id、class name 和 score。
- 由 owner review 分类结果是否符合输入图片语义。

## 下一步

完成 YOLOv8n-cls 后，应增加不同 class count、logits/probability、224/320 输入尺寸和 labels locale 的 golden output，并覆盖 YOLOv11 classification。每个模型都要记录真实 output shape 和 score 语义，不能只复制 `topK=5`。

随后在 clean package consumer 中重复同一分类输入，补充 package-consumer-runtime；source-tree `real-model-runtime`、Top-K 截图和 TensorRtExec report 都不能关闭发布 proof blocker。
