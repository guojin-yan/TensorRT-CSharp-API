# YoloVision LRASPP Semantic Segmentation 获取、转换与运行指南

## 适用读者

本文适合需要把语义分割模型纳入 `YoloVision` 样例、C# 业务程序和后续公众号/博客案例的部署工程师，也适合需要审核逐像素 class map、palette 和坐标还原的维护者。

## 解决问题

本文解决：

1. owner-provided semantic model、labels、palette、输入图片和预处理 tensor 如何组成可复核资产。
2. `[C,H,W]`、`[1,C,H,W]` 与按 class count 识别的 `[1,H,W,C]` 输出如何记录。
3. argmax、resize-back、void/ignore class、palette 和 map hash 如何与 JSON/SVG 关联。
4. 如何区分 build-only、semantic map runtime 候选和 package-consumer-runtime proof。

## 背景与场景

语义分割与 detection/instance segmentation 不同：输出通常是每个像素的类别 logits map，而不是 box、mask coefficients 或 prototype。`YoloVision` 的 `sem` decoder 保留 class-major 浮点 map，并在可视化阶段按像素选择最高 class；它不会把预先 argmax 的整数索引图当成等价的浮点输出，也不会替 owner 猜测 palette 或 ignore policy。

本文面向需要把语义分割模型纳入 `YoloVision` 样例和后续公众号/博客案例的开发者。目标是讲清楚 semantic segmentation 的模型获取、ONNX 导出、TensorRtExec build-only 记录、YoloVision 运行命令、输出 metadata 和 proof 边界。

## 适用场景

## 操作路径

1. 获取 owner-approved compatible semantic model、labels、palette 和输入图片，记录来源、许可证与 SHA256。
2. 固定 ONNX export 工具、opset、dynamic/static shape 和输入预处理规则。
3. 确认输出是否为 `[C,H,W]`、`[1,C,H,W]` 或按 class count 识别的 `[1,H,W,C]` 浮点 logits map。
4. 使用 TensorRtExec 构建 engine 并保存 build-only report。
5. 使用 YoloVision 的 `--semantic-output` 和 `--class-count` 运行，保存 output JSON、SVG 和日志。
6. 由 owner 回填 argmax、resize-back、palette、void class 和 map shape 证据，再执行 validator。

## 已验证模型与本地目录

本项目首个完成真实 TensorRT 闭环的 semantic 模型不是不存在的 `yolov8n-sem.pt`，而是 torchvision 官方 `LRASPP MobileNetV3 Large`。固定来源如下：

| 项目 | 固定值 |
| --- | --- |
| torchvision tag | `v0.25.0` |
| torchvision commit | `8ac84ee75afb1c327902156b5336f56ad63b7e2f` |
| 权重 URL | `https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth` |
| 权重 SHA256 | `d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280` |
| 模型许可证 | `BSD-3-Clause` |
| 输入图片 | PyTorch Hub commit `c7895df70c7767403e36f82786d6b611b7984557` 的 `images/dog.jpg` |
| ONNX SHA256 | `3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8` |

转换后的模型统一暂存到工作区外层目录，不放进 Git 仓库：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0\
  lraspp_mobilenet_v3_large-d234d4ea.pth
  lraspp-mobilenet-v3-large-320.onnx
```

这里的“工作区外层”很重要：Git 仓库是 `E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0`，`models` 与它同级，因此权重和 ONNX 不会进入源码提交。后续 Model Zoo 建立前，其他演示模型也遵守同一规则。仓库只保存获取/转换脚本、来源 URL、版本、长度、SHA256、输入输出契约和小型验证记录。

## 获取官方资产

在仓库根目录执行：

```powershell
$python = "C:\Users\guoji\.conda\envs\ultralytics\python.exe"

pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-TorchVisionLrasppOfficialAssets.ps1 `
  -AllowDownload `
  -PythonPath $python
```

脚本固定校验权重、torchvision `_meta.py`、torchvision LICENSE 和 `dog.jpg` 的长度与 SHA256。已有文件会直接复核；缺少文件时只有显式传入 `-AllowDownload` 才会联网。脚本不会上传资产、创建 Release 或发布 package。

对应的可审计清单是：

- `samples/assets/yolovision-torchvision-lraspp-official-assets.json`
- `eng/Acquire-TorchVisionLrasppOfficialAssets.ps1`

模型与图片目前都没有获得本项目公开再分发批准，因此不能把本地 `models` 或 `downloads` 内容提交到 GitHub。

## 转换为 ONNX

环境需要 PyTorch、torchvision、ONNX 和 ONNX Runtime。已验证版本为 Python `3.10.20`、PyTorch `2.10.0+cpu`、torchvision `0.25.0+cpu`、ONNX/ONNX Runtime `1.15.0`。

使用仓库脚本导出：

```powershell
$workspace = "E:\GitSpace\TensorRT-CSharp-API-4.0"
$modelRoot = "$workspace\models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0"
$assetRoot = "$workspace\downloads\lraspp-mobilenet-v3-large-torchvision-v0.25.0\source"
$referenceRoot = ".\artifacts\yolovision\semantic-lraspp-reference"

& $python .\eng\Invoke-YoloVisionSemanticReference.py `
  --weights "$modelRoot\lraspp_mobilenet_v3_large-d234d4ea.pth" `
  --image "$assetRoot\dog.jpg" `
  --onnx "$modelRoot\lraspp-mobilenet-v3-large-320.onnx" `
  --output-directory $referenceRoot `
  --export-onnx
```

脚本使用 `torchvision.models.segmentation.lraspp_mobilenet_v3_large` 创建 21 类模型，加载固定权重，通过只返回 `model(images)["out"]` 的 wrapper 导出 opset 17 静态图。导出后必须满足：

```text
images:[1,3,320,320] -> semantic:[1,21,320,320]
```

它同时执行 ONNX checker、PyTorch/ONNX Runtime 对比并生成正例 reference 与单点篡改负例。当前 PyTorch/ORT 最大绝对误差为 `9.5367431640625e-06`。

## C# 图像预处理

LRASPP 路径显式使用：

- stretch 到 `320x320`；
- RGB、NCHW、float32；
- `scale=1/255`；
- `mean=0.485,0.456,0.406`；
- `std=0.229,0.224,0.225`；
- 公式 `(pixel * scale - mean[channel]) / std[channel]`。

先把参考脚本生成的 `dog.ppm` 交给内置预处理器：

```powershell
dotnet run --project .\samples\YoloVision -c Release -- `
  --task sem `
  --class-count 21 `
  --input-shape 1x3x320x320 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize stretch `
  --scale 0.003921568627451 `
  --mean 0.485,0.456,0.406 `
  --std 0.229,0.224,0.225 `
  --image "$referenceRoot\dog.ppm" `
  --preprocessed-output "$referenceRoot\input-csharp-imagenet-1x3x320x320.fp32.bin" `
  --preprocess-only
```

当前 C# tensor SHA256 是 `d8f631d10bcc75645455f313e3b888acd68ba138a87b3ca380993062e507115a`，preprocess contract SHA256 是 `9f2dc5f182e1db1ea124cfcf8cf11c685dcb836b7695a67b14405851d6910a39`。C# 与 Pillow 的 resize 数值不完全相同，因此严格 TensorRT reference 必须由“同一个 C# tensor 经 ONNX Runtime”生成，不能拿另一套 resize 的 tensor 冒充。

## TensorRT 运行与完整产物

设置本机 TensorRT 与 bridge 后执行：

```powershell
$env:TENSORRT_PATH = "D:\Program Files\TensorRT-10.11.0.33-cu12"
$env:JYPPX_TENSORRT_ROOT = $env:TENSORRT_PATH
$env:JYPPX_NATIVE_BRIDGE_PATH = ".\build-out\win-x64-trt10-cuda12-release\bin\Release\jyppxtrtbridge.dll"
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "true"

dotnet run --project .\samples\YoloVision -c Release -- `
  --model "$modelRoot\lraspp-mobilenet-v3-large-320.onnx" `
  --labels "$referenceRoot\voc-semantic.names" `
  --image "$referenceRoot\dog.ppm" `
  --preprocessed-output "$referenceRoot\input-csharp-imagenet-1x3x320x320.fp32.bin" `
  --input-shape 1x3x320x320 `
  --input-name images `
  --output-name semantic `
  --tensor-rt-line 10 `
  --noTF32 `
  --family custom `
  --task sem `
  --class-count 21 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize stretch `
  --scale 0.003921568627451 `
  --mean 0.485,0.456,0.406 `
  --std 0.229,0.224,0.225 `
  --reference-outputs "semantic:$referenceRoot\semantic.reference.json" `
  --reference-abs-tolerance 0.0001 `
  --reference-rel-tolerance 0.0001 `
  --output "$referenceRoot\yolovision-semantic-output.json" `
  --semantic-artifact-output-directory "$referenceRoot\tensorrt-semantic-artifacts" `
  --visualization "$referenceRoot\yolovision-semantic.svg"
```

`--noTF32` 只在用户显式请求严格 FP32 parity 时关闭 TensorRT TF32；默认构建行为保持不变。本次 TensorRT 10.11 实测结果：

| 检查项 | 结果 |
| --- | ---: |
| 比较 logits | `2,150,400` |
| raw mismatch | `0` |
| raw 最大绝对误差 | `2.670288E-05` |
| class-index 像素 | `102,400` |
| argmax mismatch | `0` |
| background | `65,193` |
| dog（class 12） | `37,207` |

`--semantic-artifact-output-directory` 写出完整的 `semantic-class-index.i32.bin` 和 `semantic-map-artifacts.manifest.json`。二进制格式是 `int32 little-endian`、row-major `[H,W]`；manifest 记录 shape、长度、SHA256、完整 class histogram 和 proof boundary。它不是降采样 SVG，也不是只记录一个 `valueCount` 的摘要。

受控负例把 reference 的第 0 个 logit 增加 `10`，结果必须为退出码 `1`、mismatch `1`、first mismatch `0`。轻量真实运行记录位于 `samples/assets/yolovision-torchvision-lraspp-real-model-runtime-evidence.json`。

## 场景

语义分割与 detection/instance segmentation 不同：输出通常是每个像素的类别 logits map，而不是 box、mask coefficients 或 prototype。`YoloVision` 的 `sem` 任务把这个输出归一到 `SemanticMap`，并要求 owner 明确 `classCount`、`semanticMapShape`、`classMapLayout`、palette 和 void class 策略。

## 模型与资产

建议 owner 准备：

- owner-provided、与 `YoloVision sem` decoder 兼容的 semantic segmentation 权重；不要把不存在的官方 `yolov8n-sem.pt` 当作固定下载地址。
- `models/semantic-classes.names`，每行一个类别名。
- `models/semantic-palette.json`，记录类别到 RGB 颜色的映射。
- 一张可公开授权的测试图片。
- 预处理后的 `models/yolov8n-sem-fp32.bin`。
- 所有资产的 SHA256、来源 URL、许可证和 owner review 记录。

## 导出 ONNX

示例命令：

```powershell
yolo export model=.\models\owner-approved-semantic-model.pt format=onnx opset=17 dynamic=False simplify=True imgsz=512
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
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-sem.onnx --labels .\models\semantic-classes.names --input-data .\models\yolov8n-sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21 --preflight --preflight-report .\models\yolov8n-sem-preflight.json
```

该报告只允许 `yolovision-preflight.v1`/`proofClassification=precheck`，并要求所有 execution 与 promotion flag 为 `false`；它不能替代真实 semantic map 输出和 owner review。

## 运行 YoloVision

示例命令：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-sem.onnx --labels .\models\semantic-classes.names --input-data .\models\yolov8n-sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21 --output-json .\artifacts\yolovision\yolov8n-sem-output.json --visualization-svg .\artifacts\yolovision\yolov8n-sem-output.svg
```

`--semantic-map-shape`、`--class-map-layout` 和 `--palette` 不是当前 YoloVision CLI 参数。它们应作为 owner metadata 保存；命令行只声明实际支持的输出 role 和 class count。若模型输出不是浮点 class logits，或需要特殊 palette/resize-back 逻辑，必须先在 owner-approved preprocessing/postprocess adapter 中转换并记录版本、命令与 hash。

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

## 可复用资产目录与完整验证

建议为语义分割 case 建立独立的 E 盘 workspace，模型、labels、palette、输入图、预处理 tensor、engine、报告和日志互相隔离，避免把大文件和临时包落到系统盘：

E:\TensorRtSharpAssets\cases\yolov8n-sem\models
E:\TensorRtSharpAssets\cases\yolov8n-sem\labels
E:\TensorRtSharpAssets\cases\yolov8n-sem\images
E:\TensorRtSharpAssets\cases\yolov8n-sem\tensors
E:\TensorRtSharpAssets\cases\yolov8n-sem\engines
E:\TensorRtSharpAssets\cases\yolov8n-sem\reports
E:\TensorRtSharpAssets\cases\yolov8n-sem\logs

从 `samples/assets/yolovision-yolov8-sem-candidate.template.json` 开始回填 `model.sourceUrl`、`model.downloadUrl`、`model.license`、`model.sha256`、`labels.sourceUrl`、`labels.sha256`、`labels.palettePath`、`labels.classCount`、`input.imageSha256`、`input.preprocessedTensorSha256`，以及 `outputMetadata.semanticOutput`、`outputMetadata.semanticOutputRole`、`outputMetadata.classCount`、`outputMetadata.mapWidth`、`outputMetadata.mapHeight`、`outputMetadata.semanticMapShape`、`outputMetadata.classMapLayout`、`outputMetadata.argmaxRule`、`outputMetadata.postprocessMetadata.resizeBackRule` 和 `outputMetadata.postprocessMetadata.ignoreIndex`。palette 文件也要单独记录 `paletteSha256`，不能只把颜色写进截图。

模型、labels、palette、原图、预处理 tensor、engine、build report、preflight report、output JSON、overlay SVG 和 run log 分别计算 SHA256：

```powershell
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-sem\models\yolov8n-sem.onnx
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-sem\labels\semantic-classes.names
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-sem\labels\semantic-palette.json
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-sem\images\street.ppm
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-sem\tensors\street-fp32.bin
```

先只做预处理，确认输入 shape、颜色顺序和 tensor hash：

```powershell
dotnet run --project .\samples\YoloVision -- --preprocess-only --image E:\TensorRtSharpAssets\cases\yolov8n-sem\images\street.ppm --preprocessed-output E:\TensorRtSharpAssets\cases\yolov8n-sem\tensors\street-fp32.bin --input-shape 1x3x512x512 --tensor-layout NCHW --color-order RGB --resize letterbox
```

运行时保留显式 semantic output role、class count、JSON 和 SVG：

```powershell
dotnet run --project .\samples\YoloVision -- --model E:\TensorRtSharpAssets\cases\yolov8n-sem\models\yolov8n-sem.onnx --labels E:\TensorRtSharpAssets\cases\yolov8n-sem\labels\semantic-classes.names --input-data E:\TensorRtSharpAssets\cases\yolov8n-sem\tensors\street-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21 --output-json E:\TensorRtSharpAssets\cases\yolov8n-sem\reports\yolov8n-sem-output.json --visualization-svg E:\TensorRtSharpAssets\cases\yolov8n-sem\reports\yolov8n-sem-output.svg
```

当前 `yolovision-output.v1` 的 semantic prediction 包含 `task=sem`、`classCount`、`width`、`height`、`valueCount`、`classIndexValueCount`、dominant class 和完整 `classHistogram`；output tensor summary 还保存实际 shape 与 value hash。`YoloVision` 不会自动应用 owner palette 或猜测 resize-back。需要完整逐像素复核时必须同时保存 `semantic-class-index.i32.bin` 和 manifest，并关联 `modelSha256`、`labelsSha256`、`paletteSha256`、`imageSha256`、`preprocessedTensorSha256`、run log hash、`classMapLayout` 和 `argmaxRule`。

建议按以下顺序验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
```

只有 owner 回填真实兼容模型、labels、palette、输入 hash、class/map metadata、`YoloVision Passed=True`、stdout/stderr summary、run log hash 并通过 validator 后，才能形成 `real-model-runtime` 候选。TensorRtExec build-only、preflight、semantic overlay、截图、local feed 和 direct `.nupkg` 仍不是 `package-consumer-runtime` proof。

## 代码与文件入口

- `samples/YoloVision/YoloSampleRunner.cs`：`DecodeSemanticMap` 与 `[C,H,W]`/`[1,C,H,W]`/按 class count 识别的 NHWC map 路由。
- `samples/YoloVision/YoloVisionOutputReport.cs`：semantic prediction 的 classCount、width、height 和 valueCount 输出。
- `samples/YoloVision/YoloVisionVisualizationWriter.cs`：按像素 argmax 的语义 SVG 网格。
- `samples/YoloVision/YoloSemanticMapArtifactWriter.cs`：完整 class-index 二进制、SHA256、histogram 与 manifest。
- `samples/YoloVision/yolovision-semantic-map-artifacts.schema.json`：完整语义图产物 schema。
- `eng/Acquire-TorchVisionLrasppOfficialAssets.ps1`：官方资产获取与固定 hash 校验。
- `eng/Invoke-YoloVisionSemanticReference.py`：ONNX 导出、ORT reference、argmax 与负例比较。
- `samples/YoloVision/yolovision-task-output-contract.json`：semantic output role 与必填 metadata。
- `samples/YoloVision/Program.cs`：`--task sem`、`--semantic-output`、`--class-count`、输出参数入口。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：semantic map、palette 和 owner 证据字段验证。

## 图示建议

建议准备：

1. Netron 中 semantic output tensor shape 和 class-major/NHWC 对照。
2. labels、palette、void/ignore policy 与 class count 表格。
3. 原图、预处理输入、semantic map overlay 和 JSON 摘要。
4. logits -> argmax -> resize-back -> palette 的数据流图。
5. output tensor hash、output JSON 和 run log hash 的关联图。

截图和颜色 overlay 只用于人工理解，不能替代真实 class map、结构化输出、run log、hash 和 validator。

## 边界说明（Proof Boundary）

本文、`samples/assets/yolovision-article-case-pack.json`、TensorRtExec build-only report、YoloVision matrix、sidecar-only report、dry-run、local feed、ProjectReference 和 direct `.nupkg` 都不是 runtime proof。

只有在 owner 提供真实模型、labels、palette、输入图片、预处理 tensor、运行日志、输出 JSON、SHA256、stdout/stderr 摘要，并通过对应 validator 后，样例证据才可以晋级 `real-model-runtime`。它仍然不能替代 `package-consumer-runtime` proof。

## Owner Backfill Checklist

- 填写模型来源、许可证和 SHA256。
- 填写 labels、palette、输入图片和预处理 tensor SHA256。
- 保存 TensorRtExec build-only report 和 engine SHA256。
- 保存 YoloVision run log，并确认 `YoloVision Passed=True`。
- 保存 output JSON 和 semantic map 摘要。
- 用 owner proof input/template 或 sample-run-evidence validator 做严格验证。

## 下一步

完成 semantic case 后，应增加 `[1,C,H,W]` 与 `[1,H,W,C]` golden output、不同 class count、palette/ignoreIndex、resize-back 和自定义 owner model 的对照测试，并覆盖 YOLOv26/custom semantic exports。不能把一个 map shape 或一个颜色表推广到所有模型。

随后在 clean package consumer 中重复同一 semantic 输入，补充 package-consumer-runtime；source-tree `real-model-runtime`、semantic SVG 和 TensorRtExec report 都不能关闭发布 proof blocker。
