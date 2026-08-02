# YoloVision 总览：一个样例覆盖 YOLO 多系列多任务

早期检测样例命名太窄，只能让人想到 detection。TensorRtSharp4.0 现在把视觉样例统一到 `samples/YoloVision`：它的定位不是“跑一个 YOLOv8 检测 demo”，而是用一个样例承载 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26、YOLOX 和 custom 模型的多任务教程、模型资产回填、输出 JSON、可视化、owner evidence 与文章案例。

这篇文章面向公众号、博客和项目主页读者。它要说明 YoloVision 为什么值得做成统一样例，也要把 proof boundary 讲清楚：模板、support matrix、preflight、local package consumer、截图和可视化都不是 package-consumer-runtime proof。

## 适合谁阅读

- 希望在 .NET 中统一跑 YOLO 多任务模型的视觉工程师。
- 想理解 det、cls、seg、obb、pose、sem 六类任务输出差异的模型部署同学。
- 准备为 YOLOv8n、YOLOv10n、YOLOX-S 或自有模型回填真实资产的维护者。
- 需要把 TensorRtExec、OnnxToEngine、YoloVision 和 release proof 串成文章系列的作者。

## 目标范围

YoloVision 的长期目标是支持：

```text
families = YOLOv5 / YOLOv6 / YOLOv7 / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO26 / YOLOX / custom
tasks = det / cls / seg / obb / pose / sem
input = ONNX model / TensorRT engine / image / preprocessed float tensor
output = yolovision-output.v1 JSON / SVG visualization / owner evidence row
boundary = sample evidence, not package-consumer-runtime proof
```

机器可读总览在：

```text
samples/YoloVision/yolo-model-matrix.json
samples/YoloVision/yolo-model-matrix.md
samples/YoloVision/yolovision-task-output-contract.json
samples/YoloVision/yolovision-output.schema.json
samples/YoloVision/yolovision-preflight.schema.json
samples/assets/yolovision-family-task-real-asset-roadmap.json
samples/assets/yolovision-article-case-pack.json
samples/assets/yolovision-real-asset-owner-backfill-pack.json
```

这些文件让 README、公开文章、owner backfill pack 和 validator 共用同一套 family/task/output 语义，避免每篇文章自己猜输出 tensor。

## 为什么要做成统一样例

YOLO 系列模型的问题不是“能不能加载 ONNX”这么简单。不同任务的输出 tensor 语义差异很大：

- det：box、class score、objectness、class-aware/class-agnostic NMS。
- cls：logits/topK、labels、softmax 是否已应用。
- seg：detection rows、mask coefficients、mask prototypes、crop/resize rule。
- obb：rotated box、angle output、angle unit、rotated NMS 边界。
- pose：keypoint count、keypoint stride、visibility/confidence、skeleton metadata。
- sem：semantic map、class count、palette、resize-back rule。

统一样例可以把这些差异放在明确 metadata 中，而不是散落到每篇文章或每个 demo 的临时代码里。它也让 YoloVision 能和 `samples/OnnxToEngine`、`applications/TensorRtExec` 共享 evidence ladder：先构建 engine，再运行样例，再由 owner 回填真实日志和 hash。

## 核心代码路径

YoloVision 的实现分成 runner、preprocess、runtime output、managed postprocess、report 和 visualization：

```text
samples/YoloVision/Program.cs
samples/YoloVision/YoloVision.csproj
samples/YoloVision/YoloSampleRunner.cs
samples/YoloVision/YoloVisionResult.cs
samples/YoloVision/YoloVisionOutputReport.cs
samples/YoloVision/YoloVisionPreflightReport.cs
samples/YoloVision/YoloImagePreprocessor.cs
samples/YoloVision/YoloRuntimeOutputSet.cs
samples/YoloVision/YoloRuntimeOutputTensor.cs
samples/YoloVision/YoloRuntimeOutputRoleResolver.cs
samples/YoloVision/YoloMultiOutputMetadata.cs
samples/YoloVision/YoloVisionVisualizationWriter.cs
```

任务专属 managed postprocess 包括：

```text
YoloDetectionDecoder
YoloClassificationPrediction
YoloMaskComposer
YoloSegmentationPrediction
YoloObbDecoder
YoloObbDetection
YoloPoseDecoder
YoloPosePrediction
YoloSemanticMap
YoloEndToEndOutput
YoloXOutputDecoder
```

这些类的共同目标是：用户拿到的是有语义的 detection/classification/mask/pose/semantic result，而不是无语义的 `IntPtr`、裸 buffer 或隐式生命周期。

## 离线 preflight 与 managed smoke

没有 CUDA、TensorRT、ONNX 或模型文件时，也可以跑 deterministic managed smoke：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --self-test-end2end
```

它应输出：

```text
ManagedSmoke=YOLOv10EndToEnd Passed=True
ApplyNms=False
ManagedSmokeBoundary=managed-array-decode-only
```

这只证明 managed array decode contract，不是 TensorRT execution、real-model-runtime 或 package-consumer-runtime proof。

准备 owner handoff 或文章案例时可以先跑 preflight：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- `
  --preflight `
  --family v8 `
  --task seg `
  --model E:\TensorRtSharpAssets\models\yolov8n-seg.onnx `
  --labels E:\TensorRtSharpAssets\models\coco.names `
  --input-data E:\TensorRtSharpAssets\tensors\yolov8n-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --preflight-report E:\TensorRtSharpAssets\reports\yolov8n-seg-preflight.json
```

`state=ready-for-runtime-precheck` 表示输入和 metadata 具备预检查条件；`owner-action-required` 表示还需要 owner 补真实模型、labels、输入图、tensor 或任务 metadata；`invalid` 用于 strict preflight blocker。`--dryRun` 和 `--previewOnly` 只是 alias，报告边界仍是 `proofClassification=precheck`、`isRuntimeProof=false`、`canPromoteRealModelRuntime=false`。

## 图像预处理路径

真实文章案例不应只用 synthetic tensor。YoloVision 支持把 `.bmp` / `.ppm` 图像预处理成 float32 tensor：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- `
  --preprocess-only `
  --image E:\TensorRtSharpAssets\images\dog.ppm `
  --preprocessed-output E:\TensorRtSharpAssets\tensors\dog-yolo-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox
```

命令会记录 `ImagePreprocess`、`ImagePreprocessConfig`、source/tensor SHA256、source/target dimensions、resize scale、padding、normalization、layout、color order 和 output element count。它是 preprocessing evidence only，直到同一个 tensor 被成功 TensorRT enqueue 并进入 sample-run evidence 才能继续晋级。

## 六任务命令骨架

任务文章和 owner proof record 应保持命令显式，不要靠“auto”掩盖模型输出语义。

Detection：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-det.onnx --labels E:\TensorRtSharpAssets\models\coco.names --image E:\TensorRtSharpAssets\images\det.ppm --preprocessed-output E:\TensorRtSharpAssets\tensors\det-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45 --output E:\TensorRtSharpAssets\reports\yolo-det-output.json
```

YOLOv10 end-to-end detection：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolov10n.onnx --labels E:\TensorRtSharpAssets\models\coco.names --image E:\TensorRtSharpAssets\images\det.ppm --preprocessed-output E:\TensorRtSharpAssets\tensors\yolov10n-fp32.bin --input-shape 1x3x640x640 --family v10 --task det --layout end2end --class-count 80 --confidence 0.25 --output E:\TensorRtSharpAssets\reports\yolov10n-output.json
```

Classification：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-cls.onnx --labels E:\TensorRtSharpAssets\models\labels.txt --input-data E:\TensorRtSharpAssets\tensors\cls-fp32.bin --input-shape 1x3x224x224 --family custom --task cls --classification-output output0 --classification-score-mode probabilities --confidence 0 --top-k 5 --output E:\TensorRtSharpAssets\reports\yolo-cls-output.json
```

Segmentation：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-seg.onnx --labels E:\TensorRtSharpAssets\models\coco.names --input-data E:\TensorRtSharpAssets\tensors\seg-fp32.bin --input-shape 1x3x640x640 --family v8 --task seg --output-role-map boxes:det,proto:mask-prototypes --mask-coefficient-count 32 --output E:\TensorRtSharpAssets\reports\yolo-seg-output.json
```

Oriented bounding box：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-obb.onnx --labels E:\TensorRtSharpAssets\models\labels.txt --input-data E:\TensorRtSharpAssets\tensors\obb-fp32.bin --input-shape 1x3x1024x1024 --family v8 --task obb --output-role-map boxes:det,angles:obb-angle --obb-angle-output angles --output E:\TensorRtSharpAssets\reports\yolo-obb-output.json
```

Pose：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-pose.onnx --labels E:\TensorRtSharpAssets\models\labels.txt --input-data E:\TensorRtSharpAssets\tensors\pose-fp32.bin --input-shape 1x3x640x640 --family v8 --task pose --output-role-map boxes:det,keypoints:pose-keypoints --pose-keypoint-count 17 --output E:\TensorRtSharpAssets\reports\yolo-pose-output.json
```

Semantic segmentation：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --model E:\TensorRtSharpAssets\models\yolo-sem.onnx --labels E:\TensorRtSharpAssets\models\labels.txt --input-data E:\TensorRtSharpAssets\tensors\sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21 --output E:\TensorRtSharpAssets\reports\yolo-sem-output.json
```

Dedicated role options 也可直接使用：`--detection-output`、`--classification-output`、`--semantic-output`、`--mask-prototypes-output`、`--pose-keypoints-output`、`--obb-angle-output`。如果没有显式 role，runner 会使用保守 tensor-name heuristics，例如 `proto`、`keypoint`、`angle`、`semantic`、`logits`、`box`、`detect`。

## 输出 JSON 与可视化

`--output` 或 `--output-json` 会写出 `yolovision-output.v1` JSON。报告包含：

```text
task/family metadata
copied output tensor shapes
per-output valueSha256
preview values
postprocess thresholds
prediction summaries
labels path/class count/SHA256
model/input SHA256
bindingMetadata
boundary.isRuntimeProof=false
forbidden substitutes
```

`--visualization` 或 `--visualization-svg` 会写 lightweight SVG visualization，覆盖 detection boxes、classification bars、segmentation masks/boxes、OBB rotation、pose keypoints 和 semantic maps。SVG 很适合文章截图和 owner review，但它仍是 derived evidence，必须绑定真实模型、真实输入、preprocessed tensor、output JSON hash、run log hash 和 owner review。

六任务最小输出示例位于：

```text
samples/YoloVision/examples/yolovision-output-det.example.json
samples/YoloVision/examples/yolovision-output-cls.example.json
samples/YoloVision/examples/yolovision-output-seg.example.json
samples/YoloVision/examples/yolovision-output-obb.example.json
samples/YoloVision/examples/yolovision-output-pose.example.json
samples/YoloVision/examples/yolovision-output-sem.example.json
```

验证示例或 owner 产出的 output JSON，脚本路径是 `eng/Test-YoloVisionOutputReport.ps1`：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
```

这个 validator 检查 task-specific prediction metadata、copied output tensor summaries、`boundary.isRuntimeProof=false` 和 forbidden substitute list。它是 owner-review infrastructure，不是 runtime proof。

## 案例矩阵与证据回填顺序

每篇 YOLO 案例都应先从 yolo-model-matrix.json 选择 family/task，再从 yolovision-task-output-contract.json 读取该任务的 requiredMetadata；不要先下载一个模型、跑出一张截图，再倒推输出语义。推荐顺序是：

1. **选择矩阵条目**：记录 family、task、supportedTasks、status、inputShape、output roles 和当前 proof boundary。future-family-planning、planned-runtime-proof 和 managed-postprocess-ready 都表示仍需 owner 资产或真实运行证据。
2. **获取并固定来源**：优先使用官方或 owner-approved 下载入口，记录 modelSourceUrl、modelVersion、license、checkpoint、exporterVersion 和 modelSha256；模型权重、图片和 labels 留在 E 盘外部 workspace。
3. **导出并做静态检查**：保存 export command、opset、dynamic/static shape、input/output tensor names、class count 和 graph-side NMS 说明；不把导出成功写成 runtime proof。
4. **预检输出契约**：用 --preflight 填写 outputRoleMap 和任务专属字段，保存 preflight report；owner-action-required 或 canPromoteRealModelRuntime=false 必须原样保留。
5. **构建 engine**：用 TensorRtExec 或 OnnxToEngine 执行 build-only，保存 command、report、engine SHA256 和 readback；build-only 只进入 build evidence。
6. **运行样例并校验输出**：提供真实输入图或 tensor、labels、预处理配置、阈值和 reference output，保存 YoloVision Passed=True、stdout/stderr、output JSON、visualization 和 run log SHA256。
7. **回填并审查**：运行 sample-run-evidence validator，补 host metadata、owner review 和失败诊断；只有 real-model-runtime evidence 完整后，才可继续讨论外部 clean consumer，不能由本地样例直接晋级 package-consumer-runtime。

六类任务不能省略的字段也不同：

| Task | 必须明确的输出语义 | 常见遗漏 |
| --- | --- | --- |
| det | box format、score rule、class count、objectness、NMS mode、end-to-end column order | 把 [1,N,6] 和 raw head 当成同一布局 |
| cls | classification score mode、Top-K、labels path、class count、graph softmax 状态 | 只展示 top-1，不记录 labels 和 score 规则 |
| seg | boxes、mask coefficients、prototype shape/layout、crop/resize policy、mask threshold | 只画 mask，不保存 prototype 和缩放规则 |
| obb | angle output、angle unit/range、rotated box format、rotated NMS | 把角度当作普通 box 坐标或忽略单位 |
| pose | keypoint count、stride、coordinate layout、visibility/score、skeleton metadata | 只画点，不记录 keypoint tensor layout |
| sem | semantic map shape、class count、argmax rule、palette、ignore/void policy | 把 class-index map 和 logits map 混为一谈 |

当前矩阵中，YOLOv5/v6/v7/v8/v9/v10/YOLO11/YOLO26/YOLOX/custom 的任务覆盖并不相同；例如 YOLOv10 的官方 end-to-end detection 使用 [1,300,6]，YOLOX 是 detection-only 的 [1,8400,85] raw output，YOLO26 仍需 owner-approved output contract。文章必须展示这种差异，而不是用“支持全部 YOLO”替代具体证据。

所有候选模板都应保留 owner-action-required、canPromoteRealModelRuntime=false 和 canPromotePackageConsumerRuntime=false，直到真实模型、真实输入、输出 JSON、日志 hash、许可证和 owner review 全部通过 validator。support matrix、preflight、build report、SVG、GUI screenshot、local feed 和 direct nupkg 都不能改变这些字段。

## 官方资产与文章案例

仓库不会把大型模型、图片和 label 直接塞进源码，因为这些资产有体积和 license 限制。已有的 acquisition/backfill 入口都要求 owner 明确来源、license、SHA256 和输出日志。

YOLOv10n 官方路径：

```text
eng/Acquire-YoloV10OfficialAssets.ps1
samples/assets/yolovision-yolov10-official-assets.json
docs/articles/zh-cn/yolovision-yolov10-end-to-end-output-guide.md
artifacts/interface-coverage/yolov10-official-runtime-proof-closure.json
```

官方 THU-MIG YOLOv10n v1.1 ONNX 使用 `[1,300,6]` output，列为 `x1,y1,x2,y2,score,classId`，适合 `--layout end2end`。它证明 source-tree real-model-runtime 的局部路径，不是 package-consumer-runtime，不是 public redistribution approval，也不是 publish readiness。

YOLOX-S 官方路径：

```text
eng/Acquire-YoloXOfficialAssets.ps1
samples/assets/yolovision-yolox-official-assets.json
docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md
docs/articles/zh-cn/yolovision-yolox-local-package-consumer-tutorial.md
```

YOLOX 是 detection-only。它使用 `[1,8400,85]` raw output，通过 `(xy + grid) * stride` 和 `exp(wh) * stride` 解码，再做 objectness scoring 和 NMS。它同样不批准 public asset redistribution。

YOLOv8n 六任务文章包：

```text
samples/assets/yolovision-article-case-pack.json
samples/assets/yolovision-yolov8-det-candidate.template.json
samples/assets/yolovision-yolov8-seg-candidate.template.json
samples/assets/yolovision-yolov8-pose-candidate.template.json
samples/assets/yolovision-yolov8-obb-candidate.template.json
samples/assets/yolovision-yolov8-cls-candidate.template.json
samples/assets/yolovision-yolov8-sem-candidate.template.json
```

这些模板是 `owner-action-required`。它们定义文章、导出命令、preflight、TensorRtExec build-only report、YoloVision run command、expected evidence line、required SHA256 fields 和 owner review 字段，但没有真实资产回填前不能晋级。

## Owner proof 路径

真实模型晋级至少需要：

```text
model source URL / license / export command / model SHA256
labels source / license / class count / labels SHA256
input image or preprocessed tensor source / license / SHA256
TensorRtExec or OnnxToEngine build-only report path and SHA256
YoloVision run command
Expected real run evidence line: YoloVision Passed=True
output JSON path and SHA256
stdout/stderr summaries
run log SHA256
host metadata
owner review
sample-run-evidence record
```

相关脚本：

```text
eng/Export-YoloVisionRealAssetOwnerBackfillPack.ps1
eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1
eng/Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1
eng/Test-YoloVisionRealAssetOwnerProofInput.ps1
eng/Import-YoloVisionRealAssetOwnerProofInput.ps1
eng/Test-YoloVisionRealAssetCandidate.ps1
eng/Test-SampleRunEvidenceRecord.ps1
```

`samples/assets/yolovision-real-asset-owner-backfill-pack.json` 和生成的 sample-run evidence template 只是 owner backfill scaffold。只有真实 hashes、logs、host metadata、`YoloVision Passed=True` 和 owner review 都存在，并通过 validator，才可能形成 real-model-runtime evidence。它仍然不是 package-consumer-runtime proof。

## Local PackageReference consumer

`YoloVision.csproj` 可以打包成 `JYPPX.TensorRT.CSharp.API.YoloVision`，暴露 pointer-free 的 `YoloVisionCommand.Run(string[] args)`。仓库外应用可以复用同一套 CLI、preprocess、decode、NMS、report 和 visualization 路径，而不需要 `ProjectReference`。

本地验证入口：

```text
samples/YoloVision.PackageConsumer
eng/Test-YoloVisionLocalPackageConsumer.ps1
eng/Test-YoloVisionLocalPackageConsumerMatrix.ps1
eng/Test-YoloVisionPublicPackageProof.ps1
eng/Test-YoloVisionPublicPackageConsumer.ps1
```

本地 PackageReference consumer 使用 local file feed 和 E 盘隔离 NuGet cache，它可证明 `local-package-consumer-runtime`，但不是 public package source、post-publish verification 或 package-consumer-runtime proof。

## Proof 边界

YoloVision 可以支撑 real-model-runtime proof，但不能直接支撑 package-consumer-runtime proof。原因很简单：样例运行证明的是某个模型资产在某个 host 上可以走通样例路径；package proof 必须来自干净外部 consumer 使用公开 NuGet 包执行。

以下内容都不能替代 package-consumer-runtime proof：

- support matrix。
- `yolovision-task-output-contract.json`。
- preflight report。
- preprocessing output。
- output JSON。
- SVG visualization。
- local feed。
- ProjectReference。
- direct `.nupkg`。
- GitHub Actions dry-run。
- TensorRtExec build-only report。
- OnnxToEngine report。
- dependency-probe-only。
- GUI screenshot。
- command preview。
- package-consumer-runtime string in sample evidence。

一句话边界：YoloVision 的 matrix、contract、template、preflight、visualization、local package consumer 和 sample-run evidence scaffold 都不是 package-consumer-runtime proof，不能授权发布，也不能关闭 release issue。

## 配图建议

- YOLO 六任务输出 tensor 到 `YoloVisionResult` 的流程图。
- YoloVision CLI 参数到 task decoder 的映射图。
- det/seg/pose/obb/cls/sem 六格结果示意图，后续用 owner-approved 真实图片替换。
- Evidence ladder：TensorRtExec build report -> YoloVision run log -> output JSON/SVG -> sample-run validator -> owner review -> clean external package consumer。

## 下一步

下一步应把 YOLOv8n 六任务模板继续回填真实 owner 资产，并扩展到 YOLOv5、YOLOv6、YOLOv7、YOLOv9、YOLOv10、YOLO11、YOLO26 和 custom 模型。每篇模型案例文章都要包含模型获取、license、导出命令、TensorRtExec/OnnxToEngine build-only report、YoloVision run command、输出 JSON、可视化截图和 proof boundary；在 owner 提供真实 public package source、clean consumer、hash、host metadata 和 strict validator 前，不要把任何样例证据写成 package-consumer-runtime proof。
