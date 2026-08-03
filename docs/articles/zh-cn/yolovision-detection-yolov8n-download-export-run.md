# YoloVision YOLOv8n Detection 下载、导出与运行

> 本文的 owner 执行路径已经在 2026-08-02 使用官方 Ultralytics `v8.3.0` `yolov8n.pt`、真实 bus 图片、TensorRT 10.11 与 CUDA 12.9 完成。实际清单、reference 工具、短证据和逐值/逐框结果分别位于 `samples/assets/yolovision-yolov8n-det-official-assets.json`、`eng/Invoke-YoloVisionDetectionReference.py`、`samples/assets/yolovision-yolov8n-det-real-model-runtime-evidence.json` 和 `yolovision-yolov8-det-real-asset-tutorial.md`。本文后续通用命令仍可用于其他模型，但不再代表当前官方案例“等待 owner 回填”。

## 适用读者

本文适合以下读者：

- 希望从 YOLOv8n 权重开始完成 ONNX 导出、TensorRT engine 构建和 C# 推理的开发者。
- 正在评估 TensorRtSharp4.0、TensorRtExec 与 YoloVision 是否适合真实视觉模型部署的工程团队。
- 需要为模型来源、许可证、输入输出、运行日志和 SHA256 建立可审计证据链的维护者。

读者应具备基本的 Python、ONNX、CUDA/TensorRT 和 .NET 命令行使用经验。首次接触 TensorRT 的用户可以先阅读源码编译、版本矩阵和 Windows 安装文章，再执行本文命令。

## 解决问题

本文解决四个常见问题：

1. YOLOv8n detection 模型从哪里获取，如何记录许可证和 hash。
2. 如何导出具有明确输入 shape 和输出 layout 的 ONNX。
3. 如何分别使用 TensorRtExec 完成 build-only、使用 YoloVision 完成真实输入运行与检测后处理。
4. 如何区分 build-only、real-model-runtime、package-consumer-runtime 和 post-publish proof，避免把一份本地报告写成发布证明。

## 背景与场景

这篇文章面向想把 YOLOv8n detection 从 PyTorch 权重一路跑到 TensorRT + C# 样例的开发者。目标不是展示一条“看起来能跑”的命令，而是把模型来源、ONNX 导出、TensorRtExec build-only、YoloVision 运行、输出字段和 proof 边界都放到同一个可复查路径里。

YoloVision 是统一 YOLO-family 样例入口，覆盖 YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det、cls、seg、obb、pose、sem 任务。YOLOv8n detection 适合作为第一篇真实资产文章：模型小、输出 layout 常见、COCO labels 容易核对，能帮助用户确认 CUDA、TensorRT、C# wrapper 和样例后处理是否协同工作。

## 适用场景

当你已经有一台可运行 TensorRT 的 Windows 或 Linux 兼容主机，并希望验证一个真实 YOLO 检测模型时，可以从本文开始。本文假设模型资产由 owner 自行下载和确认许可证；仓库不内置权重、图片、engine 或私有日志。

这条路径适合做三类工作：一是本地验证 TensorRtSharp4.0 的模型部署体验；二是为公众号或博客写一篇完整的 YOLOv8n deployment walkthrough；三是复现仓库已有的 `real-model-runtime` 证据。换用其他模型、图片或导出器时仍需建立独立记录。

## 操作路径

完整路径建议固定为以下顺序：

1. 从明确来源获取 YOLOv8n 权重和 COCO labels，记录许可证与 SHA256。
2. 使用固定 Python/Ultralytics/opset 参数导出 ONNX。
3. 检查 ONNX 输入名、输入 shape、输出名、输出 shape 和是否包含 graph-side NMS。
4. 使用 TensorRtExec 生成 build-only report 与 engine。
5. 将真实图片按训练/导出约定预处理为 NCHW float32 tensor。
6. 使用 YoloVision 执行真实 inference、decode 和 NMS。
7. 保存模型、labels、图片、预处理 tensor、engine、输出 JSON 和日志的 SHA256。
8. 通过 owner validator 后再决定是否可以标记为 `real-model-runtime`。

## 模型与许可证

建议 owner 明确记录这些字段：

- `modelSourceUrl`：模型权重的来源页面或 release 地址。
- `modelLicense`：模型和训练数据对应的许可证。
- `modelSha256`：下载后的 `.pt` 或导出后的 `.onnx` SHA256。
- `labelsSha256`：`coco.names` 或自定义 labels 文件 SHA256。
- `exportToolVersion`：Ultralytics 包版本、Python 版本、opset 和导出参数。

这些字段必须写入真实资产记录。缺少任何一项，都不能把文章、模板、截图或 matrix 晋级为 runtime proof。

## 导出 ONNX

示例命令如下：

```powershell
yolo export model=.\models\yolov8n.pt format=onnx opset=12 dynamic=True simplify=True imgsz=640
```

导出后建议立即记录：

- 输入张量名，例如 `images`。
- 输入 shape，例如 `1x3x640x640`。
- 输出张量名、shape 和 layout。
- 是否包含 graph-side NMS。
- class count、score 字段和 objectness 字段。

YoloVision 可以使用 `--layout auto` 和 `--has-objectness auto` 做保守推断，但真正可发布的文章应该把实际输出说明写清楚。

## TensorRtExec Build-Only

先用 TensorRtExec 生成 engine 或 build report：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n.onnx `
  --saveEngine .\models\yolov8n.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\yolov8n-build-report.json
```

这一步可以验证 ONNX parser、profile shape、precision flag、engine serialization 和 report schema。它仍然只是 build-only evidence。即使 report 中记录了 TensorRT 版本、CUDA 版本、engine 路径和 hash，也不能替代 YoloVision 真实输入运行日志。

## YoloVision 离线 Preflight

在准备真实运行前，先生成离线配置和资产预检报告：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n.onnx --labels .\models\coco.names --input-data .\models\yolov8n-det-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --has-objectness auto --nms-mode class-aware --preflight --preflight-report .\models\yolov8n-det-preflight.json
```

报告必须标记为 `yolovision-preflight.v1` 和 `proofClassification=precheck`，并明确 `TensorRT/ONNX parser/engine/inference` 均未执行。它只帮助 owner 发现路径、SHA256 和输出 metadata 缺口，不能替代后面的 `YoloVision Passed=True` 真实运行日志。

## YoloVision 运行

准备一个真实图片，预处理成 NCHW float32 tensor 后运行：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8n.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8n-det-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware
```

真实运行日志至少应包含：

- `Profile Family=v8 Task=det`
- `InputSource=external`
- `Postprocess Task=det`
- `Detection Class=... Score=... BoxCxCyWh=...`
- owner 提供的 YoloVision 成功标记日志行

如果使用 synthetic tensor，只能证明托管后处理路径没有崩溃，不能证明模型能检测真实图片。

## 输出字段解释

Detection 输出建议按如下字段记录：

- `classCount`：labels 数量，必须与模型输出类别数一致。
- `outputLayout`：例如 `[1,84,8400]` 或 `[1,8400,84]`。
- `hasObjectness`：是否存在 objectness 分支。
- `scoreThreshold`：候选框保留阈值。
- `iouThreshold`：NMS 阈值。
- `nmsMode`：`class-aware` 或 `class-agnostic`。
- `topDetections`：用于人工复核的前若干个检测框。

这些字段应该进入样例输出 JSON 或 sample-run-evidence record，便于后续 owner review。

## 代码与文件入口

仓库中的关键入口如下：

- `samples/YoloVision/Program.cs`：CLI 参数解析、模型 profile 和运行入口。
- `samples/YoloVision/YoloVisionRuntimePipeline.cs`：真实 runtime 输出到统一视觉结果的管线。
- `samples/YoloVision/YoloVisionDetectionDecoder.cs`：检测输出 layout、score 与 box 解码。
- `samples/YoloVision/YoloVisionNms.cs`：class-aware 与 class-agnostic NMS。
- `samples/YoloVision/yolovision-task-output-contract.json`：det/cls/seg/obb/pose/sem 输出角色契约。
- `samples/assets/yolovision-yolox-s-example.json`：模型 profile 示例，可按 YOLOv8n 实际输出调整。
- `applications/TensorRtExec`：ONNX build、engine serialization 与 report 输出。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：真实资产候选字段验证。

文章中的命令应该与这些源码入口保持一致。新增参数时，需要同步更新 CLI、README、文章、JSON schema 和 ProjectQuality 测试。

## 可复用资产目录与完整运行产物

为了让文章命令可以重复执行，建议为 YOLOv8n detection 建立独立的 E 盘 case workspace：

..\downloads\cases\yolov8n-det\models
..\downloads\cases\yolov8n-det\labels
..\downloads\cases\yolov8n-det\images
..\downloads\cases\yolov8n-det\tensors
..\downloads\cases\yolov8n-det\engines
..\downloads\cases\yolov8n-det\reports
..\downloads\cases\yolov8n-det\logs

从 samples/assets/yolovision-yolov8-det-candidate.template.json 复制候选记录后，先回填 model.sourceUrl、model.downloadUrl、model.license、model.licenseEvidence、model.sha256、model.opset 和 model.onnxExportCommand。labels 和 input 也必须分别记录来源、许可证、路径和 SHA256，不能只记录 ONNX。

下载和导出后至少计算这些 hash：

Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-det\models\yolov8n.pt
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-det\models\yolov8n.onnx
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-det\labels\coco.names
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-det\images\dog.ppm

使用真实图片时，先把预处理步骤独立保存：

dotnet run --project .\samples\YoloVision -- --preprocess-only --image ..\downloads\cases\yolov8n-det\images\dog.ppm --preprocessed-output ..\downloads\cases\yolov8n-det\tensors\dog-fp32.bin --input-shape 1x3x640x640 --tensor-layout NCHW --color-order RGB --resize letterbox

运行 YoloVision 时同时导出结构化结果和可视化：

dotnet run --project .\samples\YoloVision -- --model ..\downloads\cases\yolov8n-det\models\yolov8n.onnx --labels ..\downloads\cases\yolov8n-det\labels\coco.names --input-data ..\downloads\cases\yolov8n-det\tensors\dog-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45 --output-json ..\downloads\cases\yolov8n-det\reports\yolov8n-det-output.json --visualization-svg ..\downloads\cases\yolov8n-det\reports\yolov8n-det-output.svg

运行结束后，应对 preprocessed tensor、engine、build report、output JSON、SVG 和 stdout/stderr log 计算 SHA256。output JSON 至少要能回答 classCount、outputLayout、hasObjectness、scoreThreshold、iouThreshold、nmsMode、modelSha256、labelsSha256、imageSha256 和 preprocessedTensorSha256。

建议按以下顺序验证：

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1

candidate template 中的 proofChecklist.requiredEvidenceLines、proofChecklist.requiredHashes、stdoutSummary、stderrSummary 和 packageConsumerBoundary 都必须由 owner 真实回填。只有 validator 接受 YoloVision Passed=True、真实 hash、日志和 owner review 后，才可以形成 real-model-runtime 候选；E 盘 case workspace、output JSON、SVG 和截图本身仍不是 package-consumer-runtime proof。

## 图示建议

发布到微信公众号或博客时，建议至少准备六张图：

1. 模型下载页、release tag 与许可证位置截图。
2. ONNX 导出命令和成功输出截图。
3. Netron 中输入 `images` 与输出 tensor shape 截图。
4. TensorRtExec build report 的 engine、TensorRT、CUDA 和 profile 字段截图。
5. YoloVision 终端中的真实输入、top detections 和运行耗时截图。
6. 原图与检测框可视化结果，附模型、图片和输出 JSON 的短 hash。

截图只能辅助说明，不能替代原始日志、结构化输出、SHA256 或 runtime proof validator。

## 常见问题

如果输出检测框为空，优先检查 labels 数量、输入归一化、RGB/BGR、letterbox 参数和 layout。很多“模型跑通但没结果”的问题，根因是预处理和导出脚本不一致。

如果 TensorRtExec 可以 build，但 YoloVision 不能 run，应分别检查 engine/runtime package 版本、输入 shape profile、CUDA driver、TensorRT DLL/SO 搜索路径和 ONNX 输出张量名。Build-only 成功只能证明构建路径，不保证样例后处理正确。

如果输出框数量异常多，通常要检查 objectness、class score 乘法、score threshold、NMS 模式和 graph-side NMS 是否重复执行。

## 边界说明（Proof Boundary）

本文、`samples/assets/yolovision-article-case-pack.json`、YoloVision matrix、TensorRtExec report、OnnxToEngine report、sidecar-only report、screenshot、template、dry-run、build-only、local feed、ProjectReference、direct `.nupkg`、readonly diagnostics 都不是 runtime proof。

只有 owner 提供真实模型、labels、输入、运行日志、SHA256、stdout/stderr summary、许可证说明，并通过对应 validator 后，才可能晋级为 `real-model-runtime`。`package-consumer-runtime` 和 post-publish proof 还需要独立的公开包与外部 consumer 证据。

## Owner Backfill Checklist

- 填写模型来源、许可证、下载时间和 SHA256。
- 保存 ONNX 导出命令、导出工具版本和导出日志。
- 保存 TensorRtExec build-only 命令、report JSON 和 engine hash。
- 保存 YoloVision 真实运行命令、run log 和 `YoloVision Passed=True`。
- 填写输入图片、预处理 tensor、labels 和输出 JSON 的 SHA256。
- 记录 host OS、GPU、driver、CUDA、TensorRT 和 runtime package 信息。
- 明确 owner review 结论：能否作为 `real-model-runtime` 候选。

## 下一步

YOLOv8n detection、classification、segmentation、pose 与 OBB 的源码树真实案例均已完成。下一步建议继续沿同一证据结构扩展：

1. 为 semantic segmentation 选择许可证与输出合同明确的真实模型。
2. 为 YOLOv5/v6/v7/v9/v11/v26 逐个记录实际输出 layout 差异，而不是仅修改 family 字符串。
3. 使用 [YOLOv8n Detection 本地包消费教程](yolovision-yolov8n-det-local-package-consumer-tutorial.md) 复现仓库外三包运行；其 `local-package-consumer-runtime` 证据已经完成，但仍不是公共包证明。
4. 公开包发布后再建立 post-publish proof；本地源码树结果不能替代它。

只有真实模型、真实输入、可复现命令、完整输出和 validator 同时成立，文章中的运行结果才可以作为公开案例引用。
