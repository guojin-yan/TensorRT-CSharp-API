# YoloVision 真实资产接入教程

> 2026-07-19 状态更新：官方 YOLOX-S 0.1.1rc0 已完成 E 盘 hash 固定获取、内置预处理、TRT10 build、真实图片 enqueue、raw grid/stride 解码和严格 `real-model-runtime` 校验。当前可直接复现的完整流程见 [YoloVision 官方 YOLOX-S 下载、构建与真实图片运行教程](yolovision-yolox-official-runtime-tutorial.md)。本文后续内容继续保留为其他 YOLO family、自定义模型和 owner 自备资产的通用回填方法。

本文把 `applications/YoloVision` 从“synthetic input 管线样例”推进到“可接真实模型的操作流程”。仓库不会直接打包 YOLO 权重、COCO labels 或测试图片；这些资产有独立许可证、体积和再分发要求。正确做法是把资产选择、下载、hash、转换、构建和运行日志全部记录在 manifest 和报告里。

读完本文后，你应该能得到三类材料：

- 一份 `YoloVision` 可执行命令，说明模型、labels、预处理 tensor、shape、layout 和后处理参数如何进入样例。
- 一份 TensorRtExec build-only 报告，说明 ONNX parser/builder 是否能处理该模型。
- 一组证据记录，包括 asset manifest、evidence sidecar 和 sample run evidence record，说明哪些证据已齐全、哪些仍是 owner action。

仓库仍不会提交 YOLOX-S 权重、engine 或图片；这些大文件保留在外层 E 盘下载工作区。但官方链的真实 SHA256、`YoloVision Passed=True` 日志、JSON/SVG 和严格 validator 已记录在 `artifacts/yolovision/yolox-official-runtime`。其他模型不得照抄这些值，仍应按本文格式产生自己的真实记录。

## 环境准备

建议从仓库根目录运行命令：

```powershell
cd .
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

你需要准备：

- 可用的 NVIDIA driver、CUDA、TensorRT runtime。
- 能运行本仓库 .NET 项目的 SDK。
- YOLOX 源仓库或已导出的 ONNX 模型。
- 与模型匹配的 COCO labels 或自定义 labels。
- 一张许可证允许用于教程或内部验证的测试图片。
- 由该图片生成的 float32 预处理 tensor，shape 必须与 `--input-shape` 一致。

如果 CUDA driver/runtime 不匹配，样例可能会报告 `blocked-by-cuda-driver`。这属于环境阻塞，不应写成 TensorRtSharp API 缺口。

## 候选模型

本轮选择 YOLOX-S 作为候选示例，而不是直接内置模型。选择理由：

- YOLOX 官方仓库公开，仓库许可证为 Apache-2.0。
- 官方文档包含 ONNXRuntime 转换流程和 ONNX 模型表。
- 官方文档也有 TensorRT demo 方向，适合后续扩展到 TensorRtSharp 样例。

需要注意：代码仓库许可证不自动等于权重、labels、图片都可以随包再分发。发布 owner 必须复核权重下载页、测试图片来源和 labels 来源，再决定是否只写教程、是否允许打包、是否只能用户自备。

参考：

- YOLOX 仓库：<https://github.com/Megvii-BaseDetection/YOLOX>
- YOLOX 许可证：<https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE>
- YOLOX ONNXRuntime 文档：<https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html>
- YOLOX TensorRT 文档：<https://yolox.readthedocs.io/en/latest/demo/trt_py_readme.html>

## 资产清单

先复制示例清单：

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Copy-Item .\samples\assets\yolovision-yolox-s-example.json .\models\yolox_s.assets.json
```

然后补齐这些字段：

- `model.downloadUrl`
- `model.sha256`
- `labels.sha256`
- `input.sourceUrl`
- `input.license`
- `input.sha256`
- `inputTensor.localPath`
- `inputTensor.sha256`
- `inputTensor.command`
- `tensor.inputName`
- `tensor.outputName`
- `tensor.outputShape`
- `postprocess.hasObjectness`
- `evidence.lastRunLog`

在这些字段没有补齐前，`isSmokePassed` 必须保持 `false`。

建议同时生成证据模板：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
```

然后把下面两份模板复制到你的 `models` 目录旁边，作为 owner 回填记录：

- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolox-s.template.json`
- `artifacts/user-acceptance/sample-run-evidence-record.yolox-s.template.json`

第一份连接 build report 和模型资产；第二份连接真实 `YoloVision` 运行日志和样例证据。

## 导出 ONNX

在 YOLOX 仓库内按官方文档导出：

```powershell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```

如果实际使用的是自定义 exp 文件，命令应改成：

```powershell
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth
```

导出后计算 hash：

```powershell
Get-FileHash .\models\yolox_s.onnx -Algorithm SHA256
```

把结果写入 manifest。不要只记录文件名。

同样需要计算 labels、测试图片和后续运行日志的 hash：

```powershell
Get-FileHash .\models\coco.names -Algorithm SHA256
Get-FileHash .\models\yolox-test.jpg -Algorithm SHA256
Get-FileHash .\models\yolox_s-preprocessed-fp32.bin -Algorithm SHA256
```

运行日志的 SHA256 要等 `YoloVision` 实跑后再计算。不要提前填占位 hash。

## 先用 TensorRtExec 构建 engine

外部模型第一步先做 build-only，确认 ONNX parser 和 builder 能处理模型：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1

dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolox_s.onnx `
  --saveEngine .\models\yolox_s.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --fp16 `
  --workspace 512 `
  --buildOnly `
  --evidenceSidecar .\models\yolox_s-evidence.sidecar.json `
  --exportReport .\models\yolox_s-build-report.json
```

这份报告是 build evidence，不是检测质量 proof。只有当 `InferenceRan=true` 且输出语义、后处理和真实图片日志都存在时，才能讨论 runtime execution proof。

`models/yolox_s-evidence.sidecar.json` 可以从 `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolox-s.template.json` 起步。初始 `proofClassification` 应保持 `template-only` 或在 build 成功后改为 `build-only`；不要在 build-only 阶段写成 `real-model-runtime`。

## 准备预处理 Tensor

`YoloVision` 不在 runner 内硬编码 YOLOX 的图片解码、resize、letterbox、BGR/RGB 或 padding 规则。请用模型导出脚本、Python/OpenCV、训练仓库里的 preprocessing 逻辑，先把 `yolox-test.jpg` 转成与模型输入完全一致的 float32 tensor，例如：

```powershell
python .\tools\preprocess_yolox_input.py `
  --image .\models\yolox-test.jpg `
  --output .\models\yolox_s-preprocessed-fp32.bin `
  --shape 1x3x640x640
```

这条命令只是 owner 需要替换的占位示例，不能直接当作仓库已提供脚本。真实使用时要把实际命令写入 `inputTensor.command`，并记录 `yolox_s-preprocessed-fp32.bin` 的 SHA256。

## 运行 YoloVision

确认 input/output 名称、shape、layout 后运行：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolox_s.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolox_s-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family custom `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45
```

如果输出 shape 无法自动判断，显式设置 layout 和 objectness。不要把 `auto` 当作模型规范。

建议把运行输出保存成日志：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolox_s.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolox_s-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family custom `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45 *> .\models\yolox_s-sample-run.log
```

日志中至少应能看到：

- `YoloVision TensorRtLine=...`
- `Profile Family=... Task=... Layout=... Nms=... NmsMode=...`
- `InputSource=external InputFile=...`
- `Input=... Output=...`
- `Detection Class=... Score=... BoxCxCyWh=...` 或 `Detections=0 ...`
- `YoloVision Passed=True`

如果没有 `Passed=True`，不要把 manifest 或 sample run evidence record 晋级。

## 验证记录

一次可提升为真实样例 evidence 的运行至少需要：

- build report JSON 或 Markdown。
- `models/yolox_s-evidence.sidecar.json`，并通过 `eng/Test-OnnxEngineBuildEvidenceSidecar.ps1`。
- `YoloVision Passed=True` 日志。
- 至少一张可再分发测试图片。
- 与该图片对应的预处理 float tensor、生成命令和 SHA256。
- 检测输出中 class、score、box 坐标可读。
- manifest 中 hash、来源、许可证字段完整。
- 文档写明该模型是否可以随包再分发。

如果本机因为 CUDA driver/runtime mismatch 失败，应记录为环境阻塞，不要写成 API 缺口，也不要把 build-only 报告改写成 smoke passed。

推荐的收尾命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 `
  -InputPath .\models\yolox_s-sample-run-evidence.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

`sample-run-evidence-record.yolox-s.template.json` 中的 `canPromoteRealModelRuntime` 只有在以下条件同时成立时才应设为 `true`：

- `proofClassification=real-model-runtime`
- `isSmokePassed=true`
- `modelSha256`、`labelsSha256`、`inputAssetSha256`、`sampleRunLogSha256` 都是 64 位 SHA256。
- `preprocessedInputTensorSha256` 已记录，且与 `--input-data` 文件一致。
- `sampleRunLogPath` 指向真实日志。
- `stdoutSummary` 或 `stderrSummary` 已人工摘要。
- asset manifest、evidence sidecar、sample run evidence record 的 hash 和许可证说明一致。

## 常见问题

1. `Parsed=False`：先用 TensorRtExec 报告定位 parser 错误，再检查 opset、unsupported op 或 plugin 需求。
2. `EngineSaved=False`：检查 workspace、FP16/INT8 设置、dynamic shape profile 是否完整。
3. 没有检测框：确认输出 layout、objectness、class count、confidence threshold 和 NMS 规则。不要只降低阈值来制造结果。
4. 输出 shape 无法自动判断：把 manifest 里的 `tensor.outputShape` 和 `postprocess` 字段补完整，再显式传入 layout/objectness。
5. CUDA driver mismatch：记录为 `blocked-by-cuda-driver`，不要写成 sample failed 或 API bug。
6. 权重许可证不清楚：保持 `isRedistributableInRepository=false`，文章只写用户自备流程。

## 小结

`YoloVision` 的价值不是把一个固定 YOLO 权重塞进仓库，而是把 YOLO-family 部署的关键变量拆开：模型来源、许可证、ONNX 导出、TensorRT build、输入图片、labels、输出 layout、后处理、运行日志和证据记录。这样写出来的教程既能落地，也能经得起后续换 YOLOv5/v8/v11/v26 或 det/seg/pose/obb/sem 任务时继续复用。
