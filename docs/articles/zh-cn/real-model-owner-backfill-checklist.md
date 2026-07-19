# 真实模型 Owner 回填 Checklist：Classification、YoloVision 与 YOLOX-S

> 官方 YOLOX-S 的源码树真实运行链已于 2026-07-19 完成，见 `docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md` 和 `artifacts/yolovision/yolox-official-runtime`。本 checklist 仍用于 Classification、其他 YOLO family、自定义资产、公开再分发审批以及 package consumer 等尚需 owner 输入的路径；不得用已完成的源码树运行替代公开发布审批。

这份 checklist 给 release owner 或样例维护者使用。它不下载模型、不替你确认许可证，也不把候选资产写成已通过；它的作用是把真实模型样例从“模板存在”推进到“证据齐全、可审计、可被 release evidence bundle 聚合”的状态。

适用范围：

- `samples/Classification`
- `samples/YoloVision`
- `samples/assets/classification-assets.template.json`
- `samples/assets/yolovision-assets.template.json`
- `samples/assets/yolovision-yolox-s-example.json`

## 总原则

每个真实模型样例都需要五类证据：

1. 资产来源：模型、labels、输入图片的 source URL、download URL、license。
2. 文件完整性：模型、labels、输入图片、预处理后 tensor、运行日志的 SHA256。
3. build 证据：`TensorRtExec` 或 `OnnxToEngine` 的 build-only report。
4. runtime 证据：`Classification Passed=True` 或 `YoloVision Passed=True` 的真实 runner 日志。
5. 聚合证据：manifest audit、sidecar audit、sample run evidence validation、user acceptance catalog、release evidence bundle。

不能用下面任意一项替代真实样例 runtime：

- build-only report。
- DependencyProbe。
- synthetic input runtime。
- sidecar。
- acquisition plan。
- owner 手写说明。
- `docs/_site` 生成成功。

## 本地文件布局

建议把资产放在仓库根目录下的本地 `models\` 文件夹，默认不提交：

```text
models/
  classifier.onnx
  classifier.labels.txt
  classifier.input.png
  classifier.plan
  classifier-build-report.json
  classifier-evidence.sidecar.json
  classifier-sample-run-evidence.json
  classifier-run.log

  yolo.onnx
  coco.names
  yolo.input.jpg
  yolo-preprocessed-fp32.bin
  yolo.plan
  yolo-build-report.json
  yolovision-evidence.sidecar.json
  yolovision-sample-run-evidence.json
  yolovision-run.log

  yolox_s.onnx
  yolox-test.jpg
  yolox_s-preprocessed-fp32.bin
  yolox_s.plan
  yolox_s-build-report.json
  yolox_s-evidence.sidecar.json
  yolox_s-sample-run-evidence.json
  yolox_s-run.log
```

提交前先复核 license 和体积。ONNX、engine、图片和原始日志通常只保留在 owner 本机。

## Classification 回填

从模板生成本地记录：

```powershell
Copy-Item .\samples\assets\classification-assets.template.json .\models\classifier.assets.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
```

准备并记录：

- `classifier.onnx`
- `classifier.labels.txt`
- `classifier.input.png`
- 模型 source URL、download URL、license。
- labels source URL、license、class count。
- 图片 source URL、license。
- input tensor name、output tensor name、input shape、output shape。
- preprocess：resize、crop、color order、scale、mean、std。

计算 SHA256：

```powershell
Get-FileHash .\models\classifier.onnx -Algorithm SHA256
Get-FileHash .\models\classifier.labels.txt -Algorithm SHA256
Get-FileHash .\models\classifier.input.png -Algorithm SHA256
```

生成 build-only report：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\classifier.onnx `
  --saveEngine .\models\classifier.plan `
  --minShapes input:1x3x224x224 `
  --optShapes input:1x3x224x224 `
  --maxShapes input:1x3x224x224 `
  --buildOnly `
  --exportReport .\models\classifier-build-report.json `
  --evidenceSidecar .\models\classifier-evidence.sidecar.json
```

运行样例并保存日志：

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\classifier.labels.txt `
  --input .\models\classifier.input.png `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --top-k 5 *> .\models\classifier-run.log
```

日志必须包含：

```text
Classification TensorRtLine=...
TopK Index=... Label=... Score=...
Classification Passed=True
```

然后计算日志 hash：

```powershell
Get-FileHash .\models\classifier-run.log -Algorithm SHA256
```

## YoloVision 回填

从模板生成本地记录：

```powershell
Copy-Item .\samples\assets\yolovision-assets.template.json .\models\yolo.assets.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
```

准备并记录：

- `yolo.onnx`
- `coco.names`
- `yolo.input.jpg`
- YOLO family：`v5|v6|v7|v8|v9|v10|v11|v26|custom`
- task：`det|cls|seg|obb|pose|sem`
- input tensor name、output tensor names、output roles、shape、layout。
- preprocess：letterbox、padding、scale、color order、mean/std，以及生成 `--input-data` 所需 float tensor 的命令。
- postprocess：layout、objectness、class count、confidence、IoU、NMS mode。
- seg/pose/obb/sem 的 metadata。

计算 SHA256：

```powershell
Get-FileHash .\models\yolo.onnx -Algorithm SHA256
Get-FileHash .\models\coco.names -Algorithm SHA256
Get-FileHash .\models\yolo.input.jpg -Algorithm SHA256
Get-FileHash .\models\yolo-preprocessed-fp32.bin -Algorithm SHA256
```

生成 build-only report：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolo.onnx `
  --saveEngine .\models\yolo.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --buildOnly `
  --exportReport .\models\yolo-build-report.json `
  --evidenceSidecar .\models\yolovision-evidence.sidecar.json
```

运行样例并保存日志：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolo-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45 *> .\models\yolovision-run.log
```

日志必须包含：

```text
YoloVision TensorRtLine=...
Profile Family=... Task=... Layout=... Nms=... NmsMode=...
InputSource=external InputFile=...
Postprocess Task=...
YoloVision Passed=True
```

`--input-data` 应指向已经完成 resize、letterbox、通道顺序、scale、mean/std 的 float32 tensor。原始图片路径仍要写入 manifest 用于 license 和可复现性审计，但 runner 不再把图片路径伪装成已预处理 tensor。

## YOLOX-S 回填

YOLOX-S 走 `samples/assets/yolovision-yolox-s-example.json`，它默认是 `build-only` 候选，不是 runtime proof。owner 必须自己复核：

- YOLOX 权重来源。
- YOLOX 权重 license。
- COCO labels license。
- 测试图片 license。
- 预处理 tensor 生成命令、shape、hash。
- ONNX export command。
- output tensor shape 和 objectness 规则。

示例命令：

```powershell
Copy-Item .\samples\assets\yolovision-yolox-s-example.json .\models\yolox_s.assets.json

dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolox_s.onnx `
  --saveEngine .\models\yolox_s.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --fp16 `
  --workspace 512 `
  --buildOnly `
  --exportReport .\models\yolox_s-build-report.json `
  --evidenceSidecar .\models\yolox_s-evidence.sidecar.json

dotnet run --project .\samples\YoloVision -- `
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
  --iou-threshold 0.45 *> .\models\yolox_s-run.log
```

## Sidecar 回填

从模板复制：

```powershell
Copy-Item .\artifacts\user-acceptance\onnx-engine-build-evidence-sidecar.classification.template.json .\models\classifier-evidence.sidecar.json
Copy-Item .\artifacts\user-acceptance\onnx-engine-build-evidence-sidecar.yolovision.template.json .\models\yolovision-evidence.sidecar.json
Copy-Item .\artifacts\user-acceptance\onnx-engine-build-evidence-sidecar.yolox-s.template.json .\models\yolox_s-evidence.sidecar.json
```

回填：

- `proofClassification`
- `stdoutSummary`
- `stderrSummary`
- `modelEvidence.modelSha256`
- `modelEvidence.modelLicense`
- `modelEvidence.inputAssetSha256`
- `modelEvidence.preprocessedInputTensorSha256`，适用于 YoloVision。

校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1 -SidecarPath .\models\classifier-evidence.sidecar.json
```

## Sample Run Evidence Record 回填

从模板复制：

```powershell
Copy-Item .\artifacts\user-acceptance\sample-run-evidence-record.classification.template.json .\models\classifier-sample-run-evidence.json
Copy-Item .\artifacts\user-acceptance\sample-run-evidence-record.yolovision.template.json .\models\yolovision-sample-run-evidence.json
Copy-Item .\artifacts\user-acceptance\sample-run-evidence-record.yolox-s.template.json .\models\yolox_s-sample-run-evidence.json
```

真实晋级时必须设置：

- `recordKind=sample-run-evidence-record`
- `templateOnly=false`
- `proofClassification=real-model-runtime`
- `modelSha256`
- `labelsSha256`
- `inputAssetSha256`
- `preprocessedInputTensorSha256`，适用于 YoloVision。
- `sampleRunLogPath`
- `sampleRunLogSha256`
- `stdoutSummary` 或 `stderrSummary`
- `isSmokePassed=true`
- `canPromoteRealModelRuntime=true`

校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -InputPath .\models\classifier-sample-run-evidence.json -RequireExistingLog
```

`package-consumer-runtime` 禁止出现在 sample run evidence record 中。

## Manifest Cross-Check

回填 manifest 后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

manifest audit 会检查：

- `sampleName` 是否能映射到 `samples/<sampleName>/<sampleName>.csproj`。
- `sampleRunEvidenceRecord` 是否存在。
- record 存在时 `sampleName` 是否匹配。
- record 存在时 model/labels/input SHA256 是否匹配。
- record 是否错误声明 `package-consumer-runtime`。

模板中 record 缺失时是 `owner-action-required`，不是 error。

## Real Model Owner Handoff

`Export-RealModelOwnerHandoff.ps1` 会读取 `samples/assets/*.template.json`、`samples/assets/*-example.json`、manifest audit、asset acquisition plan、sample run evidence validation 和 sidecar audit，生成：

- `artifacts/user-acceptance/real-model-owner-handoff.json`
- `artifacts/user-acceptance/real-model-owner-handoff.md`

这份 handoff 把每个样例的 owner 动作集中在一处：

- license review。
- 模型、labels、输入图片获取。
- 模型、labels、输入图片、预处理后 tensor、运行日志 SHA256。
- TensorRtExec/OnnxToEngine build-only report。
- Classification/YoloVision 真实 runner 命令。
- sample run evidence record 从模板复制到本地 record 的路径。
- `Test-SampleRunEvidenceRecord.ps1`、`Test-SampleAssetManifest.ps1`、`Export-UserAcceptanceSampleCatalog.ps1`、`Export-ReleaseEvidenceBundle.ps1` 的校验链。

默认状态仍然是：

- `handoffState=owner-action-required`
- `performsDownload=false`
- `performsSampleRun=false`
- `canPromoteRealModelRuntime=false`

它是 owner 回填指南，不是 `Classification Passed=True` 或 `YoloVision Passed=True` 的证据。只有真实日志、hash、sidecar、sample run evidence record 和 validator 都齐全时，样例才可能晋级 `real-model-runtime`。

## Release Evidence Bundle 检查

最后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

检查 `artifacts/final-release/release-evidence-bundle.json`：

- `sampleRunEvidenceValidationState`
- `sampleRunEvidenceCanPromoteRealModelRuntime`
- `sampleRunEvidenceProofClassification`
- `realModelOwnerHandoffState`
- `realModelOwnerHandoffCanPromoteRealModelRuntime`
- `sample-asset-manifest-audit`
- `sample-run-evidence-validation`
- `real-model-owner-handoff`
- `onnx-engine-build-evidence-sidecar-audit`

只有真实文件、hash、license、日志和 validator 都齐全时，Classification/YoloVision 才能推进到 `real-model-runtime`。即使如此，它仍然不是 `package-consumer-runtime`。

## 禁止事项

- 不要提交未经许可的模型、图片或日志。
- 不要伪造 SHA256。
- 不要把 build-only report 当成样例 runtime。
- 不要把 `owner-action-required` 写成通过。
- 不要在 sample manifest 或 sample run evidence record 中写 `package-consumer-runtime`。
- 不要把 `docs/_site` 是否生成成功当成模型运行成功。

## 完成判据

单个样例的真实模型证据完成，需要同时满足：

- manifest 为真实 owner 回填记录。
- sidecar 校验通过。
- sample run evidence record 校验通过。
- manifest audit 无 error。
- user acceptance catalog 显示 runner evidence 可晋级真实模型 runtime。
- release evidence bundle 聚合到对应状态。
- 文章和 README 不再称它只是候选资产。

在这些条件满足前，保持 `isSmokePassed=false`，保持 `owner-action-required`，这是对用户负责的边界。
