# Classification 真实资产接入教程：从 ResNet18 获取和 ONNX 转换到可审计运行证据

`samples/Classification` 是一个真实可运行的分类样例，但仓库不会内置模型、labels 或图片。原因很直接：权重、ImageNet labels、测试图片各自有许可证和体积边界。本文给出 owner 侧完整接入路径，让你把一个分类模型从“候选资产”推进到“可审计的真实模型 runtime 证据”，同时不把未实跑内容写成 smoke passed。

## 目标读者

你适合从这篇开始，如果你想回答这些问题：

- 我应该把 ONNX、labels、图片放在哪里。
- `TensorRtExec` build-only 报告和 `Classification Passed=True` 有什么区别。
- 什么时候可以把 manifest 改成 `real-model-runtime`。
- 如何让 release evidence bundle 看到 Classification 仍是 owner action required，或者看到它已经有真实 runner 证据。

## 模型获取与 ONNX 转换

第一版固定基线是 TorchVision ResNet18 `IMAGENET1K_V1`：torchvision `v0.25.0`、源码 commit
`8ac84ee75afb1c327902156b5336f56ad63b7e2f`、权重
`https://download.pytorch.org/models/resnet18-f37072fd.pth`。权重长度 `46,830,571` bytes，SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

从仓库根目录执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AllowDownload `
  -ExportOnnx `
  -PythonPath C:\Users\guoji\.conda\envs\ultralytics\python.exe
```

脚本通过 `eng/Export-ClassificationResNet18Onnx.py` 使用 PyTorch `2.10.0+cpu`、torchvision `0.25.0+cpu` 和 opset 17
导出 `images:[1,3,224,224] -> logits:[1,1000]`，并生成 1000 行 ImageNet labels。当前 ONNX 长度
`46,748,553` bytes，SHA256 `ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903`。

默认文件放在 Git 仓库外：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models\Classification\resnet18-torchvision-v0.25.0\
```

TorchVision 源码许可证是 BSD-3-Clause，但 pretrained weights、labels 和测试图片仍要由 owner 复核。模型、权重、labels
和图片都不能提交到 GitHub。固定记录见 `samples/assets/classification-resnet18-official-assets.json`；全部演示模型总表见
`docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`。

### 独立 reference 与当前源树实跑结果

先对 C# 内置预处理生成的固定 float32 tensor 运行独立 PyTorch/ONNX Runtime CPU reference：

```powershell
$root = 'E:\GitSpace\TensorRT-CSharp-API-4.0'
$case = '.\artifacts\classification\resnet18-torchvision-v0.25.0'
& C:\Users\guoji\.conda\envs\ultralytics\python.exe .\eng\Invoke-ClassificationResNet18Reference.py `
  --weights "$root\downloads\resnet18-torchvision-v0.25.0\source\resnet18-f37072fd.pth" `
  --onnx "$root\models\Classification\resnet18-torchvision-v0.25.0\resnet18-imagenet1k-v1.onnx" `
  --labels "$root\models\Classification\resnet18-torchvision-v0.25.0\imagenet1k.names" `
  --input-tensor "$case\dog-input.fp32.bin" `
  --output-directory "$case\reference"
```

脚本重算 model、input、preprocess、output contract、labels 和 task semantics 六个 SHA256 指纹，生成 raw logits reference、
Softmax probabilities 任务 reference 和只把 index 0 加 `0.125` 的受控负例。PyTorch/ONNX Runtime 的 1000 个 logits 最大
绝对误差为 `7.62939453125e-6`，argmax 相同，概率和为 `0.9999999947211421`。

2026-08-03（Asia/Shanghai）使用 TensorRT 10.11、关闭 TF32、内置 `shorter-side-center-crop` 与 ImageNet mean/std 完成
源树实跑。raw logits 与任务 probabilities 各比较 1000 个值，均为 mismatch 0：raw 最大绝对误差 `9.536743e-6`，任务
概率最大绝对误差 `2.9802322e-7`。JSON 记录 `outputValidated=true`，日志结束于 `Classification Passed=True`；Top-1 是
`Samoyed`，score `0.8799871`。

受控负例保持 raw reference 正确，只篡改任务 probability index 0，得到 mismatch 1、first mismatch 0、最大绝对误差
`0.125`、exit code 1 和 `Classification Passed=False`。小型证据记录是
`samples/assets/classification-resnet18-real-model-runtime-evidence.json`。这证明源树真实模型主路径与 fail-closed 比较，仍不
替代 owner-reviewed golden、package consumer、公开包、再分发授权或发布后验证。

运行证据建议在同一外层用例目录保存为：

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
```

## 资产清单

复制模板作为 owner 本地 manifest：

```powershell
Copy-Item .\samples\assets\classification-assets.template.json .\models\classifier.assets.json
```

至少补齐：

- `model.sourceUrl`
- `model.license`
- `model.downloadUrl`
- `model.sha256`
- `model.opset`
- `labels.sourceUrl`
- `labels.license`
- `labels.sha256`
- `input.sourceUrl`
- `input.license`
- `input.sha256`
- `tensor.inputName`
- `tensor.outputName`
- `preprocess.resize/crop/mean/std`
- `evidence.evidenceSidecar`
- `evidence.sampleRunEvidenceRecord`

模板中的 `proofClassification=template-only`、`status=candidate-not-downloaded`、`isSmokePassed=false` 必须保持，直到真实运行证据齐全。

## SHA256

用 PowerShell 记录 hash：

```powershell
Get-FileHash .\models\classifier.onnx -Algorithm SHA256
Get-FileHash .\models\classifier.labels.txt -Algorithm SHA256
Get-FileHash .\models\classifier.input.png -Algorithm SHA256
```

真实运行后还要记录日志：

```powershell
Get-FileHash .\models\classifier-run.log -Algorithm SHA256
```

不要提前填假 hash。校验器只接受 64 位十六进制字符串，但真正的发布意义来自 hash 和文件实际一致。

## TensorRtExec Build-Only

先确认 ONNX 能构建 engine：

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

这一步只能证明 parser/builder/build report 路径，不能证明分类结果正确。build report 常见 classification 是 `build-only`，不是 sample runtime proof。

## Evidence Sidecar

生成模板：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
```

从 `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.classification.template.json` 复制到：

```text
models/classifier-evidence.sidecar.json
```

回填模型 SHA256、license、输入图片 SHA256、stdout/stderr summary 后校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1 -SidecarPath .\models\classifier-evidence.sidecar.json
```

sidecar 不能把 build-only report 晋级成 `package-consumer-runtime`。它只是把 build report 和真实资产信息连接起来。

## Classification 真实运行

下面的命令与仓库内 ResNet18 证据记录使用同一模型、图片预处理和独立 reference，并保存完整日志：

```powershell
$root = 'E:\GitSpace\TensorRT-CSharp-API-4.0'
$modelRoot = "$root\models\Classification\resnet18-torchvision-v0.25.0"
$case = '.\artifacts\classification\resnet18-torchvision-v0.25.0'
$image = '.\artifacts\yolovision\semantic-lraspp-reference\dog.ppm'

dotnet .\samples\Classification\bin\Release\net8.0\Classification.dll `
  --model "$modelRoot\resnet18-imagenet1k-v1.onnx" `
  --labels "$modelRoot\imagenet1k.names" `
  --image $image `
  --preprocessed-output "$case\dog-input.fp32.bin" `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --image-resize shorter-side-center-crop `
  --resize-shorter-side 256 `
  --tensor-layout NCHW `
  --color-order RGB `
  --scale 0.00392156862745098 `
  --mean 0.485,0.456,0.406 `
  --std 0.229,0.224,0.225 `
  --score-transform softmax `
  --top-k 5 `
  --noTF32 `
  --reference-output "$case\reference\classification.onnxruntime.reference.json" `
  --reference-abs 0.00001 `
  --reference-rel 0.0001 `
  --reference-outputs "logits:$case\reference\logits.onnxruntime.reference.json" `
  --reference-abs-tolerance 0.0001 `
  --reference-rel-tolerance 0.0001 `
  --output-json "$case\classification-positive-output.json" `
  *> "$case\classification-positive-run.log"
```

日志里至少应能看到：

```text
Classification TensorRtLine=...
TopK Index=... Label=... Score=...
Classification Passed=True
```

只有真实模型、真实 labels、真实输入图片和真实日志都存在时，才能考虑把 sample 证据晋级到 `real-model-runtime`。

## Sample Run Evidence Record

生成模板：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
```

从 `artifacts/user-acceptance/sample-run-evidence-record.classification.template.json` 复制到：

```text
models/classifier-sample-run-evidence.json
```

真实回填时设置：

- `recordKind=sample-run-evidence-record`
- `templateOnly=false`
- `sampleName=Classification`
- `proofClassification=real-model-runtime`
- `modelPath/modelSha256/modelLicense`
- `labelsPath/labelsSha256/labelsLicense`
- `inputAssetPath/inputAssetSha256/inputAssetLicense`
- `preprocessedInputTensorPath/preprocessedInputTensorSha256/preprocessedInputTensorElementCount`
- `evidenceSidecarPath` 和 `buildReportPath`
- `sampleRunCommand` 和 `expectedEvidenceLines`
- `sampleRunLogPath/sampleRunLogSha256`
- `stdoutSummary` 或 `stderrSummary`
- `isSmokePassed=true`
- `canPromoteRealModelRuntime=true`

然后校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 `
  -InputPath .\models\classifier-sample-run-evidence.json `
  -RequireExistingLog `
  -FailOnNotProof
```

sample run evidence record 不允许写 `package-consumer-runtime`。NuGet/runtime package consumer proof 是 release proof record 的职责。
仓库内固定 ResNet18 记录执行同一严格命令后得到 `ValidationState=real-model-runtime`、`ErrorCount=0`、
`OwnerActionRequiredCount=0`。

## Manifest 与 Catalog

回填后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

`Test-SampleAssetManifest.ps1` 会在 sample run evidence record 存在时检查：

- record `sampleName` 是否等于 `Classification`
- record `modelSha256` 是否匹配 manifest `model.sha256`
- record `labelsSha256` 是否匹配 manifest `labels.sha256`
- record `inputAssetSha256` 是否匹配 manifest `input.sha256`
- record 是否错误声明 `package-consumer-runtime`

catalog 会展示 runner evidence state；release evidence bundle 会把 manifest audit、sidecar audit、sample run validation 和 user acceptance catalog 一起聚合。

## 常见误区

- `TensorRtExec --buildOnly` 成功不是 `Classification Passed=True`。
- synthetic input 通过不是图片分类质量证明。
- sidecar 不是 release proof record。
- sample run evidence record 最多晋级到 `real-model-runtime`。
- 没有真实 run log 和 SHA256 时，不要把 `isSmokePassed` 改为 true。

## 小结

Classification 真实资产接入的关键不是“找一个模型跑一下”，而是把模型来源、许可证、hash、build report、sidecar、真实 runner 日志和 release evidence 聚合串成一条可复核证据链。这样文章、样例和发布检查才能一致，用户也能按同样结构替换成自己的分类模型。
