# Classification Model Assets

`samples/Classification` 不随仓库分发模型、labels 或图片。本文说明完整分类 demo 需要准备哪些资产，以及如何避免把未验证模型写成已通过。

## 需要的文件

- `model.onnx`：单输入 float 分类模型。
- `labels.txt`：每行一个类别名，顺序必须与模型输出一致。
- `input.*`：有再分发权利的小图。
- `preprocess.json` 或等价说明：resize、crop、RGB/BGR、scale、mean、std、layout。

## 模型元数据

在提交样例资产前，至少记录：

- 模型来源 URL 和许可证。
- 输入 tensor 名称、shape、layout、dtype。
- 输出 tensor 名称、shape、类别数。
- dynamic shape 是否需要 min/opt/max profile。
- 预处理是否与训练配置一致。

推荐先选择小体积、许可清晰、输入稳定的 ImageNet 分类模型，例如 MobileNet 系列或 ResNet 小模型。仓库文档只记录选择标准，不把任何未实际跑通的外部模型写成项目自带能力。

第一版已经固定并实跑 TorchVision `v0.25.0` ResNet18 `IMAGENET1K_V1`。获取和 opset 17 导出合同在
`samples/assets/classification-resnet18-official-assets.json`，实际模型位于 Git 仓库外的
`..\models\Classification\resnet18-torchvision-v0.25.0`。运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AllowDownload -ExportOnnx -PythonPath python
```

`eng/Invoke-ClassificationResNet18Reference.py` 从精确 C# 输入 tensor 生成独立 ONNX Runtime raw/task references。TensorRT
10.11 的 raw logits 与 Softmax probabilities 各比较 1000 个值，mismatch 均为 0；单值负例以 exit code 1 fail closed。
小型记录是 `samples/assets/classification-resnet18-real-model-runtime-evidence.json`。模型、图片、tensor、reference 和日志
不进入 Git，也不进入 NuGet 或 GitHub Release。

## 运行命令

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\labels.txt `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --top-k 5
```

动态 batch 需要补 profile：

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --input-shape 1x3x224x224 `
  --min-shape 1x3x224x224 `
  --opt-shape 4x3x224x224 `
  --max-shape 8x3x224x224
```

## 证据要求

完整 demo 至少要保存：

- 运行命令。
- 模型和 labels 的 SHA256。
- Top-K 输出。
- TensorRT line。
- 是否使用真实图片预处理。

如果当前只使用 synthetic input，只能说明 pipeline 可以跑通，不能说明模型精度或图片分类结果正确。

## Reference provenance

`samples/assets/cross-task-reference-provenance-contract.json` 把 independent reference 的准入条件拆成公共 provenance 与
任务语义两层。Classification 除 model/labels/image/tensor/provider/reference/comparison/Owner 字段外，还必须记录 resize、
crop、RGB/BGR、scale、mean/std、输出是 raw logits 还是 probabilities、score transform、label mapping SHA256、Top-K 和
argmax rule。

已有 MNIST ONNX Runtime CPU reference 不能直接作为本样例的 golden：它的模型、输入、预处理、输出 tensor、labels 和
任务语义均不同。跨任务复用只有在 `modelSha256`、`inputTensorSha256`、`preprocessContractSha256`、
`outputTensorContractSha256`、`labelsSha256`、`taskSemanticsSha256` 六项全部存在且完全一致，并获得 Owner golden/
redistribution 决定后才允许。使用下面的命令查看当前真实缺口：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CrossTaskReferenceProvenanceMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CrossTaskReferenceProvenanceMatrix.ps1 -Strict
```

通用模板矩阵仍可显示 `owner-action-required`，因为它面向任意用户模型；这不是 validator 失败。官方 ResNet18 用例已经有独立
reference 和 source-tree real-model-runtime 记录，但仍不具备 owner-reviewed golden、公开再分发或 package consumer 证明。
