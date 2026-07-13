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
