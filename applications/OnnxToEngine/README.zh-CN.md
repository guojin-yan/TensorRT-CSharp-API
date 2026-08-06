# OnnxToEngine

[English](README.md) | 简体中文

`OnnxToEngine` 是面向模型转换与诊断的完整应用。它能够解析 ONNX、配置动态 Shape profile、构建和反序列化 TensorRT Engine、绑定 Tensor，并输出 trtexec-like 参数与结构化报告。

程序内置的最小动态 identity ONNX 用于验证 parser、profile、序列化、反序列化和读回，因此基础流程不依赖外部模型。对任意外部 ONNX，默认只声明 build-only；只有定义了模型输入、输出和任务语义的专用 runner 才能形成真实模型证据。

## 基础运行

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = '1'
dotnet .\applications\OnnxToEngine\bin\Release\net8.0\OnnxToEngine.dll --tensor-rt-line 10
```

外部动态 ONNX 构建示例：

```powershell
dotnet run --project .\applications\OnnxToEngine -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512 `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --exportReport .\artifacts\onnx-to-engine\build-report.json `
  --buildOnly
```

`--previewOnly` 或 `--dryRun` 只归一化参数并写预检报告，不加载 CUDA/TensorRT、不读取 ONNX、也不构建 Engine。

## MNIST 真实模型流程

模型来自用户安装的 TensorRT 示例数据 `data/mnist/README.md` 所指向的 ONNX Model Zoo opset 8 MNIST。该文件本身已经是 ONNX，无需再次从框架转换。把它暂存到 `<workspace-root>/models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx`，并核对 SHA256：

```text
2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf
```

项目脚本生成自有数字 7 图片，然后执行模型专用 runner：

```powershell
$workspaceRoot = Split-Path -Parent $PWD
$model = Join-Path $workspaceRoot 'models\OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx'
$assetRoot = Join-Path $workspaceRoot 'downloads\mnist-owner-generated'

powershell -NoProfile -ExecutionPolicy Bypass -File .\eng\New-MnistOwnerGeneratedDigit.ps1

dotnet .\applications\OnnxToEngine\bin\Release\net8.0\OnnxToEngine.dll `
  --mnist `
  --tensor-rt-line 10 `
  --onnx $model `
  --mnistInput (Join-Path $assetRoot 'digit-7.pgm') `
  --expectedDigit 7 `
  --minimumConfidence 0.5 `
  --saveEngine (Join-Path $assetRoot 'digit-7.plan') `
  --exportReport (Join-Path $assetRoot 'digit-7-report.json') `
  --exportOutput (Join-Path $assetRoot 'digit-7-output.json') `
  --exportPreprocessedInput (Join-Path $assetRoot 'digit-7-input.fp32.bin') `
  --visualization (Join-Path $assetRoot 'digit-7-result.svg')
```

当前 TensorRT 10.11 记录预测数字 7，置信度 `0.99945575`，10 个 logits 与 ONNX Runtime CPU 独立参考在 `1e-4` 内一致；把期望数字改为 6 的受控负例会以非零码退出。

完整的模型获取边界、Engine 构建、独立参考、真实终端截图和标注结果图见 [OnnxToEngine MNIST 完整实战](../../docs/articles/zh-cn/onnxtoengine-mnist-owner-generated-tutorial.md)。模型、Engine、Tensor 和原始日志均保存在 Git 仓库外。

## 证据边界

`--exportReport` 可输出 JSON 或 Markdown；报告记录 parser/build、TensorRT 主线、归一化命令、builder readback 和 proof boundary。build-only、dependency probe、dry-run、只读 Engine 诊断或输出文件哈希都不能替代真实输入 enqueue、结果校验和独立参考。

对 YOLO、分类或其他业务模型，应由对应任务应用定义预处理、输出解码和准确率判断；`OnnxToEngine` 只提供转换和构建证据，不能推断任务语义。
