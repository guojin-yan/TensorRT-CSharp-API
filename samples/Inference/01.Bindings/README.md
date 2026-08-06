# InferenceBindings

English | [简体中文](README.zh-CN.md)

Status: runnable common sample.

This sample builds a small explicit-batch identity network directly in C# and demonstrates:

- optimization profile setup
- `TensorRtInferenceBindings` input/output workflow
- host-to-device copy
- tensor binding
- enqueue and output readback

The sample creates its Identity network in C# memory. It does not download a model and does not require ONNX conversion.

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\Inference\01.Bindings\bin\Release\net8.0\InferenceBindings.dll --tensor-rt-line 10 --batch 2
```

Expected evidence includes:

- `BindingReport Ready=True`
- `Readiness Ready=True Bound=True`
- `Execution ... OutputMatch=True`
- `InferenceBindings Passed=True`

The complete Chinese walkthrough, real Windows Terminal screenshot, and runtime evidence are available in [使用 TensorRtSharp4.0 管理推理输入、显存绑定与 GPU 输出读回](../../../docs/articles/zh-cn/inference-bindings-tutorial.md).
