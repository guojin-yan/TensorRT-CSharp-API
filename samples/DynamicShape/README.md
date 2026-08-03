# DynamicShape

Status: runnable common sample.

This sample builds a small dynamic-batch identity network directly in C# and demonstrates:

- explicit-batch network creation
- dynamic input shape with `-1` batch dimension
- optimization profile min/opt/max configuration
- `TensorRtInferenceBindings` input/output workflow
- CUDA stream based enqueue and output readback

This sample creates its Identity network in C# memory. It does not download a model and does not require ONNX conversion.

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\DynamicShape\bin\Release\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 3
```

Options:

- `--tensor-rt-line <8|10|11>` selects the TensorRT adapter line. Default: `10`.
- `--batch <1..4>` selects a runtime batch inside the optimization profile. Default: `3`.

Expected evidence includes:

- `Profile Index=... Min=... Opt=... Max=... Valid=True`
- `Readiness Ready=True Bound=True`
- `BindingReport Ready=True Inputs=1 Outputs=1`
- `Execution ... OutputMatch=True`
- `DynamicShape Passed=True`

If runtime, builder, CUDA, or vendor dependency probing is unavailable, the sample prints `DynamicShape=Skipped` with a diagnostic reason. That skip should be treated as environment evidence, not as a deferred API completion shortcut.

The complete Chinese walkthrough, real Windows Terminal screenshot, and runtime evidence are available in [使用 TensorRtSharp4.0 完成 Dynamic Shape 推理](../../docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md).
