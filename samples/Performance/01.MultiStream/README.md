# MultiStream

English | [简体中文](README.zh-CN.md)

Status: runnable common sample.

This sample demonstrates CUDA stream and event primitives through the safe C# wrapper surface:

- two non-blocking CUDA streams
- async device memory fill and copy
- pinned host memory readback
- CUDA event record/synchronize
- cross-stream wait ordering

The sample operates on raw device memory. It does not download a model, require ONNX conversion, or build a TensorRT engine.

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\Performance\01.MultiStream\bin\Release\net8.0\MultiStream.dll
```

Expected evidence includes:

- `IndependentStreams=True`
- `CrossStreamWait=True`
- `StreamIds A=... B=...`
- `MultiStream Passed=True`

If the local CUDA driver/runtime cannot initialize, the sample prints `MultiStream=Skipped` with the diagnostic reason. That skip is an environment/runtime compatibility signal, not a managed API completion claim.

The complete Chinese walkthrough, real Windows Terminal screenshot, and runtime evidence are available in [使用 TensorRtSharp4.0 在 C# 中实现 CUDA 多流与 Event 同步](../../../docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md).
