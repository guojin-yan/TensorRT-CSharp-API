# Samples

This directory is now reserved for user-facing examples and common adoption scenarios.

Smoke-oriented validation projects have been moved out to:

- `smoke/`

That split keeps:

- `samples/` focused on readable reference cases for adopters
- `smoke/` focused on CI diagnostics, package validation, and release gates

## Current Common Examples

| Directory | Purpose | Status |
| --- | --- | --- |
| `MultiStream` | CUDA multi-stream and cross-stream wait example | runnable |
| `DynamicShape` | TensorRT dynamic-shape/profile/binding example | runnable |
| `OnnxToEngine` | user-facing ONNX to engine walkthrough | runnable |

## Roadmap Directories

| Directory | Purpose | Current state |
| --- | --- | --- |
| `Classification` | Image classification walkthrough | waiting on redistributable model/assets |
| `YoloDet` | Object detection walkthrough | waiting on redistributable model/assets |
| `CustomKernelPreprocess` | CUDA preprocessing walkthrough | blocked on safe public module/kernel wrappers |

## Environment

Run examples from the repository root and let the existing C# path resolver probe `build-out`, `third_party/nvidia`, and standard CUDA install locations:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Only set `JYPPX_NATIVE_BRIDGE_PATH`, `JYPPX_TENSORRT_ROOT`, `JYPPX_CUDA_ROOT`, or `JYPPX_CUDNN_ROOT` when you intentionally want to override the default probing behavior.

## Local Packaging Gate

Before treating example evidence as release-ready, validate the package path first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.0 `
  -WindowsRuntimeKeys win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -WindowsRuntimeDeliveryMode split `
  -RunWindowsSmoke `
  -SignWindowsConsumerOutput `
  -TrustWindowsConsumerSigningCertificate `
  -TrustWindowsConsumerSigningCertificateRoot
```

If you want validation-oriented runners after that, use `smoke/README.md`.
