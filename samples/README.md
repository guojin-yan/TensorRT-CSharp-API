# Samples

This directory is organized into three groups so the project reads like a publishable library instead of an internal scratchpad:

- Runnable examples: user-facing, deployment-oriented samples that are reasonable first stops for adopters.
- Smoke runners: focused validation tools that print stable evidence lines for release gates and regression triage.
- Roadmap-only directories: documented topics that still depend on redistributable models, assets, or future wrappers.

## Quick Start

Use this order when validating a local machine:

1. `CudaSmokeRunner`
2. `MultiStream`
3. `TensorRtSmokeRunner`
4. `LifecycleSmokeRunner`
5. `OnnxToEngineSmokeRunner`
6. `DynamicShape`
7. `NetworkBuilderSmokeRunner`
8. Layer-specific network runners

Most projects here are intentionally smoke runners rather than polished demos. They exist to prove packaging, native loading, inference, and deployment behavior with compact, CI-friendly output.

## Environment

Run samples from the repository root and let the existing C# path resolver probe `build-out`, `third_party/nvidia`, and standard CUDA install locations:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Only set `JYPPX_NATIVE_BRIDGE_PATH`, `JYPPX_TENSORRT_ROOT`, `JYPPX_CUDA_ROOT`, or `JYPPX_CUDNN_ROOT` when you intentionally want to override the default probing behavior. If Windows application control blocks freshly built Debug binaries with `0x800711C7`, build Release and sign local development outputs before smoke validation.

## Runnable Examples

| Directory | Purpose | Status |
| --- | --- | --- |
| `MultiStream` | CUDA multi-stream/event ordering example | runnable |
| `DynamicShape` | TensorRT dynamic-shape/profile/binding example | runnable |
| `OnnxToEngineSmokeRunner` | ONNX parse/build/serialize/deserialize path | runnable |

## Smoke Runners

### CUDA foundation

| Directory | Purpose |
| --- | --- |
| `CudaSmokeRunner` | CUDA bridge/device/memory/stream/event/pool coverage |
| `CudaGraphSmokeRunner` | CUDA stream capture and graph launch validation |
| `InferenceBindingsSmokeRunner` | Managed inference-binding validation |

### TensorRT core path

| Directory | Purpose |
| --- | --- |
| `TensorRtSmokeRunner` | High-level TensorRT object-chain and deployment diagnostics |
| `LifecycleSmokeRunner` | Repeated create/use/dispose regression checks |
| `RefitWeightsSmokeRunner` | Refittable engine workflow validation |

### Direct network construction

| Directory | Purpose |
| --- | --- |
| `NetworkBuilderSmokeRunner` | Identity-network creation without ONNX |
| `NetworkLayersSmokeRunner` | Constant and elementwise layer validation |
| `NetworkShapeOpsSmokeRunner` | Shuffle/reduce/shape-layer validation |
| `NetworkConcatSliceSmokeRunner` | Slice and concatenation validation |
| `NetworkSoftmaxTopKSmokeRunner` | Softmax/top-k/unary/gather validation |
| `NetworkActivationPoolingResizeSmokeRunner` | Activation/pooling/resize plus metadata validation |
| `NetworkMatrixFillSelectSmokeRunner` | Matrix multiply/fill/select validation |
| `NetworkConvolutionScaleSmokeRunner` | Convolution/scale validation |
| `NetworkDeconvolutionSmokeRunner` | Deconvolution validation |
| `NetworkLrnSmokeRunner` | LRN validation |
| `NetworkQuantizeDequantizeSmokeRunner` | Quantize/dequantize validation |
| `NetworkCompatLayerMetadataSmokeRunner` | Cross-version compatibility metadata checks |

### TensorRT 11 focused runners

| Directory | Purpose |
| --- | --- |
| `NetworkTrt11ModernLayersSmokeRunner` | TRT11 modern-layer execution path |
| `NetworkTrt11ModernLayerMetadataRunner` | TRT11 modern-layer metadata probes |
| `NetworkTrt11AdvancedLayersSmokeRunner` | TRT11 advanced deployment-layer metadata probes |

## Roadmap-Only Directories

These directories are intentionally documented rather than shipped as empty projects:

| Directory | Purpose | Current state |
| --- | --- | --- |
| `Classification` | Image classification walkthrough | waiting on redistributable model/assets |
| `OnnxToEngine` | User-facing topic directory | redirects conceptually to `OnnxToEngineSmokeRunner` |
| `YoloDet` | Object detection walkthrough | waiting on redistributable model/assets |
| `CustomKernelPreprocess` | CUDA preprocessing walkthrough | blocked on safe public module/kernel wrappers |

## Local Packaging Gate

Before treating any sample evidence as release-ready, run the local package path first:

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

This validates documentation, managed packing, native bridge build, runtime asset collection, runtime nupkg creation, and package-consumer validation on the local machine. If you only want the lower-level runtime lane, `eng\Invoke-LocalRuntimePackage.ps1` and `eng\Invoke-LocalSplitRuntimePackage.ps1` remain available.

## Validation Notes

- `eng/Invoke-WindowsLifecycleSmoke.ps1` accepts `-Configuration Debug|Release`. Use `-Configuration Release` when WDAC or other Windows application control policies block Debug sample assemblies with `0x800711C7`.
- Current TensorRT 11 `buildSerializedNetwork(..., kernelText)` wiring is kept as a documented vendor/API boundary when the local TensorRT stack returns a null object.
- Samples should prefer deployment-safe wrappers and should not depend on deferred unsafe native boundaries unless they are explicitly documenting one.
