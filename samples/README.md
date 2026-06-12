# Samples

This directory contains Windows-first smoke runners and deployment-oriented samples for the managed TensorRT/CUDA API.

## Recommended Validation Order

Use this order when checking whether a local machine is ready for deployment work:

1. `CudaSmokeRunner`: validates CUDA bridge loading, device discovery, memory transfer, streams, events, memory pools, graphs-adjacent prerequisites, and diagnostics.
2. `MultiStream`: validates two non-blocking CUDA streams plus event-based cross-stream ordering.
3. `TensorRtSmokeRunner`: validates TensorRT environment discovery and representative high-level object chains across TensorRT 8/10/11 when the selected native bridge supports them.
4. `LifecycleSmokeRunner`: repeats CUDA and TensorRT lifecycle operations to catch disposal and native-loader regressions.
5. `OnnxToEngineSmokeRunner`: validates the ONNX parse -> build -> serialize -> deserialize -> bind -> enqueue flow.
6. `DynamicShape`: validates dynamic batch profiles, runtime input shape selection, inference bindings, enqueue, and output validation.
7. `NetworkBuilderSmokeRunner`: validates direct TensorRT network construction without ONNX.
8. Layer-specific network runners: validate the broader deployment layer surface after the core path is healthy.

Most samples are smoke runners rather than polished product demos. That is intentional: each runner prints compact evidence lines that are stable enough for release validation, issue triage, and CI logs.

## Environment

Set these variables when running a sample directly from a build tree:

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = "E:\TensorRtSharp\TensorRtSharp4.0\build-out\win-x64-trt11-cuda13-release\bin\Release"
$env:JYPPX_TENSORRT_ROOT = "E:\TensorRtSharp\TensorRtSharp4.0\third_party\nvidia\TensorRT-11.0.0.114-cuda 13.2"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
```

Use a matching preset/root combination for TensorRT 8, TensorRT 10, CUDA 11, or CUDA 12. If Windows application control blocks freshly built Debug binaries with `0x800711C7`, build Release and run the helper scripts that sign local development outputs before smoke validation.

## Current Coverage Context

As of 2026-06-12 the interface coverage matrix reports no missing TensorRT or CUDA rows for the scanned local headers. Some uncommon or risky CUDA APIs remain as explicit deferred boundaries in the native manifest because they require raw descriptors, callbacks, external handles, IPC ownership, driver entrypoint pointers, or other unsafe lifetime policy. Samples should prefer deployment-safe wrappers and should not depend on those deferred APIs unless the sample is explicitly documenting a boundary.

## Current Smoke Runners

- `CudaSmokeRunner`: CUDA device, stream/event flags, CUDA error mapping, event timing, pinned memory flags, async copy, async device-memory allocation/free, selected-pool async allocation, owned/default/current memory-pool checks, memory-pool access descriptor query/update, memory-pool high-water reset/query, `cudaMemcpyDefault`, pointer attributes, pitched 2D fill, pitched 3D sync/async/device-to-device copy, optional peer copy, device-to-device copy, and memory info checks.
- `CudaGraphSmokeRunner`: CUDA stream capture, graph instantiate, graph launch, and pinned host/device memory round-trip validation.
- `MultiStream`: deployment-oriented CUDA multi-stream sample that submits independent async fill/copy work to two non-blocking streams and demonstrates event-based cross-stream ordering.
- `TensorRtSmokeRunner`: representative TensorRT object chain checks, including execution-context max-output-size query, tensor debug-state boundary handling, refitter entry enumeration when the generated engine is refittable, and TensorRT 11 builder-config / network debug / engine inspector / execution-context runtime-control diagnostics. The TensorRT 11 path also validates deployment boundary controls such as builder max-thread/reset/error-recorder APIs, `isNetworkSupported`, network `removeTensor`, TopK V2 indices-type creation, engine aliased-input query, engine/context/network error-recorder clear/query APIs, direct engine build, plugin serialization list setting, host-memory data type, optimization-profile shape-values V2 boundary, execution-context address clearing, input-consumed event clearing, device-memory clearing, and aux-stream clearing. The `buildSerializedNetwork(..., kernelText)` path is wired but currently records a vendor/API boundary on the local TensorRT 11.0 + CUDA 12.9 stack because TensorRT returns a null object.
- `RefitWeightsSmokeRunner`: TensorRT 10 refit smoke that builds a refittable scale network, enumerates refit entries, sets new scale weights, runs `RefitCudaEngine`, and verifies changed output. TensorRT 8 remains explicit and reports a skip when the built engine is not refittable.
- `LifecycleSmokeRunner`: repeated TensorRT object lifecycle and enqueue checks.
- `OnnxToEngineSmokeRunner`: dynamic ONNX identity model parse, profile setup, max-output-size query, tensor debug-state boundary handling, engine build, deserialize, enqueue, and output validation.
- `DynamicShape`: direct TensorRT identity network with an explicit dynamic batch dimension, optimization profile, runtime shape selection, managed inference bindings, enqueue, and output validation.

## Asset-Dependent And Roadmap Directories

These directories are intentionally documented rather than shipped as empty placeholder projects:

- `Classification`: reserved for an image-classification walkthrough that requires a redistributable ONNX classifier, labels, image assets, and preprocessing metadata.
- `OnnxToEngine`: user-facing topic directory that redirects to the runnable `OnnxToEngineSmokeRunner`, which generates a minimal ONNX model in process.
- `YoloDet`: reserved for an object-detection walkthrough that requires a detector ONNX model, labels, images, decoding metadata, and often plugin/NMS diagnostics.
- `CustomKernelPreprocess`: roadmap directory blocked on safe public CUDA module/kernel wrappers; current CUDA samples cover the memory, stream, and graph primitives that future GPU preprocessing will use.

Validation note:

- `eng/Invoke-WindowsLifecycleSmoke.ps1` accepts `-Configuration Debug|Release`. Use `-Configuration Release` when Windows application control blocks freshly built Debug sample assemblies with `0x800711C7`.
- `NetworkBuilderSmokeRunner`: direct C# network build with identity layer.
- `NetworkLayersSmokeRunner`: constant and elementwise layer validation.
- `NetworkShapeOpsSmokeRunner`: shuffle reshape, shuffle zero-placeholder, reduce, and shape layer validation.
- `NetworkConcatSliceSmokeRunner`: slice and concatenation layer validation.
- `NetworkSoftmaxTopKSmokeRunner`: softmax, top-k, unary, and gather layer validation.
- `NetworkActivationPoolingResizeSmokeRunner`: activation, pooling, resize, editable layer metadata, engine metadata, and execution-context metadata validation.
- `NetworkTrt11ModernLayersSmokeRunner`: TensorRT 11 + CUDA 12.9 modern-layer execution validation with squeeze/unsqueeze, tensor dimension names, shape/execution tensor metadata, serialized engine build, deserialize, tensor binding, enqueue, and output round trip.
- `NetworkTrt11ModernLayerMetadataRunner`: TensorRT 11 modern-layer metadata probe for scatter, one-hot, cumulative, assertion, grid-sample, normalizationV2, and dynamic-quantizeV2. The cumulative probe now uses a 0D build-time constant shape tensor for the axis argument, validates shape-output marking/unmarking, and validates 7/7 modern-layer metadata probes on TensorRT 11.0 + CUDA 12.9.
- `NetworkTrt11AdvancedLayersSmokeRunner`: TensorRT 11 advanced deployment-layer metadata probe for cast, non-zero, ragged softmax, NMS, reverse sequence, einsum, loop control-flow, if-conditional, and fill Int64 metadata. Current design creates 9/9 probes; if a signed Debug build is still blocked with `0x800711C7`, record it as a local WDAC/application-control policy blocker and rerun on a policy-compatible machine.
- `NetworkMatrixFillSelectSmokeRunner`: matrix multiply, fill, select, and engine tensor metadata validation.
- `NetworkDeconvolutionSmokeRunner`: deconvolution layer and metadata validation with serialized engine build, tensor binding, enqueue, and output round trip.
- `NetworkLrnSmokeRunner`: LRN layer validation with metadata checks, serialized engine build, tensor binding, enqueue, and output round trip.
