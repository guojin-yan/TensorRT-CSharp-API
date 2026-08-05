# Samples

This directory is now reserved for user-facing examples and common adoption scenarios.

Deep-learning model binaries are staged outside this Git repository under
`<workspace-root>/models`. The complete model/source/export/hash map is
`samples/assets/demo-model-inventory.json`, with the reproducible acquisition and ONNX conversion guide at
`docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`. No ONNX or weight file from that workspace-level directory is
uploaded with the source repository or included in project packages.

Smoke-oriented validation projects have been moved out to:

- `smoke/`

That split keeps:

- `samples/` focused on readable reference cases for adopters
- `smoke/` focused on CI diagnostics, package validation, and release gates

## Current Common Examples

| Directory | Purpose | Status |
| --- | --- | --- |
| `MultiStream` | CUDA multi-stream and cross-stream wait example | runnable |
| `CudaRuntimeCompilation` | owner-safe NVRTC compile, copied PTX/CUBIN/LTO IR, failure log, determinism, named typed-kernel launch, owner retention, and GPU readback | runnable with CUDA 12.9+ runtime bridge; CUDA 13.2 compile-only/load-rejected boundary is recorded |
| `DynamicShape` | TensorRT dynamic-shape/profile/binding example | runnable |
| `InferenceBindings` | TensorRtInferenceBindings host/device workflow example | runnable |
| `OnnxToEngine` | user-facing ONNX to engine walkthrough with trtexec-like option parsing | runnable |
| `RefittedPlan.PackageConsumer` | PackageReference-only persisted refitted-plan reload, enqueue, raw output hash, and owner cleanup example | runnable through the local-package proof script |
| `GpuAllocator.PackageConsumer` | PackageReference-only TensorRT GPU allocator attach, callback, zero-leak, rejection, and exception example | runnable through `eng/Test-GpuAllocatorLocalPackageConsumer.ps1` with host-installed CUDA and TensorRT |
| `OutputAllocator.PackageConsumer` | PackageReference-only TensorRT dynamic output allocation, release, detach, and rejection example | runnable through `eng/Test-OutputAllocatorLocalPackageConsumer.ps1` with host-installed CUDA and TensorRT |
| `DebugListener.PackageConsumer` | PackageReference-only TensorRT debug tensor callback, copied metadata, detach, and rejection example | runnable through `eng/Test-DebugListenerLocalPackageConsumer.ps1` with host-installed CUDA and TensorRT |
| `StreamReader.PackageConsumer` | PackageReference-only TensorRT IStreamReaderV2 read/seek, sequential reuse, deferred disposal, and truncated-plan example | runnable through `eng/Test-StreamReaderLocalPackageConsumer.ps1` with host-installed CUDA and TensorRT |
| `Classification` | External ONNX classifier inference with legacy single-input and strict named multi-input binding, raw/task reference validation, and Top-K output | runnable with user-provided ONNX assets |
| `YoloVision` | External YOLO-family ONNX vision sample with strict named multi-input binding, all-output reference validation, family/task profiles, preprocessing, and det/cls/seg/pose/OBB/semantic helpers | runnable with user-provided ONNX assets |
| `YoloVision.ManagedPackageConsumer` | Repository-external template used to validate the managed API and YoloVision extension through PackageReference-only restore/build/run | runnable through `eng/Test-YoloVisionManagedPackageDryRun.ps1`; no NVIDIA runtime required |

`YoloVision` also exposes an offline capability matrix for documentation, smoke, and asset-planning workflows:

```powershell
dotnet run --project .\samples\YoloVision -- --list-capabilities
```

This command does not require CUDA, TensorRT, ONNX models, labels, or images. It lists the supported family/task matrix for `custom`, YOLOv5/v6/v7/v8/v9/v10/v11/v26 and `det`/`cls`/`seg`/`obb`/`pose`/`sem`, including each task's managed decode path, auxiliary metadata boundary, and evidence level.

For an owner-ready asset/configuration checklist without invoking TensorRT, use `samples/YoloVision --preflight`. The resulting `yolovision-preflight.v1` report records profile values, asset existence/SHA256, output metadata, normalized command hash, and an explicit `precheck` boundary. It never replaces a real `YoloVision Passed=True` run log or `real-model-runtime` evidence.

Recommended YoloVision documentation starts at `docs/articles/zh-cn/yolovision-sample-overview.md`, then continues through preprocess/postprocess, engine build/run, and troubleshooting. These articles keep the old detection-only naming out of the user-facing path and treat all asset-dependent runs as sample evidence until real owner logs and hashes are supplied.

For the broader publishable article route, use `docs/articles/zh-cn/yolovision-series-roadmap.md`. That roadmap is the owner-facing checklist for YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom and `det`/`cls`/`seg`/`obb`/`pose`/`sem`. It intentionally keeps TensorRtExec build reports, screenshots, templates, sidecars, local package-feed results, project-reference runs, and direct `.nupkg` runs out of runtime evidence promotion.

CUDA runtime compilation and the CUDA 12.9+ owner-bound named-kernel launch/readback path are executable through `CudaRuntimeCompilation`. CUDA 11.8/12.1 Driver ownership, Linux runtime proof, package-consumer proof, and post-publish verification remain tracked separately. See [CUDA Kernel Wrapper Roadmap](../docs/articles/en/cuda-kernel-roadmap.md) and [CUDA Runtime Compilation Roadmap](../docs/articles/en/cuda-runtime-compilation-roadmap.md).

## Evidence Ladder For Asset-Dependent Samples

Classification and YoloVision are intentionally runnable with user-provided assets instead of bundled model files. Treat their evidence as a ladder:

| Evidence level | Meaning | Promotion boundary |
| --- | --- | --- |
| `precheck` | command, shape, report, or manifest template can be parsed | not runtime proof |
| `build-only` | ONNX parser/builder produced build evidence or a conversion report | not inference proof |
| `synthetic-input-runtime` | a sample pipeline executed with synthetic input | not real model quality proof |
| `real-model-runtime` | real model, labels, input asset, hashes, license notes, runner log, and sample-run-evidence all agree | sample-level proof only |
| `package-consumer-runtime` | clean external package consumer runtime smoke with validated release proof record | release proof records only |

`sample-run-evidence` files and asset manifests can promote a sample to real-model evidence only. package-consumer-runtime belongs to release proof records, and `blocked-by-cuda-driver` is an environment compatibility blocker rather than smoke passed.

## Environment

Run examples from the repository root. Development probing may discover `build-out` and standard CUDA installations; select user-installed TensorRT and cuDNN with explicit `JYPPX_*_ROOT` variables:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Only set `JYPPX_NATIVE_BRIDGE_PATH`, `JYPPX_TENSORRT_ROOT`, `JYPPX_CUDA_ROOT`, or `JYPPX_CUDNN_ROOT` when you intentionally want to override the default probing behavior.

## Local Packaging Gate

Before treating example evidence as release-ready, validate the package path first:

This is a focused single-key example. Omit `-WindowsRuntimeKeys` when you want the full Windows runtime matrix from `eng/Invoke-LocalReleaseBundle.ps1`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.0 `
  -WindowsRuntimeKeys <runtime-key> `
  -WindowsRuntimeDeliveryMode split `
  -RunWindowsSmoke `
  -SignWindowsConsumerOutput `
  -TrustWindowsConsumerSigningCertificate `
  -TrustWindowsConsumerSigningCertificateRoot
```

If you want validation-oriented runners after that, use `smoke/README.md`.
