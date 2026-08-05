# Release Candidate Gate

## Current Baseline

As of 2026-06-12 the release candidate gate starts from a zero-missing interface baseline:

- TensorRT interface coverage: `0` missing rows.
- CUDA runtime interface coverage: `0` missing rows.
- Manifest inventory: `3271` API records across `102` manifests.
- Deferred APIs are explicit manifest/native boundaries and must not be treated as safe public wrappers.

The release candidate gate is split into two paths:

- Hosted gate: source build, runtime manifest validation, binding generator determinism, workflow contracts, DocFX, managed package content, project quality tests, and report exports.
- Runtime gate: native build, runtime asset collection, runtime nupkg, package consumer validation, optional GPU smoke, and publish readiness for the selected runtime key. Windows uses the self-hosted Windows runner; Linux Ubuntu 20.04/22.04/24.04 x64 use GitHub-hosted runners with distro-matched Ubuntu job containers, while future ARM/Jetson lines require dedicated package lines and runners.

Current primary Windows runtime keys:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

At least one locally validated runtime key should pass consumer smoke before a release candidate is considered usable. CUDA `12.9` and TensorRT 11 Windows keys are now eligible only when their exact CUDA/TensorRT/cuDNN dependency chain and package consumer smoke evidence are present.

Current package-consumer evidence on 2026-06-12:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`: restore/build/native-copy/smoke passed with `16/16` native assets; probe output TensorRT `10.11.0`, CUDA `11.8`.
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: restore/build/native-copy/smoke passed with `19/19` native asset patterns; probe output TensorRT `10.11.0`, CUDA `12.9`.
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: restore/build/native-copy/smoke passed with `19/19` native asset patterns; probe output TensorRT `11.0.0`, CUDA `12.9`.
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: 2026-06-14 full split package set and collection package packed locally; restore/build/native-copy passed with `19/19` native asset patterns, but package consumer smoke remains pending and publish readiness is blocked until CUDA 13 runtime/builder validation is available.

## Required Local Quality Gate

Run this gate before considering any release-candidate tag:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

If native, manifest, or generated binding files changed, also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

Expected local gate result:

- TensorRT missing rows remain `0`.
- CUDA runtime missing rows remain `0`.
- `dotnet build` reports `0` warnings and `0` errors.
- Public API XML comments contain both English and Chinese text.
- `JYPPX.ProjectQuality.Tests` passes.
- DocFX reports `0` warnings and `0` errors.

## Recommended Smoke Order

After the local quality gate, run validation-oriented smoke runners from `smoke/README.md` in this order:

1. `CudaSmokeRunner`
2. `MultiStream`
3. `TensorRtSmokeRunner`
4. `LifecycleSmokeRunner`
5. `OnnxToEngineSmokeRunner`
6. `DynamicShape`
7. `InferenceBindings`
8. `NetworkBuilderSmokeRunner`
9. Layer-specific network runners

Asset-dependent sample projects are executable but require user-provided model metadata:

- `Classification`
- `YoloVision`

Those projects are not release blockers by themselves as long as their README files describe required external assets and runnable substitutes. CUDA custom-kernel preprocessing remains a documentation roadmap item until safe public `CudaModule` / `CudaKernel` wrappers are available.

## Runtime Package Gate

For any runtime package selected for release:

- Validate explicit TensorRT, CUDA, and cuDNN roots before asset collection.
- Collect runtime assets from the matching CMake preset output.
- Pack the core managed API and the matching bridge package. YoloVision and Classification are sample-only projects and are not publication artifacts.
- Run package consumer validation for restore/build/native asset copy.
- Run package consumer smoke when the machine has a compatible driver/GPU/runtime stack.
- Require `local-validated` before private-feed or split-delivery readiness can be treated as ready.
- Record WDAC/application-control blockers separately from package layout failures.

## Remote Publication Prerequisites

Before using the remote publication lanes:

- `package-managed.yml` always packs and validates exactly `JYPPX.TensorRT.CSharp.API`. `YoloVision` and `Classification` remain sample-only projects and are never uploaded. Publication requires `owner_publish_approved=true`, the formal repository owner, and an exact package ID/version/source-commit allowlist. With `publish_to_nuget=true`, `NUGET_API_KEY` must be a plain-text ASCII nuget.org API key with push permission for the managed package ID or its owning account/organization. A nuget.org `403` is non-retryable until the package owner replaces the invalid, expired, or under-scoped key.
- `runtime-windows.yml` requires the Windows self-hosted runner to stay online with the labels `self-hosted`, `windows`, and `x64`.
- `runtime-linux.yml` can publish Ubuntu 20.04, Ubuntu 22.04, and Ubuntu 24.04 x64 through GitHub-hosted runners with distro-matched Ubuntu job containers. Ubuntu 20.04 uses `runner_mode=hosted-container`; Ubuntu 24.04 only covers the modern combinations; ARM/Jetson targets need separate package lines before publication.
