# TensorRtSharp4.0

TensorRtSharp4.0 is the next-generation workspace for a production-oriented TensorRT and CUDA bridge on .NET.

## Scope

- Managed assemblies: `JYPPX.TensorRtSharp`, `JYPPX.CudaSharp`
- NuGet package: `JYPPX.TensorRT.CSharp.API`
- Native bridge library: `jyppxtrtbridge`
- First release targets: Windows x64 and Linux x64
- TensorRT lines: 8.x, 10.x, and 11.x
- CUDA lines: 11.x, 12.x, and 13.x

## Current Verified Status

Status as of 2026-06-12:

- TensorRT interface coverage matrix: `0` missing rows.
- CUDA runtime interface coverage matrix: `0` missing rows.
- Manifest API inventory: `3271` API records across `102` manifests.
- Latest coverage report: `artifacts/interface-coverage/interface-coverage-summary.md`.
- Native high-version validation: `win-x64-trt11-cuda13-release` configures and builds.
- Managed validation: solution build and project quality tests pass.
- DocFX validation: documentation builds with `0` warnings and `0` errors.

The project now has manifest/native-source coverage for the scanned TensorRT 8/10/11 and CUDA 11/12/13 headers in this workspace. A subset of unsafe or rarely used CUDA runtime APIs is intentionally recorded as deferred boundaries instead of being exposed as high-level managed APIs. Deferred entries include APIs that require callback lifetime policy, raw driver entrypoint pointers, external resource descriptors, IPC ownership, CUDA library JIT option arrays, texture/surface descriptors, green/execution-context resource handles, or user-object destructor ownership.

## Current Stage

The raw interface coverage phase is complete for the locally scanned headers. Active work is release hardening:

- keep coverage matrix, build, tests, and DocFX green
- keep sample runners accurate and reproducible
- keep asset-dependent sample directories documented instead of empty
- validate runtime-package collection and package-consumer restore/build/run paths
- promote high-value deferred boundaries only when ABI, ownership, version guards, and managed lifetimes are clear

## Deployment Validation Path

For a local Windows deployment sanity check, use this order:

```powershell
dotnet restore .\TensorRtSharp.sln
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

If manifest/native/generated files changed, also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

Then run the common examples from `samples/README.md` and the validation runners from `smoke/README.md`.

Recommended common-example order:

1. `MultiStream`
2. `DynamicShape`
3. `OnnxToEngine`

Recommended smoke order:

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. Layer-specific network runners

## Samples

Runnable deployment samples are under `samples/`. Validation-oriented smoke runners are under `smoke/`.

Recent sample maturity updates:

- `MultiStream` is a real CUDA multi-stream/event ordering sample and is included in the solution.
- `DynamicShape` is a real TensorRT dynamic-shape/profile/binding sample and is included in the solution.
- `MultiStream` is a real CUDA multi-stream/event ordering sample and is included in the solution.
- `DynamicShape` is a real TensorRT dynamic-shape/profile/binding sample and is included in the solution.
- `OnnxToEngine` is now a runnable common ONNX-to-engine example and is included in the solution.
- `Classification`, `CustomKernelPreprocess`, and `YoloDet` remain documented README/roadmap directories rather than empty `.gitkeep` placeholders.

## Runtime Packages

Runtime packages carry native deployment assets for one explicit TensorRT / CUDA / cuDNN combination. Current Windows runtime package keys:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

Current local status:

- TensorRT 10 + CUDA 11.8 is the stable real vendor-backed smoke path and has package-consumer smoke evidence.
- TensorRT 10 + CUDA 12.9 and TensorRT 11 + CUDA 12.9 have local runtime/package validation and package-consumer smoke evidence.
- TensorRT 11 + CUDA 13.2 bridge builds, collects assets, packs, and passes package consumer restore/build/native-copy; runtime/builder smoke is pending on a CUDA 13-capable driver/runtime environment.
- Linux package entries and workflows are structurally prepared, but Linux runtime validation still requires self-hosted Linux x64 runner evidence.

See:

- `docs/articles/en/runtime-packages.md`
- `docs/articles/en/runtime-distribution-strategy.md`
- `docs/articles/en/package-consumer-validation.md`
- `docs/articles/en/release-candidate-gate.md`

## Release Automation

GitHub Actions does not need a GitHub-hosted machine for every job. In this repository:

- `docs-release.yml` and the hosted part of `package-managed.yml` run on GitHub-hosted runners.
- `runtime-windows.yml` runs on the local self-hosted Windows runner when dispatched from GitHub.
- `release-bundle.yml` fans out to managed package, Windows runtime, and Linux runtime modules.

That means there are two supported execution modes:

1. Dispatch the workflow through GitHub with `gh`, then let the self-hosted runner on this machine execute the Windows runtime job.
2. Run the local scripts directly when you want a true workstation-only validation loop without creating a GitHub Actions run record.

Remote dispatch example:

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.0 `
  -f windows_runtime_keys=win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -f windows_runtime_delivery_mode=split `
  -f run_windows_smoke=true `
  -f sign_windows_consumer_output=true `
  -f publish_managed_to_nuget=false `
  -f publish_to_github_packages=false `
  -f attach_runtime_to_github_release=false
```

Local bundle example:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.0 `
  -WindowsRuntimeKeys win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -WindowsRuntimeDeliveryMode split `
  -RunWindowsSmoke `
  -SignWindowsConsumerOutput `
  -TrustWindowsConsumerSigningCertificate `
  -TrustWindowsConsumerSigningCertificateRoot
```

On WDAC / application-control machines, the local and self-hosted Windows runtime validation path can sign the generated consumer output before smoke. This helps when `PackageConsumerSmoke.exe` would otherwise be blocked even though package restore, native asset copy, and build succeeded.

`release-bundle.yml` now treats an empty `linux_runtime_keys` input as a no-op Linux module, which keeps the bundle usable on repositories that do not yet have a Linux self-hosted runner.

For `nuget.org` publication, store a plain-text ASCII NuGet API key in the repository secret `NUGET_API_KEY`. Do not reuse an encrypted local credential blob or other machine-generated token format.

The managed-package workflow now validates this secret before the `publish-nuget` job downloads artifacts. If the secret contains non-ASCII characters or embedded whitespace, the job fails immediately with a configuration error instead of spending time on a doomed publish attempt.

## Repository Layout

```text
build/      CMake modules and build helpers
docs/       DocFX site and conceptual documentation
eng/        automation and dependency discovery scripts
native/     C ABI bridge and TensorRT/CUDA adapters
pack/       NuGet packaging projects
samples/    user-facing common examples and documented sample roadmaps
smoke/      validation runners for release gates, packaging, and regression checks
src/        managed libraries
tests/      managed integration and unit tests
third_party/local dependency drop folder (not committed)
```

## Build Prerequisites

- .NET SDK 10.0.300 or later
- CMake 3.27 or later
- Visual Studio C++ toolchain on Windows
- Matching local TensorRT / CUDA / cuDNN roots for native builds and runtime-package validation

## Quick Start

```powershell
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

To build the current high-version native preset:

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## Dependency Discovery

Use one of the scripts below to inspect local TensorRT/CUDA/cuDNN roots:

- `eng/Get-Dependencies.ps1`
- `eng/get-dependencies.sh`

Windows local roots are intentionally not stored in the public runtime manifest. Use `pack/runtime/runtime-packages.local.json` for machine-specific root overrides; start from `pack/runtime/runtime-packages.local.example.json`.

Managed runtime loading is production-first:

- normal probing checks the app base directory and `runtimes/<rid>/native`
- explicit bridge loading uses `JYPPX_NATIVE_BRIDGE_PATH`
- explicit vendor roots use `JYPPX_TENSORRT_ROOT` and `JYPPX_CUDA_ROOT`
- local `build-out` / `third_party` development scanning requires `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`

## Documentation Entry Points

- `docs/index.md`
- `docs/articles/en/getting-started.md`
- `docs/articles/en/installation-layout.md`
- `docs/articles/en/api-coverage-and-deferred-boundaries.md`
- `docs/articles/en/sample-runners.md`
- `docs/articles/en/runtime-packages.md`
- `docs/articles/en/package-consumer-validation.md`
- `docs/articles/en/release-candidate-gate.md`

Build documentation with:

```powershell
dotnet docfx .\docs\docfx.json
```

## Notes

- NVIDIA binaries are intentionally not committed.
- `third_party/` is only a local drop location.
- Runtime packages are intended to ship the bridge together with matching TensorRT, CUDA, and cuDNN dynamic libraries only after redistribution and package-size review.
- Public hand-written C# wrappers should include useful XML documentation; generated APIs may use generated comments.
