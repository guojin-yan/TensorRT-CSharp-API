# Runtime Packages

Runtime packages carry the native deployment assets for one explicit TensorRT / CUDA / cuDNN combination:

- the JYPPX native bridge library
- the matching CUDA runtime dynamic libraries
- the matching TensorRT dynamic libraries
- the matching cuDNN dynamic libraries when the selected TensorRT line needs them
- optional deployment dependencies such as cuBLAS for TensorRT 8 parser/plugin support

## Naming Rule

Runtime package keys and NuGet package IDs include dependency `major.minor` versions:

- runtime key: `win-x64-trt10.11-cuda11.8-cudnn8.9`
- package ID: `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9`

The public runtime manifest still stores full vendor versions, for example TensorRT `10.11.0.33` and cuDNN `8.9.7.29`. This keeps package names readable while keeping the exact binary provenance auditable.

Do not use ambiguous package names such as `win-x64-trt10-cuda11` or `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.cuda11`.

## Windows Matrix

Current Windows runtime targets:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`: TensorRT `8.6.1.6`, CUDA `11.8`, cuDNN `8.9.7.29`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`: TensorRT `8.6.1.6`, CUDA `12.1`, cuDNN `8.9.7.29`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`: TensorRT `10.11.0.33`, CUDA `11.8`, cuDNN `8.9.7.29`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: TensorRT `10.11.0.33`, CUDA `12.9`, cuDNN `9.22.0`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: TensorRT `11.0.0.114`, CUDA `12.9`, cuDNN `9.22.0`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: TensorRT `11.0.0.114`, CUDA `13.2`, cuDNN `9.22.0`

The stable TensorRT 10 / CUDA 11.8 path has current package-consumer evidence. On 2026-06-12, `win-x64-trt10.11-cuda11.8-cudnn8.9` restored from local packages, built a consumer app, copied `16/16` native assets, and passed smoke with TensorRT `10.11.0`, CUDA `11.8`, and one CUDA device.

CUDA `12.9` is installed on the current workstation and is now used by the CUDA `12.9` target presets. The `win-x64-trt10.11-cuda12.9-cudnn9.22` and `win-x64-trt11.0-cuda12.9-cudnn9.22` packages have completed local runtime asset collection, runtime packing, package consumer validation, and package consumer smoke.

TensorRT 11 packages are active in the matrix. The Windows `trt11.0-cuda12.9-cudnn9.22` path now has native minimal adapter smoke validation and package consumer smoke validation for logger/runtime/builder/config/network/serialized-engine/deserialize/context. The `trt11.0-cuda13.2-cudnn9.22` bridge compiles, collects assets, packs, and passes package consumer restore/build/native-copy validation locally, but runtime smoke remains pending until a CUDA 13-capable driver/runtime stack is available.

Current package consumer validation:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`: `16/16` native assets copied, package consumer smoke passed.
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: `19/19` native asset patterns copied, package consumer smoke passed.
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: `19/19` native asset patterns copied, package consumer smoke passed.
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: `19/19` native asset patterns copied, package consumer restore/build passed, package consumer smoke not requested because CUDA 13 runtime validation is pending.

## Local Roots

Windows local roots are intentionally not stored in the public manifest. Use `pack/runtime/runtime-packages.local.json` for machine-specific root overrides; that file is ignored by Git. Start from `pack/runtime/runtime-packages.local.example.json`.

Before packaging a Windows runtime package, validate all explicit inputs:

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot
```

## Asset Collection

Assets are collected by:

- `eng/Collect-RuntimeAssets.ps1`
- `pack/runtime/runtime-packages.manifest.json`

Current native bridge output layout:

- `build-out/<preset>/bin/<Configuration>/`
- `build-out/<preset>/lib/<Configuration>/`

Bridge binaries are isolated per CMake preset and should always be collected using the matching `buildPreset` from the runtime manifest.

TensorRT 8 Windows runtime packages collect additional parser/plugin dependencies:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`: `cublas64_11.dll`, `cublasLt64_11.dll`, and the full `cudnn*_8.dll` split runtime set
- `win-x64-trt8.6-cuda12.1-cudnn8.9`: `cublas64_12.dll`, `cublasLt64_12.dll`, and the full `cudnn*_8.dll` split runtime set

TensorRT 11 Windows packages use a different runtime layout from older TensorRT packages: DLLs are under `bin`, while import libraries are under `lib`. The runtime manifest collects the DLLs.

## Split-Delivery Prototype

TensorRT 10 packages are modeled with a design-only split-delivery prototype under `pack/runtime-split`.

Prototype package IDs:

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Extensions`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Extensions`

The `Core` role carries the bridge, CUDA runtime, and core TensorRT runtime libraries. The `Extensions` role carries builder resources, plugin libraries, parser libraries, and related optional assets.

Split packages are design-only until split consumer validation, package-size policy, and NVIDIA redistribution review are complete.

## Linux Status

Linux runtime package entries mirror the same TensorRT / CUDA / cuDNN major.minor matrix and include wildcard `.so` asset patterns. Linux packaging remains structurally prepared but not validated on this Windows workstation.

Current Linux workflow modules target future self-hosted Linux x64 runners:

- `runtime-linux.yml`
- `release-bundle.yml`

Linux packages must remain `dry-run-only` until a real Linux runner validates build, asset collection, package restore, native `.so` copy, and optional GPU smoke.

## Publication Risk

Runtime packages can become very large because they may include TensorRT builder resources, plugins, parser libraries, CUDA runtime assets, cuBLAS, and cuDNN.

Before public distribution:

- NVIDIA TensorRT / CUDA / cuDNN redistribution terms must be reviewed.
- NuGet.org package-size constraints must be evaluated.
- GitHub artifact / release hosting strategy must be decided.
- Large TensorRT 10 and TensorRT 11 packages may need private-feed or split-delivery strategies.
