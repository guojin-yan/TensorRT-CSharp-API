# Runtime Packages

> Policy update (2026-07-30): NVIDIA runtime packages are retired. TensorRtSharp publishes only the managed C# package, project-owned `.Bridge` packages, and tracked source archives. Consumers install CUDA, cuDNN, TensorRT, and optional NVRTC themselves. See `pack/external-vendor-runtime-policy.json`.

Each bridge package carries one project-owned native binary compiled for an explicit TensorRT / CUDA / cuDNN compatibility key:

- `jyppxtrtbridge.dll` on Windows; or
- `libjyppxtrtbridge.so` on Linux.

CUDA, TensorRT, cuDNN, parser, plugin, builder-resource, NVRTC, and NVRTC-builtins libraries are never package assets. The key selects bridge build headers/import libraries and documents the compatible machine-installed dependency line.

## Naming Rule

Runtime package keys and NuGet package IDs include dependency `major.minor` versions:

- runtime key: `win-x64-trt10.11-cuda11.8-cudnn8.9`
- package ID: `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Bridge`

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

Historical full/vendor-package runs from June 2026 remain useful only as migration diagnostics. They are not evidence for the current bridge-only publication policy and cannot be reused as post-publish proof.

The validated Windows maintainer environment uses CUDA `12.9` for the CUDA `12.9` target presets. Current validation builds the matching bridge, consumes managed plus bridge packages outside the repository, and resolves NVIDIA dependencies from the host.

TensorRT 11 bridge packages are active in the matrix. The Windows `trt11.0-cuda12.9-cudnn9.22` path has native minimal adapter smoke validation for logger/runtime/builder/config/network/serialized-engine/deserialize/context. A CUDA 13.2 bridge can be compiled independently, but runtime smoke still requires a CUDA 13-capable host driver/runtime stack.

Current proof records must distinguish bridge package copy evidence from host dependency discovery. A successful restore must copy exactly one project-owned bridge asset; TensorRT/CUDA/cuDNN/NVRTC paths and versions are recorded separately from the nupkg listing.

## Local Roots

Windows local roots are intentionally not stored in the public manifest. Use `pack/runtime/runtime-packages.local.json` for machine-specific root overrides; that file is ignored by Git. Start from `pack/runtime/runtime-packages.local.example.json`.

`eng/Invoke-LocalRuntimePackage.ps1` is retired and fails closed. Use the runtime manifest only to resolve compatible local headers, import libraries, and runtime smoke prerequisites; it is not a vendor-package publication manifest.

To record user-installed TensorRT/CUDA/cuDNN roots in the profile override used by self-hosted runs, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Sync-LocalRuntimeRoots.ps1 `
  -RuntimePackageKey <runtime-key> `
  -TensorRtRoot $env:JYPPX_TENSORRT_ROOT `
  -CudaRoot $env:JYPPX_CUDA_ROOT `
  -CudnnRoot $env:JYPPX_CUDNN_ROOT
```

Before compiling a Windows bridge package, validate all explicit inputs:

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 `
  -SplitPackageRole bridge
```

## Asset Collection

Bridge build inputs and outputs are resolved by:

- `eng/Invoke-LocalSplitRuntimePackage.ps1`
- `pack/runtime/runtime-packages.manifest.json`
- `pack/runtime-split/split-runtime-packages.manifest.json`

Current native bridge output layout:

- `build-out/<preset>/bin/<Configuration>/`
- `build-out/<preset>/lib/<Configuration>/`

Bridge binaries are isolated per CMake preset and should always be collected using the matching `buildPreset` from the runtime manifest.

TensorRT 8 and TensorRT 11 use different vendor layouts, but those differences affect compilation and host probing only. They must not change the one-bridge-binary nupkg boundary.

## Split Runtime Components

Bridge packages are modeled under `pack/runtime-split`.

Component roles:

- `Bridge`: carries only the local C ABI bridge and is the sole packable native role.

Historical `CudaCudnn`, `TensorRt`, `CudaRtc`, collection, meta, and full-runtime identities remain in selected manifests for cleanup and compatibility audit. Their projects are non-packable and policy gates reject them from upload.

## Linux Status

Linux runtime package identities include the distribution version and CPU architecture because NVIDIA publishes different apt repositories and binary sets per target. Do not publish generic `linux-x64-trt...` packages; use a distro-qualified key such as `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22`.

Current Linux matrix:

- Ubuntu 22.04 x64: hosted default for all six configured TensorRT / CUDA / cuDNN combinations.
- Ubuntu 24.04 x64: hosted for `trt10.11-cuda12.9-cudnn9.22`, `trt11.0-cuda12.9-cudnn9.22`, and `trt11.0-cuda13.2-cudnn9.22`; NVIDIA does not publish the older TensorRT 8.6 / CUDA 11.8 lines for Ubuntu 24.04.
- Ubuntu 20.04 x64: hosted-container line for the older TensorRT 8.6 and TensorRT 10.11 combinations that exist in the NVIDIA Ubuntu 20.04 repo. It runs on a GitHub-hosted runner with an `ubuntu:20.04` job container.
- Linux arm64/SBSA and Jetson/L4T are separate future package lines. SBSA server ARM and Jetson are not interchangeable, and neither should share the x64 Ubuntu package IDs.
- Other Linux distributions such as RHEL/Rocky should be added only after a matching NVIDIA repository and runner image are modeled explicitly.

`pack/runtime/linux-runtime-targets.manifest.json` is the target catalog for these Linux lines. It records the modeled Ubuntu targets, their `runtime_key_set` aliases, and the future package lines that still need dedicated package IDs, runners, official NVIDIA dependency plans, and package consumer evidence. `Resolve-RuntimeKeySet.ps1` recognizes future aliases such as `arm64-sbsa`, `jetson-l4t`, and `non-ubuntu`, but intentionally fails with readiness guidance instead of dispatching an unsupported package run.

Current Linux workflow modules:

- `runtime-linux.yml`
- `release-bundle.yml`

Ubuntu 20.04 x64, Ubuntu 22.04 x64, and Ubuntu 24.04 x64 have historical remote runs. New publication evidence must be regenerated with bridge-only packages. Linux arm64/SBSA and Jetson/L4T still need separate bridge package lines and runners.

Historical remote vendor-package map from 2026-06-17:

- `v4.0.6156`: Windows x64 runtime matrix, all six Windows combinations.
- `v4.0.6167`: Linux x64 Ubuntu 22.04 runtime matrix, all six hosted Ubuntu 22.04 combinations.
- `v4.0.6169`: Linux x64 Ubuntu 24.04 runtime matrix, the three modern hosted Ubuntu 24.04 combinations.
- `v4.0.6170`: managed package only.
- `v4.0.6171`: Linux x64 Ubuntu 20.04 runtime matrix, the three hosted-container Ubuntu 20.04 combinations.

Use `eng/Test-LinuxRuntimeTargetCoverage.ps1` to regenerate the target coverage report under `artifacts/linux-target-coverage`. The report is also uploaded by `release-publication-audit.yml` so publication evidence shows which Linux targets are modeled and which future ARM/Jetson/non-Ubuntu lines are intentionally held.

The vendor package versions and matching Release assets in this historical map were removed on 2026-07-30 after Owner review. Publication coverage must now count only managed, `.Bridge`, and source assets; old coverage reports cannot authorize republishing retired identities.

Use `eng/Export-RuntimePublicationIndex.ps1` after `eng/Test-GitHubPublicationInventory.ps1` to generate `artifacts/publication-index/runtime-publication-index.md`. This index shows which runtime combinations live under each Release tag and confirms the matching GitHub Packages entries, which is easier to read than the GitHub Packages package list.

## Publication Risk

Bridge packages remain small because they contain no NVIDIA runtime assets.

Current publication strategy:

- Publish `JYPPX.TensorRT.CSharp.API`, matching `.Bridge` packages, and tracked source archives only.
- Treat GitHub Release assets as downloadable package files, not as a NuGet feed. Verify their immutable URLs and digests before isolated restore staging.
- Require managed and bridge nuspec files to name the formal repository and the same source commit for promotable public asset evidence.
- Run `eng/Test-ExternalVendorRuntimePackagePolicy.ps1` on every pack and upload path.
- Record machine-installed NVIDIA dependency versions and paths as host evidence, never as package contents.
