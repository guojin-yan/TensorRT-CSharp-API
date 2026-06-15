# Runtime Distribution Strategy

## Current reality

Runtime packages must now be versioned by the exact major/minor deployment combination:

- TensorRT major/minor, for example `trt10.11`
- CUDA major/minor, for example `cuda11.8`
- cuDNN major/minor, for example `cudnn8.9`

Example package key:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`

Example package id:

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9`

Do not use ambiguous names such as `win-x64-trt10-cuda11` for runtime packages. CMake preset names may still use line-based labels such as `win-x64-trt10-cuda11-release` because they describe a build line rather than a NuGet runtime identity.

## Windows runtime matrix

Current Windows runtime package targets:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

CUDA `12.9` is installed on the current Windows development machine. Runtime packages targeting `cuda12.9` must now build and validate against CUDA `12.9`; the earlier CUDA `12.3` fallback is retired and must not be used for locally validated packages.

Current Windows package-consumer readiness:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`: package consumer smoke passed.
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: package consumer smoke passed.
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: package consumer smoke passed.
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: 2026-06-14 full split package set and collection package packed locally, restore/build/native-copy passed with `19/19` native asset patterns, but readiness remains blocked until CUDA 13 runtime/builder smoke is available.

## Linux runtime matrix

Linux package structure is kept in the manifest and workflows, but Linux packaging is not the current development focus. Linux package keys mirror the Windows matrix:

- `linux-x64-trt8.6-cuda11.8-cudnn8.9`
- `linux-x64-trt8.6-cuda12.1-cudnn8.9`
- `linux-x64-trt10.11-cuda11.8-cudnn8.9`
- `linux-x64-trt10.11-cuda12.9-cudnn9.22`
- `linux-x64-trt11.0-cuda12.9-cudnn9.22`
- `linux-x64-trt11.0-cuda13.2-cudnn9.22`

Linux packages stay `dry-run-only` until a real Linux x64 runner validates build, asset collection, pack, and package consumer restore/build.

## Distribution lanes

Current lane guidance:

- `TRT8` Windows packages are public-preview candidates only after NVIDIA redistribution terms and package size limits are reviewed.
- `TRT10` Windows packages are private-feed or split-delivery candidates because builder resources, plugins, and parser assets can be large. Both Windows TRT10 package-consumer smoke paths have 2026-06-12 local evidence.
- `TRT11` Windows CUDA `12.9` is a private-feed candidate with package-consumer smoke evidence. Windows CUDA `13.2` remains blocked until driver/runtime-compatible smoke is available.
- Linux packages stay dry-run candidates until a real Linux runner validates them.

## nuget.org size boundary

nuget.org has an approximately `250 MB` per-package size limit. Windows split runtime packages should be audited per release because CUDA/cuDNN and TensorRT packages can still exceed that size. Therefore:

- The `JYPPX.TensorRT.CSharp.API` managed package can be published to nuget.org.
- Small `Bridge` and collection packages can be published to nuget.org or GitHub Packages when needed.
- Most CUDA/cuDNN and TensorRT dependency component packages are not suitable for nuget.org and should normally stay on GitHub Packages when a NuGet feed is required, or as GitHub Release assets when direct `.nupkg` download is acceptable.
- GitHub Release assets are not queried by NuGet restore. If stable dependency packages stay only on a Release, validation and consumers must download the matching `.nupkg` files into a local package source first.
- If only the local C ABI bridge or C# wrapper changes later, republish the `Bridge`, collection, and managed packages. Do not republish `CudaCudnn` or `TensorRt` packages unless the corresponding NVIDIA dependency set changes.

## Engineering rules

Each runtime package entry in `pack/runtime/runtime-packages.manifest.json` records:

- `tensorRtVersion`
- `cudaVersion`
- `cudnnVersion`
- `cudnnMajor`
- `distributionTier`
- `validationState`
- `distributionNotes`

Validation rules:

- Package key and package id must include TensorRT, CUDA, and cuDNN major/minor fragments.
- Full vendor patch versions stay in manifest metadata and documentation, not in package identity.
- CUDA `12.9` target packages can only be marked `local-validated` when the exact CUDA `12.9` toolkit and matching TensorRT/cuDNN assets were used.
- Private-feed and split-delivery readiness require `local-validated`; `pending-local-validation` and `dry-run-only` must stay blocked.
- NVIDIA binary dependencies must not be committed to Git.

Supporting scripts:

- `eng/Validate-RuntimeManifest.ps1`
- `eng/Validate-WindowsRuntimeInputs.ps1`
- `eng/Collect-RuntimeAssets.ps1`
- `eng/Export-RuntimeDistributionReport.ps1`
- `eng/Export-RuntimeDeliveryStrategy.ps1`
- `eng/Test-RuntimePublishReadiness.ps1`
- `eng/Export-ReleaseCandidateChecklist.ps1`

## Split Runtime Components

The split runtime model applies to Windows runtime combinations that are too large or too stable to republish with every managed-code release:

- `Bridge`: local C ABI bridge, republished when native wrapper code changes.
- `CudaCudnn`: CUDA runtime, cuDNN, and related shared assets, republished only when the CUDA/cuDNN dependency set changes.
- `TensorRt`: TensorRT runtime, parser, plugin, and builder-resource assets, republished only when the TensorRT dependency set changes.
- collection package: the original runtime package ID, republished when a new tested component-version combination should be advertised.

The managed package can release independently from these runtime component packages. Routine C# or bridge changes should publish only the managed package, `Bridge`, and collection package while pinning the existing `CudaCudnn` and `TensorRt` package versions.

## Release boundary

Formal public distribution is blocked until these issues are resolved:

- NVIDIA CUDA / cuDNN / TensorRT redistribution terms are reviewed.
- NuGet.org package-size practicality is reviewed.
- Runtime package keys, package IDs, component package versions, and manifest metadata are aligned.
- Package consumer validation covers the intended release packages.
- TensorRT 11 packages pass real adapter-backed runtime/package smoke on the intended CUDA driver/runtime stack.
