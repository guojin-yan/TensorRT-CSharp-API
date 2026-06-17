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

Linux package keys must include the distribution version and architecture. The default hosted Linux release line is Ubuntu 22.04 x64:

- `linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9`
- `linux-x64-ubuntu22.04-trt8.6-cuda12.1-cudnn8.9`
- `linux-x64-ubuntu22.04-trt10.11-cuda11.8-cudnn8.9`
- `linux-x64-ubuntu22.04-trt10.11-cuda12.9-cudnn9.22`
- `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22`
- `linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22`

Ubuntu 24.04 x64 is modeled only for the modern combinations that exist in NVIDIA's Ubuntu 24.04 repo. Ubuntu 20.04 x64 is self-hosted only. arm64/SBSA, Jetson/L4T, and non-Ubuntu distributions must be added as separate package lines after the matching NVIDIA repository and runner strategy are modeled.

The `runtime-linux` workflow can resolve package lines through `runtime_key_set`:

- `ubuntu22-hosted`: the default hosted publication line with all six Ubuntu 22.04 x64 combinations.
- `hosted-all`: every hosted Linux line, currently the six Ubuntu 22.04 x64 combinations plus the three modern Ubuntu 24.04 x64 combinations.
- `ubuntu24-hosted`: only the modern Ubuntu 24.04 x64 combinations.
- `self-hosted-ubuntu20`: only the modeled Ubuntu 20.04 x64 self-hosted combinations; dispatch with `runner_mode=self-hosted`.
- `custom`: requires explicit `runtime_keys`.

Explicit `runtime_keys` always win. When `runtime_keys` is empty, `runtime_key_set` selects the package line. This keeps the normal release path on Ubuntu 22.04 hosted while still making hosted-all and Ubuntu 20.04 self-hosted publication explicit and repeatable.

The `release-bundle` workflow now has two Linux orchestration lanes:

- `run_linux_runtime_packaging`: hosted Linux publication, defaulting to `hosted-all` so Ubuntu 22.04 x64 and the modeled Ubuntu 24.04 x64 lines are dispatched together.
- `run_linux_self_hosted_ubuntu20_runtime_packaging`: Ubuntu 20.04 x64 self-hosted publication, defaulting to `self-hosted-ubuntu20` and always dispatching `runner_mode=self-hosted`.

Use separate release-bundle inputs for Linux split roles and stable dependency versions. Routine bridge or managed changes can publish Linux `bridge,collection` while pinning already-published `CudaCudnn` and `TensorRt` versions; NVIDIA dependency refreshes should use `cuda-cudnn`, `tensorrt`, or `all`. When one dispatch spans multiple dependency publication versions, such as `hosted-all`, pin the dependencies with runtime-key maps instead of one global version. The current hosted Linux bridge/collection refresh maps `linux-x64-ubuntu22.04-*` to `4.0.6167` and `linux-x64-ubuntu24.04-*` to `4.0.6169`; the default release tag is `v<resolved package version>` unless a release-tag map is supplied. Less common overrides, such as delivery mode, release tags for stable dependency assets, bridge/meta package versions, and skip-validation toggles, are passed through `release_config_json` to keep the manual GitHub Actions form under the `workflow_dispatch` input limit.

Prefer `eng/Invoke-RemoteReleaseBundle.ps1` when dispatching releases from a workstation. The script keeps supported top-level workflow inputs as `-f key=value` flags and serializes advanced release settings into `release_config_json`, which avoids accidental dispatch failures from undeclared workflow inputs.

Linux packages stay `dry-run-only` until a matching Linux runner validates build, asset collection, pack, and package consumer restore/build.

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
- Use `cuda_cudnn_package_version_map` and `tensorrt_package_version_map` when a collection package references stable dependency packages that were published under different versions for different runtime keys.

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
