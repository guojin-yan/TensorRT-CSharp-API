# split runtime packages

This directory contains split runtime component packages for Windows runtime combinations that are too large to publish as a single nupkg.

Current goals:

- keep the original runtime package ID available through a lightweight collection package
- move native binaries into component packages that each stay below GitHub package and release size limits
- validate that consumers can install the collection package and still receive the full native runtime layout
- allow bridge packages to be republished independently from stable CUDA/cuDNN/TensorRT packages

Current split strategies:

- TensorRT 8: three-part split (`Bridge` + `CudaCudnn` + `TensorRtRuntime`) for both Windows public sample lines
- TensorRT 10: component split (`Bridge` + `CudaCudnn` + `TensorRtRuntime` + `TensorRtExtensions`) with a matching collection package per runtime line
- TensorRT 11: publish-oriented component split for `win-x64-trt11.0-cuda12.9-cudnn9.22`
  - `Bridge`
  - `CudaCudnn`
  - `TensorRtRuntime`
  - `TensorRtBuilder.Sm75Sm86`
  - `TensorRtBuilder.Sm89Sm90`
  - `TensorRtBuilder.Sm100Sm120Ptx`
- TensorRT 11: mirrored component split for `win-x64-trt11.0-cuda13.2-cudnn9.22`
  - keeps the same package layout as the CUDA 12.9 line
  - remains blocked on CUDA 13-capable runtime smoke before public-ready validation

These packages still require:

- NVIDIA redistribution review
- package-size review for each package host
- consumer validation proof for the split package set

Publication guidance:

- nuget.org is suitable for the managed package and tiny bridge/collection packages only. Most CUDA/cuDNN/TensorRT component packages are larger than nuget.org's package-size limit.
- GitHub Packages can host the split component packages as a NuGet feed when the package stays under its registry limit.
- GitHub Releases are the fallback for large runtime assets and public release attachment.
- Use the same runtime package version for a full NVIDIA dependency refresh, for example `4.0.0`.
- Use a newer bridge and collection package version when the local C ABI bridge changes, for example `4.0.1`, while pinning `CudaCudnn` and `TensorRtRuntime` to the previously published vendor version.

Examples:

```powershell
# Full split package set for one runtime key.
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole all `
  -RunSmoke

# First-time or dependency-upgrade vendor component refresh.
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole vendor

# Native bridge refresh that publishes a new collection package but reuses vendor packages.
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.1 `
  -SplitPackageRole bridge,collection `
  -VendorPackageVersion 4.0.0 `
  -AdditionalPackageSource https://nuget.pkg.github.com/<owner>/index.json `
  -AdditionalPackageSourceUsername <owner-or-actor> `
  -AdditionalPackageSourcePassword <token>
```

Supporting scripts:

- `eng/Validate-SplitRuntimePackages.ps1`
- `eng/Collect-SplitRuntimeAssets.ps1`
- `eng/Invoke-LocalSplitRuntimePackage.ps1`
