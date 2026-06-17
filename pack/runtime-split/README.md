# split runtime packages

This directory contains split runtime component packages for runtime combinations that are too large or too stable to republish as a single nupkg.

Windows split projects are checked in because they are packaged on the self-hosted Windows runner. Linux split projects are generated dynamically from `pack/runtime/runtime-packages.manifest.json` by `eng/Invoke-LocalSplitRuntimePackage.ps1`, so Linux runtime split packages do not require checked-in per-combination project directories.

Current goals:

- keep the original runtime package ID available through a lightweight collection package
- move native binaries into component packages that each stay below GitHub package and release size limits
- validate that consumers can install the collection package and still receive the full native runtime layout
- allow bridge packages to be republished independently from stable CUDA/cuDNN/TensorRT packages

Current split strategy:

- most Windows runtime lines use the same three-component split:
- `Bridge`: the local `jyppxtrtbridge.dll` C ABI wrapper, republished when local native wrapper code changes
- `CudaCudnn`: CUDA runtime, cuDNN, and cuBLAS assets needed by older TensorRT lines, republished only when the CUDA/cuDNN dependency set changes
- `TensorRt`: TensorRT runtime, parser, plugin, and builder-resource binaries, republished only when the TensorRT dependency set changes
- TensorRT 11 packages may split TensorRT further into `TensorRtRuntime` plus builder-resource packages such as `TensorRtBuilder.Sm75Sm86`, `TensorRtBuilder.Sm89Sm90`, and `TensorRtBuilder.Sm100Sm120Ptx` when a single TensorRT component would exceed release host limits
- the original runtime package ID remains a lightweight collection package that references one `Bridge`, one `CudaCudnn`, and all required TensorRT component packages
- `win-x64-trt11.0-cuda13.2-cudnn9.22` remains blocked on CUDA 13-capable runtime smoke before public-ready validation

These packages still require:

- NVIDIA redistribution review
- package-size review for each package host
- consumer validation proof for the split package set

Publication guidance:

- nuget.org is suitable for the managed package and tiny bridge/collection packages only. Most CUDA/cuDNN/TensorRT component packages are larger than nuget.org's package-size limit.
- GitHub Packages can host the split component packages as a NuGet feed when the package stays under its registry limit.
- GitHub Releases are the fallback for large runtime assets and public release attachment. Release assets are not a NuGet feed, so consumers or validation scripts must first download the `.nupkg` files into a local package source.
- Use the same runtime package version for a full NVIDIA dependency refresh, for example `4.0.0`.
- Use a newer bridge and collection package version when the local C ABI bridge changes, for example `4.0.1`, while pinning `CudaCudnn` and `TensorRt` to their previously published dependency versions.

Examples:

```powershell
# Full split package set for one runtime key.
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole all `
  -RunSmoke

# First-time or dependency-upgrade stable dependency component refresh.
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole cuda-cudnn,tensorrt

# Native bridge refresh that publishes a new collection package but reuses stable dependency packages.
gh release download v4.0.6156 `
  --pattern "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.*.4.0.6156.nupkg" `
  --dir .\artifacts\stable-runtime-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22 `
  --repo guojin-yan/TensorRT-CSharp-API

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.1 `
  -SplitPackageRole bridge,collection `
  -CudaCudnnPackageVersion 4.0.6156 `
  -TensorRtPackageVersion 4.0.6156 `
  -AdditionalPackageSource .\artifacts\stable-runtime-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22
```

Supporting scripts:

- `eng/Validate-SplitRuntimePackages.ps1`
- `eng/Collect-SplitRuntimeAssets.ps1`
- `eng/Invoke-LocalSplitRuntimePackage.ps1`
