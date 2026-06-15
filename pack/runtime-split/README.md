# split runtime packages

This directory contains split runtime component packages for Windows runtime combinations that are too large to publish as a single nupkg.

Current goals:

- keep the original runtime package ID available through a lightweight collection package
- move native binaries into component packages that each stay below GitHub package and release size limits
- validate that consumers can install the collection package and still receive the full native runtime layout
- allow bridge packages to be republished independently from stable CUDA/cuDNN/TensorRT packages

Current split strategies:

- every Windows runtime line uses the same two-component split:
- `Bridge`: the local `jyppxtrtbridge.dll` C ABI wrapper, republished when local native wrapper code changes
- `Vendor`: CUDA runtime, cuDNN, TensorRT runtime, parser, plugin, and builder-resource binaries, republished only when the NVIDIA dependency set changes
- the original runtime package ID remains a lightweight collection package that references one `Bridge` package version and one `Vendor` package version
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
- Use a newer bridge and collection package version when the local C ABI bridge changes, for example `4.0.1`, while pinning `Vendor` to the previously published NVIDIA dependency version.

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
gh release download v4.0.6142 `
  --pattern "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Vendor.4.0.6142.nupkg" `
  --dir .\artifacts\vendor-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22 `
  --repo guojin-yan/TensorRT-CSharp-API

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.1 `
  -SplitPackageRole bridge,collection `
  -VendorPackageVersion 4.0.6142 `
  -AdditionalPackageSource .\artifacts\vendor-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22
```

Supporting scripts:

- `eng/Validate-SplitRuntimePackages.ps1`
- `eng/Collect-SplitRuntimeAssets.ps1`
- `eng/Invoke-LocalSplitRuntimePackage.ps1`
