# Runtime compatibility matrix

The former full-runtime package projects have been removed. This directory now contains compatibility manifests and fail-closed policy metadata only; it contains no runtime package project.

`runtime-packages.manifest.json` remains the canonical TensorRT/CUDA/cuDNN compatibility and build-input matrix. Its vendor file lists describe locally installed dependencies used for build and diagnostic probes. They are not package assets and must never be copied to NuGet packages or GitHub Release assets.

Consumers install matching NVIDIA dependencies themselves:

- TensorRT
- CUDA Toolkit/runtime
- cuDNN when required by the selected TensorRT line
- NVRTC and matching builtins when runtime compilation is used

Use `runtime-packages.local.example.json` as the starting point for machine-specific roots. The real `runtime-packages.local.json` remains ignored by Git.

Build a package through the bridge-only entry point:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge
```

`eng/Invoke-LocalRuntimePackage.ps1` and `eng/Collect-RuntimeAssets.ps1` remain fail-closed compatibility guards. The enforced policy is `pack/external-vendor-runtime-policy.json`.
