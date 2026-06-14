# Runtime Packages

This directory contains runtime-specific NuGet package projects for explicit TensorRT / CUDA / cuDNN combinations.

Runtime package identities use dependency `major.minor` fragments, for example:

- Key: `win-x64-trt10.11-cuda11.8-cudnn8.9`
- Package ID: `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9`

Before packing a Windows runtime package, validate the explicit local roots and collect assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 `
  -TensorRtRoot "<TensorRT root>" `
  -CudaRoot "<CUDA root>" `
  -CudnnRoot "<cuDNN root>"
powershell -ExecutionPolicy Bypass -File .\eng\Collect-RuntimeAssets.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9
```

Then pack the managed package and matching runtime package:

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj -c Release -o .\artifacts\managed
dotnet pack .\pack\runtime\win-x64-trt10.11-cuda11.8-cudnn8.9\JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.csproj -c Release -o .\artifacts\runtime-nupkg
```

Validate a consumer project from the local packages:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 -RunSmoke -SmokeRuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9
```

Latest local evidence on 2026-06-12:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`: restored, built, copied `16/16` native assets, and passed smoke with TensorRT `10.11.0`, CUDA `11.8`, and one CUDA device.
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: restored, built, copied `19/19` native asset patterns, and passed smoke with TensorRT `10.11.0`, CUDA `12.9`, and one CUDA device.
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: restored, built, copied `19/19` native asset patterns, and passed smoke with TensorRT `11.0.0`, CUDA `12.9`, and one CUDA device.
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: restored, built, and copied `19/19` native asset patterns; smoke remains pending until a CUDA 13-capable driver/runtime stack is available.

Machine-specific roots belong in `runtime-packages.local.json`, which is ignored by Git. Public package metadata belongs in `runtime-packages.manifest.json`.

Start from `runtime-packages.local.example.json` for Windows or Linux runners. CUDA, cuDNN, and TensorRT binaries must be downloaded from official NVIDIA distributions and installed or unpacked on the self-hosted runner; the workflows resolve those roots and do not commit or fetch vendor binaries from Git.
