# Getting Started

TensorRtSharp4.0 currently has two verified local baselines:

- CUDA smoke path through `CudaSmokeRunner`
- TensorRT vendor-backed minimum path through `TensorRtSmokeRunner`

## Prerequisites

- .NET SDK `10.0.300` or newer
- CMake `3.27` or newer
- Windows x64 or Linux x64

Windows local development currently assumes:

- CUDA is installed in a standard NVIDIA Toolkit location and is discoverable through `CUDA_PATH*` or `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`
- TensorRT Windows C++ packages are unpacked under `<repo-root>\third_party\nvidia`

## First commands

```powershell
dotnet restore
dotnet build TensorRtSharp.sln -c Debug
powershell -ExecutionPolicy Bypass -File .\eng\Get-Dependencies.ps1
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

## Verified sample commands

CUDA smoke:

```powershell
dotnet .\samples\CudaSmokeRunner\bin\Debug\net8.0\CudaSmokeRunner.dll
```

TensorRT minimum smoke:

```powershell
dotnet .\samples\TensorRtSmokeRunner\bin\Debug\net8.0\TensorRtSmokeRunner.dll
```

## Read next

- [Installation Layout](installation-layout.md)
- [Runtime Packages](runtime-packages.md)
- [Sample Runners](sample-runners.md)
