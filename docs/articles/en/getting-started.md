# Getting Started

TensorRtSharp4.0 currently has two verified local validation baselines:

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

## Verified common examples

MultiStream:

```powershell
dotnet .\samples\MultiStream\bin\Debug\net8.0\MultiStream.dll
```

DynamicShape:

```powershell
dotnet .\samples\DynamicShape\bin\Debug\net8.0\DynamicShape.dll --tensor-rt-line 10
```

InferenceBindings:

```powershell
dotnet .\samples\InferenceBindings\bin\Debug\net8.0\InferenceBindings.dll --tensor-rt-line 10 --batch 2
```

OnnxToEngine:

```powershell
dotnet .\samples\OnnxToEngine\bin\Debug\net8.0\OnnxToEngine.dll --tensor-rt-line 10
```

Classification with your own ONNX classifier assets:

```powershell
dotnet run --project .\samples\Classification -- --model .\models\classifier.onnx --labels .\models\labels.txt --input-shape 1x3x224x224 --tensor-rt-line 10
```

YOLO-family detection with your own ONNX detector assets:

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolo.onnx --labels .\models\coco.names --input-shape 1x3x640x640 --tensor-rt-line 10
```

## Verified smoke commands

CUDA smoke:

```powershell
dotnet .\smoke\CudaSmokeRunner\bin\Debug\net8.0\CudaSmokeRunner.dll
```

TensorRT minimum smoke:

```powershell
dotnet .\smoke\TensorRtSmokeRunner\bin\Debug\net8.0\TensorRtSmokeRunner.dll
```

## Read next

- [Installation Layout](installation-layout.md)
- [Runtime Packages](runtime-packages.md)
- [Sample Runners](sample-runners.md)
