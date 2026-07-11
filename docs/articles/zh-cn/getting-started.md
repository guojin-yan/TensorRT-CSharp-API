# 快速开始

当前仓库已经完成本机扫描头文件的接口覆盖追平：

- TensorRT interface coverage：`0` missing rows。
- CUDA runtime interface coverage：`0` missing rows。
- Manifest inventory：`3271` 条 API records，`102` 份 manifests。

当前重点已经从“继续补 missing API”转为发布加固：样例可运行性、文档一致性、runtime package 验证和 package consumer 证据。

## 前置条件

- .NET SDK `10.0.300` 或更高。
- CMake `3.27` 或更高。
- Windows x64 或 Linux x64。
- 如需 native/runtime package 验证，需要匹配的 TensorRT、CUDA、cuDNN 本机根目录。

## 本地质量门

```powershell
dotnet restore .\TensorRtSharp.sln
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如果修改了 manifest、native 或 generated 文件，还需要运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## 推荐常用案例顺序

1. `MultiStream`
2. `DynamicShape`
3. `InferenceBindings`
4. `OnnxToEngine`
5. `Classification`，使用你自己的分类 ONNX 资产
6. `YoloVision`，使用你自己的 YOLO-family ONNX 资产

推荐 smoke 顺序：

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. 各类 layer-specific network runners

`MultiStream`、`DynamicShape`、`InferenceBindings` 和 `OnnxToEngine` 都是真实可运行案例并加入解决方案。`Classification` 和 `YoloVision` 也是可执行项目，但需要你提供可再分发的 ONNX model / labels / 输入尺寸等资产。CUDA custom-kernel preprocessing 先保留为文档路线图，等待安全 public CUDA module/kernel wrapper 后再加入可运行 sample。

分类模型示例：

```powershell
dotnet run --project .\samples\Classification -- --model .\models\classifier.onnx --labels .\models\labels.txt --input-shape 1x3x224x224 --tensor-rt-line 10
```

YOLO 检测模型示例：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolo.onnx --labels .\models\coco.names --input-shape 1x3x640x640 --tensor-rt-line 10
```

## 继续阅读

- [安装布局说明](installation-layout.md)
- [API 覆盖与 deferred 边界](../en/api-coverage-and-deferred-boundaries.md)
- [样例运行说明](sample-runners.md)
- [运行时包说明](runtime-packages.md)
- [发布候选质量门禁](release-candidate-gate.md)
