# 样例运行说明

本文描述的 validation runner 现在位于 `smoke/`，而面向普通用户的常用案例位于 `samples/`。

当前样例优先面向 Windows x64，用于验证 TensorRT / CUDA 托管高层对象、原生 bridge 加载、内存传输、stream/event、模型构建、tensor binding 和 enqueue 关键链路。
## 统一本地启动方式

请在仓库根目录执行样例命令，并让现有 C# `NativeBridgePathResolver` 自动探测 `build-out`、`third_party/nvidia` 和标准 CUDA 安装目录：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

只有在你明确要覆盖默认解析行为时，才需要设置 `JYPPX_NATIVE_BRIDGE_PATH`、`JYPPX_TENSORRT_ROOT`、`JYPPX_CUDA_ROOT` 或 `JYPPX_CUDNN_ROOT`。
## 推荐顺序

1. `CudaSmokeRunner`
2. `MultiStream`
3. `TensorRtSmokeRunner`
4. `LifecycleSmokeRunner`
5. `OnnxToEngineSmokeRunner`
6. `DynamicShape`
7. `InferenceBindings`
8. `NetworkBuilderSmokeRunner`
9. 各类 layer-specific network runners

## CudaSmokeRunner

用途：

- 验证 CUDA bridge 加载和设备发现。
- 验证 stream、event、device memory、pinned memory、managed memory、pitched memory、pointer attributes、peer copy 和常用 copy 路径。
- 验证 async device-memory allocation/free、memory pool、memory pressure 和 CUDA error-name/error-string 映射。

运行示例：

```powershell
dotnet .\smoke\CudaSmokeRunner\bin\Debug\net8.0\CudaSmokeRunner.dll
```

关键输出：

- `DeviceCount=...`
- `RoundTrip=True`
- `PinnedAsyncRoundTrip=True`
- `PitchedMemoryAsync RoundTrip=True`
- `PointerAttributes Type=Device Device=...`
- `CudaPeekLastError=0:cudaSuccess`

## MultiStream

用途：

- 验证两个 non-blocking CUDA stream。
- 验证不同 stream 上的异步 fill/copy。
- 验证 event-based cross-stream ordering。

运行示例：

```powershell
dotnet .\samples\MultiStream\bin\Debug\net8.0\MultiStream.dll
```

关键输出：

- `IndependentStreams=True`
- `CrossStreamWait=True`
- `MultiStream Passed=True`

## TensorRtSmokeRunner

用途：

- 验证 TensorRT bridge 加载。
- 验证 TensorRT 8 / 10 / 11 的代表性高层对象链路。
- 验证 logger、runtime、builder、config、network、engine、context、inspector、refitter 等封装边界。

运行 TensorRT 10：

```powershell
dotnet .\smoke\TensorRtSmokeRunner\bin\Debug\net8.0\TensorRtSmokeRunner.dll --tensor-rt-line 10
```

关键输出：

- `HighLevelChain10=True`
- `TryCreateRuntime10=True`
- `TryCreateBuilder10=True`
- `OutputSizing=[...]`

## LifecycleSmokeRunner

用途：

- 重复创建和释放 CUDA stream、event、memory。
- 重复执行 TensorRT 高层 build / serialize / deserialize / enqueue。
- 提前发现生命周期、释放顺序和 native loader 回归。

运行示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-WindowsLifecycleSmoke.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 -Iterations 2 -Configuration Release
```

关键输出：

- `CUDA LifecycleIterations=... Passed=True`
- `TensorRT Line=... LifecycleIterations=... Passed=True`
- `LifecycleSmokeRunner Passed=True`

## OnnxToEngineSmokeRunner

用途：

- 进程内生成最小动态 ONNX identity model。
- 验证 ONNX parser、optimization profile、timing cache、tensor address binding、enqueue 和输出回读。

运行示例：

```powershell
dotnet .\smoke\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
```

关键输出：

- `Parsed=True`
- `ProfileIndex=0`
- `EngineFileRoundTrip=True`
- `BindingReport Ready=True`
- `OutputMatch=True`

## DynamicShape

用途：

- 直接用 C# 构建 dynamic batch identity network。
- 配置 min/opt/max optimization profile。
- 使用 `TensorRtInferenceBindings` 完成输入复制、输出分配、tensor bind、enqueue 和输出读取。

运行示例：

```powershell
dotnet .\samples\DynamicShape\bin\Debug\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 3
```

关键输出：

- `Profile Index=... Min=... Opt=... Max=... Valid=True`
- `Readiness Ready=True Bound=True`
- `BindingReport Ready=True Inputs=1 Outputs=1`
- `OutputMatch=True`
- `DynamicShape Passed=True`

## InferenceBindings

用途：

- 直接用 C# 构建最小 explicit-batch identity network。
- 配置 min/opt/max optimization profile。
- 使用 `TensorRtInferenceBindings` 完成输入复制、输出分配、tensor bind、enqueue 和输出读取。
- 验证输出是否与输入完全一致。

运行示例：

```powershell
dotnet .\samples\InferenceBindings\bin\Debug\net8.0\InferenceBindings.dll --tensor-rt-line 10 --batch 2
```

关键输出：

- `BindingReport Ready=True Inputs=1 Outputs=1`
- `Readiness Ready=True Bound=True`
- `Execution ... OutputMatch=True`
- `InferenceBindings Passed=True`

## 直接网络样例

直接网络样例不依赖 ONNX parser，直接用 C# 构建 TensorRT network：

- `NetworkBuilderSmokeRunner`
- `NetworkLayersSmokeRunner`
- `NetworkShapeOpsSmokeRunner`
- `NetworkConcatSliceSmokeRunner`
- `NetworkSoftmaxTopKSmokeRunner`
- `NetworkActivationPoolingResizeSmokeRunner`
- `NetworkMatrixFillSelectSmokeRunner`
- `NetworkConvolutionScaleSmokeRunner`
- `NetworkDeconvolutionSmokeRunner`
- `NetworkLrnSmokeRunner`
- `NetworkQuantizeDequantizeSmokeRunner`
- `NetworkTrt11ModernLayersSmokeRunner`
- `NetworkTrt11ModernLayerMetadataRunner`
- `NetworkTrt11AdvancedLayersSmokeRunner`

## 资产依赖和 roadmap 目录

以下目录当前是 README/roadmap 状态，而不是可执行项目：

- `Classification`：需要可再分发分类模型、labels、输入图片和 preprocessing metadata。
- `CustomKernelPreprocess`：等待安全 public CUDA module/kernel wrapper，不直接暴露 raw kernel launch。
- `InferenceBindings`：现在提供用户侧常用的 tensor binding / enqueue 示例。
- `OnnxToEngine`：现在提供用户侧常用的 ONNX 转 engine 示例。
- `YoloDet`：需要 detector ONNX、labels、输入图片、decode/NMS metadata，部分模型还需要 plugin 诊断。

这些目录不是发布阻塞项，只要 README 说明清楚资产要求、当前替代 runner 和 roadmap。

