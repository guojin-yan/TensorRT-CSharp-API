# Sample Runners

The validation-oriented runner projects described in this article now live under `smoke/`, while user-facing common examples live under `samples/`.
## Shared Local Setup

Run sample commands from the repository root and let the existing C# `NativeBridgePathResolver` discover `build-out`, `third_party/nvidia`, and standard CUDA install locations:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Only set `JYPPX_NATIVE_BRIDGE_PATH`, `JYPPX_TENSORRT_ROOT`, `JYPPX_CUDA_ROOT`, or `JYPPX_CUDNN_ROOT` when you intentionally want to override the default resolver behavior.
## CudaSmokeRunner

Purpose:

- validate CUDA bridge loading
- validate device discovery
- validate stream, event, memory, memcpy, memset, managed memory, pinned host memory, registered host memory, async copy, device-to-device copy, optional peer copy, pointer attributes, and memory info paths
- validate async memory allocation/free, owned memory-pool create/destroy, default memory-pool attribute query paths, and memory-pool access descriptors
- validate `CudaDeviceScope` and float array round trips for deployment-style input/output buffers
- validate selected device attributes and optional peer-access capability
- validate stream/event/pinned-memory flags and CUDA error-name/error-string mapping

Run:

```powershell
dotnet .\smoke\CudaSmokeRunner\bin\Debug\net8.0\CudaSmokeRunner.dll
```

Expected signals:

- `DeviceCount=...`
- `RoundTrip=True`
- `FloatRoundTrip=True`
- `Fill=True`
- `StreamAndEvent=True`
- `DeviceToDevice=True`
- `MemcpyDefault Sync=True Async=True`
- `PointerAttributes Type=Device Device=...`
- `OwnedMemoryPool Device=... PoolAsyncAllocation=True ... ResetHigh=True Trim=True`
- `DefaultMemoryPool Device=... ResetHigh=True`
- `MemoryPoolAccess Self=... Peer=...`
- `PeerCopy Sync=True Async=True` on multi-GPU systems where peer access is available
- `StreamFlags=NonBlocking`
- `PinnedFlags`
- `CudaPeekLastError=0:cudaSuccess`
- `MemoryPressure Device=... Free=... Total=... FreeRatio=...`
- `PinnedAsyncRoundTrip=True`
- `RegisteredHostMemoryAsyncRoundTrip=True` when `HostRegisterSupported=True`
- `DeviceAttribute ManagedMemory=...`
- `ManagedMemoryRoundTrip=True` when supported by the selected device
- `PeerAccess` or `PeerAccess Skipped=True`
- `StreamReadyAfterSync=True`
- `StreamWaitEvent=True`
- `PinnedFloatAsyncRoundTrip=True`
- `DefaultMemoryPool Device=...`
- `AsyncMemoryPoolAllocation=True` when supported by the selected CUDA runtime
- `CudaGetLastError=0:cudaSuccess`

Recent CUDA deployment signals:

- `PitchedMemory SyncRoundTrip=True`
- `PitchedMemory Fill2D=True`
- `PitchedMemory DeviceToDevice=True`
- `PitchedMemoryAsync RoundTrip=True`
- `PitchedMemoryAsync Fill2DAsync=True`
- `ManagedMemoryAdvice=Skipped ...` is acceptable on devices or driver policies that reject the selected `cudaMemAdvise` combination, as long as `ManagedMemoryRoundTrip=True` remains true.

## TensorRtSmokeRunner

Purpose:

- validate TensorRT bridge loading
- validate the TensorRT 10 minimum vendor-backed path
- validate build -> serialize -> deserialize -> engine I/O metadata -> execution context creation
- validate execution-context max-output-size query and tensor debug-state boundary handling

Run:

```powershell
dotnet .\smoke\TensorRtSmokeRunner\bin\Debug\net8.0\TensorRtSmokeRunner.dll
```

Expected signals:

- `TRT10 Adapter Vendor=True`
- `TryCreateRuntime10=True`
- `TryRunMinimalBuildChain10=True`
- `HighLevelChain10=True`
- `Refitter=Created All=... Missing=...` for refittable engines, including entry enumeration output
- `OutputSizing=[...]`
- `TensorDebug=Skipped...` when TensorRT refuses to enable debug state for the minimal tensor

When the bridge is built against a TensorRT 8 preset, the same runner should also report:

- `TryCreateRuntime8=True`
- `TryCreateBuilder8=True`
- `TryRunMinimalBuildChain8=True`
- `HighLevelChain8=True`
- `TensorDebug=Unsupported`
- `Refitter=Skipped NonRefittable` is acceptable for the current minimal TensorRT 8 identity engine.

## MultiStream

Purpose:

- validate two independent non-blocking CUDA streams
- validate asynchronous fill and device-to-pinned-host copies on separate streams
- validate event-based cross-stream ordering with `CudaStream.WaitFor`
- provide a compact deployment pattern for overlapping preprocessing, postprocessing, and host transfer work

Run:

```powershell
dotnet .\samples\MultiStream\bin\Debug\net8.0\MultiStream.dll
```

Expected signals:

- `IndependentStreams=True`
- `CrossStreamWait=True`
- `MultiStream Passed=True`

## DynamicShape

Purpose:

- build a direct TensorRT identity network with an explicit dynamic batch dimension
- configure min/opt/max optimization profile shapes for the input tensor
- select the runtime input shape from command-line batch size
- use `TensorRtInferenceBindings` to copy host input, allocate output, bind tensors, enqueue, and read output
- validate that dynamic-shape inference preserves the input values

Run:

```powershell
dotnet .\samples\DynamicShape\bin\Debug\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 3
```

Expected signals:

- `Profile Index=... Min=... Opt=... Max=... Valid=True`
- `RuntimeShape=...`
- `Readiness Ready=True Bound=True`
- `BindingReport Ready=True Inputs=1 Outputs=1`
- `OutputMatch=True`
- `DynamicShape Passed=True`

## RefitWeightsSmokeRunner

Purpose:

- build a small direct TensorRT scale network with the `Refit` builder flag
- enumerate all refittable and missing weight entries
- set explicit host-buffer scale weights through `TensorRtRefitter.SetWeights`
- run `RefitCudaEngine`
- verify that inference output changes after refit

Run against TensorRT 10:

```powershell
dotnet .\smoke\RefitWeightsSmokeRunner\bin\Debug\net8.0\RefitWeightsSmokeRunner.dll --tensor-rt-line 10
```

Expected TensorRT 10 signals:

- `Engine Refittable=True`
- `RefitEntries All=... Missing=...`
- `RefitWeights Set=True Refit=True`
- `OutputChanged=True`

Current local TensorRT 8 behavior is explicit:

- `RefitWeights=Skipped Reason=EngineNotRefittable`

## LifecycleSmokeRunner

Purpose:

- repeatedly create and dispose CUDA stream, event, and memory wrappers
- repeatedly run the TensorRT logger/runtime/builder/minimal-build-chain path
- catch obvious lifecycle, release-order, and native loader regressions before release

Run against TensorRT 10:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-WindowsLifecycleSmoke.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 -Iterations 3 -Configuration Release
```

Run against TensorRT 8:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-WindowsLifecycleSmoke.ps1 -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 -Iterations 3 -Configuration Release
```

The lifecycle helper uses the selected preset's `build-out` bridge by default. Pass `-UseRuntimeAssets` only when you explicitly want to validate the collected runtime-package native asset directory. Use `-Configuration Release` when Windows application-control policies block freshly built Debug sample assemblies with `0x800711C7`. Some Windows application-control policies may block copied runtime assets even when the freshly built bridge is allowed.

Expected signals:

- `CUDA LifecycleIterations=... Passed=True`
- `TensorRT Line=... LifecycleIterations=... Passed=True`
- `LifecycleSmokeRunner Passed=True`
- `BindingReport:2:True` in the TensorRT advanced API readiness summary

## OnnxToEngineSmokeRunner

Purpose:

- generate a minimal dynamic ONNX identity model in-process
- validate ONNX parser creation and parse-from-memory
- validate optimization profile setup, profile shape query, active profile selection, max-output-size query, and tensor debug-state boundary handling
- validate builder config workspace limit and builder flags
- validate timing cache create, attach, and serialize
- validate parser error-summary plumbing
- validate serialized host-memory byte/file round trip and runtime deserialize from file
- validate engine deserialization, tensor address binding, stream enqueue, and output round trip

Run against TensorRT 10:

```powershell
dotnet .\smoke\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Parsed=True`
- `ParserErrors=0`
- `ProfileIndex=0`
- `ProfileShapes Min=[1, 4] Opt=[2, 4] Max=[4, 4]`
- `ActiveAfter=0`
- `EnqueueEmitsProfileToggle=False`
- `WorkspaceLimit=67108864`
- `Tf32=True`
- `TimingCacheBytes=...`
- `EngineFileRoundTrip=True`
- `MaxOutputSize=...`
- `TensorDebug=...`
- `ParserErrorSummary=...`
- `BindingReport Ready=True Profile=0 Tensors=2`
- `OutputMatch=True`

## Asset-Dependent Sample Directories

The repository also keeps a few user-facing sample topic directories that are documented instead of executable today:

- `samples/Classification`: requires a redistributable classifier ONNX model, labels, input image, and preprocessing metadata. Use `DynamicShape` and `OnnxToEngineSmokeRunner` to validate the deployment foundation first.
- `samples/OnnxToEngine`: now provides the user-facing common ONNX-to-engine example.
- `samples/YoloDet`: requires a redistributable detector ONNX model, labels, input image, postprocessing metadata, and possibly TensorRT plugin diagnostics.
- `samples/CustomKernelPreprocess`: blocked on safe public CUDA module/kernel wrappers; use `CudaSmokeRunner` and `MultiStream` for the current memory/stream preprocessing primitives.

## NetworkBuilderSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddInput`, `TensorRtNetworkDefinition.AddIdentity`, `TensorRtNetworkDefinition.MarkOutput`, `TensorRtTensor`, and `TensorRtLayer`
- validate builder config optimization level, profiling verbosity, and max auxiliary streams
- build a serialized engine, deserialize it, bind input/output tensors, enqueue, and verify output round trip

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkBuilderSmokeRunner\bin\Debug\net8.0\NetworkBuilderSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkBuilderSmokeRunner\bin\Debug\net8.0\NetworkBuilderSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Network Inputs=1 Outputs=1`
- `Input=input:Float:[-1, 4]`
- `Output=output:Float:[-1, 4]`
- `BuilderConfig OptLevel=3`
- `BindingReport Ready=True Profile=0 Tensors=2`
- tensor format entries such as `input:Input:Linear:-1:Row major linear FP32 format`
- `Enqueue=True`
- `OutputMatch=True`

## NetworkLayersSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtWeights`, `TensorRtNetworkDefinition.AddConstant`, and `TensorRtNetworkDefinition.AddElementWise`
- validate layer metadata for constant and elementwise layers, including name, type, input count, and output count
- build a serialized engine, deserialize it, bind input/output tensors, enqueue, and verify `output = input + constant`

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkLayersSmokeRunner\bin\Debug\net8.0\NetworkLayersSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkLayersSmokeRunner\bin\Debug\net8.0\NetworkLayersSmokeRunner.dll --tensor-rt-line 8
```

The explicit cuDNN root is only required when running directly from the build output for TensorRT 8 parser/plugin dependencies. Refreshed TensorRT 8 runtime packages collect the full `cudnn*_8.dll` split runtime set as part of their native assets.

Expected signals:

- `LayerMetadata Constant=bias_constant:Constant:I0:O1`
- `Sum=sum:ElementWise:I2:O1`
- `Network Inputs=1 Outputs=1`
- `ElementWiseOutputMatch=True`

## NetworkShapeOpsSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddShuffle`, `TensorRtLayer.SetShuffleReshapeDimensions`, and `TensorRtNetworkDefinition.AddReduce`
- validate shuffle and reduce metadata, including layer type, reshape dimensions, reduce operation, axes, and keep-dimensions
- build a serialized engine, deserialize it, bind input/output tensors, enqueue, and verify a small reshape-plus-reduce result

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkShapeOpsSmokeRunner\bin\Debug\net8.0\NetworkShapeOpsSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkShapeOpsSmokeRunner\bin\Debug\net8.0\NetworkShapeOpsSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `ShapeOpsMetadata Shuffle=reshape_2x2x2:Shuffle:Reshape=[2, 2, 2]`
- `Reduce=sum_last_dim:Reduce:Op=Sum:Axes=4:Keep=True`
- `Output=reduced_output:Output:Float:[2, 2, 1]`
- `ShapeOpsOutputMatch=True`

## NetworkConcatSliceSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddSlice`, `TensorRtNetworkDefinition.AddConcatenation`, and `TensorRtBuilderConfig.SetAverageTimingIterations`
- validate slice start/size/stride metadata and concatenation axis metadata
- build a serialized engine, deserialize it, bind input/output tensors, enqueue, and verify a small slice-plus-concat result

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkConcatSliceSmokeRunner\bin\Debug\net8.0\NetworkConcatSliceSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkConcatSliceSmokeRunner\bin\Debug\net8.0\NetworkConcatSliceSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Config AvgTiming=1`
- `ConcatSliceMetadata Slice=slice_first_two_columns:SliceTrt...:Start=[0, 0]:Size=[2, 2]:Stride=[1, 1]`
- `Concat=concat_axis_1:Concatenation:Axis=1`
- `Output=concat_output:Output:Float:[2, 4]`
- `ConcatSliceOutputMatch=True`

## NetworkSoftmaxTopKSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddSoftMax`, `AddTopK`, `AddUnary`, and `AddGather`
- validate `TensorRtBuilderConfig.GetTacticSources` / `SetTacticSources`
- validate softmax axes, top-k operation/k/axes, unary operation, and gather axis metadata
- build a serialized engine, deserialize it, bind four output tensors, enqueue, and verify all outputs against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkSoftmaxTopKSmokeRunner\bin\Debug\net8.0\NetworkSoftmaxTopKSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkSoftmaxTopKSmokeRunner\bin\Debug\net8.0\NetworkSoftmaxTopKSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `LayerMetadata SoftMax=softmax_axis_1:SoftMax:Axes=2`
- `TopK=topk_max_axis_1:TopK:Op=Max:K=1:Axes=2`
- `Unary=unary_abs:Unary:Op=Abs`
- `Gather=gather_columns:Gather:Axis=1`
- `SoftMaxTopKOutputMatch=True UnaryOutputMatch=True GatherOutputMatch=True`

## NetworkActivationPoolingResizeSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddActivation`, `AddPooling`, and `AddResize`
- validate activation type, pooling window/stride/padding, and resize mode/output-dimensions metadata
- build a serialized engine, deserialize it, bind tensors, enqueue, and verify output against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkActivationPoolingResizeSmokeRunner\bin\Debug\net8.0\NetworkActivationPoolingResizeSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkActivationPoolingResizeSmokeRunner\bin\Debug\net8.0\NetworkActivationPoolingResizeSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `LayerMetadata Activation=relu:Activation:Type=Relu`
- `Pooling=max_pool_2x2:Pooling:Type=Max:Window=[2, 2]:Stride=[2, 2]:Padding=[0, 0]`
- `Resize=resize_identity_shape:ResizeTrt...:Mode=Nearest:OutputDims=[1, 1, 2, 2]`
- `ActivationPoolingResizeOutputMatch=True`

## NetworkMatrixFillSelectSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddMatrixMultiply`, `AddFill`, and `AddSelect`
- validate matrix multiply operation metadata, fill dimensions/operation/alpha/beta metadata, select layer metadata, and engine tensor metadata
- build a serialized engine, deserialize it, bind tensors, enqueue, and verify output against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkMatrixFillSelectSmokeRunner\bin\Debug\net8.0\NetworkMatrixFillSelectSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkMatrixFillSelectSmokeRunner\bin\Debug\net8.0\NetworkMatrixFillSelectSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Matrix=matrix_identity:MatrixMultiplyTrt...:Op0=None:Op1=None`
- `Fill=fill_linspace:FillTrt...:Dims=[2, 2]:Op=Linspace:Alpha=10:Beta=0`
- `Select=select_mix:SelectTrt...`
- `EngineMetadata DeviceMemory=... Profiles=...`
- `EngineTensor Index=... Type=Float Shape=[2, 2] Mode=Output`
- `MatrixFillSelectOutputMatch=True`

## NetworkConvolutionScaleSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddConvolution`, `AddScale`, and `AddPadding`
- validate convolution output-map/group/stride/padding/dilation metadata, scale mode/channel-axis metadata, and padding metadata
- validate execution-context deployment helpers such as persistent cache limit and input-consumed event where the selected TensorRT line supports the operation
- build a serialized engine, deserialize it, bind tensors, enqueue, and verify output against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkConvolutionScaleSmokeRunner\bin\Debug\net8.0\NetworkConvolutionScaleSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkConvolutionScaleSmokeRunner\bin\Debug\net8.0\NetworkConvolutionScaleSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Convolution=...`
- `Scale=...`
- `Padding=...`
- `ContextPersistentCache=...`
- `ConvolutionScaleOutputMatch=True`

If Windows application control blocks a freshly rebuilt `jyppxtrtbridge.dll` with `0x800711C7`, record this runner as blocked rather than passed.

## NetworkDeconvolutionSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddDeconvolution`
- validate deconvolution metadata helpers for output maps, groups, kernel size, stride, dilation, padding, and padding mode
- validate a second complex deconvolution metadata probe with 2x2 kernel, stride, and pre-padding
- validate a small deployment-style deconvolution identity network
- build a serialized engine, deserialize it, bind tensors, enqueue, and verify output against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkDeconvolutionSmokeRunner\bin\Debug\net8.0\NetworkDeconvolutionSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkDeconvolutionSmokeRunner\bin\Debug\net8.0\NetworkDeconvolutionSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `Deconvolution=...`
- `DeconvolutionMetadata OutputMaps=1 Groups=1 Kernel=[1, 1] Stride=[1, 1] Dilation=[1, 1]`
- `DeconvolutionComplexMetadata OutputMaps=1 Groups=1 Kernel=[2, 2] Stride=[2, 2] Dilation=[1, 1]`
- `DeconvolutionIdentityOutputMatch=True`
- `Layers=2`

## NetworkQuantizeDequantizeSmokeRunner

Purpose:

- build a TensorRT network directly from C# without the ONNX parser
- validate `TensorRtNetworkDefinition.AddQuantize` and `TensorRtNetworkDefinition.AddDequantize`
- validate Q/DQ layer type mapping and axis metadata helpers for TensorRT 8 and TensorRT 10
- build a serialized engine with the Int8 builder flag, deserialize it, bind tensors, enqueue, and verify output against CPU reference values

Run against TensorRT 10:

```powershell
dotnet .\smoke\NetworkQuantizeDequantizeSmokeRunner\bin\Release\net8.0\NetworkQuantizeDequantizeSmokeRunner.dll --tensor-rt-line 10
```

Run against TensorRT 8:

```powershell
dotnet .\smoke\NetworkQuantizeDequantizeSmokeRunner\bin\Release\net8.0\NetworkQuantizeDequantizeSmokeRunner.dll --tensor-rt-line 8
```

Expected signals:

- `QdqMetadata QuantizeType=... QuantizeAxis=-1 DequantizeType=... DequantizeAxis=-1`
- `QdqOutputMatch=True`
- `Layers=3`

If Debug sample assemblies are blocked by Windows application control with `0x800711C7`, run the Release runner and record the Debug path as environment-blocked rather than failed.

