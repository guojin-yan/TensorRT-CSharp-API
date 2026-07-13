using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets the native address value of the CUDA event TensorRT uses to signal input-consumption completion.
    /// 获取 TensorRT 用于通知输入消费完成的 CUDA event 原生地址诊断值。
    /// </summary>
    /// <remarks>
    /// The value is diagnostic only. It is intentionally exposed as an integer instead of a user-owned pointer.
    /// 该值仅用于诊断；它有意以整数形式暴露，而不是用户可拥有或解引用的指针。
    /// </remarks>
    public ulong InputConsumedEventAddressValue => NativeBridgeApi.GetExecutionContextInputConsumedEventAddressValue(Line, _handle);

    /// <summary>
    /// Gets the allocation strategy from the runtime config attached to this TensorRT 11 execution context.
    /// 获取附加到当前 TensorRT 11 execution context 的 runtime config allocation strategy。
    /// </summary>
    /// <exception cref="TensorRtException">
    /// Thrown when TensorRT does not expose a runtime config for this context.
    /// 当 TensorRT 未为该 context 暴露 runtime config 时抛出。
    /// </exception>
    public TensorRtExecutionContextAllocationStrategy RuntimeConfigAllocationStrategy => NativeBridgeApi.GetExecutionContextRuntimeConfigAllocationStrategy(Line, _handle);

    /// <summary>
    /// Gets the engine name reached through this execution context.
    /// 获取通过当前 execution context 反查到的 engine 名称。
    /// </summary>
    public string EngineName => NativeBridgeApi.GetExecutionContextEngineName(Line, _handle);

    /// <summary>
    /// Gets the engine I/O tensor count reached through this execution context.
    /// 获取通过当前 execution context 反查到的 engine I/O tensor 数量。
    /// </summary>
    public int EngineIOTensorCount => NativeBridgeApi.GetExecutionContextEngineIOTensorCount(Line, _handle);

    /// <summary>
    /// Gets the engine layer count reached through this execution context.
    /// 获取通过当前 execution context 反查到的 engine layer 数量。
    /// </summary>
    public int EngineLayerCount => NativeBridgeApi.GetExecutionContextEngineLayerCount(Line, _handle);

    /// <summary>
    /// Gets the engine optimization profile count reached through this execution context.
    /// 获取通过当前 execution context 反查到的 engine optimization profile 数量。
    /// </summary>
    public int EngineOptimizationProfileCount => NativeBridgeApi.GetExecutionContextEngineOptimizationProfileCount(Line, _handle);

    /// <summary>
    /// Builds a TensorRT 11 deployment snapshot for this execution context.
    /// 为当前 execution context 构建 TensorRT 11 部署快照。
    /// </summary>
    /// <param name="engine">The engine used to enumerate tensor metadata. 用于枚举 tensor 元数据的 engine。</param>
    /// <returns>A deployment snapshot with context readiness, address state, and runtime metadata. 包含 context 就绪状态、地址状态和运行时元数据的部署快照。</returns>
    public TensorRtExecutionContextDeploymentSnapshot GetDeploymentSnapshot(TensorRtEngine engine)
    {
        if (engine == null)
        {
            throw new ArgumentNullException(nameof(engine));
        }

        if (engine.Line != Line)
        {
            throw new ArgumentException("Execution context and engine must belong to the same TensorRT API line.", nameof(engine));
        }

        List<string> diagnostics = new List<string>();
        List<TensorRtTensorBindingState> states = new List<TensorRtTensorBindingState>();
        List<TensorRtExecutionContextRuntimeDiagnosticSnapshot> runtimeDiagnostics = new List<TensorRtExecutionContextRuntimeDiagnosticSnapshot>();
        foreach (TensorRtTensorInfo tensor in engine.GetIOTensors())
        {
            TensorRtDims? contextShape = TryCollect($"ContextShape[{tensor.Name}]", diagnostics, () => GetTensorShape(tensor.Name), (TensorRtDims?)null);
            TensorRtDims? contextStrides = TryCollect($"ContextStrides[{tensor.Name}]", diagnostics, () => GetTensorStrides(tensor.Name), (TensorRtDims?)null);
            bool isBound = TryCollect($"AddressBound[{tensor.Name}]", diagnostics, () => IsTensorAddressBound(tensor.Name), false);
            long? maxOutputSize = tensor.IOMode == TensorRtIOMode.Output
                ? TryCollect($"MaxOutputSize[{tensor.Name}]", diagnostics, () => (long?)GetMaxOutputSize(tensor.Name), null)
                : null;

            states.Add(new TensorRtTensorBindingState(
                tensor.Index,
                tensor.Name,
                tensor.DataType,
                tensor.IOMode,
                tensor.Shape,
                contextShape,
                contextStrides,
                isBound,
                maxOutputSize,
                null));

            if (tensor.IOMode == TensorRtIOMode.Output)
            {
                runtimeDiagnostics.Add(TryCollect(
                    $"RuntimeDiagnosticSnapshot[{tensor.Name}]",
                    diagnostics,
                    () => GetRuntimeDiagnosticSnapshot(tensor.Name),
                    CreateUnavailableRuntimeDiagnosticSnapshot(tensor.Name)));
            }
        }

        bool hasRuntimeConfig = TryCollect("HasRuntimeConfig", diagnostics, () => HasRuntimeConfig, false);
        TensorRtExecutionContextAllocationStrategy? allocationStrategy = hasRuntimeConfig
            ? TryCollect("RuntimeConfigAllocationStrategy", diagnostics, () => (TensorRtExecutionContextAllocationStrategy?)RuntimeConfigAllocationStrategy, null)
            : null;

        return new TensorRtExecutionContextDeploymentSnapshot(
            TryCollect("Name", diagnostics, () => Name, string.Empty),
            TryCollect("EngineName", diagnostics, () => EngineName, string.Empty),
            TryCollect("EngineIOTensorCount", diagnostics, () => EngineIOTensorCount, 0),
            TryCollect("EngineLayerCount", diagnostics, () => EngineLayerCount, 0),
            TryCollect("EngineOptimizationProfileCount", diagnostics, () => EngineOptimizationProfileCount, 0),
            TryCollect("OptimizationProfileIndex", diagnostics, () => OptimizationProfileIndex, -1),
            TryCollect("DebugSync", diagnostics, () => DebugSync, false),
            TryCollect("AllInputDimensionsSpecified", diagnostics, () => AllInputDimensionsSpecified, false),
            TryCollect("AllInputShapesSpecified", diagnostics, () => AllInputShapesSpecified, false),
            TryCollect("DeviceMemorySizeInBytes", diagnostics, () => DeviceMemorySizeInBytes, 0UL),
            TryCollect("PersistentCacheLimitInBytes", diagnostics, () => PersistentCacheLimitInBytes, 0UL),
            TryCollect("EnqueueEmitsProfile", diagnostics, () => EnqueueEmitsProfile, false),
            TryCollect("IsInputConsumedEventSet", diagnostics, () => IsInputConsumedEventSet, false),
            TryCollect("InputConsumedEventAddressValue", diagnostics, () => InputConsumedEventAddressValue, 0UL),
            TryCollect("HasTemporaryStorageAllocator", diagnostics, () => HasTemporaryStorageAllocator, false),
            TryCollect("HasDebugListener", diagnostics, () => HasDebugListener, false),
            TryCollect("HasProfiler", diagnostics, () => HasProfiler, false),
            hasRuntimeConfig,
            allocationStrategy,
            TryCollect("NvtxVerbosity", diagnostics, GetNvtxVerbosity, TensorRtProfilingVerbosity.LayerNamesOnly),
            TryCollect("UnfusedTensorsDebugState", diagnostics, GetUnfusedTensorsDebugState, false),
            states,
            runtimeDiagnostics,
            diagnostics);
    }

    private TensorRtExecutionContextRuntimeDiagnosticSnapshot CreateUnavailableRuntimeDiagnosticSnapshot(string outputTensorName)
    {
        return new TensorRtExecutionContextRuntimeDiagnosticSnapshot(
            Line,
            outputTensorName,
            hasErrorRecorder: false,
            isInputConsumedEventSet: false,
            inputConsumedEventAddressValue: 0UL,
            hasOutputAllocator: false,
            isOutputTensorAddressSet: false,
            outputTensorAddressValue: 0UL,
            hasTemporaryStorageAllocator: false,
            hasDebugListener: false,
            hasManagedProfiler: false,
            hasNativeProfiler: false,
            hasRuntimeConfig: false,
            nvtxVerbosity: TensorRtProfilingVerbosity.LayerNamesOnly,
            unfusedTensorsDebugState: false,
            callbackState: CreateUnavailableCallbackStateSnapshot(outputTensorName),
            diagnostics: new[] { "Runtime diagnostic snapshot unavailable." });
    }

    private static T TryCollect<T>(string fieldName, List<string> diagnostics, Func<T> getter, T fallback)
    {
        try
        {
            return getter();
        }
        catch (Exception ex)
        {
            diagnostics.Add($"{fieldName}: {ex.Message}");
            return fallback;
        }
    }
}
