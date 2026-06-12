using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures TensorRT execution-context state relevant to dynamic shapes, tensor addresses, and enqueue readiness.
/// 捕获与动态 shape、tensor 地址绑定和 enqueue 就绪状态相关的 TensorRT execution-context 状态。
/// </summary>
public sealed class TensorRtExecutionContextDeploymentSnapshot
{
    internal TensorRtExecutionContextDeploymentSnapshot(
        string contextName,
        string engineName,
        int engineIOTensorCount,
        int engineLayerCount,
        int engineOptimizationProfileCount,
        int optimizationProfileIndex,
        bool debugSync,
        bool allInputDimensionsSpecified,
        bool allInputShapesSpecified,
        ulong deviceMemorySizeInBytes,
        ulong persistentCacheLimitInBytes,
        bool enqueueEmitsProfile,
        bool isInputConsumedEventSet,
        ulong inputConsumedEventAddressValue,
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool hasProfiler,
        bool hasRuntimeConfig,
        TensorRtExecutionContextAllocationStrategy? runtimeConfigAllocationStrategy,
        TensorRtProfilingVerbosity nvtxVerbosity,
        bool unfusedTensorsDebugState,
        IReadOnlyList<TensorRtTensorBindingState> tensorStates,
        IReadOnlyList<string> diagnostics)
    {
        ContextName = contextName;
        EngineName = engineName;
        EngineIOTensorCount = engineIOTensorCount;
        EngineLayerCount = engineLayerCount;
        EngineOptimizationProfileCount = engineOptimizationProfileCount;
        OptimizationProfileIndex = optimizationProfileIndex;
        DebugSync = debugSync;
        AllInputDimensionsSpecified = allInputDimensionsSpecified;
        AllInputShapesSpecified = allInputShapesSpecified;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        PersistentCacheLimitInBytes = persistentCacheLimitInBytes;
        EnqueueEmitsProfile = enqueueEmitsProfile;
        IsInputConsumedEventSet = isInputConsumedEventSet;
        InputConsumedEventAddressValue = inputConsumedEventAddressValue;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        HasProfiler = hasProfiler;
        HasRuntimeConfig = hasRuntimeConfig;
        RuntimeConfigAllocationStrategy = runtimeConfigAllocationStrategy;
        NvtxVerbosity = nvtxVerbosity;
        UnfusedTensorsDebugState = unfusedTensorsDebugState;
        TensorStates = tensorStates;
        Diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets the execution-context name.
    /// 获取 execution context 名称。
    /// </summary>
    public string ContextName { get; }

    /// <summary>
    /// Gets the engine name reached through the execution context.
    /// 获取通过 execution context 反查到的 engine 名称。
    /// </summary>
    public string EngineName { get; }

    /// <summary>
    /// Gets the engine I/O tensor count reached through the context.
    /// 获取通过 context 反查到的 engine I/O tensor 数量。
    /// </summary>
    public int EngineIOTensorCount { get; }

    /// <summary>
    /// Gets the engine layer count reached through the context.
    /// 获取通过 context 反查到的 engine layer 数量。
    /// </summary>
    public int EngineLayerCount { get; }

    /// <summary>
    /// Gets the engine optimization-profile count reached through the context.
    /// 获取通过 context 反查到的 engine optimization profile 数量。
    /// </summary>
    public int EngineOptimizationProfileCount { get; }

    /// <summary>
    /// Gets the active optimization profile index.
    /// 获取当前 active optimization profile 索引。
    /// </summary>
    public int OptimizationProfileIndex { get; }

    /// <summary>
    /// Gets whether TensorRT debug synchronization is enabled.
    /// 获取 TensorRT debug synchronization 是否启用。
    /// </summary>
    public bool DebugSync { get; }

    /// <summary>
    /// Gets whether TensorRT reports all input dimensions specified.
    /// 获取 TensorRT 是否报告所有输入维度已指定。
    /// </summary>
    public bool AllInputDimensionsSpecified { get; }

    /// <summary>
    /// Gets whether TensorRT reports all input shape tensors specified.
    /// 获取 TensorRT 是否报告所有输入 shape tensor 已指定。
    /// </summary>
    public bool AllInputShapesSpecified { get; }

    /// <summary>
    /// Gets execution-context device-memory requirement.
    /// 获取 execution context 设备内存需求。
    /// </summary>
    public ulong DeviceMemorySizeInBytes { get; }

    /// <summary>
    /// Gets persistent-cache limit.
    /// 获取 persistent cache 限制。
    /// </summary>
    public ulong PersistentCacheLimitInBytes { get; }

    /// <summary>
    /// Gets whether enqueue emits profiling data.
    /// 获取 enqueue 是否发出 profiling 数据。
    /// </summary>
    public bool EnqueueEmitsProfile { get; }

    /// <summary>
    /// Gets whether an input-consumed CUDA event is set.
    /// 获取是否设置了 input-consumed CUDA event。
    /// </summary>
    public bool IsInputConsumedEventSet { get; }

    /// <summary>
    /// Gets the input-consumed CUDA event native address as a diagnostic integer value.
    /// 获取 input-consumed CUDA event 原生地址的诊断整数值。
    /// </summary>
    public ulong InputConsumedEventAddressValue { get; }

    /// <summary>
    /// Gets whether a temporary-storage allocator is attached.
    /// 获取是否已绑定 temporary-storage allocator。
    /// </summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>
    /// Gets whether a debug listener is attached.
    /// 获取是否已绑定 debug listener。
    /// </summary>
    public bool HasDebugListener { get; }

    /// <summary>
    /// Gets whether a profiler is attached.
    /// 获取是否已绑定 profiler。
    /// </summary>
    public bool HasProfiler { get; }

    /// <summary>
    /// Gets whether TensorRT exposes a runtime config for this context.
    /// 获取 TensorRT 是否为该 context 暴露 runtime config。
    /// </summary>
    public bool HasRuntimeConfig { get; }

    /// <summary>
    /// Gets runtime-config allocation strategy when available.
    /// 获取可用时的 runtime config allocation strategy。
    /// </summary>
    public TensorRtExecutionContextAllocationStrategy? RuntimeConfigAllocationStrategy { get; }

    /// <summary>
    /// Gets current NVTX verbosity.
    /// 获取当前 NVTX verbosity。
    /// </summary>
    public TensorRtProfilingVerbosity NvtxVerbosity { get; }

    /// <summary>
    /// Gets TensorRT 11 unfused-tensor debug state.
    /// 获取 TensorRT 11 unfused tensor debug state。
    /// </summary>
    public bool UnfusedTensorsDebugState { get; }

    /// <summary>
    /// Gets per-tensor context binding states.
    /// 获取逐 tensor 的 context 绑定状态。
    /// </summary>
    public IReadOnlyList<TensorRtTensorBindingState> TensorStates { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{ContextName} engine={EngineName} profile={OptimizationProfileIndex} tensors={TensorStates.Count} memory={DeviceMemorySizeInBytes} runtimeConfig={HasRuntimeConfig} diagnostics={Diagnostics.Count}";
    }
}
