using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Compact pointer-free summary of TensorRT execution-context deployment state.
/// TensorRT execution-context deployment 状态的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtExecutionContextDeploymentSummary
{
    internal TensorRtExecutionContextDeploymentSummary(
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
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool hasProfiler,
        bool hasRuntimeConfig,
        TensorRtExecutionContextAllocationStrategy? runtimeConfigAllocationStrategy,
        TensorRtProfilingVerbosity nvtxVerbosity,
        bool unfusedTensorsDebugState,
        int copiedTensorStateCount,
        int copiedRuntimeDiagnosticCount,
        int runtimeDiagnosticsWithCallbackStateCount,
        int runtimeDiagnosticsWithOutputAllocatorCount,
        int diagnosticCount)
    {
        ContextName = contextName ?? string.Empty;
        EngineName = engineName ?? string.Empty;
        EngineIOTensorCount = engineIOTensorCount < 0 ? 0 : engineIOTensorCount;
        EngineLayerCount = engineLayerCount < 0 ? 0 : engineLayerCount;
        EngineOptimizationProfileCount = engineOptimizationProfileCount < 0 ? 0 : engineOptimizationProfileCount;
        OptimizationProfileIndex = optimizationProfileIndex;
        DebugSync = debugSync;
        AllInputDimensionsSpecified = allInputDimensionsSpecified;
        AllInputShapesSpecified = allInputShapesSpecified;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        PersistentCacheLimitInBytes = persistentCacheLimitInBytes;
        EnqueueEmitsProfile = enqueueEmitsProfile;
        IsInputConsumedEventSet = isInputConsumedEventSet;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        HasProfiler = hasProfiler;
        HasRuntimeConfig = hasRuntimeConfig;
        RuntimeConfigAllocationStrategy = runtimeConfigAllocationStrategy;
        NvtxVerbosity = nvtxVerbosity;
        UnfusedTensorsDebugState = unfusedTensorsDebugState;
        CopiedTensorStateCount = copiedTensorStateCount < 0 ? 0 : copiedTensorStateCount;
        CopiedRuntimeDiagnosticCount = copiedRuntimeDiagnosticCount < 0 ? 0 : copiedRuntimeDiagnosticCount;
        RuntimeDiagnosticsWithCallbackStateCount = runtimeDiagnosticsWithCallbackStateCount < 0 ? 0 : runtimeDiagnosticsWithCallbackStateCount;
        RuntimeDiagnosticsWithOutputAllocatorCount = runtimeDiagnosticsWithOutputAllocatorCount < 0 ? 0 : runtimeDiagnosticsWithOutputAllocatorCount;
        DiagnosticCount = diagnosticCount < 0 ? 0 : diagnosticCount;
    }

    /// <summary>Gets the copied context name. 获取已复制 context 名称。</summary>
    public string ContextName { get; }

    /// <summary>Gets the copied engine name. 获取已复制 engine 名称。</summary>
    public string EngineName { get; }

    /// <summary>Gets copied engine I/O tensor count. 获取已复制 engine I/O tensor 数量。</summary>
    public int EngineIOTensorCount { get; }

    /// <summary>Gets copied engine layer count. 获取已复制 engine layer 数量。</summary>
    public int EngineLayerCount { get; }

    /// <summary>Gets copied engine optimization profile count. 获取已复制 engine optimization profile 数量。</summary>
    public int EngineOptimizationProfileCount { get; }

    /// <summary>Gets copied active optimization profile index. 获取已复制 active optimization profile 索引。</summary>
    public int OptimizationProfileIndex { get; }

    /// <summary>Gets copied debug-sync state. 获取已复制 debug-sync 状态。</summary>
    public bool DebugSync { get; }

    /// <summary>Gets copied all-input-dimensions-specified state. 获取已复制 all-input-dimensions-specified 状态。</summary>
    public bool AllInputDimensionsSpecified { get; }

    /// <summary>Gets copied all-input-shapes-specified state. 获取已复制 all-input-shapes-specified 状态。</summary>
    public bool AllInputShapesSpecified { get; }

    /// <summary>Gets copied context device-memory requirement. 获取已复制 context device-memory 需求。</summary>
    public ulong DeviceMemorySizeInBytes { get; }

    /// <summary>Gets copied persistent-cache limit. 获取已复制 persistent-cache limit。</summary>
    public ulong PersistentCacheLimitInBytes { get; }

    /// <summary>Gets copied enqueue-emits-profile state. 获取已复制 enqueue-emits-profile 状态。</summary>
    public bool EnqueueEmitsProfile { get; }

    /// <summary>Gets copied input-consumed-event presence. 获取已复制 input-consumed-event 是否存在。</summary>
    public bool IsInputConsumedEventSet { get; }

    /// <summary>Gets copied temporary-storage allocator presence. 获取已复制 temporary-storage allocator 是否存在。</summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>Gets copied debug-listener presence. 获取已复制 debug-listener 是否存在。</summary>
    public bool HasDebugListener { get; }

    /// <summary>Gets copied profiler presence. 获取已复制 profiler 是否存在。</summary>
    public bool HasProfiler { get; }

    /// <summary>Gets copied runtime-config presence. 获取已复制 runtime-config 是否存在。</summary>
    public bool HasRuntimeConfig { get; }

    /// <summary>Gets copied runtime-config allocation strategy when available. 获取可用时已复制 runtime-config allocation strategy。</summary>
    public TensorRtExecutionContextAllocationStrategy? RuntimeConfigAllocationStrategy { get; }

    /// <summary>Gets copied NVTX verbosity. 获取已复制 NVTX verbosity。</summary>
    public TensorRtProfilingVerbosity NvtxVerbosity { get; }

    /// <summary>Gets copied unfused tensor debug state. 获取已复制 unfused tensor debug state。</summary>
    public bool UnfusedTensorsDebugState { get; }

    /// <summary>Gets copied tensor-state count. 获取已复制 tensor-state 数量。</summary>
    public int CopiedTensorStateCount { get; }

    /// <summary>Gets copied runtime diagnostic count. 获取已复制 runtime diagnostic 数量。</summary>
    public int CopiedRuntimeDiagnosticCount { get; }

    /// <summary>Gets copied runtime diagnostic count with callback status. 获取包含 callback status 的 runtime diagnostic 数量。</summary>
    public int RuntimeDiagnosticsWithCallbackStateCount { get; }

    /// <summary>Gets copied runtime diagnostic count with output allocator. 获取包含 output allocator 的 runtime diagnostic 数量。</summary>
    public int RuntimeDiagnosticsWithOutputAllocatorCount { get; }

    /// <summary>Gets diagnostic count collected while building the snapshot. 获取构建快照时收集的诊断数量。</summary>
    public int DiagnosticCount { get; }

    /// <summary>Gets whether copied tensor states cover the engine I/O tensor count. 获取已复制 tensor state 是否覆盖 engine I/O tensor 数量。</summary>
    public bool CopiedTensorStatesMatchEngineIOTensorCount => CopiedTensorStateCount == EngineIOTensorCount;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether deferred records can be deleted because of this summary. 获取是否可因该摘要删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for smoke output and logs. 将该摘要格式化为 smoke 输出和日志。</summary>
    public override string ToString()
    {
        return $"{ContextName} engine={EngineName} profile={OptimizationProfileIndex} tensors={CopiedTensorStateCount}/{EngineIOTensorCount} runtimeDiagnostics={CopiedRuntimeDiagnosticCount} callbacks={RuntimeDiagnosticsWithCallbackStateCount} outputAllocators={RuntimeDiagnosticsWithOutputAllocatorCount} memory={DeviceMemorySizeInBytes} runtimeConfig={HasRuntimeConfig} diagnostics={DiagnosticCount} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
