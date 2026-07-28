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
        IReadOnlyList<TensorRtExecutionContextRuntimeDiagnosticSnapshot> runtimeDiagnostics,
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
        RuntimeDiagnostics = runtimeDiagnostics;
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
    /// Gets pointer-free runtime diagnostic snapshots collected for output tensors.
    /// 获取为 output tensor 采集的 pointer-free 运行时诊断快照。
    /// </summary>
    public IReadOnlyList<TensorRtExecutionContextRuntimeDiagnosticSnapshot> RuntimeDiagnostics { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Converts this copied execution-context deployment snapshot into a compact pointer-free summary.
    /// 将当前已复制 execution-context deployment 快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native context pointers,
    /// or promote local deployment diagnostics to runtime proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露原生 context 指针，也不会将本地部署诊断晋级为 runtime proof。
    /// </remarks>
    public TensorRtExecutionContextDeploymentSummary ToSummary()
    {
        int runtimeDiagnosticsWithCallbacks = 0;
        int runtimeDiagnosticsWithOutputAllocator = 0;
        for (int index = 0; index < RuntimeDiagnostics.Count; index++)
        {
            TensorRtExecutionContextRuntimeDiagnosticSummary summary = RuntimeDiagnostics[index].ToSummary();
            if (summary.CallbackStateLastStatus != 0)
            {
                runtimeDiagnosticsWithCallbacks++;
            }

            if (summary.HasOutputAllocator)
            {
                runtimeDiagnosticsWithOutputAllocator++;
            }
        }

        return new TensorRtExecutionContextDeploymentSummary(
            ContextName,
            EngineName,
            EngineIOTensorCount,
            EngineLayerCount,
            EngineOptimizationProfileCount,
            OptimizationProfileIndex,
            DebugSync,
            AllInputDimensionsSpecified,
            AllInputShapesSpecified,
            DeviceMemorySizeInBytes,
            PersistentCacheLimitInBytes,
            EnqueueEmitsProfile,
            IsInputConsumedEventSet,
            HasTemporaryStorageAllocator,
            HasDebugListener,
            HasProfiler,
            HasRuntimeConfig,
            RuntimeConfigAllocationStrategy,
            NvtxVerbosity,
            UnfusedTensorsDebugState,
            TensorStates.Count,
            RuntimeDiagnostics.Count,
            runtimeDiagnosticsWithCallbacks,
            runtimeDiagnosticsWithOutputAllocator,
            Diagnostics.Count);
    }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{ContextName} engine={EngineName} profile={OptimizationProfileIndex} tensors={TensorStates.Count} runtimeDiagnostics={RuntimeDiagnostics.Count} memory={DeviceMemorySizeInBytes} runtimeConfig={HasRuntimeConfig} diagnostics={Diagnostics.Count}";
    }
}

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
