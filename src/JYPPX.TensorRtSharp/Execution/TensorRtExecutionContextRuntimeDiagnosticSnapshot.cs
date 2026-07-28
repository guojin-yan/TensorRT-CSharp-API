using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures pointer-free execution-context runtime diagnostics for one output tensor.
/// 捕获单个输出 tensor 对应的 execution context 运行时诊断快照；不暴露任何 borrowed pointer。
/// </summary>
public sealed class TensorRtExecutionContextRuntimeDiagnosticSnapshot
{
    internal TensorRtExecutionContextRuntimeDiagnosticSnapshot(
        TensorRtApiLine line,
        string outputTensorName,
        bool hasErrorRecorder,
        bool isInputConsumedEventSet,
        ulong inputConsumedEventAddressValue,
        bool hasOutputAllocator,
        bool isOutputTensorAddressSet,
        ulong outputTensorAddressValue,
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool hasManagedProfiler,
        bool hasNativeProfiler,
        bool hasRuntimeConfig,
        TensorRtProfilingVerbosity nvtxVerbosity,
        bool unfusedTensorsDebugState,
        TensorRtExecutionContextCallbackStateSnapshot callbackState,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        OutputTensorName = outputTensorName;
        HasErrorRecorder = hasErrorRecorder;
        IsInputConsumedEventSet = isInputConsumedEventSet;
        InputConsumedEventAddressValue = inputConsumedEventAddressValue;
        HasOutputAllocator = hasOutputAllocator;
        IsOutputTensorAddressSet = isOutputTensorAddressSet;
        OutputTensorAddressValue = outputTensorAddressValue;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        HasManagedProfiler = hasManagedProfiler;
        HasNativeProfiler = hasNativeProfiler;
        HasRuntimeConfig = hasRuntimeConfig;
        NvtxVerbosity = nvtxVerbosity;
        UnfusedTensorsDebugState = unfusedTensorsDebugState;
        CallbackState = callbackState;
        Diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets the TensorRT API line used for the query.
    /// 获取执行查询时使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the output tensor name used for output-address and callback-state diagnostics.
    /// 获取用于输出地址与回调状态诊断的 output tensor 名称。
    /// </summary>
    public string OutputTensorName { get; }

    /// <summary>
    /// Gets whether TensorRT reports an error recorder attached to the execution context.
    /// 获取 TensorRT 是否报告 execution context 已绑定 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets whether TensorRT reports an input-consumed event.
    /// 获取 TensorRT 是否报告已设置 input-consumed event。
    /// </summary>
    public bool IsInputConsumedEventSet { get; }

    /// <summary>
    /// Gets the input-consumed event native address as a diagnostic integer value.
    /// 获取 input-consumed event 原生地址的诊断整数值。
    /// </summary>
    public ulong InputConsumedEventAddressValue { get; }

    /// <summary>
    /// Gets whether TensorRT reports an output allocator for <see cref="OutputTensorName"/>.
    /// 获取 TensorRT 是否为 <see cref="OutputTensorName"/> 报告 output allocator。
    /// </summary>
    public bool HasOutputAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reports an explicit output tensor address.
    /// 获取 TensorRT 是否报告该 output tensor 已绑定显式地址。
    /// </summary>
    public bool IsOutputTensorAddressSet { get; }

    /// <summary>
    /// Gets the output tensor native address as a diagnostic integer value.
    /// 获取 output tensor 原生地址的诊断整数值。
    /// </summary>
    public ulong OutputTensorAddressValue { get; }

    /// <summary>
    /// Gets whether TensorRT reports a temporary-storage allocator attached to the context.
    /// 获取 TensorRT 是否报告 context 已绑定 temporary-storage allocator。
    /// </summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reports a debug listener attached to the context.
    /// 获取 TensorRT 是否报告 context 已绑定 debug listener。
    /// </summary>
    public bool HasDebugListener { get; }

    /// <summary>
    /// Gets whether the managed wrapper currently holds a profiler borrow.
    /// 获取托管 wrapper 当前是否持有 profiler 借用。
    /// </summary>
    public bool HasManagedProfiler { get; }

    /// <summary>
    /// Gets whether TensorRT reports a non-null native profiler pointer.
    /// 获取 TensorRT 是否报告非空 native profiler 指针；该指针不会被暴露。
    /// </summary>
    public bool HasNativeProfiler { get; }

    /// <summary>
    /// Gets whether TensorRT exposes a runtime config object for this context.
    /// 获取 TensorRT 是否为该 context 暴露 runtime config 对象。
    /// </summary>
    public bool HasRuntimeConfig { get; }

    /// <summary>
    /// Gets the current NVTX verbosity when readable.
    /// 获取可读时的当前 NVTX verbosity。
    /// </summary>
    public TensorRtProfilingVerbosity NvtxVerbosity { get; }

    /// <summary>
    /// Gets the TensorRT 11 unfused-tensor debug state when readable.
    /// 获取可读时的 TensorRT 11 unfused tensor debug state。
    /// </summary>
    public bool UnfusedTensorsDebugState { get; }

    /// <summary>
    /// Gets the copied callback boundary state for <see cref="OutputTensorName"/>.
    /// 获取 <see cref="OutputTensorName"/> 对应的回调边界复制快照。
    /// </summary>
    public TensorRtExecutionContextCallbackStateSnapshot CallbackState { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact pointer-free summary from this copied execution-context runtime diagnostic snapshot.
    /// 从当前已复制的 execution-context runtime 诊断快照创建简短无指针摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, invoke callbacks, or promote runtime proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露 native 指针、调用回调，也不会提升 runtime proof。
    /// </remarks>
    /// <returns>A pointer-free execution-context runtime diagnostic summary. 无指针 execution-context runtime 诊断摘要。</returns>
    public TensorRtExecutionContextRuntimeDiagnosticSummary ToSummary()
    {
        return new TensorRtExecutionContextRuntimeDiagnosticSummary(
            Line,
            !string.IsNullOrWhiteSpace(OutputTensorName),
            HasErrorRecorder,
            IsInputConsumedEventSet,
            InputConsumedEventAddressValue != 0UL,
            HasOutputAllocator,
            IsOutputTensorAddressSet,
            OutputTensorAddressValue != 0UL,
            HasTemporaryStorageAllocator,
            HasDebugListener,
            HasManagedProfiler,
            HasNativeProfiler,
            HasRuntimeConfig,
            NvtxVerbosity,
            UnfusedTensorsDebugState,
            CallbackState.HasOutputAllocator,
            CallbackState.HasTemporaryStorageAllocator,
            CallbackState.HasDebugListener,
            CallbackState.OutputAllocatorInterfaceInfoAvailable,
            CallbackState.TemporaryStorageAllocatorInterfaceInfoAvailable,
            CallbackState.DebugListenerInterfaceInfoAvailable,
            CallbackState.LastStatus,
            !string.IsNullOrWhiteSpace(CallbackState.LastOperation),
            Diagnostics.Count);
    }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{Line}:{OutputTensorName}:errorRecorder={HasErrorRecorder}:outputAllocator={HasOutputAllocator}:outputAddress={IsOutputTensorAddressSet}:debugListener={HasDebugListener}:nativeProfiler={HasNativeProfiler}:diagnostics={Diagnostics.Count}";
    }
}

/// <summary>
/// Compact pointer-free summary for an execution-context runtime diagnostic snapshot.
/// execution-context runtime 诊断快照的简短无指针摘要。
/// </summary>
public sealed class TensorRtExecutionContextRuntimeDiagnosticSummary
{
    internal TensorRtExecutionContextRuntimeDiagnosticSummary(
        TensorRtApiLine line,
        bool hasOutputTensorName,
        bool hasErrorRecorder,
        bool isInputConsumedEventSet,
        bool hasInputConsumedEventAddressValue,
        bool hasOutputAllocator,
        bool isOutputTensorAddressSet,
        bool hasOutputTensorAddressValue,
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool hasManagedProfiler,
        bool hasNativeProfiler,
        bool hasRuntimeConfig,
        TensorRtProfilingVerbosity nvtxVerbosity,
        bool unfusedTensorsDebugState,
        bool callbackStateHasOutputAllocator,
        bool callbackStateHasTemporaryStorageAllocator,
        bool callbackStateHasDebugListener,
        bool callbackStateOutputAllocatorInterfaceInfoAvailable,
        bool callbackStateTemporaryStorageAllocatorInterfaceInfoAvailable,
        bool callbackStateDebugListenerInterfaceInfoAvailable,
        BridgeStatusCode callbackStateLastStatus,
        bool callbackStateLastOperationAvailable,
        int diagnosticCount)
    {
        Line = line;
        HasOutputTensorName = hasOutputTensorName;
        HasErrorRecorder = hasErrorRecorder;
        IsInputConsumedEventSet = isInputConsumedEventSet;
        HasInputConsumedEventAddressValue = hasInputConsumedEventAddressValue;
        HasOutputAllocator = hasOutputAllocator;
        IsOutputTensorAddressSet = isOutputTensorAddressSet;
        HasOutputTensorAddressValue = hasOutputTensorAddressValue;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        HasManagedProfiler = hasManagedProfiler;
        HasNativeProfiler = hasNativeProfiler;
        HasRuntimeConfig = hasRuntimeConfig;
        NvtxVerbosity = nvtxVerbosity;
        UnfusedTensorsDebugState = unfusedTensorsDebugState;
        CallbackStateHasOutputAllocator = callbackStateHasOutputAllocator;
        CallbackStateHasTemporaryStorageAllocator = callbackStateHasTemporaryStorageAllocator;
        CallbackStateHasDebugListener = callbackStateHasDebugListener;
        CallbackStateOutputAllocatorInterfaceInfoAvailable = callbackStateOutputAllocatorInterfaceInfoAvailable;
        CallbackStateTemporaryStorageAllocatorInterfaceInfoAvailable = callbackStateTemporaryStorageAllocatorInterfaceInfoAvailable;
        CallbackStateDebugListenerInterfaceInfoAvailable = callbackStateDebugListenerInterfaceInfoAvailable;
        CallbackStateLastStatus = callbackStateLastStatus;
        CallbackStateLastOperationAvailable = callbackStateLastOperationAvailable;
        DiagnosticCount = diagnosticCount;
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect the source snapshot.
    /// 获取采集源快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the source snapshot had a non-empty output tensor name.
    /// 获取源快照是否包含非空 output tensor 名称。
    /// </summary>
    public bool HasOutputTensorName { get; }

    /// <summary>
    /// Gets whether TensorRT reported an error recorder.
    /// 获取 TensorRT 是否报告 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets whether TensorRT reported an input-consumed event.
    /// 获取 TensorRT 是否报告 input-consumed event。
    /// </summary>
    public bool IsInputConsumedEventSet { get; }

    /// <summary>
    /// Gets whether the copied input-consumed event address diagnostic was non-zero.
    /// 获取复制出的 input-consumed event 地址诊断值是否非零。
    /// </summary>
    public bool HasInputConsumedEventAddressValue { get; }

    /// <summary>
    /// Gets whether TensorRT reported an output allocator.
    /// 获取 TensorRT 是否报告 output allocator。
    /// </summary>
    public bool HasOutputAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reported an explicit output tensor address.
    /// 获取 TensorRT 是否报告显式 output tensor 地址。
    /// </summary>
    public bool IsOutputTensorAddressSet { get; }

    /// <summary>
    /// Gets whether the copied output tensor address diagnostic was non-zero.
    /// 获取复制出的 output tensor 地址诊断值是否非零。
    /// </summary>
    public bool HasOutputTensorAddressValue { get; }

    /// <summary>
    /// Gets whether TensorRT reported a temporary-storage allocator.
    /// 获取 TensorRT 是否报告 temporary-storage allocator。
    /// </summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reported a debug listener.
    /// 获取 TensorRT 是否报告 debug listener。
    /// </summary>
    public bool HasDebugListener { get; }

    /// <summary>
    /// Gets whether the managed wrapper holds a managed profiler borrow.
    /// 获取托管 wrapper 是否持有 managed profiler 借用。
    /// </summary>
    public bool HasManagedProfiler { get; }

    /// <summary>
    /// Gets whether TensorRT reported a native profiler.
    /// 获取 TensorRT 是否报告 native profiler。
    /// </summary>
    public bool HasNativeProfiler { get; }

    /// <summary>
    /// Gets whether TensorRT exposes a runtime config object.
    /// 获取 TensorRT 是否暴露 runtime config 对象。
    /// </summary>
    public bool HasRuntimeConfig { get; }

    /// <summary>
    /// Gets the copied NVTX verbosity value.
    /// 获取复制出的 NVTX verbosity 值。
    /// </summary>
    public TensorRtProfilingVerbosity NvtxVerbosity { get; }

    /// <summary>
    /// Gets the copied unfused-tensor debug state.
    /// 获取复制出的 unfused tensor debug state。
    /// </summary>
    public bool UnfusedTensorsDebugState { get; }

    /// <summary>
    /// Gets whether the copied callback state reported an output allocator.
    /// 获取复制出的 callback state 是否报告 output allocator。
    /// </summary>
    public bool CallbackStateHasOutputAllocator { get; }

    /// <summary>
    /// Gets whether the copied callback state reported a temporary-storage allocator.
    /// 获取复制出的 callback state 是否报告 temporary-storage allocator。
    /// </summary>
    public bool CallbackStateHasTemporaryStorageAllocator { get; }

    /// <summary>
    /// Gets whether the copied callback state reported a debug listener.
    /// 获取复制出的 callback state 是否报告 debug listener。
    /// </summary>
    public bool CallbackStateHasDebugListener { get; }

    /// <summary>
    /// Gets whether output allocator interface metadata was copied.
    /// 获取是否复制到了 output allocator interface 元数据。
    /// </summary>
    public bool CallbackStateOutputAllocatorInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets whether temporary-storage allocator interface metadata was copied.
    /// 获取是否复制到了 temporary-storage allocator interface 元数据。
    /// </summary>
    public bool CallbackStateTemporaryStorageAllocatorInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets whether debug listener interface metadata was copied.
    /// 获取是否复制到了 debug listener interface 元数据。
    /// </summary>
    public bool CallbackStateDebugListenerInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets the last bridge status copied by the callback-state snapshot.
    /// 获取 callback-state 快照复制出的最后 bridge 状态。
    /// </summary>
    public BridgeStatusCode CallbackStateLastStatus { get; }

    /// <summary>
    /// Gets whether the callback-state snapshot had a non-empty last operation label.
    /// 获取 callback-state 快照是否包含非空 last operation 标签。
    /// </summary>
    public bool CallbackStateLastOperationAvailable { get; }

    /// <summary>
    /// Gets the number of diagnostics collected while creating the snapshot.
    /// 获取创建快照时收集到的诊断数量。
    /// </summary>
    public int DiagnosticCount { get; }

    /// <summary>
    /// Converts the summary to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Line}:outputName={HasOutputTensorName}:errorRecorder={HasErrorRecorder}:outputAllocator={HasOutputAllocator}:outputAddress={IsOutputTensorAddressSet}:tempAllocator={HasTemporaryStorageAllocator}:debugListener={HasDebugListener}:managedProfiler={HasManagedProfiler}:nativeProfiler={HasNativeProfiler}:runtimeConfig={HasRuntimeConfig}:callbackStatus={CallbackStateLastStatus}:diagnostics={DiagnosticCount}";
    }
}
