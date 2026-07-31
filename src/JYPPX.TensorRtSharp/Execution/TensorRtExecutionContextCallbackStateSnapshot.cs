using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Copied read-only snapshot of execution-context callback boundary state.
/// execution context 回调边界状态的只读复制快照。
/// </summary>
public sealed class TensorRtExecutionContextCallbackStateSnapshot
{
    internal TensorRtExecutionContextCallbackStateSnapshot(
        TensorRtApiLine line,
        bool hasOutputAllocator,
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool outputAllocatorInterfaceInfoAvailable,
        bool temporaryStorageAllocatorInterfaceInfoAvailable,
        bool debugListenerInterfaceInfoAvailable,
        bool outputAllocatorClearSupported,
        bool temporaryStorageAllocatorClearSupported,
        bool debugListenerClearSupported,
        bool outputAllocatorCleared,
        bool temporaryStorageAllocatorCleared,
        bool debugListenerCleared,
        TensorRtInterfaceInfo outputAllocatorInterfaceInfo,
        TensorRtInterfaceInfo temporaryStorageAllocatorInterfaceInfo,
        TensorRtInterfaceInfo debugListenerInterfaceInfo,
        BridgeStatusCode lastStatus,
        string lastOperation,
        string diagnostic)
    {
        Line = line;
        HasOutputAllocator = hasOutputAllocator;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        OutputAllocatorInterfaceInfoAvailable = outputAllocatorInterfaceInfoAvailable;
        TemporaryStorageAllocatorInterfaceInfoAvailable = temporaryStorageAllocatorInterfaceInfoAvailable;
        DebugListenerInterfaceInfoAvailable = debugListenerInterfaceInfoAvailable;
        OutputAllocatorClearSupported = outputAllocatorClearSupported;
        TemporaryStorageAllocatorClearSupported = temporaryStorageAllocatorClearSupported;
        DebugListenerClearSupported = debugListenerClearSupported;
        OutputAllocatorCleared = outputAllocatorCleared;
        TemporaryStorageAllocatorCleared = temporaryStorageAllocatorCleared;
        DebugListenerCleared = debugListenerCleared;
        OutputAllocatorInterfaceInfo = outputAllocatorInterfaceInfo;
        TemporaryStorageAllocatorInterfaceInfo = temporaryStorageAllocatorInterfaceInfo;
        DebugListenerInterfaceInfo = debugListenerInterfaceInfo;
        LastStatus = lastStatus;
        LastOperation = lastOperation ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this snapshot.
    /// 获取采集该快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether TensorRT reported an output allocator for the requested output tensor.
    /// 获取 TensorRT 是否报告指定输出 tensor 已绑定 output allocator。
    /// </summary>
    public bool HasOutputAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reported a temporary-storage allocator.
    /// 获取 TensorRT 是否报告已绑定 temporary-storage allocator。
    /// </summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>
    /// Gets whether TensorRT reported a debug listener.
    /// 获取 TensorRT 是否报告已绑定 debug listener。
    /// </summary>
    public bool HasDebugListener { get; }

    /// <summary>
    /// Gets whether output allocator interface metadata was copied.
    /// 获取是否复制到了 output allocator interface 元数据。
    /// </summary>
    public bool OutputAllocatorInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets whether temporary-storage allocator interface metadata was copied.
    /// 获取是否复制到了 temporary-storage allocator interface 元数据。
    /// </summary>
    public bool TemporaryStorageAllocatorInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets whether debug listener interface metadata was copied.
    /// 获取是否复制到了 debug listener interface 元数据。
    /// </summary>
    public bool DebugListenerInterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets whether output allocator clearing is supported for this snapshot line.
    /// 获取该版本线是否支持清除 output allocator。
    /// </summary>
    public bool OutputAllocatorClearSupported { get; }

    /// <summary>
    /// Gets whether temporary-storage allocator clearing is supported for this snapshot line.
    /// 获取该版本线是否支持清除 temporary-storage allocator。
    /// </summary>
    public bool TemporaryStorageAllocatorClearSupported { get; }

    /// <summary>
    /// Gets whether debug listener clearing is supported for this snapshot line.
    /// 获取该版本线是否支持清除 debug listener。
    /// </summary>
    public bool DebugListenerClearSupported { get; }

    /// <summary>
    /// Gets whether the clear snapshot reported that output allocator clear was accepted.
    /// 获取 clear 快照是否报告 output allocator 清理被接受。
    /// </summary>
    public bool OutputAllocatorCleared { get; }

    /// <summary>
    /// Gets whether the clear snapshot reported that temporary-storage allocator clear was accepted.
    /// 获取 clear 快照是否报告 temporary-storage allocator 清理被接受。
    /// </summary>
    public bool TemporaryStorageAllocatorCleared { get; }

    /// <summary>
    /// Gets whether the clear snapshot reported that debug listener clear was accepted.
    /// 获取 clear 快照是否报告 debug listener 清理被接受。
    /// </summary>
    public bool DebugListenerCleared { get; }

    /// <summary>
    /// Gets copied output allocator interface metadata. No allocator pointer is exposed or retained.
    /// 获取复制出的 output allocator interface 元数据；不会暴露或保留 allocator 指针。
    /// </summary>
    public TensorRtInterfaceInfo OutputAllocatorInterfaceInfo { get; }

    /// <summary>
    /// Gets copied temporary-storage allocator interface metadata. No allocator pointer is exposed or retained.
    /// 获取复制出的 temporary-storage allocator interface 元数据；不会暴露或保留 allocator 指针。
    /// </summary>
    public TensorRtInterfaceInfo TemporaryStorageAllocatorInterfaceInfo { get; }

    /// <summary>
    /// Gets copied debug listener interface metadata. No listener pointer is exposed or retained.
    /// 获取复制出的 debug listener interface 元数据；不会暴露或保留 listener 指针。
    /// </summary>
    public TensorRtInterfaceInfo DebugListenerInterfaceInfo { get; }

    /// <summary>
    /// Gets the last native bridge status copied into the snapshot.
    /// 获取复制到快照中的最后一个原生 bridge 状态。
    /// </summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>
    /// Gets whether every native snapshot phase completed successfully.
    /// 获取所有原生快照阶段是否都已成功完成。
    /// </summary>
    public bool IsComplete => LastStatus == BridgeStatusCode.Ok;

    /// <summary>
    /// Gets the last native operation label copied into the snapshot.
    /// 获取复制到快照中的最后一个原生操作标签。
    /// </summary>
    public string LastOperation { get; }

    /// <summary>
    /// Gets a copied diagnostic string.
    /// 获取复制出的诊断字符串。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Converts the snapshot to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    public override string ToString()
    {
        return $"{Line}:outputAllocator={HasOutputAllocator}:tempAllocator={HasTemporaryStorageAllocator}:debugListener={HasDebugListener}:operation={LastOperation}:status={LastStatus}";
    }
}
