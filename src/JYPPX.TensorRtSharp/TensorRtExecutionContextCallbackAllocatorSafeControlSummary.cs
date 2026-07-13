using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free summary of execution-context callback allocator safe controls.
/// execution context callback allocator 安全控制面的无指针摘要。
/// </summary>
/// <remarks>
/// This type contains copied metadata only. Borrowed pointer not exposed/owned, no callback invocation is attempted,
/// and the summary is not runtime proof of TensorRT callback execution.
/// 该类型只包含复制出的元数据；不会暴露或拥有 borrowed pointer，不会尝试 callback 调用，也不是 TensorRT callback
/// 已真实执行的 runtime proof。
/// </remarks>
public sealed class TensorRtExecutionContextCallbackAllocatorSafeControlSummary
{
    private readonly string[] _diagnostics;

    internal TensorRtExecutionContextCallbackAllocatorSafeControlSummary(
        TensorRtApiLine line,
        string outputTensorName,
        bool hasOutputAllocator,
        bool hasTemporaryStorageAllocator,
        bool hasDebugListener,
        bool outputAllocatorInterfaceInfoAvailable,
        bool temporaryStorageAllocatorInterfaceInfoAvailable,
        bool debugListenerInterfaceInfoAvailable,
        TensorRtInterfaceInfo outputAllocatorInterfaceInfo,
        TensorRtInterfaceInfo temporaryStorageAllocatorInterfaceInfo,
        TensorRtInterfaceInfo debugListenerInterfaceInfo,
        string outputAllocatorDiagnostic,
        string temporaryStorageAllocatorDiagnostic,
        string debugListenerDiagnostic,
        TensorRtExecutionContextCallbackStateSnapshot callbackState,
        string[] diagnostics)
    {
        Line = line;
        OutputTensorName = outputTensorName ?? string.Empty;
        HasOutputAllocator = hasOutputAllocator;
        HasTemporaryStorageAllocator = hasTemporaryStorageAllocator;
        HasDebugListener = hasDebugListener;
        OutputAllocatorInterfaceInfoAvailable = outputAllocatorInterfaceInfoAvailable;
        TemporaryStorageAllocatorInterfaceInfoAvailable = temporaryStorageAllocatorInterfaceInfoAvailable;
        DebugListenerInterfaceInfoAvailable = debugListenerInterfaceInfoAvailable;
        OutputAllocatorInterfaceInfo = outputAllocatorInterfaceInfo;
        TemporaryStorageAllocatorInterfaceInfo = temporaryStorageAllocatorInterfaceInfo;
        DebugListenerInterfaceInfo = debugListenerInterfaceInfo;
        OutputAllocatorDiagnostic = outputAllocatorDiagnostic ?? string.Empty;
        TemporaryStorageAllocatorDiagnostic = temporaryStorageAllocatorDiagnostic ?? string.Empty;
        DebugListenerDiagnostic = debugListenerDiagnostic ?? string.Empty;
        CallbackState = callbackState;
        _diagnostics = diagnostics == null ? Array.Empty<string>() : (string[])diagnostics.Clone();
    }

    /// <summary>Gets the evidence marker used by smoke and package-consumer diagnostics. 获取 smoke 与 package-consumer 诊断使用的 evidence marker。</summary>
    public string EvidenceKind => "execution-context-callback-allocator-safe-control-summary";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "copied-interface-info-safe-controls";

    /// <summary>Gets whether this summary represents real TensorRT callback runtime proof. 获取该摘要是否代表真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether this summary may be promoted as real callback runtime proof. 获取该摘要是否可提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line used for the safe-control query. 获取执行安全控制查询时使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the output tensor name used for output allocator queries. 获取用于 output allocator 查询的输出 tensor 名称。</summary>
    public string OutputTensorName { get; }

    /// <summary>Gets whether TensorRT reported an output allocator for the output tensor. 获取 TensorRT 是否为输出 tensor 报告 output allocator。</summary>
    public bool HasOutputAllocator { get; }

    /// <summary>Gets whether TensorRT reported a temporary-storage allocator. 获取 TensorRT 是否报告 temporary-storage allocator。</summary>
    public bool HasTemporaryStorageAllocator { get; }

    /// <summary>Gets whether TensorRT reported a debug listener. 获取 TensorRT 是否报告 debug listener。</summary>
    public bool HasDebugListener { get; }

    /// <summary>Gets whether output allocator interface metadata was copied. 获取是否复制到 output allocator interface 元数据。</summary>
    public bool OutputAllocatorInterfaceInfoAvailable { get; }

    /// <summary>Gets whether temporary-storage allocator interface metadata was copied. 获取是否复制到 temporary-storage allocator interface 元数据。</summary>
    public bool TemporaryStorageAllocatorInterfaceInfoAvailable { get; }

    /// <summary>Gets whether debug listener interface metadata was copied. 获取是否复制到 debug listener interface 元数据。</summary>
    public bool DebugListenerInterfaceInfoAvailable { get; }

    /// <summary>Gets copied output allocator interface metadata. 获取复制出的 output allocator interface 元数据。</summary>
    public TensorRtInterfaceInfo OutputAllocatorInterfaceInfo { get; }

    /// <summary>Gets copied temporary-storage allocator interface metadata. 获取复制出的 temporary-storage allocator interface 元数据。</summary>
    public TensorRtInterfaceInfo TemporaryStorageAllocatorInterfaceInfo { get; }

    /// <summary>Gets copied debug listener interface metadata. 获取复制出的 debug listener interface 元数据。</summary>
    public TensorRtInterfaceInfo DebugListenerInterfaceInfo { get; }

    /// <summary>Gets the copied output allocator diagnostic. 获取复制出的 output allocator 诊断。</summary>
    public string OutputAllocatorDiagnostic { get; }

    /// <summary>Gets the copied temporary-storage allocator diagnostic. 获取复制出的 temporary-storage allocator 诊断。</summary>
    public string TemporaryStorageAllocatorDiagnostic { get; }

    /// <summary>Gets the copied debug listener diagnostic. 获取复制出的 debug listener 诊断。</summary>
    public string DebugListenerDiagnostic { get; }

    /// <summary>Gets the copied callback-state snapshot consumed by this summary. 获取该摘要消费的 callback-state 复制快照。</summary>
    public TensorRtExecutionContextCallbackStateSnapshot CallbackState { get; }

    /// <summary>Gets non-fatal diagnostics collected while building this summary. 获取构建摘要时收集的非致命诊断。</summary>
    public ReadOnlyCollection<string> Diagnostics => Array.AsReadOnly(_diagnostics ?? Array.Empty<string>());

    /// <summary>Gets the number of copied interface-info records available in this summary. 获取该摘要中可用 interface-info 副本数量。</summary>
    public int CopiedInterfaceInfoCount =>
        (OutputAllocatorInterfaceInfoAvailable ? 1 : 0) +
        (TemporaryStorageAllocatorInterfaceInfoAvailable ? 1 : 0) +
        (DebugListenerInterfaceInfoAvailable ? 1 : 0);

    /// <summary>Gets the copied diagnostic count. 获取复制出的诊断数量。</summary>
    public int DiagnosticCount => (_diagnostics ?? Array.Empty<string>()).Length;

    /// <summary>Gets whether the public surface is pointer-free. 获取 public surface 是否保持无指针。</summary>
    public bool PointerFreeSurfaceReady => true;

    /// <summary>Gets whether this summary attempted to invoke TensorRT callbacks. 获取该摘要是否尝试调用 TensorRT callback。</summary>
    public bool CallbackInvocationAttempted => false;

    /// <summary>Gets whether callback invocation proof is complete. 获取 callback invocation proof 是否完成。</summary>
    public bool CallbackInvocationProofComplete => false;

    /// <summary>Gets whether real runtime invocation proof is complete. 获取真实 runtime invocation proof 是否完成。</summary>
    public bool IsRuntimeInvocationProofComplete => false;

    /// <summary>Gets a compact copied diagnostic summary. 获取紧凑的复制式诊断摘要。</summary>
    public string Summary =>
        "execution-context-callback-allocator-safe-control-summary; RuntimeEvidenceKind=copied-interface-info-safe-controls; " +
        "RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False; PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "CopiedInterfaceInfoCount=" + CopiedInterfaceInfoCount + "; DiagnosticCount=" + DiagnosticCount + "; " +
        "CallbackInvocationAttempted=False; IsRuntimeInvocationProofComplete=False.";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    /// <returns>A pointer-free diagnostic string. 无指针诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={Line}:output={OutputTensorName}:interfaces={CopiedInterfaceInfoCount}:diagnostics={DiagnosticCount}:runtimeProof={IsRuntimeInvocationProofComplete}";
    }
}
