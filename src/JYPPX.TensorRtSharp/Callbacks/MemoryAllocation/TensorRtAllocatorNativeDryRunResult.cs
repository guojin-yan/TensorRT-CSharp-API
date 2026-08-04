using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied diagnostics from the native allocator owner dry-run C ABI.
/// 表示从 native allocator owner dry-run C ABI 复制出的诊断信息。
/// </summary>
/// <remarks>
/// This result does not expose or retain the native owner handle. It does not represent an enabled TensorRT allocator
/// callback and never contains a device pointer.
/// 该结果不会暴露或保留 native owner 句柄；它不表示 TensorRT allocator callback 已启用，也不会包含 device pointer。
/// </remarks>
public readonly struct TensorRtAllocatorNativeDryRunResult
{
    internal TensorRtAllocatorNativeDryRunResult(
        TensorRtApiLine line,
        ulong invocationCount,
        ulong failureCount,
        BridgeStatusCode lastStatus,
        bool isAttached,
        ulong lastSize,
        ulong lastAlignment,
        string diagnostic)
    {
        Line = line;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        LastSize = lastSize;
        LastAlignment = lastAlignment;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the native dry-run owner.
    /// 获取 native dry-run owner 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the native dry-run invocation count copied from the owner.
    /// 获取从 native owner 复制出的 dry-run 调用次数。
    /// </summary>
    public ulong InvocationCount { get; }

    /// <summary>
    /// Gets the native dry-run failure count copied from the owner.
    /// 获取从 native owner 复制出的 dry-run 失败次数。
    /// </summary>
    public ulong FailureCount { get; }

    /// <summary>
    /// Gets the last bridge status recorded by the native dry-run owner.
    /// 获取 native dry-run owner 记录的最近一次 bridge status。
    /// </summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>
    /// Gets whether the native dry-run owner is attached to TensorRT.
    /// 获取 native dry-run owner 是否绑定到 TensorRT。
    /// </summary>
    /// <remarks>
    /// This is always <see langword="false"/> in the dry-run C ABI skeleton.
    /// 在 dry-run C ABI 骨架中，该值始终为 <see langword="false"/>。
    /// </remarks>
    public bool IsAttached { get; }

    /// <summary>
    /// Gets the last dry-run size copied from the native owner.
    /// 获取从 native owner 复制出的最近一次 dry-run size。
    /// </summary>
    public ulong LastSize { get; }

    /// <summary>
    /// Gets the last dry-run alignment copied from the native owner.
    /// 获取从 native owner 复制出的最近一次 dry-run alignment。
    /// </summary>
    public ulong LastAlignment { get; }

    /// <summary>
    /// Gets the copied native diagnostic message.
    /// 获取复制出的 native 诊断消息。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Gets whether the native dry-run diagnostic completed successfully.
    /// 获取 native dry-run 诊断是否成功完成。
    /// </summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0;

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Line}:status={LastStatus}:invocations={InvocationCount}:failures={FailureCount}:attached={IsAttached}:{Diagnostic}";
    }
}
