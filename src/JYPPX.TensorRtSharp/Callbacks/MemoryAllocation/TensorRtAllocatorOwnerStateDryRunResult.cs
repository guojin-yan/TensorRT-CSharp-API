using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied native allocator owner state intent data from the dry-run C ABI.
/// 表示从 dry-run C ABI 复制出的 native allocator owner 状态意图数据。
/// </summary>
/// <remarks>
/// This result copies the synthetic owner state only. It does not expose any native handle, device pointer, or real
/// TensorRT allocator callback.
/// 该结果只复制合成的 owner 状态，不暴露任何 native handle、device pointer 或真实 TensorRT allocator callback。
/// </remarks>
public readonly struct TensorRtAllocatorOwnerStateDryRunResult
{
    internal TensorRtAllocatorOwnerStateDryRunResult(
        TensorRtApiLine line,
        ulong ownerId,
        ulong stateTransitionCount,
        ulong ledgerAllocationCount,
        ulong ledgerReleaseCount,
        ulong ledgerFailureCount,
        ulong lastAllocationId,
        ulong lastReleaseAllocationId,
        ulong lastSize,
        ulong lastAlignment,
        ulong lastStreamValue,
        int attachState,
        BridgeStatusCode lastStatus,
        bool isAttached,
        bool hasLiveAllocation,
        string lastOperation,
        string diagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        StateTransitionCount = stateTransitionCount;
        LedgerAllocationCount = ledgerAllocationCount;
        LedgerReleaseCount = ledgerReleaseCount;
        LedgerFailureCount = ledgerFailureCount;
        LastAllocationId = lastAllocationId;
        LastReleaseAllocationId = lastReleaseAllocationId;
        LastSize = lastSize;
        LastAlignment = lastAlignment;
        LastStreamValue = lastStreamValue;
        AttachState = attachState;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        HasLiveAllocation = hasLiveAllocation;
        LastOperation = lastOperation ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT API line used by the native dry-run owner. 获取 native dry-run owner 使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the synthetic owner id copied from native. 获取从 native 复制出的合成 owner id。</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets the number of state transitions copied from native. 获取从 native 复制出的状态转移次数。</summary>
    public ulong StateTransitionCount { get; }

    /// <summary>Gets the number of recorded allocation intents. 获取记录的分配意图次数。</summary>
    public ulong LedgerAllocationCount { get; }

    /// <summary>Gets the number of recorded release intents. 获取记录的释放意图次数。</summary>
    public ulong LedgerReleaseCount { get; }

    /// <summary>Gets the number of ledger failures copied from native. 获取从 native 复制出的 ledger 失败次数。</summary>
    public ulong LedgerFailureCount { get; }

    /// <summary>Gets the last synthetic allocation id. 获取最近一次合成 allocation id。</summary>
    public ulong LastAllocationId { get; }

    /// <summary>Gets the last synthetic release allocation id. 获取最近一次释放意图的 allocation id。</summary>
    public ulong LastReleaseAllocationId { get; }

    /// <summary>Gets the last copied size. 获取最近一次复制出的 size。</summary>
    public ulong LastSize { get; }

    /// <summary>Gets the last copied alignment. 获取最近一次复制出的 alignment。</summary>
    public ulong LastAlignment { get; }

    /// <summary>Gets the last copied stream value. 获取最近一次复制出的 stream value。</summary>
    public ulong LastStreamValue { get; }

    /// <summary>Gets the native attach state. 获取 native attach state。</summary>
    public int AttachState { get; }

    /// <summary>Gets the last bridge status recorded by native. 获取 native 记录的最近一次 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether the synthetic owner is attached. 获取合成 owner 是否处于 attached 状态。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the synthetic ledger still has a live allocation. 获取合成 ledger 是否仍有 live allocation。</summary>
    public bool HasLiveAllocation { get; }

    /// <summary>Gets the copied last operation name. 获取复制出的最近一次操作名。</summary>
    public string LastOperation { get; }

    /// <summary>Gets the copied native diagnostic string. 获取复制出的 native 诊断字符串。</summary>
    public string Diagnostic { get; }

    /// <summary>Gets whether the copied state indicates success. 获取复制出的状态是否表示成功。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && LedgerFailureCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Line}:owner={OwnerId}:transitions={StateTransitionCount}:allocations={LedgerAllocationCount}:releases={LedgerReleaseCount}:failures={LedgerFailureCount}:live={HasLiveAllocation}:{LastOperation}:{Diagnostic}";
    }
}
