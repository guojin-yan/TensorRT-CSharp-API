using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied diagnostics for the output allocator callback owner design gate.
/// 表示 output allocator callback owner 设计门禁复制出的诊断信息。
/// </summary>
/// <remarks>
/// This snapshot intentionally exposes no native handle, callback owner pointer, output buffer pointer, or borrowed
/// TensorRT object pointer. It is not proof that TensorRT has invoked <c>IOutputAllocator::notifyShape</c> or
/// <c>IOutputAllocator::reallocateOutput</c>.
/// 该快照有意不暴露 native handle、callback owner pointer、output buffer pointer 或 borrowed TensorRT object pointer。
/// 它不证明 TensorRT 已调用 <c>IOutputAllocator::notifyShape</c> 或 <c>IOutputAllocator::reallocateOutput</c>。
/// </remarks>
public readonly struct TensorRtOutputAllocatorCallbackOwnerSnapshot
{
    internal TensorRtOutputAllocatorCallbackOwnerSnapshot(
        TensorRtApiLine line,
        TensorRtOutputAllocatorRuntimeGateResult gate,
        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger,
        BridgeStatusCode nativeLedgerStatus,
        string nativeLedgerDiagnostic,
        bool nativeLedgerAvailable)
    {
        Line = line;
        OwnerId = gate.OwnerId;
        Operation = gate.Operation;
        TensorName = gate.TensorName;
        RequestedSize = gate.RequestedSize;
        Alignment = gate.Alignment;
        ShapeRank = gate.ShapeRank;
        ShapeSummary = gate.ShapeSummary;
        HasCurrentMemory = gate.HasCurrentMemory;
        RuntimeGateStatus = gate.LastStatus;
        NativeLedgerStatus = nativeLedgerStatus;
        LastStatus = nativeLedgerStatus == BridgeStatusCode.Ok ? gate.LastStatus : nativeLedgerStatus;
        InvocationCount = gate.InvocationCount;
        NotifyShapeCount = gate.NotifyShapeCount;
        ReallocateOutputCount = gate.ReallocateOutputCount;
        FailureCount = gate.FailureCount;
        InFlightCallbackCount = gate.InFlightCallbackCount;
        MaxInFlightCallbackCount = gate.MaxInFlightCallbackCount;
        ActiveGateCallCount = gate.ActiveGateCallCount;
        ReleaseHookCount = gate.ReleaseHookCount;
        CallbackStatePinned = gate.CallbackStatePinned;
        DelegatePinned = gate.DelegatePinned;
        DisposeRequested = gate.DisposeRequested;
        IsAttached = false;
        NativeLedgerAvailable = nativeLedgerAvailable;
        NativeOwnerId = nativeLedger?.OwnerId ?? 0UL;
        StateTransitionCount = nativeLedger?.StateTransitionCount ?? 0UL;
        LedgerAllocationCount = nativeLedger?.LedgerAllocationCount ?? 0UL;
        LedgerReleaseCount = nativeLedger?.LedgerReleaseCount ?? 0UL;
        LedgerFailureCount = nativeLedger?.LedgerFailureCount ?? (nativeLedgerStatus == BridgeStatusCode.Ok ? 0UL : 1UL);
        LastAllocationId = nativeLedger?.LastAllocationId ?? 0UL;
        LastReleaseAllocationId = nativeLedger?.LastReleaseAllocationId ?? 0UL;
        LastStreamValue = nativeLedger?.LastStreamValue ?? 0UL;
        HasLiveAllocation = nativeLedger?.HasLiveAllocation ?? false;
        NativeLastOperation = nativeLedger?.LastOperation ?? string.Empty;
        NativeLedgerDiagnostic = nativeLedger?.Diagnostic ?? nativeLedgerDiagnostic ?? string.Empty;
        LastDiagnostic = ComposeDiagnostic(gate.LastDiagnostic, NativeLedgerDiagnostic, nativeLedgerAvailable);
        ReleaseDiagnostic = gate.ReleaseDiagnostic;
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取 readiness 用于识别该门禁的 marker。</summary>
    public string EvidenceKind => "output-allocator-callback-owner-design";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "output-allocator-prototype";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "not-present";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line used for the native ledger diagnostic attempt. 获取 native ledger 诊断尝试使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the managed output allocator owner id copied from the runtime gate. 获取从 runtime gate 复制出的托管 output allocator owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the synthetic native owner id copied from the ledger dry-run. 获取从 ledger dry-run 复制出的合成 native owner id。</summary>
    public ulong NativeOwnerId { get; }

    /// <summary>Gets the last copied operation. 获取最近一次复制出的操作。</summary>
    public string Operation { get; }

    /// <summary>Gets the copied output tensor name. 获取复制出的输出 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied requested output buffer size. 获取复制出的请求输出缓冲区大小。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets the copied requested output buffer alignment. 获取复制出的请求输出缓冲区对齐。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets the copied output shape rank. 获取复制出的输出 shape rank。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets a compact copied output shape summary. 获取紧凑的复制输出 shape 摘要。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether TensorRT reported an existing current memory pointer. 获取 TensorRT 是否报告已有 current memory pointer。</summary>
    public bool HasCurrentMemory { get; }

    /// <summary>Gets the combined status for the gate and ledger diagnostics. 获取 gate 与 ledger 诊断合并后的状态。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the status returned by the managed runtime gate. 获取托管 runtime gate 返回的状态。</summary>
    public BridgeStatusCode RuntimeGateStatus { get; }

    /// <summary>Gets the status returned by the native ledger diagnostic attempt. 获取 native ledger 诊断尝试返回的状态。</summary>
    public BridgeStatusCode NativeLedgerStatus { get; }

    /// <summary>Gets whether the native ledger diagnostic completed and produced copied state. 获取 native ledger 诊断是否完成并产生复制状态。</summary>
    public bool NativeLedgerAvailable { get; }

    /// <summary>Gets the total copied runtime gate invocation count. 获取复制出的 runtime gate 调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied notifyShape diagnostic count. 获取复制出的 notifyShape 诊断次数。</summary>
    public long NotifyShapeCount { get; }

    /// <summary>Gets the copied reallocateOutput diagnostic count. 获取复制出的 reallocateOutput 诊断次数。</summary>
    public long ReallocateOutputCount { get; }

    /// <summary>Gets the copied managed callback failure count. 获取复制出的托管 callback 失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active gate call count. 获取复制出的 active gate 调用数量。</summary>
    public int ActiveGateCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this design owner is attached to a TensorRT execution context. 获取该设计 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets the copied native owner state transition count. 获取复制出的 native owner 状态转移次数。</summary>
    public ulong StateTransitionCount { get; }

    /// <summary>Gets the copied ledger allocation intent count. 获取复制出的 ledger allocation intent 次数。</summary>
    public ulong LedgerAllocationCount { get; }

    /// <summary>Gets the copied ledger release intent count. 获取复制出的 ledger release intent 次数。</summary>
    public ulong LedgerReleaseCount { get; }

    /// <summary>Gets the copied ledger failure count. 获取复制出的 ledger 失败次数。</summary>
    public ulong LedgerFailureCount { get; }

    /// <summary>Gets the last synthetic allocation id. 获取最近一次合成 allocation id。</summary>
    public ulong LastAllocationId { get; }

    /// <summary>Gets the last synthetic release allocation id. 获取最近一次合成 release allocation id。</summary>
    public ulong LastReleaseAllocationId { get; }

    /// <summary>Gets the last copied synthetic stream value. 获取最近一次复制出的合成 stream 值。</summary>
    public ulong LastStreamValue { get; }

    /// <summary>Gets whether the synthetic ledger still has a live allocation. 获取合成 ledger 是否仍有 live allocation。</summary>
    public bool HasLiveAllocation { get; }

    /// <summary>Gets whether an output buffer pointer is exposed by this public API. 获取该 public API 是否暴露 output buffer pointer。</summary>
    public bool OutputBufferPointerExposed => false;

    /// <summary>Gets whether an output buffer pointer was produced by this diagnostic gate. 获取该诊断门禁是否产生 output buffer pointer。</summary>
    public bool OutputBufferPointerProduced => false;

    /// <summary>Gets the copied native ledger last operation. 获取复制出的 native ledger 最近操作。</summary>
    public string NativeLastOperation { get; }

    /// <summary>Gets the copied combined diagnostic message. 获取复制出的合并诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied native ledger diagnostic message. 获取复制出的 native ledger 诊断消息。</summary>
    public string NativeLedgerDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether the diagnostic completed without gate or ledger failures. 获取诊断是否未出现 gate 或 ledger 失败。</summary>
    public bool Succeeded =>
        RuntimeGateStatus == BridgeStatusCode.Ok &&
        NativeLedgerStatus == BridgeStatusCode.Ok &&
        FailureCount == 0 &&
        LedgerFailureCount == 0 &&
        InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:notify={NotifyShapeCount}:reallocate={ReallocateOutputCount}:ledger={LedgerAllocationCount}/{LedgerReleaseCount}:proof={IsRealCallbackRuntimeProof}";
    }

    private static string ComposeDiagnostic(string gateDiagnostic, string nativeLedgerDiagnostic, bool nativeLedgerAvailable)
    {
        string ledgerPrefix = nativeLedgerAvailable ? "native-ledger=" : "native-ledger-unavailable=";
        return "output-allocator-callback-owner-design; " +
            (gateDiagnostic ?? string.Empty) +
            "; " +
            ledgerPrefix +
            (nativeLedgerDiagnostic ?? string.Empty) +
            "; RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False.";
    }
}
