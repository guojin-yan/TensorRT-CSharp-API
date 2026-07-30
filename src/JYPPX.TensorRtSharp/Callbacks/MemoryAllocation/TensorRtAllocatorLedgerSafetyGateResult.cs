using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied allocator ledger safety gate diagnostics.
/// 表示复制出的 allocator ledger 安全门禁诊断信息。
/// </summary>
/// <remarks>
/// This result is a safety gate only. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> are always <see langword="false"/> until a full package consumer smoke
/// emits complete <c>real-callback-runtime</c> evidence.
/// 该结果只是安全门禁。除非 full package consumer smoke 输出完整 <c>real-callback-runtime</c> 证据，
/// <see cref="RealCallbackRuntime"/> 与 <see cref="IsRealCallbackRuntimeProof"/> 始终为 <see langword="false"/>。
/// </remarks>
public readonly struct TensorRtAllocatorLedgerSafetyGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtAllocatorLedgerSafetyGateResult(
        TensorRtApiLine line,
        TensorRtAllocatorInternalRuntimePrototypeResult prototype,
        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger,
        BridgeStatusCode nativeLedgerStatus,
        string nativeLedgerDiagnostic,
        bool nativeLedgerAvailable)
    {
        Line = line;
        OwnerId = prototype.OwnerId;
        Operation = prototype.Operation;
        InternalPrototypeStatus = prototype.LastStatus;
        NativeLedgerStatus = nativeLedgerStatus;
        LastStatus = prototype.LastStatus == BridgeStatusCode.Ok ? nativeLedgerStatus : prototype.LastStatus;
        InvocationCount = prototype.InvocationCount;
        FailureCount = prototype.FailureCount;
        InFlightCallbackCount = prototype.InFlightCallbackCount;
        MaxInFlightCallbackCount = prototype.MaxInFlightCallbackCount;
        ActivePrototypeCallCount = prototype.ActivePrototypeCallCount;
        ReleaseHookCount = prototype.ReleaseHookCount;
        CallbackStatePinned = prototype.CallbackStatePinned;
        DelegatePinned = prototype.DelegatePinned;
        DisposeRequested = prototype.DisposeRequested;
        IsAttached = prototype.IsAttached;
        LastDiagnostic = prototype.LastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = prototype.ReleaseDiagnostic ?? string.Empty;
        NativeLedgerAvailable = nativeLedgerAvailable;
        NativeOwnerId = nativeLedger?.OwnerId ?? 0UL;
        StateTransitionCount = nativeLedger?.StateTransitionCount ?? 0UL;
        LedgerAllocationCount = nativeLedger?.LedgerAllocationCount ?? 0UL;
        LedgerReleaseCount = nativeLedger?.LedgerReleaseCount ?? 0UL;
        LedgerFailureCount = nativeLedger?.LedgerFailureCount ?? (nativeLedgerStatus == BridgeStatusCode.Ok ? 0UL : 1UL);
        LastAllocationId = nativeLedger?.LastAllocationId ?? 0UL;
        LastReleaseAllocationId = nativeLedger?.LastReleaseAllocationId ?? 0UL;
        LastStreamValue = nativeLedger?.LastStreamValue ?? 0UL;
        NativeAttachState = nativeLedger?.AttachState ?? 0;
        HasLiveAllocation = nativeLedger?.HasLiveAllocation ?? false;
        NativeLastOperation = nativeLedger?.LastOperation ?? string.Empty;
        NativeLedgerDiagnostic = nativeLedger?.Diagnostic ?? nativeLedgerDiagnostic ?? string.Empty;
        _blockedPrerequisites = BuildBlockedPrerequisites(
            prototype,
            nativeLedger,
            nativeLedgerStatus,
            nativeLedgerAvailable);
    }

    /// <summary>Gets the marker used by readiness to identify this safety gate. 获取 readiness 用于识别该安全门禁的 marker。</summary>
    public string EvidenceKind => "allocator-owner-ledger-safety-gate";

    /// <summary>Gets the callback kind represented by this diagnostic. 获取该诊断代表的 callback 类型。</summary>
    public string CallbackKind => "sync-allocator-ledger-safety";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "ledger-safety-gate";

    /// <summary>Gets whether this result proves real TensorRT callback runtime. 获取该结果是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 获取 readiness 是否可将该结果提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line used by the native ledger diagnostic. 获取 native ledger 诊断使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the managed owner id copied from the internal prototype. 获取从 internal prototype 复制出的托管 owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied operation label. 获取复制出的操作标签。</summary>
    public string Operation { get; }

    /// <summary>Gets the combined status for the safety gate. 获取安全门禁的合并状态。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the internal runtime prototype status. 获取 internal runtime prototype 状态。</summary>
    public BridgeStatusCode InternalPrototypeStatus { get; }

    /// <summary>Gets the native ledger diagnostic status. 获取 native ledger 诊断状态。</summary>
    public BridgeStatusCode NativeLedgerStatus { get; }

    /// <summary>Gets whether native ledger copied state was available. 获取 native ledger 复制状态是否可用。</summary>
    public bool NativeLedgerAvailable { get; }

    /// <summary>Gets the internal prototype invocation count. 获取 internal prototype 调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the internal prototype failure count. 获取 internal prototype 失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the current in-flight callback count. 获取当前 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the maximum observed in-flight callback count. 获取观察到的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the active prototype call count. 获取 active prototype 调用数量。</summary>
    public int ActivePrototypeCallCount { get; }

    /// <summary>Gets the release hook count. 获取 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether the callback state remains pinned. 获取 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the delegate remains pinned. 获取 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this diagnostic owner is attached to TensorRT. 获取该诊断 owner 是否已绑定到 TensorRT。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets the synthetic native owner id copied from the ledger. 获取从 ledger 复制出的合成 native owner id。</summary>
    public ulong NativeOwnerId { get; }

    /// <summary>Gets copied native owner state transition count. 获取复制出的 native owner 状态转移次数。</summary>
    public ulong StateTransitionCount { get; }

    /// <summary>Gets copied ledger allocation count. 获取复制出的 ledger allocation 次数。</summary>
    public ulong LedgerAllocationCount { get; }

    /// <summary>Gets copied ledger release count. 获取复制出的 ledger release 次数。</summary>
    public ulong LedgerReleaseCount { get; }

    /// <summary>Gets copied ledger failure count. 获取复制出的 ledger 失败次数。</summary>
    public ulong LedgerFailureCount { get; }

    /// <summary>Gets the last synthetic allocation id. 获取最近一次合成 allocation id。</summary>
    public ulong LastAllocationId { get; }

    /// <summary>Gets the last synthetic release allocation id. 获取最近一次合成 release allocation id。</summary>
    public ulong LastReleaseAllocationId { get; }

    /// <summary>Gets the last copied synthetic stream value. 获取最近一次复制出的合成 stream 值。</summary>
    public ulong LastStreamValue { get; }

    /// <summary>Gets the copied native attach state. 获取复制出的 native attach 状态。</summary>
    public int NativeAttachState { get; }

    /// <summary>Gets whether the synthetic ledger has live allocation state. 获取合成 ledger 是否存在 live allocation 状态。</summary>
    public bool HasLiveAllocation { get; }

    /// <summary>Gets the copied native ledger last operation. 获取复制出的 native ledger 最近操作。</summary>
    public string NativeLastOperation { get; }

    /// <summary>Gets the copied internal prototype diagnostic. 获取复制出的 internal prototype 诊断。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied native ledger diagnostic. 获取复制出的 native ledger 诊断。</summary>
    public string NativeLedgerDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic. 获取复制出的 release 诊断。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether the managed keep-alive state is ready before dispose. 获取 dispose 前托管 keep-alive 状态是否就绪。</summary>
    public bool ManagedKeepAliveReady =>
        CallbackStatePinned &&
        DelegatePinned &&
        !DisposeRequested &&
        InFlightCallbackCount == 0 &&
        !IsAttached;

    /// <summary>Gets whether dispose release evidence is present. 获取 dispose release 证据是否存在。</summary>
    public bool DisposeReleaseReady =>
        DisposeRequested &&
        ReleaseHookCount > 0 &&
        !CallbackStatePinned &&
        !DelegatePinned &&
        InFlightCallbackCount == 0;

    /// <summary>Gets whether native ledger copied-state evidence is clean. 获取 native ledger copied-state 证据是否干净。</summary>
    public bool NativeLedgerDesignReady =>
        NativeLedgerAvailable &&
        NativeLedgerStatus == BridgeStatusCode.Ok &&
        StateTransitionCount >= 2UL &&
        LedgerAllocationCount > 0UL &&
        LedgerAllocationCount == LedgerReleaseCount &&
        LedgerFailureCount == 0UL &&
        !HasLiveAllocation;

    /// <summary>Gets whether the public safety gate surface is pointer-free. 获取 public safety gate 表面是否不含 pointer。</summary>
    public bool PointerFreeSurfaceReady => true;

    /// <summary>Gets whether line-specific TensorRT attach/detach is ready. 获取按 TensorRT line 区分的 attach/detach 是否就绪。</summary>
    public bool LineSpecificAttachDetachReady => false;

    /// <summary>Gets whether runtime device pointer ledgering is ready. 获取 runtime device pointer ledger 是否就绪。</summary>
    public bool DevicePointerLedgerRuntimeReady => false;

    /// <summary>Gets whether stream lifetime and async allocator semantics are ready. 获取 stream lifetime 与 async allocator 语义是否就绪。</summary>
    public bool StreamLifetimeReady => false;

    /// <summary>Gets whether full package consumer runtime evidence is ready. 获取 full package consumer runtime 证据是否就绪。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady => false;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 获取直接 deferred callback 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 获取是否满足尝试真实 runtime proof 的全部前置条件。</summary>
    public bool CanAttemptRuntimeProof =>
        ManagedKeepAliveReady &&
        NativeLedgerDesignReady &&
        DisposeReleaseReady &&
        PointerFreeSurfaceReady &&
        LineSpecificAttachDetachReady &&
        DevicePointerLedgerRuntimeReady &&
        StreamLifetimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 获取真实 runtime proof 是否仍被阻断。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets copied blocked prerequisites. 获取复制出的阻断前置条件。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取复制出的阻断前置条件数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the safety gate status. 获取安全门禁状态。</summary>
    public string Status => RuntimeProofBlocked ? "safety-gate-blocked" : "can-attempt-runtime-proof";

    /// <summary>Gets a copied diagnostic summary. 获取复制出的诊断摘要。</summary>
    public string Diagnostic =>
        "allocator-owner-ledger-safety-gate; RuntimeEvidenceKind=ledger-safety-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeLedgerDesignReady=" + NativeLedgerDesignReady + "; " +
        "ManagedKeepAliveReady=" + ManagedKeepAliveReady + "; DisposeReleaseReady=" + DisposeReleaseReady + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:owner={OwnerId}:status={Status}:nativeLedger={NativeLedgerDesignReady}:proof={IsRealCallbackRuntimeProof}";
    }

    private static string[] BuildBlockedPrerequisites(
        TensorRtAllocatorInternalRuntimePrototypeResult prototype,
        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger,
        BridgeStatusCode nativeLedgerStatus,
        bool nativeLedgerAvailable)
    {
        List<string> blockers = new List<string>();
        bool managedKeepAliveReady =
            prototype.CallbackStatePinned &&
            prototype.DelegatePinned &&
            !prototype.DisposeRequested &&
            prototype.InFlightCallbackCount == 0 &&
            !prototype.IsAttached;
        bool disposeReleaseReady =
            prototype.DisposeRequested &&
            prototype.ReleaseHookCount > 0 &&
            !prototype.CallbackStatePinned &&
            !prototype.DelegatePinned &&
            prototype.InFlightCallbackCount == 0;
        bool nativeLedgerDesignReady =
            nativeLedgerAvailable &&
            nativeLedgerStatus == BridgeStatusCode.Ok &&
            nativeLedger.HasValue &&
            nativeLedger.Value.StateTransitionCount >= 2UL &&
            nativeLedger.Value.LedgerAllocationCount > 0UL &&
            nativeLedger.Value.LedgerAllocationCount == nativeLedger.Value.LedgerReleaseCount &&
            nativeLedger.Value.LedgerFailureCount == 0UL &&
            !nativeLedger.Value.HasLiveAllocation;

        if (prototype.LastStatus != BridgeStatusCode.Ok || prototype.FailureCount != 0 || prototype.InFlightCallbackCount != 0)
        {
            blockers.Add("internal allocator runtime prototype did not produce clean no-throw copied diagnostics.");
        }

        if (!managedKeepAliveReady && !disposeReleaseReady)
        {
            blockers.Add("managed owner keep-alive or dispose release evidence is not clean.");
        }

        if (!nativeLedgerDesignReady)
        {
            blockers.Add("native allocator owner ledger copied-state evidence is not available or not clean.");
        }

        blockers.Add("line-specific setGpuAllocator attach/detach is not implemented.");
        blockers.Add("runtime device pointer ownership ledger is not implemented.");
        blockers.Add("CUDA stream lifetime and IGpuAsyncAllocator semantics are not implemented.");
        blockers.Add("full package consumer smoke has not emitted real-callback-runtime evidence.");
        return blockers.ToArray();
    }
}
