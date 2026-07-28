using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener callback in-flight accounting gate evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate reports source-visible enter/leave accounting and release-after-drain scaffold evidence only. It does not
/// attach a listener to TensorRT and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerInFlightAccountingGate
{
    /// <summary>
    /// Evaluates in-flight accounting evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free in-flight accounting gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerInFlightAccountingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
    }

    /// <summary>
    /// Evaluates in-flight accounting evidence from copied owner and exception/status mapping evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="exceptionStatusMappingGate">The copied exception/status mapping gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free in-flight accounting gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerInFlightAccountingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate)
    {
        bool exceptionStatusMappingGateReady = exceptionStatusMappingGate.ExceptionStatusMappingGateReady;
        bool callbackEnterAccountingGateReady = ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool callbackLeaveAccountingGateReady = ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool callbackInFlightNeverNegativeReady = ownerDesignSnapshot.InFlightCallbackCount >= 0;
        bool releaseAfterDrainGateReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool callbackStateUnpinAfterDrainGateReady =
            releaseAfterDrainGateReady &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned;
        const bool accountingAddressExposed = false;
        const bool accountingPointerProduced = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, exceptionStatusMappingGateReady, "debug-listener-exception-status-mapping-gate is not ready for in-flight accounting evaluation.");
        AddBlockerIfFalse(blockers, callbackEnterAccountingGateReady, "DebugListener callback enter accounting scaffold has not observed copied diagnostic metadata.");
        AddBlockerIfFalse(blockers, callbackLeaveAccountingGateReady, "DebugListener callback leave accounting did not drain to zero.");
        AddBlockerIfFalse(blockers, callbackInFlightNeverNegativeReady, "DebugListener in-flight callback count is invalid.");
        AddBlockerIfFalse(blockers, releaseAfterDrainGateReady, "DebugListener release-after-drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackStateUnpinAfterDrainGateReady, "DebugListener callback state/delegate unpin after drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, !accountingAddressExposed, "DebugListener in-flight accounting gate exposes a native address.");
        AddBlockerIfFalse(blockers, !accountingPointerProduced, "DebugListener in-flight accounting gate produces a native pointer.");
        AddBlockerIfFalse(blockers, exceptionStatusMappingGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime in-flight accounting evidence.");

        foreach (string blocker in exceptionStatusMappingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerInFlightAccountingGateResult(
            ownerDesignSnapshot.Line,
            exceptionStatusMappingGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.ProcessDebugTensorCount,
            ownerDesignSnapshot.InFlightCallbackCount,
            ownerDesignSnapshot.MaxInFlightCallbackCount,
            ownerDesignSnapshot.ReleaseHookCount,
            ownerDesignSnapshot.CallbackStatePinned,
            ownerDesignSnapshot.DelegatePinned,
            ownerDesignSnapshot.DisposeRequested,
            exceptionStatusMappingGateReady,
            callbackEnterAccountingGateReady,
            callbackLeaveAccountingGateReady,
            callbackInFlightNeverNegativeReady,
            releaseAfterDrainGateReady,
            callbackStateUnpinAfterDrainGateReady,
            accountingAddressExposed,
            accountingPointerProduced,
            exceptionStatusMappingGate.NativeAttachEntryLocated,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }

    private static void AddBlockerIfFalse(List<string> blockers, bool condition, string blocker)
    {
        if (!condition)
        {
            AddBlocker(blockers, blocker);
        }
    }

    private static void AddBlocker(List<string> blockers, string blocker)
    {
        if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
        {
            blockers.Add(blocker);
        }
    }
}

/// <summary>
/// Reports copied DebugListener in-flight accounting diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerInFlightAccountingGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerInFlightAccountingGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        long processDebugTensorCount,
        long inFlightCallbackCount,
        long maxInFlightCallbackCount,
        long releaseHookCount,
        bool callbackStatePinned,
        bool delegatePinned,
        bool disposeRequested,
        bool exceptionStatusMappingGateReady,
        bool callbackEnterAccountingGateReady,
        bool callbackLeaveAccountingGateReady,
        bool callbackInFlightNeverNegativeReady,
        bool releaseAfterDrainGateReady,
        bool callbackStateUnpinAfterDrainGateReady,
        bool accountingAddressExposed,
        bool accountingPointerProduced,
        bool nativeAttachEntryLocated,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        ProcessDebugTensorCount = processDebugTensorCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        ReleaseHookCount = releaseHookCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        ExceptionStatusMappingGateReady = exceptionStatusMappingGateReady;
        CallbackEnterAccountingGateReady = callbackEnterAccountingGateReady;
        CallbackLeaveAccountingGateReady = callbackLeaveAccountingGateReady;
        CallbackInFlightNeverNegativeReady = callbackInFlightNeverNegativeReady;
        ReleaseAfterDrainGateReady = releaseAfterDrainGateReady;
        CallbackStateUnpinAfterDrainGateReady = callbackStateUnpinAfterDrainGateReady;
        AccountingAddressExposed = accountingAddressExposed;
        AccountingPointerProduced = accountingPointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-inflight-accounting-gate";

    /// <summary>Gets the callback kind represented by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "inflight-accounting-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied processDebugTensor diagnostic count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long ProcessDebugTensorCount { get; }

    /// <summary>Gets the copied in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied max in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied release hook count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether copied callback state remains pinned. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether copied delegate state remains pinned. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested in the copied snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether exception/status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionStatusMappingGateReady { get; }

    /// <summary>Gets whether callback enter accounting scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackEnterAccountingGateReady { get; }

    /// <summary>Gets whether callback leave accounting scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackLeaveAccountingGateReady { get; }

    /// <summary>Gets whether in-flight count cannot go negative in copied evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightNeverNegativeReady { get; }

    /// <summary>Gets whether release-after-drain evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseAfterDrainGateReady { get; }

    /// <summary>Gets whether callback state unpin after drain evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDrainGateReady { get; }

    /// <summary>Gets whether the accounting gate exposes a native address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AccountingAddressExposed { get; }

    /// <summary>Gets whether the accounting gate produces a native pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AccountingPointerProduced { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingGateReady =>
        ExceptionStatusMappingGateReady &&
        CallbackEnterAccountingGateReady &&
        CallbackLeaveAccountingGateReady &&
        CallbackInFlightNeverNegativeReady &&
        ReleaseAfterDrainGateReady &&
        CallbackStateUnpinAfterDrainGateReady &&
        !AccountingAddressExposed &&
        !AccountingPointerProduced &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        NativeAttachEntryLocated &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the in-flight accounting gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => InFlightAccountingGateReady ? "inflight-accounting-gate-ready" : "inflight-accounting-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-inflight-accounting-gate; RuntimeEvidenceKind=inflight-accounting-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; InFlightAccountingGateReady=" + InFlightAccountingGateReady + "; " +
        "ExceptionStatusMappingGateReady=" + ExceptionStatusMappingGateReady + "; " +
        "CallbackEnterAccountingGateReady=" + CallbackEnterAccountingGateReady + "; " +
        "CallbackLeaveAccountingGateReady=" + CallbackLeaveAccountingGateReady + "; " +
        "CallbackInFlightNeverNegativeReady=" + CallbackInFlightNeverNegativeReady + "; " +
        "ReleaseAfterDrainGateReady=" + ReleaseAfterDrainGateReady + "; " +
        "CallbackStateUnpinAfterDrainGateReady=" + CallbackStateUnpinAfterDrainGateReady + "; " +
        "AccountingAddressExposed=" + AccountingAddressExposed + "; " +
        "AccountingPointerProduced=" + AccountingPointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:inflight={InFlightCallbackCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
