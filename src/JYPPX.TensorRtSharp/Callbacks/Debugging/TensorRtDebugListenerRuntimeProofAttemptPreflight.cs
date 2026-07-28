using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates whether DebugListener real callback runtime proof work may be attempted.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This preflight consumes pointer-free runtime proof precheck evidence only. It does not enable
/// <c>setDebugListener(non-null)</c>, does not install a native <c>IDebugListener</c> vtable, and does not invoke
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerRuntimeProofAttemptPreflight
{
    /// <summary>
    /// Evaluates proof-attempt readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free proof-attempt preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofAttemptPreflightResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        return Evaluate(TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(ownerDesignSnapshot));
    }

    /// <summary>
    /// Evaluates proof-attempt readiness from a runtime proof precheck result.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="precheck">The pointer-free runtime proof precheck result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free proof-attempt preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofAttemptPreflightResult Evaluate(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck)
    {
        bool canEnableSetDebugListenerNonNull =
            precheck.NativeAttachBridgeShapeGateReady &&
            precheck.AttachBridgeNoThrowBoundaryReady &&
            precheck.AttachBridgeVersionGuardReady &&
            precheck.AttachBridgeOwnershipDiagnosticsReady &&
            precheck.AttachBridgePointerFree &&
            precheck.NativeAttachEntryLocated &&
            !precheck.NonNullAttachStillDisabled &&
            precheck.NativeOwnerLifecycleReady &&
            precheck.CanImplementNativeAttach;

        bool canInstallNativeVTable =
            precheck.NativeVTableReady &&
            precheck.NoThrowVTableDesignReady &&
            precheck.NativeVTableTrampolineReady &&
            precheck.CallbackExceptionCaptureReady &&
            precheck.CallbackStatusMappingReady &&
            precheck.CallbackInFlightAccountingReady &&
            precheck.NativeNoThrowVTableScaffoldGateReady &&
            precheck.NoThrowVTableScaffoldReady &&
            precheck.VTableDestructorNoThrowReady &&
            precheck.ProcessDebugTensorCallbackStubNoThrowReady &&
            !precheck.VTableAddressExposed &&
            !precheck.VTablePointerProduced;

        bool canCallProcessDebugTensorRuntime =
            canEnableSetDebugListenerNonNull &&
            canInstallNativeVTable &&
            precheck.BorrowedDebugTensorPointerEscapeBlocked &&
            precheck.BorrowedDebugTensorLifetimeReady &&
            precheck.BorrowedDebugTensorDataLifetimeReady &&
            precheck.ProcessDebugTensorRuntimeReady;

        bool canPromoteRealCallbackRuntime =
            canCallProcessDebugTensorRuntime &&
            precheck.FullPackageConsumerRuntimeEvidenceReady &&
            precheck.CanAttemptRuntimeProof &&
            precheck.RealCallbackRuntime &&
            precheck.IsRealCallbackRuntimeProof;

        string nonNullReason = BuildNonNullAttachReason(precheck, canEnableSetDebugListenerNonNull);
        string nativeVTableReason = BuildNativeVTableReason(precheck, canInstallNativeVTable);
        string runtimeReason = BuildRuntimeProofReason(
            precheck,
            canEnableSetDebugListenerNonNull,
            canInstallNativeVTable,
            canCallProcessDebugTensorRuntime,
            canPromoteRealCallbackRuntime);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, canEnableSetDebugListenerNonNull, nonNullReason);
        AddBlockerIfFalse(blockers, canInstallNativeVTable, nativeVTableReason);
        AddBlockerIfFalse(blockers, canCallProcessDebugTensorRuntime, runtimeReason);
        AddBlockerIfFalse(blockers, canPromoteRealCallbackRuntime, "real-callback-runtime promotion remains blocked by attempt preflight.");
        foreach (string blocker in precheck.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerRuntimeProofAttemptPreflightResult(
            precheck.Line,
            precheck.NativeAttachEntryLocated,
            precheck.NonNullAttachStillDisabled,
            precheck.NativeOwnerLifecycleReady,
            precheck.CanImplementNativeAttach,
            precheck.NativeVTableReady,
            precheck.NoThrowVTableDesignReady,
            precheck.NativeVTableTrampolineReady,
            precheck.CallbackExceptionCaptureReady,
            precheck.CallbackStatusMappingReady,
            precheck.CallbackInFlightAccountingReady,
            precheck.VTableAddressExposed,
            precheck.VTablePointerProduced,
            precheck.BorrowedDebugTensorPointerEscapeBlocked,
            precheck.BorrowedDebugTensorLifetimeReady,
            precheck.BorrowedDebugTensorDataLifetimeReady,
            precheck.ProcessDebugTensorRuntimeReady,
            precheck.FullPackageConsumerRuntimeEvidenceReady,
            precheck.CanAttemptRuntimeProof,
            canEnableSetDebugListenerNonNull,
            canInstallNativeVTable,
            canCallProcessDebugTensorRuntime,
            canPromoteRealCallbackRuntime,
            nonNullReason,
            nativeVTableReason,
            runtimeReason,
            blockers.ToArray());
    }

    private static string BuildNonNullAttachReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canEnableSetDebugListenerNonNull)
    {
        if (canEnableSetDebugListenerNonNull)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, precheck.NativeAttachEntryLocated, "native line-specific setDebugListener(non-null) attach entry is not implemented.");
        AddBlockerIfFalse(reasons, !precheck.NonNullAttachStillDisabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, precheck.NativeOwnerLifecycleReady, "native owner lifecycle is not ready for non-null attach.");
        AddBlockerIfFalse(reasons, precheck.CanImplementNativeAttach, "native attach implementation is still blocked.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgePointerFree, "attach bridge diagnostics are not pointer-free.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgeNoThrowBoundaryReady, "attach bridge no-throw boundary is not ready.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgeVersionGuardReady, "attach bridge TensorRT version guard is not ready.");
        return string.Join(" ", reasons);
    }

    private static string BuildNativeVTableReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canInstallNativeVTable)
    {
        if (canInstallNativeVTable)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, precheck.NativeVTableReady, "stable native owner address and no-throw native vtable are not ready.");
        AddBlockerIfFalse(reasons, precheck.NoThrowVTableDesignReady, "no-throw vtable design is not complete.");
        AddBlockerIfFalse(reasons, precheck.NativeVTableTrampolineReady, "native IDebugListener vtable trampoline is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackExceptionCaptureReady, "native callback exception capture is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackStatusMappingReady, "callback status mapping is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackInFlightAccountingReady, "callback in-flight accounting is not implemented.");
        AddBlockerIfFalse(reasons, precheck.NativeNoThrowVTableScaffoldGateReady, "native no-throw vtable scaffold gate is not ready.");
        AddBlockerIfFalse(reasons, precheck.NoThrowVTableScaffoldReady, "no-throw vtable scaffold is not ready.");
        AddBlockerIfFalse(reasons, precheck.VTableDestructorNoThrowReady, "vtable destructor no-throw scaffold is not ready.");
        AddBlockerIfFalse(reasons, precheck.ProcessDebugTensorCallbackStubNoThrowReady, "processDebugTensor callback stub no-throw scaffold is not ready.");
        AddBlockerIfFalse(reasons, !precheck.VTableAddressExposed, "native vtable address would be exposed.");
        AddBlockerIfFalse(reasons, !precheck.VTablePointerProduced, "native vtable pointer would be produced.");
        return string.Join(" ", reasons);
    }

    private static string BuildRuntimeProofReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canEnableSetDebugListenerNonNull,
        bool canInstallNativeVTable,
        bool canCallProcessDebugTensorRuntime,
        bool canPromoteRealCallbackRuntime)
    {
        if (canPromoteRealCallbackRuntime)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, canEnableSetDebugListenerNonNull, "non-null DebugListener attach is not enabled.");
        AddBlockerIfFalse(reasons, canInstallNativeVTable, "native IDebugListener vtable cannot be installed.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorLifetimeReady, "borrowed debug tensor metadata lifetime is not ready.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data lifetime is not ready.");
        AddBlockerIfFalse(reasons, precheck.ProcessDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlockerIfFalse(reasons, canCallProcessDebugTensorRuntime, "runtime callback invocation cannot be attempted.");
        AddBlockerIfFalse(reasons, precheck.FullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not produced real-callback-runtime evidence.");
        AddBlockerIfFalse(reasons, precheck.CanAttemptRuntimeProof, "runtime proof precheck is still blocked.");
        AddBlockerIfFalse(reasons, precheck.RealCallbackRuntime, "gate/precheck evidence is not real callback runtime evidence.");
        AddBlockerIfFalse(reasons, precheck.IsRealCallbackRuntimeProof, "gate/precheck evidence is not promotable runtime proof.");
        return string.Join(" ", reasons);
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
/// Reports pointer-free readiness for attempting DebugListener real callback runtime proof work.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerRuntimeProofAttemptPreflightResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerRuntimeProofAttemptPreflightResult(
        TensorRtApiLine line,
        bool nativeAttachEntryLocated,
        bool nonNullAttachStillDisabled,
        bool nativeOwnerLifecycleReady,
        bool canImplementNativeAttach,
        bool nativeVTableReady,
        bool noThrowVTableDesignReady,
        bool nativeVTableTrampolineReady,
        bool callbackExceptionCaptureReady,
        bool callbackStatusMappingReady,
        bool callbackInFlightAccountingReady,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        bool precheckCanAttemptRuntimeProof,
        bool canEnableSetDebugListenerNonNull,
        bool canInstallNativeVTable,
        bool canCallProcessDebugTensorRuntime,
        bool canPromoteRealCallbackRuntime,
        string reasonNonNullAttachStillBlocked,
        string reasonNativeVTableStillBlocked,
        string reasonRuntimeProofStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NonNullAttachStillDisabled = nonNullAttachStillDisabled;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        CanImplementNativeAttach = canImplementNativeAttach;
        NativeVTableReady = nativeVTableReady;
        NoThrowVTableDesignReady = noThrowVTableDesignReady;
        NativeVTableTrampolineReady = nativeVTableTrampolineReady;
        CallbackExceptionCaptureReady = callbackExceptionCaptureReady;
        CallbackStatusMappingReady = callbackStatusMappingReady;
        CallbackInFlightAccountingReady = callbackInFlightAccountingReady;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        PrecheckCanAttemptRuntimeProof = precheckCanAttemptRuntimeProof;
        CanEnableSetDebugListenerNonNull = canEnableSetDebugListenerNonNull;
        CanInstallNativeVTable = canInstallNativeVTable;
        CanCallProcessDebugTensorRuntime = canCallProcessDebugTensorRuntime;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        ReasonNonNullAttachStillBlocked = reasonNonNullAttachStillBlocked ?? string.Empty;
        ReasonNativeVTableStillBlocked = reasonNativeVTableStillBlocked ?? string.Empty;
        ReasonRuntimeProofStillBlocked = reasonRuntimeProofStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-runtime-proof-attempt-preflight";

    /// <summary>Gets the callback kind represented by this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "runtime-proof-attempt-preflight";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether non-null attach remains deliberately disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled { get; }

    /// <summary>Gets whether native owner lifetime and release ordering are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native attach implementation prerequisites are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach { get; }

    /// <summary>Gets whether the native owner and vtable are ready as a unit. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableReady { get; }

    /// <summary>Gets whether the native vtable no-throw design is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableDesignReady { get; }

    /// <summary>Gets whether a native IDebugListener vtable trampoline exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableTrampolineReady { get; }

    /// <summary>Gets whether native callback exception capture is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingReady { get; }

    /// <summary>Gets whether callback in-flight accounting is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightAccountingReady { get; }

    /// <summary>Gets whether a native vtable address would be exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer would be produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor metadata lifetime is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data lifetime is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is implemented. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the consumed runtime proof precheck allows a runtime proof attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PrecheckCanAttemptRuntimeProof { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull { get; }

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime { get; }

    /// <summary>Gets whether real-callback-runtime evidence can be promoted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets why non-null attach remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNonNullAttachStillBlocked { get; }

    /// <summary>Gets why native vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeVTableStillBlocked { get; }

    /// <summary>Gets why runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonRuntimeProofStillBlocked { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether real runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the copied list of prerequisites that still block a proof attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the proof-attempt preflight status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime ? "runtime-proof-attempt-ready" : "runtime-proof-attempt-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-runtime-proof-attempt-preflight; RuntimeEvidenceKind=runtime-proof-attempt-preflight; " +
        "RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "NativeVTableReady=" + NativeVTableReady + "; " +
        "NoThrowVTableDesignReady=" + NoThrowVTableDesignReady + "; " +
        "NativeVTableTrampolineReady=" + NativeVTableTrampolineReady + "; " +
        "CallbackExceptionCaptureReady=" + CallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingReady=" + CallbackStatusMappingReady + "; " +
        "CallbackInFlightAccountingReady=" + CallbackInFlightAccountingReady + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "FullPackageConsumerRuntimeEvidenceReady=" + FullPackageConsumerRuntimeEvidenceReady + "; " +
        "PrecheckCanAttemptRuntimeProof=" + PrecheckCanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + "; " +
        "ReasonNonNullAttachStillBlocked=" + ReasonNonNullAttachStillBlocked + "; " +
        "ReasonNativeVTableStillBlocked=" + ReasonNativeVTableStillBlocked + "; " +
        "ReasonRuntimeProofStillBlocked=" + ReasonRuntimeProofStillBlocked + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={CanEnableSetDebugListenerNonNull}:proof={IsRealCallbackRuntimeProof}";
    }
}
