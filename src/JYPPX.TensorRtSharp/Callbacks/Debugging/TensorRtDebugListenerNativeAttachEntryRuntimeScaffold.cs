using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach entry runtime scaffold readiness.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This scaffold records the intended attach entry parameter shape, version guard, no-throw boundary, and ownership
/// diagnostics. It does not allocate a native owner, does not call <c>setDebugListener(non-null)</c>, and is not proof
/// that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachEntryRuntimeScaffold
{
    /// <summary>
    /// Evaluates native attach entry runtime scaffold readiness from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach entry runtime scaffold result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate);
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight);
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate);
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate);
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate);
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun =
            TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate,
                nativeDetachBeforeReleaseDesignGate);
        return Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleDryRun);
    }

    /// <summary>
    /// Evaluates native attach entry runtime scaffold readiness from copied owner and native owner lifecycle dry-run evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="nativeOwnerLifecycleDryRun">The copied native owner lifecycle dry-run result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach entry runtime scaffold result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun)
    {
        bool nativeOwnerLifecycleDryRunReady = nativeOwnerLifecycleDryRun.DryRunReady;
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool attachEntryParameterShapeReady =
            nativeOwnerLifecycleDryRunReady &&
            lineSupportsDebugListener &&
            nativeOwnerLifecycleDryRun.ManagedCallbackKeepAliveDesignReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorPointerEscapeBlocked;
        bool attachEntryVersionGuardReady =
            nativeOwnerLifecycleDryRunReady &&
            lineSupportsDebugListener;
        bool attachEntryNoThrowBoundaryReady =
            nativeOwnerLifecycleDryRunReady &&
            attachEntryParameterShapeReady;
        bool attachEntryOwnershipDiagnosticsReady =
            nativeOwnerLifecycleDryRunReady &&
            attachEntryParameterShapeReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorPointerEscapeBlocked;

        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRunReady, "debug-listener-native-owner-lifecycle-dry-run is not ready for native attach entry runtime scaffold evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, attachEntryParameterShapeReady, "native DebugListener attach entry parameter shape scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryVersionGuardReady, "native DebugListener attach entry TensorRT version guard scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryNoThrowBoundaryReady, "native DebugListener attach entry no-throw/status mapping scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryOwnershipDiagnosticsReady, "native DebugListener attach entry ownership diagnostics scaffold is incomplete.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.StableNativeOwnerIdentityReady, "native DebugListener stable owner identity is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NoThrowNativeDestructorReady, "native DebugListener owner no-throw destructor is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.ProcessDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener attach entry scaffold evidence.");

        foreach (string blocker in nativeOwnerLifecycleDryRun.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult(
            ownerDesignSnapshot.Line,
            nativeOwnerLifecycleDryRun.OwnerId,
            nativeOwnerLifecycleDryRun.LastStatus,
            nativeOwnerLifecycleDryRunReady,
            nativeOwnerLifecycleDryRun.NativeAttachEntryLocated,
            nativeOwnerLifecycleDryRun.NativeDetachEntryLocated,
            attachEntryParameterShapeReady,
            attachEntryVersionGuardReady,
            attachEntryNoThrowBoundaryReady,
            attachEntryOwnershipDiagnosticsReady,
            nativeOwnerLifecycleDryRun.StableNativeOwnerIdentityReady,
            nativeOwnerLifecycleDryRun.NativeOwnerNonCopyableReady,
            nativeOwnerLifecycleDryRun.NoThrowNativeDestructorReady,
            nativeOwnerLifecycleDryRun.NativeOwnerLifecycleReady,
            nativeOwnerLifecycleDryRun.ProcessDebugTensorRuntimeReady,
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
/// Reports copied DebugListener native attach entry runtime scaffold diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is scaffold evidence. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real native attach bridge and full
/// package consumer callback runtime evidence exist.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleDryRunReady,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool attachEntryParameterShapeReady,
        bool attachEntryVersionGuardReady,
        bool attachEntryNoThrowBoundaryReady,
        bool attachEntryOwnershipDiagnosticsReady,
        bool stableNativeOwnerIdentityReady,
        bool nativeOwnerNonCopyableReady,
        bool noThrowNativeDestructorReady,
        bool nativeOwnerLifecycleReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleDryRunReady = nativeOwnerLifecycleDryRunReady;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        AttachEntryParameterShapeReady = attachEntryParameterShapeReady;
        AttachEntryVersionGuardReady = attachEntryVersionGuardReady;
        AttachEntryNoThrowBoundaryReady = attachEntryNoThrowBoundaryReady;
        AttachEntryOwnershipDiagnosticsReady = attachEntryOwnershipDiagnosticsReady;
        StableNativeOwnerIdentityReady = stableNativeOwnerIdentityReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this scaffold. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-native-attach-entry-runtime-scaffold";

    /// <summary>Gets the callback kind represented by this scaffold. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "scaffold";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether native owner lifecycle dry-run evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleDryRunReady { get; }

    /// <summary>Gets whether a line-specific native non-null DebugListener attach entry exists. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a line-specific native detach/clear entry exists. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether the attach entry parameter shape has been scaffolded without exposing pointers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachEntryParameterShapeReady { get; }

    /// <summary>Gets whether TensorRT 10/11 attach entry version guard shape has been scaffolded. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachEntryVersionGuardReady { get; }

    /// <summary>Gets whether the attach entry no-throw/status mapping boundary has been scaffolded. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachEntryNoThrowBoundaryReady { get; }

    /// <summary>Gets whether attach entry ownership diagnostics have been scaffolded. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachEntryOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether stable native owner identity is implemented. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool StableNativeOwnerIdentityReady { get; }

    /// <summary>Gets whether native owner storage is non-copyable. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether the native owner destructor is no-throw. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether native owner lifecycle is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether copied scaffold evidence is ready for native attach entry implementation work. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RuntimeScaffoldReady =>
        NativeOwnerLifecycleDryRunReady &&
        NativeDetachEntryLocated &&
        AttachEntryParameterShapeReady &&
        AttachEntryVersionGuardReady &&
        AttachEntryNoThrowBoundaryReady &&
        AttachEntryOwnershipDiagnosticsReady &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved owner lifecycle blockers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanImplementNativeAttach =>
        RuntimeScaffoldReady &&
        NativeAttachEntryLocated &&
        StableNativeOwnerIdentityReady &&
        NativeOwnerNonCopyableReady &&
        NoThrowNativeDestructorReady &&
        NativeOwnerLifecycleReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the scaffold status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => RuntimeScaffoldReady ? "scaffold-ready" : "scaffold-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-native-attach-entry-runtime-scaffold; RuntimeEvidenceKind=scaffold; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; RuntimeScaffoldReady=" + RuntimeScaffoldReady + "; " +
        "NativeOwnerLifecycleDryRunReady=" + NativeOwnerLifecycleDryRunReady + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "AttachEntryParameterShapeReady=" + AttachEntryParameterShapeReady + "; " +
        "AttachEntryVersionGuardReady=" + AttachEntryVersionGuardReady + "; " +
        "AttachEntryNoThrowBoundaryReady=" + AttachEntryNoThrowBoundaryReady + "; " +
        "AttachEntryOwnershipDiagnosticsReady=" + AttachEntryOwnershipDiagnosticsReady + "; " +
        "StableNativeOwnerIdentityReady=" + StableNativeOwnerIdentityReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={NativeAttachEntryLocated}:proof={IsRealCallbackRuntimeProof}";
    }
}
