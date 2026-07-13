using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach bridge shape evidence before non-null attach is enabled.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate reports source-visible attach bridge parameter shape, version guard, no-throw boundary, and ownership
/// diagnostics only. It does not call <c>setDebugListener(non-null)</c>, does not create a native owner, and is not
/// proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachBridgeShapeGate
{
    /// <summary>
    /// Evaluates attach bridge shape evidence from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach bridge shape gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachBridgeShapeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleGate);
    }

    /// <summary>
    /// Evaluates attach bridge shape evidence from copied owner and lifecycle gate evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="nativeOwnerLifecycleGate">The copied native owner lifecycle gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach bridge shape gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachBridgeShapeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool nativeOwnerLifecycleGateReady = nativeOwnerLifecycleGate.LifecycleGateReady;
        bool attachBridgeShapeReady =
            nativeOwnerLifecycleGateReady &&
            lineSupportsDebugListener &&
            nativeOwnerLifecycleGate.NativeDetachEntryLocated &&
            nativeOwnerLifecycleGate.ManagedDisposeSnapshotReady &&
            nativeOwnerLifecycleGate.BecausePointerFree();
        bool attachBridgeNoThrowBoundaryReady = attachBridgeShapeReady;
        bool attachBridgeVersionGuardReady = nativeOwnerLifecycleGateReady && lineSupportsDebugListener;
        bool attachBridgeOwnershipDiagnosticsReady =
            attachBridgeShapeReady &&
            nativeOwnerLifecycleGate.NativeOwnerNonCopyableReady &&
            nativeOwnerLifecycleGate.NoThrowNativeDestructorReady;
        const bool attachBridgePointerFree = true;
        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeAttachEntryLocated = false;
        const bool nativeVTableDesignReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for attach bridge shape evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGate.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, attachBridgeShapeReady, "native DebugListener attach bridge shape scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeNoThrowBoundaryReady, "native DebugListener attach bridge no-throw boundary scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeVersionGuardReady, "native DebugListener attach bridge TensorRT version guard scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeOwnershipDiagnosticsReady, "native DebugListener attach bridge ownership diagnostics scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgePointerFree, "native DebugListener attach bridge shape gate exposes a native pointer.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled before native attach bridge proof.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGate.NativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not complete beyond source-visible scaffold evidence.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable implementation is not complete.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime attach bridge evidence.");

        foreach (string blocker in nativeOwnerLifecycleGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachBridgeShapeGateResult(
            ownerDesignSnapshot.Line,
            nativeOwnerLifecycleGate.OwnerId,
            nativeOwnerLifecycleGate.LastStatus,
            nativeOwnerLifecycleGateReady,
            attachBridgeShapeReady,
            attachBridgeNoThrowBoundaryReady,
            attachBridgeVersionGuardReady,
            attachBridgeOwnershipDiagnosticsReady,
            attachBridgePointerFree,
            setDebugListenerNonNullEnabled,
            nativeAttachEntryLocated,
            nativeOwnerLifecycleGate.NativeDetachEntryLocated,
            nativeOwnerLifecycleGate.NativeOwnerLifecycleReady,
            nativeVTableDesignReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }

    private static bool BecausePointerFree(this TensorRtDebugListenerNativeOwnerLifecycleGateResult result)
    {
        return !result.NativeOwnerAddressExposed &&
            !result.NativeOwnerPointerProduced &&
            !result.DestructorAddressExposed &&
            !result.DestructorPointerProduced &&
            !result.LifecycleAddressExposed &&
            !result.LifecyclePointerProduced;
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
/// Reports copied DebugListener native attach bridge shape diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is an attach bridge shape gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real non-null attach bridge,
/// complete native vtable, and full package consumer callback runtime evidence exist.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeAttachBridgeShapeGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeAttachBridgeShapeGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool attachBridgeShapeReady,
        bool attachBridgeNoThrowBoundaryReady,
        bool attachBridgeVersionGuardReady,
        bool attachBridgeOwnershipDiagnosticsReady,
        bool attachBridgePointerFree,
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool nativeOwnerLifecycleReady,
        bool nativeVTableDesignReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        AttachBridgeShapeReady = attachBridgeShapeReady;
        AttachBridgeNoThrowBoundaryReady = attachBridgeNoThrowBoundaryReady;
        AttachBridgeVersionGuardReady = attachBridgeVersionGuardReady;
        AttachBridgeOwnershipDiagnosticsReady = attachBridgeOwnershipDiagnosticsReady;
        AttachBridgePointerFree = attachBridgePointerFree;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-native-attach-bridge-shape-gate";

    /// <summary>Gets the callback kind represented by this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "attach-bridge-shape-gate";

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

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether source-visible attach bridge shape evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeShapeReady { get; }

    /// <summary>Gets whether source-visible attach bridge no-throw boundary evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeNoThrowBoundaryReady { get; }

    /// <summary>Gets whether TensorRT 10/11 attach bridge version guard evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeVersionGuardReady { get; }

    /// <summary>Gets whether attach bridge ownership diagnostics evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether attach bridge diagnostics remain pointer-free. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgePointerFree { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null attach is still disabled. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NonNullAttachStillDisabled => !SetDebugListenerNonNullEnabled;

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether native owner lifecycle implementation is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native IDebugListener vtable implementation is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether copied attach bridge shape evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeShapeGateReady =>
        NativeOwnerLifecycleGateReady &&
        AttachBridgeShapeReady &&
        AttachBridgeNoThrowBoundaryReady &&
        AttachBridgeVersionGuardReady &&
        AttachBridgeOwnershipDiagnosticsReady &&
        AttachBridgePointerFree &&
        NonNullAttachStillDisabled &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved blockers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanImplementNativeAttach =>
        AttachBridgeShapeGateReady &&
        NativeAttachEntryLocated &&
        NativeOwnerLifecycleReady &&
        NativeVTableDesignReady;

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

    /// <summary>Gets the attach bridge shape gate status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => AttachBridgeShapeGateReady ? "attach-bridge-shape-gate-ready" : "attach-bridge-shape-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-native-attach-bridge-shape-gate; RuntimeEvidenceKind=attach-bridge-shape-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; AttachBridgeShapeGateReady=" + AttachBridgeShapeGateReady + "; " +
        "NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "AttachBridgeShapeReady=" + AttachBridgeShapeReady + "; " +
        "AttachBridgeNoThrowBoundaryReady=" + AttachBridgeNoThrowBoundaryReady + "; " +
        "AttachBridgeVersionGuardReady=" + AttachBridgeVersionGuardReady + "; " +
        "AttachBridgeOwnershipDiagnosticsReady=" + AttachBridgeOwnershipDiagnosticsReady + "; " +
        "AttachBridgePointerFree=" + AttachBridgePointerFree + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
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
