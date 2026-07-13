using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach entry minimal-safety evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This evidence reports that a source-visible, no-throw, version-guarded attach entry shape is ready for TensorRT 10
/// and TensorRT 11 review. It does not enable <c>setDebugListener(non-null)</c>, does not expose a native owner pointer,
/// and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachEntryMinimalSafety
{
    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, runtimeScaffold, lifecycleGate);
    }

    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from copied owner and runtime scaffold evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimeScaffold">The copied native attach entry runtime scaffold result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, runtimeScaffold, lifecycleGate);
    }

    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from copied owner, runtime scaffold, and lifecycle gate evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimeScaffold">The copied native attach entry runtime scaffold result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="lifecycleGate">The copied native owner lifecycle gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool runtimeScaffoldReady = runtimeScaffold.RuntimeScaffoldReady;
        bool lifecycleGateReady = lifecycleGate.LifecycleGateReady;
        bool lifecyclePointerFree =
            !lifecycleGate.NativeOwnerAddressExposed &&
            !lifecycleGate.NativeOwnerPointerProduced &&
            !lifecycleGate.DestructorAddressExposed &&
            !lifecycleGate.DestructorPointerProduced &&
            !lifecycleGate.LifecycleAddressExposed &&
            !lifecycleGate.LifecyclePointerProduced;
        bool attachEntryParameterShapeReady =
            lineSupportsDebugListener &&
            runtimeScaffoldReady &&
            runtimeScaffold.AttachEntryParameterShapeReady;
        bool attachEntryVersionGuardReady = lineSupportsDebugListener;
        bool attachEntryNoThrowReady =
            runtimeScaffoldReady &&
            lifecycleGateReady &&
            runtimeScaffold.AttachEntryNoThrowBoundaryReady;
        bool attachEntryOwnershipDiagnosticsReady =
            runtimeScaffoldReady &&
            lifecycleGateReady &&
            lifecyclePointerFree &&
            runtimeScaffold.AttachEntryOwnershipDiagnosticsReady &&
            lifecycleGate.NativeOwnerNonCopyableReady &&
            lifecycleGate.NoThrowNativeDestructorReady;
        bool nativeAttachEntryLocated =
            attachEntryParameterShapeReady &&
            attachEntryVersionGuardReady &&
            attachEntryNoThrowReady &&
            attachEntryOwnershipDiagnosticsReady;

        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeAttachWouldBeBlocked = true;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, runtimeScaffoldReady, "debug-listener-native-attach-entry-runtime-scaffold is not ready for minimal safety evaluation.");
        AddBlockerIfFalse(blockers, lifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for minimal safety evaluation.");
        AddBlockerIfFalse(blockers, lifecyclePointerFree, "native DebugListener lifecycle diagnostics are not pointer-free.");
        AddBlockerIfFalse(blockers, attachEntryParameterShapeReady, "native DebugListener attach entry parameter shape is not ready.");
        AddBlockerIfFalse(blockers, attachEntryVersionGuardReady, "native DebugListener attach entry TensorRT 10/11 version guard is not ready.");
        AddBlockerIfFalse(blockers, attachEntryNoThrowReady, "native DebugListener attach entry no-throw boundary is not ready.");
        AddBlockerIfFalse(blockers, attachEntryOwnershipDiagnosticsReady, "native DebugListener attach entry ownership diagnostics are not ready.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "source-visible native DebugListener attach entry minimal-safety shape is not located.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled during minimal-safety evaluation.");
        AddBlockerIfFalse(blockers, nativeAttachWouldBeBlocked, "native attach would not be blocked by the minimal-safety gate.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime attach entry evidence.");

        foreach (string blocker in runtimeScaffold.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in lifecycleGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        string reasonNativeAttachStillBlocked = BuildNativeAttachBlockedReason(
            setDebugListenerNonNullEnabled,
            nativeAttachWouldBeBlocked,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady);

        return new TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult(
            ownerDesignSnapshot.Line,
            runtimeScaffold.OwnerId,
            ownerDesignSnapshot.LastStatus,
            runtimeScaffoldReady,
            lifecycleGateReady,
            lifecyclePointerFree,
            nativeAttachEntryLocated,
            runtimeScaffold.NativeDetachEntryLocated,
            attachEntryParameterShapeReady,
            attachEntryNoThrowReady,
            attachEntryVersionGuardReady,
            attachEntryOwnershipDiagnosticsReady,
            setDebugListenerNonNullEnabled,
            nativeAttachWouldBeBlocked,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            reasonNativeAttachStillBlocked,
            blockers.ToArray());
    }

    private static string BuildNativeAttachBlockedReason(
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachWouldBeBlocked,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, setDebugListenerNonNullEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, !nativeAttachWouldBeBlocked, "native attach is deliberately blocked by minimal-safety evidence.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlockerIfFalse(reasons, fullPackageConsumerRuntimeEvidenceReady, "full package consumer real-callback-runtime evidence is not present.");
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
/// Reports copied DebugListener native attach entry minimal-safety diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is minimal-safety evidence. <see cref="NativeAttachEntryLocated"/> means the source-visible attach entry
/// shape is located for this result only; <see cref="SetDebugListenerNonNullEnabled"/> remains <see langword="false"/>,
/// and this result is not real callback runtime proof.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool runtimeScaffoldReady,
        bool lifecycleGateReady,
        bool lifecyclePointerFree,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool attachEntryParameterShapeReady,
        bool attachEntryNoThrowReady,
        bool attachEntryVersionGuardReady,
        bool attachEntryOwnershipDiagnosticsReady,
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachWouldBeBlocked,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string reasonNativeAttachStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        RuntimeScaffoldReady = runtimeScaffoldReady;
        LifecycleGateReady = lifecycleGateReady;
        LifecyclePointerFree = lifecyclePointerFree;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        AttachEntryParameterShapeReady = attachEntryParameterShapeReady;
        AttachEntryNoThrowReady = attachEntryNoThrowReady;
        AttachEntryVersionGuardReady = attachEntryVersionGuardReady;
        AttachEntryOwnershipDiagnosticsReady = attachEntryOwnershipDiagnosticsReady;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        NativeAttachWouldBeBlocked = nativeAttachWouldBeBlocked;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        ReasonNativeAttachStillBlocked = reasonNativeAttachStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this minimal-safety result. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-attach-entry-minimal-safety";

    /// <summary>Gets the callback kind represented by this result. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "minimal-safety";

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

    /// <summary>Gets whether runtime scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeScaffoldReady { get; }

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool LifecycleGateReady { get; }

    /// <summary>Gets whether lifecycle diagnostics remain pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool LifecyclePointerFree { get; }

    /// <summary>Gets whether a source-visible native non-null attach entry shape is located for this minimal-safety result. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native detach/clear entry is available. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether attach entry parameter shape evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryParameterShapeReady { get; }

    /// <summary>Gets whether attach entry no-throw boundary evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryNoThrowReady { get; }

    /// <summary>Gets whether TensorRT 10/11 attach entry version guard evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryVersionGuardReady { get; }

    /// <summary>Gets whether attach entry ownership diagnostics evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for minimal-safety evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null attach remains deliberately disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled => !SetDebugListenerNonNullEnabled;

    /// <summary>Gets whether a native attach attempt would still be blocked by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachWouldBeBlocked { get; }

    /// <summary>Gets whether minimal-safety evidence is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool MinimalSafetyReady =>
        RuntimeScaffoldReady &&
        LifecycleGateReady &&
        LifecyclePointerFree &&
        NativeAttachEntryLocated &&
        NativeDetachEntryLocated &&
        AttachEntryParameterShapeReady &&
        AttachEntryNoThrowReady &&
        AttachEntryVersionGuardReady &&
        AttachEntryOwnershipDiagnosticsReady &&
        NonNullAttachStillDisabled &&
        NativeAttachWouldBeBlocked &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented from this result alone. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach => false;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why native attach remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeAttachStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the minimal-safety status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => MinimalSafetyReady ? "minimal-safety-ready" : "minimal-safety-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-attach-entry-minimal-safety; RuntimeEvidenceKind=minimal-safety; " +
        "RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False; MinimalSafetyReady=" + MinimalSafetyReady + "; " +
        "RuntimeScaffoldReady=" + RuntimeScaffoldReady + "; " +
        "LifecycleGateReady=" + LifecycleGateReady + "; " +
        "LifecyclePointerFree=" + LifecyclePointerFree + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "AttachEntryParameterShapeReady=" + AttachEntryParameterShapeReady + "; " +
        "AttachEntryNoThrowReady=" + AttachEntryNoThrowReady + "; " +
        "AttachEntryVersionGuardReady=" + AttachEntryVersionGuardReady + "; " +
        "AttachEntryOwnershipDiagnosticsReady=" + AttachEntryOwnershipDiagnosticsReady + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "NativeAttachWouldBeBlocked=" + NativeAttachWouldBeBlocked + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonNativeAttachStillBlocked=" + ReasonNativeAttachStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={NativeAttachEntryLocated}:proof={IsRealCallbackRuntimeProof}";
    }
}
