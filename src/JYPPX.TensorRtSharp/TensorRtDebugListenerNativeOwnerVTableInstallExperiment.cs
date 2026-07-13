using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates disabled-by-default DebugListener native owner/vtable install experiment evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This experiment reports source-visible native owner/vtable install shape, rollback, detach-before-release, and
/// failure/status mapping diagnostics. It does not enable <c>setDebugListener(non-null)</c>, does not install a native
/// <c>IDebugListener</c> vtable, and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerVTableInstallExperiment
{
    /// <summary>
    /// Evaluates native owner/vtable install experiment evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner/vtable install experiment result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleGate);
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot, nativeAttachBridgeShapeGate);
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(
                ownerDesignSnapshot,
                nativeAttachBridgeShapeGate,
                exceptionStatusMappingGate,
                inFlightAccountingGate);
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult borrowedDebugTensorMetadataRuntimeGate =
            TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNativeVTableInstallPreflightResult nativeVTableInstallPreflight =
            TensorRtDebugListenerNativeVTableInstallPreflight.Evaluate(
                ownerDesignSnapshot,
                nativeOwnerLifecycleGate,
                nativeAttachBridgeShapeGate,
                nativeNoThrowVTableScaffoldGate,
                borrowedDebugTensorMetadataRuntimeGate);

        return Evaluate(
            ownerDesignSnapshot,
            nativeOwnerLifecycleGate,
            nativeAttachBridgeShapeGate,
            nativeNoThrowVTableScaffoldGate,
            borrowedDebugTensorMetadataRuntimeGate,
            nativeVTableInstallPreflight);
    }

    /// <summary>
    /// Evaluates native owner/vtable install experiment evidence from copied owner and prerequisite gate results.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeOwnerLifecycleGate">The copied native owner lifecycle gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeAttachBridgeShapeGate">The copied native attach bridge shape gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeNoThrowVTableScaffoldGate">The copied native no-throw vtable scaffold gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedDebugTensorMetadataRuntimeGate">The copied borrowed debug tensor metadata gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeVTableInstallPreflight">The copied native vtable install preflight result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner/vtable install experiment result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate,
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate,
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult borrowedDebugTensorMetadataRuntimeGate,
        TensorRtDebugListenerNativeVTableInstallPreflightResult nativeVTableInstallPreflight)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool nativeOwnerLifecycleGateReady = nativeOwnerLifecycleGate.LifecycleGateReady;
        bool nativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool nativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGate.VTableScaffoldGateReady;
        bool borrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataRuntimeGate.MetadataGateReady;
        bool nativeVTableInstallPreflightReady = nativeVTableInstallPreflight.NativeVTableInstallPreflightReady;
        bool experimentShapeReady =
            lineSupportsDebugListener &&
            nativeOwnerLifecycleGateReady &&
            nativeAttachBridgeShapeGateReady &&
            nativeNoThrowVTableScaffoldGateReady &&
            borrowedDebugTensorMetadataGateReady &&
            nativeVTableInstallPreflightReady;

        const bool nonNullAttachEnabled = false;
        const bool runtimeProofEnabled = false;
        const bool nativeVTableInstallAttempted = false;
        const bool nativeVTableInstalled = false;
        const bool processDebugTensorRuntimeReady = false;
        bool rollbackReady =
            nativeOwnerLifecycleGate.DisposeIdempotencyGateReady &&
            nativeOwnerLifecycleGate.ReleaseHookOrderingGateReady &&
            nativeOwnerLifecycleGate.CallbackStateUnpinAfterDetachGateReady &&
            nativeOwnerLifecycleGate.DelegateUnpinAfterDetachGateReady;
        bool detachBeforeReleaseReady =
            nativeOwnerLifecycleGate.NativeDetachEntryLocated &&
            nativeOwnerLifecycleGate.InFlightDrainGateReady &&
            nativeOwnerLifecycleGate.ManagedDisposeSnapshotReady;
        bool failureStatusMappingReady =
            nativeNoThrowVTableScaffoldGate.ExceptionEscapeBlocked &&
            nativeNoThrowVTableScaffoldGate.CallbackStatusMappingGateReady;
        bool pointerFree =
            nativeVTableInstallPreflight.VTableInstallPointerFree &&
            !nativeNoThrowVTableScaffoldGate.VTableAddressExposed &&
            !nativeNoThrowVTableScaffoldGate.VTablePointerProduced &&
            !borrowedDebugTensorMetadataRuntimeGate.DebugTensorPointerExposed &&
            !borrowedDebugTensorMetadataRuntimeGate.DebugTensorDataPointerExposed;
        bool installAttemptGuardReady =
            experimentShapeReady &&
            rollbackReady &&
            detachBeforeReleaseReady &&
            failureStatusMappingReady &&
            pointerFree &&
            !nonNullAttachEnabled &&
            !runtimeProofEnabled &&
            !nativeVTableInstallAttempted &&
            !nativeVTableInstalled;

        string reasonNativeOwnerVTableInstallStillBlocked = BuildExperimentBlockedReason(
            nonNullAttachEnabled,
            runtimeProofEnabled,
            nativeVTableInstallAttempted,
            nativeVTableInstalled,
            processDebugTensorRuntimeReady);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, nativeAttachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableScaffoldGateReady, "debug-listener-native-nothrow-vtable-scaffold-gate is not ready for native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataGateReady, "debug-listener-borrowed-debug-tensor-metadata-runtime-gate is not ready for native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, nativeVTableInstallPreflightReady, "debug-listener-native-vtable-install-preflight is not ready for native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, experimentShapeReady, "native owner/vtable install experiment shape is incomplete.");
        AddBlockerIfFalse(blockers, installAttemptGuardReady, "native owner/vtable install attempt guard is incomplete.");
        AddBlockerIfFalse(blockers, rollbackReady, "native owner/vtable install rollback diagnostics are incomplete.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "native owner/vtable install detach-before-release diagnostics are incomplete.");
        AddBlockerIfFalse(blockers, failureStatusMappingReady, "native owner/vtable install failure/status mapping diagnostics are incomplete.");
        AddBlockerIfFalse(blockers, pointerFree, "native owner/vtable install experiment is not pointer-free.");
        AddBlockerIfFalse(blockers, !nonNullAttachEnabled, "setDebugListener(non-null) unexpectedly appears enabled during native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, !runtimeProofEnabled, "runtime proof unexpectedly appears enabled during native owner/vtable install experiment.");
        AddBlockerIfFalse(blockers, !nativeVTableInstallAttempted, "native IDebugListener vtable install was unexpectedly attempted during experiment.");
        AddBlockerIfFalse(blockers, !nativeVTableInstalled, "native IDebugListener vtable unexpectedly appears installed during experiment.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlocker(blockers, reasonNativeOwnerVTableInstallStillBlocked);

        foreach (string blocker in nativeOwnerLifecycleGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in nativeAttachBridgeShapeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in nativeNoThrowVTableScaffoldGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in borrowedDebugTensorMetadataRuntimeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in nativeVTableInstallPreflight.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            nativeOwnerLifecycleGateReady,
            nativeAttachBridgeShapeGateReady,
            nativeNoThrowVTableScaffoldGateReady,
            borrowedDebugTensorMetadataGateReady,
            nativeVTableInstallPreflightReady,
            experimentShapeReady,
            installAttemptGuardReady,
            nonNullAttachEnabled,
            runtimeProofEnabled,
            nativeVTableInstallAttempted,
            nativeVTableInstalled,
            rollbackReady,
            detachBeforeReleaseReady,
            failureStatusMappingReady,
            pointerFree,
            nativeNoThrowVTableScaffoldGate.VTableAddressExposed,
            nativeNoThrowVTableScaffoldGate.VTablePointerProduced,
            borrowedDebugTensorMetadataRuntimeGate.DebugTensorPointerExposed,
            borrowedDebugTensorMetadataRuntimeGate.DebugTensorDataPointerExposed,
            processDebugTensorRuntimeReady,
            reasonNativeOwnerVTableInstallStillBlocked,
            blockers.ToArray());
    }

    private static string BuildExperimentBlockedReason(
        bool nonNullAttachEnabled,
        bool runtimeProofEnabled,
        bool nativeVTableInstallAttempted,
        bool nativeVTableInstalled,
        bool processDebugTensorRuntimeReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, nonNullAttachEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, runtimeProofEnabled, "runtime proof remains disabled by design.");
        AddBlockerIfFalse(reasons, nativeVTableInstallAttempted, "native IDebugListener vtable install attempt has not been enabled.");
        AddBlockerIfFalse(reasons, nativeVTableInstalled, "native IDebugListener vtable has not been installed.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlocker(reasons, "native-owner-vtable-install-experiment is non-proof evidence and must not be promoted to real-callback-runtime.");
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
/// Reports copied DebugListener native owner/vtable install experiment diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool nativeAttachBridgeShapeGateReady,
        bool nativeNoThrowVTableScaffoldGateReady,
        bool borrowedDebugTensorMetadataGateReady,
        bool nativeVTableInstallPreflightReady,
        bool experimentShapeReady,
        bool installAttemptGuardReady,
        bool nonNullAttachEnabled,
        bool runtimeProofEnabled,
        bool nativeVTableInstallAttempted,
        bool nativeVTableInstalled,
        bool rollbackReady,
        bool detachBeforeReleaseReady,
        bool failureStatusMappingReady,
        bool pointerFree,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool processDebugTensorRuntimeReady,
        string reasonNativeOwnerVTableInstallStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        NativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGateReady;
        BorrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataGateReady;
        NativeVTableInstallPreflightReady = nativeVTableInstallPreflightReady;
        ExperimentShapeReady = experimentShapeReady;
        InstallAttemptGuardReady = installAttemptGuardReady;
        NonNullAttachEnabled = nonNullAttachEnabled;
        RuntimeProofEnabled = runtimeProofEnabled;
        NativeVTableInstallAttempted = nativeVTableInstallAttempted;
        NativeVTableInstalled = nativeVTableInstalled;
        RollbackReady = rollbackReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        FailureStatusMappingReady = failureStatusMappingReady;
        PointerFree = pointerFree;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        ReasonNativeOwnerVTableInstallStillBlocked = reasonNativeOwnerVTableInstallStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-owner-vtable-install-experiment";

    /// <summary>Gets the callback kind represented by this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "native-owner-vtable-install-experiment";

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

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether native attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether copied borrowed debug tensor metadata gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataGateReady { get; }

    /// <summary>Gets whether native vtable install preflight evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallPreflightReady { get; }

    /// <summary>Gets whether the disabled native owner/vtable install experiment shape is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExperimentShapeReady { get; }

    /// <summary>Gets whether install attempt guards are ready and still disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InstallAttemptGuardReady { get; }

    /// <summary>Gets whether non-null setDebugListener attach is enabled. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachEnabled { get; }

    /// <summary>Gets whether runtime proof is enabled. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofEnabled { get; }

    /// <summary>Gets whether native vtable installation was attempted. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallAttempted { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether rollback diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RollbackReady { get; }

    /// <summary>Gets whether detach-before-release diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether failure/status mapping diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FailureStatusMappingReady { get; }

    /// <summary>Gets whether all experiment diagnostics remain pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFree { get; }

    /// <summary>Gets whether a native vtable address is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer is produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull => false;

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable => false;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why native owner/vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeOwnerVTableInstallStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the experiment status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => ExperimentShapeReady ? "native-owner-vtable-install-experiment-ready" : "native-owner-vtable-install-experiment-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-owner-vtable-install-experiment; RuntimeEvidenceKind=native-owner-vtable-install-experiment; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "NativeNoThrowVTableScaffoldGateReady=" + NativeNoThrowVTableScaffoldGateReady + "; " +
        "BorrowedDebugTensorMetadataGateReady=" + BorrowedDebugTensorMetadataGateReady + "; " +
        "NativeVTableInstallPreflightReady=" + NativeVTableInstallPreflightReady + "; " +
        "ExperimentShapeReady=" + ExperimentShapeReady + "; " +
        "InstallAttemptGuardReady=" + InstallAttemptGuardReady + "; " +
        "NonNullAttachEnabled=" + NonNullAttachEnabled + "; " +
        "RuntimeProofEnabled=" + RuntimeProofEnabled + "; " +
        "NativeVTableInstallAttempted=" + NativeVTableInstallAttempted + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "RollbackReady=" + RollbackReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "FailureStatusMappingReady=" + FailureStatusMappingReady + "; " +
        "PointerFree=" + PointerFree + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonNativeOwnerVTableInstallStillBlocked=" + ReasonNativeOwnerVTableInstallStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attempted={NativeVTableInstallAttempted}:proof={IsRealCallbackRuntimeProof}";
    }
}
