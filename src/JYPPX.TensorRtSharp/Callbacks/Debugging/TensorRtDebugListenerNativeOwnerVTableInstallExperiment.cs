using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

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
