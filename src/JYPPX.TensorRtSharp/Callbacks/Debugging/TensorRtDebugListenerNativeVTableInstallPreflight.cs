using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native vtable install preflight evidence before installation is enabled.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This preflight reports source-visible owner lifecycle, attach bridge, no-throw vtable scaffold, and copied debug
/// tensor metadata prerequisites. It does not install a native <c>IDebugListener</c> vtable, does not enable
/// <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerNativeVTableInstallPreflight
{
    /// <summary>
    /// Evaluates native vtable install preflight evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native vtable install preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeVTableInstallPreflightResult Evaluate(
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
        return Evaluate(
            ownerDesignSnapshot,
            nativeOwnerLifecycleGate,
            nativeAttachBridgeShapeGate,
            nativeNoThrowVTableScaffoldGate,
            borrowedDebugTensorMetadataRuntimeGate);
    }

    /// <summary>
    /// Evaluates native vtable install preflight evidence from copied owner and prerequisite gate results.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerLifecycleGate">The copied native owner lifecycle gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachBridgeShapeGate">The copied native attach bridge shape gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableScaffoldGate">The copied native no-throw vtable scaffold gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedDebugTensorMetadataRuntimeGate">The copied borrowed debug tensor metadata gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native vtable install preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeVTableInstallPreflightResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate,
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate,
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult borrowedDebugTensorMetadataRuntimeGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool nativeOwnerLifecycleGateReady = nativeOwnerLifecycleGate.LifecycleGateReady;
        bool nativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool nativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGate.VTableScaffoldGateReady;
        bool borrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataRuntimeGate.MetadataGateReady;
        bool installShapeReady =
            nativeOwnerLifecycleGateReady &&
            nativeAttachBridgeShapeGateReady &&
            nativeNoThrowVTableScaffoldGateReady &&
            borrowedDebugTensorMetadataGateReady;
        bool installVersionGuardReady =
            lineSupportsDebugListener &&
            nativeAttachBridgeShapeGate.AttachBridgeVersionGuardReady;
        bool installNoThrowBoundaryReady =
            installShapeReady &&
            nativeNoThrowVTableScaffoldGate.ExceptionEscapeBlocked &&
            nativeNoThrowVTableScaffoldGate.VTableDestructorNoThrowReady &&
            nativeNoThrowVTableScaffoldGate.ProcessDebugTensorCallbackStubNoThrowReady;
        bool installOwnershipDiagnosticsReady =
            installShapeReady &&
            nativeOwnerLifecycleGate.NativeOwnerNonCopyableReady &&
            nativeOwnerLifecycleGate.ManagedDisposeSnapshotReady &&
            nativeAttachBridgeShapeGate.AttachBridgeOwnershipDiagnosticsReady;
        bool borrowedMetadataPointerFree =
            !borrowedDebugTensorMetadataRuntimeGate.DebugTensorPointerExposed &&
            !borrowedDebugTensorMetadataRuntimeGate.DebugTensorDataPointerExposed &&
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorDataPointerEscapeBlocked;
        bool installPointerFree =
            nativeAttachBridgeShapeGate.AttachBridgePointerFree &&
            !nativeNoThrowVTableScaffoldGate.VTableAddressExposed &&
            !nativeNoThrowVTableScaffoldGate.VTablePointerProduced &&
            borrowedMetadataPointerFree;
        bool nativeVTableInstallPreflightReady =
            lineSupportsDebugListener &&
            installShapeReady &&
            installVersionGuardReady &&
            installNoThrowBoundaryReady &&
            installOwnershipDiagnosticsReady &&
            installPointerFree &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok;

        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeVTableInstalled = false;
        const bool nativeVTableInstallRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;
        string reasonNativeVTableInstallStillBlocked = BuildNativeVTableInstallBlockedReason(
            setDebugListenerNonNullEnabled,
            nativeVTableInstalled,
            nativeVTableInstallRuntimeReady,
            processDebugTensorRuntimeReady,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorLifetimeReady,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorDataLifetimeReady,
            fullPackageConsumerRuntimeEvidenceReady);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for native vtable install preflight.");
        AddBlockerIfFalse(blockers, nativeAttachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for native vtable install preflight.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableScaffoldGateReady, "debug-listener-native-nothrow-vtable-scaffold-gate is not ready for native vtable install preflight.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataGateReady, "debug-listener-borrowed-debug-tensor-metadata-runtime-gate is not ready for native vtable install preflight.");
        AddBlockerIfFalse(blockers, installShapeReady, "native IDebugListener vtable install shape preflight is incomplete.");
        AddBlockerIfFalse(blockers, installVersionGuardReady, "native IDebugListener vtable install TensorRT version guard is incomplete.");
        AddBlockerIfFalse(blockers, installNoThrowBoundaryReady, "native IDebugListener vtable install no-throw boundary is incomplete.");
        AddBlockerIfFalse(blockers, installOwnershipDiagnosticsReady, "native IDebugListener vtable install ownership diagnostics are incomplete.");
        AddBlockerIfFalse(blockers, installPointerFree, "native IDebugListener vtable install preflight is not pointer-free.");
        AddBlockerIfFalse(blockers, nativeVTableInstallPreflightReady, "native IDebugListener vtable install preflight is incomplete.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled during native vtable install preflight.");
        AddBlockerIfFalse(blockers, !nativeVTableInstalled, "native IDebugListener vtable unexpectedly appears installed during preflight.");
        AddBlockerIfFalse(blockers, nativeVTableInstallRuntimeReady, "native IDebugListener vtable install runtime is not enabled.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime vtable install evidence.");
        AddBlocker(blockers, reasonNativeVTableInstallStillBlocked);

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

        return new TensorRtDebugListenerNativeVTableInstallPreflightResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            nativeOwnerLifecycleGateReady,
            nativeAttachBridgeShapeGateReady,
            nativeNoThrowVTableScaffoldGateReady,
            borrowedDebugTensorMetadataGateReady,
            installShapeReady,
            installVersionGuardReady,
            installNoThrowBoundaryReady,
            installOwnershipDiagnosticsReady,
            installPointerFree,
            nativeAttachBridgeShapeGate.SetDebugListenerNonNullEnabled,
            setDebugListenerNonNullEnabled,
            nativeNoThrowVTableScaffoldGate.VTableAddressExposed,
            nativeNoThrowVTableScaffoldGate.VTablePointerProduced,
            borrowedDebugTensorMetadataRuntimeGate.DebugTensorPointerExposed,
            borrowedDebugTensorMetadataRuntimeGate.DebugTensorDataPointerExposed,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorPointerEscapeBlocked,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorDataPointerEscapeBlocked,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorLifetimeReady,
            borrowedDebugTensorMetadataRuntimeGate.BorrowedDebugTensorDataLifetimeReady,
            nativeVTableInstallPreflightReady,
            nativeVTableInstalled,
            nativeVTableInstallRuntimeReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            reasonNativeVTableInstallStillBlocked,
            blockers.ToArray());
    }

    private static string BuildNativeVTableInstallBlockedReason(
        bool setDebugListenerNonNullEnabled,
        bool nativeVTableInstalled,
        bool nativeVTableInstallRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, setDebugListenerNonNullEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, nativeVTableInstalled, "native IDebugListener vtable has not been installed.");
        AddBlockerIfFalse(reasons, nativeVTableInstallRuntimeReady, "native IDebugListener vtable install runtime remains disabled.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlockerIfFalse(reasons, borrowedDebugTensorLifetimeReady, "borrowed debug tensor lifetime remains unproven.");
        AddBlockerIfFalse(reasons, borrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data lifetime remains unproven.");
        AddBlockerIfFalse(reasons, fullPackageConsumerRuntimeEvidenceReady, "full package consumer real-callback-runtime evidence is not present.");
        AddBlocker(reasons, "native-vtable-install-preflight is non-proof evidence and must not be promoted to real-callback-runtime.");
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
