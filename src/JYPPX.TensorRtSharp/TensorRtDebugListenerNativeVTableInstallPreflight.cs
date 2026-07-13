using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

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

/// <summary>
/// Reports copied DebugListener native vtable install preflight diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeVTableInstallPreflightResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeVTableInstallPreflightResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool nativeAttachBridgeShapeGateReady,
        bool nativeNoThrowVTableScaffoldGateReady,
        bool borrowedDebugTensorMetadataGateReady,
        bool vTableInstallShapeReady,
        bool vTableInstallVersionGuardReady,
        bool vTableInstallNoThrowBoundaryReady,
        bool vTableInstallOwnershipDiagnosticsReady,
        bool vTableInstallPointerFree,
        bool attachBridgeSetDebugListenerNonNullEnabled,
        bool setDebugListenerNonNullEnabled,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorDataPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool nativeVTableInstallPreflightReady,
        bool nativeVTableInstalled,
        bool nativeVTableInstallRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string reasonNativeVTableInstallStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        NativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGateReady;
        BorrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataGateReady;
        VTableInstallShapeReady = vTableInstallShapeReady;
        VTableInstallVersionGuardReady = vTableInstallVersionGuardReady;
        VTableInstallNoThrowBoundaryReady = vTableInstallNoThrowBoundaryReady;
        VTableInstallOwnershipDiagnosticsReady = vTableInstallOwnershipDiagnosticsReady;
        VTableInstallPointerFree = vTableInstallPointerFree;
        AttachBridgeSetDebugListenerNonNullEnabled = attachBridgeSetDebugListenerNonNullEnabled;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorDataPointerEscapeBlocked = borrowedDebugTensorDataPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        NativeVTableInstallPreflightReady = nativeVTableInstallPreflightReady;
        NativeVTableInstalled = nativeVTableInstalled;
        NativeVTableInstallRuntimeReady = nativeVTableInstallRuntimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        ReasonNativeVTableInstallStillBlocked = reasonNativeVTableInstallStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-vtable-install-preflight";

    /// <summary>Gets the callback kind represented by this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "native-vtable-install-preflight";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether native attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether copied borrowed debug tensor metadata gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataGateReady { get; }

    /// <summary>Gets whether the install parameter shape preflight is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallShapeReady { get; }

    /// <summary>Gets whether the install path is guarded by TensorRT 10/11 support. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallVersionGuardReady { get; }

    /// <summary>Gets whether the install path has a no-throw boundary preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallNoThrowBoundaryReady { get; }

    /// <summary>Gets whether install ownership diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether all install diagnostics remain pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallPointerFree { get; }

    /// <summary>Gets whether the consumed attach bridge gate has non-null setDebugListener enabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeSetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null attach remains deliberately disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled => !SetDebugListenerNonNullEnabled;

    /// <summary>Gets whether a native vtable address is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer is produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor data pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether copied native vtable install preflight evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallPreflightReady { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether native vtable install runtime is enabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallRuntimeReady { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull => false;

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable => false;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why native vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeVTableInstallStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the native vtable install preflight status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => NativeVTableInstallPreflightReady ? "native-vtable-install-preflight-ready" : "native-vtable-install-preflight-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-vtable-install-preflight; RuntimeEvidenceKind=native-vtable-install-preflight; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeVTableInstallPreflightReady=" + NativeVTableInstallPreflightReady + "; " +
        "NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "NativeNoThrowVTableScaffoldGateReady=" + NativeNoThrowVTableScaffoldGateReady + "; " +
        "BorrowedDebugTensorMetadataGateReady=" + BorrowedDebugTensorMetadataGateReady + "; " +
        "VTableInstallShapeReady=" + VTableInstallShapeReady + "; " +
        "VTableInstallVersionGuardReady=" + VTableInstallVersionGuardReady + "; " +
        "VTableInstallNoThrowBoundaryReady=" + VTableInstallNoThrowBoundaryReady + "; " +
        "VTableInstallOwnershipDiagnosticsReady=" + VTableInstallOwnershipDiagnosticsReady + "; " +
        "VTableInstallPointerFree=" + VTableInstallPointerFree + "; " +
        "AttachBridgeSetDebugListenerNonNullEnabled=" + AttachBridgeSetDebugListenerNonNullEnabled + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorDataPointerEscapeBlocked=" + BorrowedDebugTensorDataPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "NativeVTableInstallRuntimeReady=" + NativeVTableInstallRuntimeReady + "; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "FullPackageConsumerRuntimeEvidenceReady=" + FullPackageConsumerRuntimeEvidenceReady + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonNativeVTableInstallStillBlocked=" + ReasonNativeVTableInstallStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:install={NativeVTableInstalled}:proof={IsRealCallbackRuntimeProof}";
    }
}
