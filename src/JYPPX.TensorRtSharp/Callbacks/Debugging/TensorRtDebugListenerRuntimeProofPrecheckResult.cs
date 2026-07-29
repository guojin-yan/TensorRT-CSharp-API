using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener runtime proof precheck diagnostics.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a runtime gate precheck only. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> are always <see langword="false"/> until a full package consumer smoke
/// produces complete <c>real-callback-runtime</c> evidence.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public readonly struct TensorRtDebugListenerRuntimeProofPrecheckResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerRuntimeProofPrecheckResult(
        TensorRtApiLine line,
        bool ownerDesignReady,
        bool debugTensorMetadataCopied,
        bool disposeReleaseReady,
        bool pointerFreeSurfaceReady,
        bool attachDetachDesignGateReady,
        bool attachControlAvailable,
        bool detachClearControlAvailable,
        bool managedOwnerStateMachineReady,
        bool stableNativeOwnerAddressReady,
        bool noThrowNativeVTableReady,
        bool exceptionToStatusMappingReady,
        bool borrowedTensorSafetyGateReady,
        bool attachVTableSafetyGateReady,
        bool nativeAttachNoThrowPreflightReady,
        bool nativeOwnerAddressDesignGateReady,
        bool nativeNoThrowVTableDesignGateReady,
        bool nativeAttachEntryDesignGateReady,
        bool nativeDetachBeforeReleaseDesignGateReady,
        bool nativeOwnerLifecycleDryRunReady,
        bool nativeAttachEntryRuntimeScaffoldReady,
        bool nativeOwnerStableIdentityReady,
        bool ownerIdentityDiagnosticsReady,
        bool ownerIdentityPointerFree,
        bool nativeOwnerNonCopyableStorageReady,
        bool nativeOwnerCopyBlocked,
        bool nativeOwnerMoveBlocked,
        bool nativeOwnerAddressExposed,
        bool nativeOwnerPointerProduced,
        bool nativeNoThrowDestructorGateReady,
        bool destructorNoThrowScaffoldReady,
        bool destructorExceptionEscapeBlocked,
        bool destructorAddressExposed,
        bool destructorPointerProduced,
        bool nativeOwnerLifecycleGateReady,
        bool managedDisposeSnapshotReady,
        bool lifecycleScaffoldReady,
        bool releaseHookOrderingGateReady,
        bool disposeIdempotencyGateReady,
        bool inFlightDrainGateReady,
        bool callbackStateUnpinAfterDetachGateReady,
        bool delegateUnpinAfterDetachGateReady,
        bool lifecycleAddressExposed,
        bool lifecyclePointerProduced,
        bool nativeAttachBridgeShapeGateReady,
        bool attachBridgeShapeReady,
        bool attachBridgeNoThrowBoundaryReady,
        bool attachBridgeVersionGuardReady,
        bool attachBridgeOwnershipDiagnosticsReady,
        bool attachBridgePointerFree,
        bool nonNullAttachStillDisabled,
        bool exceptionStatusMappingGateReady,
        bool nativeCallbackExceptionCaptureReady,
        bool callbackStatusMappingGateReady,
        bool exceptionEscapeBlocked,
        bool diagnosticCopyReady,
        bool inFlightAccountingGateReady,
        bool callbackEnterAccountingGateReady,
        bool callbackLeaveAccountingGateReady,
        bool callbackInFlightNeverNegativeReady,
        bool releaseAfterDrainGateReady,
        bool callbackStateUnpinAfterDrainGateReady,
        bool nativeNoThrowVTableScaffoldGateReady,
        bool noThrowVTableScaffoldReady,
        bool vTableDestructorNoThrowReady,
        bool processDebugTensorCallbackStubNoThrowReady,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool attachEntryParameterShapeReady,
        bool attachEntryNoThrowBoundaryReady,
        bool attachEntryOwnershipDiagnosticsReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool lineSpecificAttachEntryDesignReady,
        bool attachEntryNoThrowReady,
        bool attachEntryVersionGuardReady,
        bool attachEntryOwnershipReady,
        bool detachBeforeReleaseReady,
        bool releaseHookOrderingReady,
        bool disposeIdempotencyReady,
        bool inFlightDrainBeforeReleaseReady,
        bool callbackStateUnpinAfterDetachReady,
        bool delegateUnpinAfterDetachReady,
        bool stableNativeOwnerAddressDesignReady,
        bool managedCallbackKeepAliveDesignReady,
        bool nativeOwnerNonCopyableReady,
        bool nativeOwnerDisposeOrderReady,
        bool nativeOwnerReleaseHookReady,
        bool nativeOwnerInFlightDrainReady,
        bool noThrowNativeDestructorReady,
        bool noThrowVTableDesignReady,
        bool exceptionToStatusMappingDesignReady,
        bool nativeVTableTrampolineReady,
        bool callbackExceptionCaptureReady,
        bool callbackStatusMappingReady,
        bool callbackInFlightAccountingReady,
        bool canImplementNativeAttach,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool processDebugTensorRuntimeReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerDesignReady = ownerDesignReady;
        DebugTensorMetadataCopied = debugTensorMetadataCopied;
        DisposeReleaseReady = disposeReleaseReady;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        AttachDetachDesignGateReady = attachDetachDesignGateReady;
        AttachControlAvailable = attachControlAvailable;
        DetachClearControlAvailable = detachClearControlAvailable;
        ManagedOwnerStateMachineReady = managedOwnerStateMachineReady;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        NoThrowNativeVTableReady = noThrowNativeVTableReady;
        ExceptionToStatusMappingReady = exceptionToStatusMappingReady;
        BorrowedTensorSafetyGateReady = borrowedTensorSafetyGateReady;
        AttachVTableSafetyGateReady = attachVTableSafetyGateReady;
        NativeAttachNoThrowPreflightReady = nativeAttachNoThrowPreflightReady;
        NativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGateReady;
        NativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGateReady;
        NativeAttachEntryDesignGateReady = nativeAttachEntryDesignGateReady;
        NativeDetachBeforeReleaseDesignGateReady = nativeDetachBeforeReleaseDesignGateReady;
        NativeOwnerLifecycleDryRunReady = nativeOwnerLifecycleDryRunReady;
        NativeAttachEntryRuntimeScaffoldReady = nativeAttachEntryRuntimeScaffoldReady;
        NativeOwnerStableIdentityReady = nativeOwnerStableIdentityReady;
        OwnerIdentityDiagnosticsReady = ownerIdentityDiagnosticsReady;
        OwnerIdentityPointerFree = ownerIdentityPointerFree;
        NativeOwnerNonCopyableStorageReady = nativeOwnerNonCopyableStorageReady;
        NativeOwnerCopyBlocked = nativeOwnerCopyBlocked;
        NativeOwnerMoveBlocked = nativeOwnerMoveBlocked;
        NativeOwnerAddressExposed = nativeOwnerAddressExposed;
        NativeOwnerPointerProduced = nativeOwnerPointerProduced;
        NativeNoThrowDestructorGateReady = nativeNoThrowDestructorGateReady;
        DestructorNoThrowScaffoldReady = destructorNoThrowScaffoldReady;
        DestructorExceptionEscapeBlocked = destructorExceptionEscapeBlocked;
        DestructorAddressExposed = destructorAddressExposed;
        DestructorPointerProduced = destructorPointerProduced;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        ManagedDisposeSnapshotReady = managedDisposeSnapshotReady;
        LifecycleScaffoldReady = lifecycleScaffoldReady;
        ReleaseHookOrderingGateReady = releaseHookOrderingGateReady;
        DisposeIdempotencyGateReady = disposeIdempotencyGateReady;
        InFlightDrainGateReady = inFlightDrainGateReady;
        CallbackStateUnpinAfterDetachGateReady = callbackStateUnpinAfterDetachGateReady;
        DelegateUnpinAfterDetachGateReady = delegateUnpinAfterDetachGateReady;
        LifecycleAddressExposed = lifecycleAddressExposed;
        LifecyclePointerProduced = lifecyclePointerProduced;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        AttachBridgeShapeReady = attachBridgeShapeReady;
        AttachBridgeNoThrowBoundaryReady = attachBridgeNoThrowBoundaryReady;
        AttachBridgeVersionGuardReady = attachBridgeVersionGuardReady;
        AttachBridgeOwnershipDiagnosticsReady = attachBridgeOwnershipDiagnosticsReady;
        AttachBridgePointerFree = attachBridgePointerFree;
        NonNullAttachStillDisabled = nonNullAttachStillDisabled;
        ExceptionStatusMappingGateReady = exceptionStatusMappingGateReady;
        NativeCallbackExceptionCaptureReady = nativeCallbackExceptionCaptureReady;
        CallbackStatusMappingGateReady = callbackStatusMappingGateReady;
        ExceptionEscapeBlocked = exceptionEscapeBlocked;
        DiagnosticCopyReady = diagnosticCopyReady;
        InFlightAccountingGateReady = inFlightAccountingGateReady;
        CallbackEnterAccountingGateReady = callbackEnterAccountingGateReady;
        CallbackLeaveAccountingGateReady = callbackLeaveAccountingGateReady;
        CallbackInFlightNeverNegativeReady = callbackInFlightNeverNegativeReady;
        ReleaseAfterDrainGateReady = releaseAfterDrainGateReady;
        CallbackStateUnpinAfterDrainGateReady = callbackStateUnpinAfterDrainGateReady;
        NativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGateReady;
        NoThrowVTableScaffoldReady = noThrowVTableScaffoldReady;
        VTableDestructorNoThrowReady = vTableDestructorNoThrowReady;
        ProcessDebugTensorCallbackStubNoThrowReady = processDebugTensorCallbackStubNoThrowReady;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        AttachEntryParameterShapeReady = attachEntryParameterShapeReady;
        AttachEntryNoThrowBoundaryReady = attachEntryNoThrowBoundaryReady;
        AttachEntryOwnershipDiagnosticsReady = attachEntryOwnershipDiagnosticsReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        LineSpecificAttachEntryDesignReady = lineSpecificAttachEntryDesignReady;
        AttachEntryNoThrowReady = attachEntryNoThrowReady;
        AttachEntryVersionGuardReady = attachEntryVersionGuardReady;
        AttachEntryOwnershipReady = attachEntryOwnershipReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        ReleaseHookOrderingReady = releaseHookOrderingReady;
        DisposeIdempotencyReady = disposeIdempotencyReady;
        InFlightDrainBeforeReleaseReady = inFlightDrainBeforeReleaseReady;
        CallbackStateUnpinAfterDetachReady = callbackStateUnpinAfterDetachReady;
        DelegateUnpinAfterDetachReady = delegateUnpinAfterDetachReady;
        StableNativeOwnerAddressDesignReady = stableNativeOwnerAddressDesignReady;
        ManagedCallbackKeepAliveDesignReady = managedCallbackKeepAliveDesignReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NativeOwnerDisposeOrderReady = nativeOwnerDisposeOrderReady;
        NativeOwnerReleaseHookReady = nativeOwnerReleaseHookReady;
        NativeOwnerInFlightDrainReady = nativeOwnerInFlightDrainReady;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NoThrowVTableDesignReady = noThrowVTableDesignReady;
        ExceptionToStatusMappingDesignReady = exceptionToStatusMappingDesignReady;
        NativeVTableTrampolineReady = nativeVTableTrampolineReady;
        CallbackExceptionCaptureReady = callbackExceptionCaptureReady;
        CallbackStatusMappingReady = callbackStatusMappingReady;
        CallbackInFlightAccountingReady = callbackInFlightAccountingReady;
        CanImplementNativeAttach = canImplementNativeAttach;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-runtime-proof-precheck";

    /// <summary>Gets the callback kind represented by this precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "runtime-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the owner design snapshot was clean owner-design evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata was copied before the precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorMetadataCopied { get; }

    /// <summary>Gets whether dispose release hook evidence was present before the precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeReleaseReady { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether the attach/detach design gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachDetachDesignGateReady { get; }

    /// <summary>Gets whether a non-null DebugListener attach bridge is available. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachControlAvailable { get; }

    /// <summary>Gets whether the TensorRT 10/11 detach/clear control is available. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DetachClearControlAvailable { get; }

    /// <summary>Gets whether the managed owner state machine has clean dispose and in-flight drain evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ManagedOwnerStateMachineReady { get; }

    /// <summary>Gets whether line-specific execution context attach/detach is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LineSpecificAttachDetachReady => AttachControlAvailable && DetachClearControlAvailable;

    /// <summary>Gets whether a stable native owner address is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a no-throw native vtable trampoline is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeVTableReady { get; }

    /// <summary>Gets whether the native DebugListener owner and no-throw vtable are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableReady => StableNativeOwnerAddressReady && NoThrowNativeVTableReady;

    /// <summary>Gets whether managed exceptions are mapped to native status without crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionToStatusMappingReady { get; }

    /// <summary>Gets whether borrowed debug tensor/data lifetime rules are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether the borrowed tensor safety gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedTensorSafetyGateReady { get; }

    /// <summary>Gets whether the attach/vtable safety gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachVTableSafetyGateReady { get; }

    /// <summary>Gets whether native attach/no-throw preflight has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachNoThrowPreflightReady { get; }

    /// <summary>Gets whether native owner address design gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressDesignGateReady { get; }

    /// <summary>Gets whether native no-throw vtable design gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableDesignGateReady { get; }

    /// <summary>Gets whether native attach entry design gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryDesignGateReady { get; }

    /// <summary>Gets whether native detach-before-release design gate has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachBeforeReleaseDesignGateReady { get; }

    /// <summary>Gets whether native owner lifecycle dry-run has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleDryRunReady { get; }

    /// <summary>Gets whether native attach entry runtime scaffold has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryRuntimeScaffoldReady { get; }

    /// <summary>Gets whether stable owner identity diagnostics have copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerStableIdentityReady { get; }

    /// <summary>Gets whether copied owner id and diagnostic identity evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool OwnerIdentityDiagnosticsReady { get; }

    /// <summary>Gets whether the owner identity diagnostics surface remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool OwnerIdentityPointerFree { get; }

    /// <summary>Gets whether native owner non-copyable storage evidence has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableStorageReady { get; }

    /// <summary>Gets whether native owner copy construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerCopyBlocked { get; }

    /// <summary>Gets whether native owner move construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerMoveBlocked { get; }

    /// <summary>Gets whether the public surface exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressExposed { get; }

    /// <summary>Gets whether the public surface produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerPointerProduced { get; }

    /// <summary>Gets whether native owner no-throw destructor evidence has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowDestructorGateReady { get; }

    /// <summary>Gets whether the source-visible destructor scaffold is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorNoThrowScaffoldReady { get; }

    /// <summary>Gets whether destructor exception escape is blocked by the scaffold boundary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorExceptionEscapeBlocked { get; }

    /// <summary>Gets whether the destructor gate exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorAddressExposed { get; }

    /// <summary>Gets whether the destructor gate produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorPointerProduced { get; }

    /// <summary>Gets whether native owner lifecycle gate evidence has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether the managed dispose snapshot is clean enough for lifecycle gate evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ManagedDisposeSnapshotReady { get; }

    /// <summary>Gets whether source-visible lifecycle scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LifecycleScaffoldReady { get; }

    /// <summary>Gets whether release hook ordering scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseHookOrderingGateReady { get; }

    /// <summary>Gets whether dispose/release idempotency scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeIdempotencyGateReady { get; }

    /// <summary>Gets whether in-flight drain scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightDrainGateReady { get; }

    /// <summary>Gets whether callback state post-detach unpin scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDetachGateReady { get; }

    /// <summary>Gets whether delegate post-detach unpin scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegateUnpinAfterDetachGateReady { get; }

    /// <summary>Gets whether the lifecycle gate exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LifecycleAddressExposed { get; }

    /// <summary>Gets whether the lifecycle gate produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LifecyclePointerProduced { get; }

    /// <summary>Gets whether native attach bridge shape gate evidence has copied evidence ready for precheck consumption. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether native attach bridge shape scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeShapeReady { get; }

    /// <summary>Gets whether native attach bridge no-throw boundary scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeNoThrowBoundaryReady { get; }

    /// <summary>Gets whether native attach bridge TensorRT version guard scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeVersionGuardReady { get; }

    /// <summary>Gets whether native attach bridge ownership diagnostics scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether native attach bridge diagnostics stay pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgePointerFree { get; }

    /// <summary>Gets whether non-null setDebugListener remains disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled { get; }

    /// <summary>Gets whether exception-to-status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionStatusMappingGateReady { get; }

    /// <summary>Gets whether native callback exception capture scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeCallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingGateReady { get; }

    /// <summary>Gets whether callback exceptions are blocked from crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionEscapeBlocked { get; }

    /// <summary>Gets whether callback diagnostics are copied into pointer-free status records. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DiagnosticCopyReady { get; }

    /// <summary>Gets whether in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingGateReady { get; }

    /// <summary>Gets whether callback enter accounting scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackEnterAccountingGateReady { get; }

    /// <summary>Gets whether callback leave accounting scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackLeaveAccountingGateReady { get; }

    /// <summary>Gets whether in-flight callback count is protected from negative values. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightNeverNegativeReady { get; }

    /// <summary>Gets whether release-after-drain scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseAfterDrainGateReady { get; }

    /// <summary>Gets whether callback state unpin after drain evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDrainGateReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether source-visible no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableScaffoldReady { get; }

    /// <summary>Gets whether vtable destructor no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableDestructorNoThrowReady { get; }

    /// <summary>Gets whether processDebugTensor callback stub no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorCallbackStubNoThrowReady { get; }

    /// <summary>Gets whether the vtable scaffold exposes a native address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether the vtable scaffold produces a native pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether the native attach entry parameter shape scaffold is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryParameterShapeReady { get; }

    /// <summary>Gets whether the native attach entry no-throw/status boundary scaffold is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryNoThrowBoundaryReady { get; }

    /// <summary>Gets whether the native attach entry ownership diagnostics scaffold is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether TensorRT 10 and 11 line-specific attach entry design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LineSpecificAttachEntryDesignReady { get; }

    /// <summary>Gets whether the native attach entry has a no-throw boundary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryNoThrowReady { get; }

    /// <summary>Gets whether the native attach entry is guarded by TensorRT version support. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryVersionGuardReady { get; }

    /// <summary>Gets whether the native attach entry ownership contract is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryOwnershipReady { get; }

    /// <summary>Gets whether native detach-before-release ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether native release hooks detach before releasing callback state. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseHookOrderingReady { get; }

    /// <summary>Gets whether native dispose/release is idempotent. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeIdempotencyReady { get; }

    /// <summary>Gets whether in-flight callbacks drain before native release. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightDrainBeforeReleaseReady { get; }

    /// <summary>Gets whether callback state is unpinned only after native detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether delegate handles are unpinned only after native detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether stable native owner address design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerAddressDesignReady { get; }

    /// <summary>Gets whether managed callback keep-alive design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ManagedCallbackKeepAliveDesignReady { get; }

    /// <summary>Gets whether native owner storage has a non-copyable design. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether native owner dispose ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerDisposeOrderReady { get; }

    /// <summary>Gets whether native owner release hooks are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerReleaseHookReady { get; }

    /// <summary>Gets whether native owner in-flight callback drain is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerInFlightDrainReady { get; }

    /// <summary>Gets whether the native owner destructor is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether native owner lifecycle design is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady =>
        StableNativeOwnerAddressReady &&
        StableNativeOwnerAddressDesignReady &&
        NativeOwnerNonCopyableReady &&
        NativeOwnerDisposeOrderReady &&
        NativeOwnerReleaseHookReady &&
        NativeOwnerInFlightDrainReady &&
        NoThrowNativeDestructorReady;

    /// <summary>Gets whether no-throw native vtable design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableDesignReady { get; }

    /// <summary>Gets whether exception-to-status mapping design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionToStatusMappingDesignReady { get; }

    /// <summary>Gets whether native IDebugListener vtable trampoline implementation exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableTrampolineReady { get; }

    /// <summary>Gets whether native callback exceptions are captured into diagnostics without crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback failure status mapping is implemented. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingReady { get; }

    /// <summary>Gets whether native in-flight callback accounting is implemented. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightAccountingReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved design blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady => false;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        OwnerDesignReady &&
        DebugTensorMetadataCopied &&
        DisposeReleaseReady &&
        PointerFreeSurfaceReady &&
        AttachDetachDesignGateReady &&
        ManagedOwnerStateMachineReady &&
        LineSpecificAttachDetachReady &&
        NativeVTableReady &&
        ExceptionToStatusMappingReady &&
        BorrowedTensorSafetyGateReady &&
        AttachVTableSafetyGateReady &&
        NativeAttachNoThrowPreflightReady &&
        NativeOwnerAddressDesignGateReady &&
        NativeNoThrowVTableDesignGateReady &&
        NativeAttachEntryDesignGateReady &&
        NativeDetachBeforeReleaseDesignGateReady &&
        NativeOwnerLifecycleDryRunReady &&
        NativeAttachEntryRuntimeScaffoldReady &&
        NativeOwnerStableIdentityReady &&
        OwnerIdentityDiagnosticsReady &&
        OwnerIdentityPointerFree &&
        NativeOwnerNonCopyableStorageReady &&
        NativeOwnerCopyBlocked &&
        NativeOwnerMoveBlocked &&
        !NativeOwnerAddressExposed &&
        !NativeOwnerPointerProduced &&
        NativeNoThrowDestructorGateReady &&
        DestructorNoThrowScaffoldReady &&
        DestructorExceptionEscapeBlocked &&
        !DestructorAddressExposed &&
        !DestructorPointerProduced &&
        NativeOwnerLifecycleGateReady &&
        ManagedDisposeSnapshotReady &&
        LifecycleScaffoldReady &&
        ReleaseHookOrderingGateReady &&
        DisposeIdempotencyGateReady &&
        InFlightDrainGateReady &&
        CallbackStateUnpinAfterDetachGateReady &&
        DelegateUnpinAfterDetachGateReady &&
        !LifecycleAddressExposed &&
        !LifecyclePointerProduced &&
        NativeAttachBridgeShapeGateReady &&
        AttachBridgeShapeReady &&
        AttachBridgeNoThrowBoundaryReady &&
        AttachBridgeVersionGuardReady &&
        AttachBridgeOwnershipDiagnosticsReady &&
        AttachBridgePointerFree &&
        NonNullAttachStillDisabled &&
        ExceptionStatusMappingGateReady &&
        NativeCallbackExceptionCaptureReady &&
        CallbackStatusMappingGateReady &&
        ExceptionEscapeBlocked &&
        DiagnosticCopyReady &&
        InFlightAccountingGateReady &&
        CallbackEnterAccountingGateReady &&
        CallbackLeaveAccountingGateReady &&
        CallbackInFlightNeverNegativeReady &&
        ReleaseAfterDrainGateReady &&
        CallbackStateUnpinAfterDrainGateReady &&
        NativeNoThrowVTableScaffoldGateReady &&
        NoThrowVTableScaffoldReady &&
        VTableDestructorNoThrowReady &&
        ProcessDebugTensorCallbackStubNoThrowReady &&
        !VTableAddressExposed &&
        !VTablePointerProduced &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        CanImplementNativeAttach &&
        BorrowedDebugTensorLifetimeReady &&
        BorrowedDebugTensorDataLifetimeReady &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets the copied list of prerequisites that still block real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied number of blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the precheck status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => CanAttemptRuntimeProof ? "can-attempt-runtime-proof" : "precheck-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-runtime-proof-precheck; RuntimeEvidenceKind=runtime-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "AttachDetachDesignGateReady=" + AttachDetachDesignGateReady + "; " +
        "AttachControlAvailable=" + AttachControlAvailable + "; " +
        "DetachClearControlAvailable=" + DetachClearControlAvailable + "; " +
        "StableNativeOwnerAddressReady=" + StableNativeOwnerAddressReady + "; " +
        "NoThrowNativeVTableReady=" + NoThrowNativeVTableReady + "; " +
        "ExceptionToStatusMappingReady=" + ExceptionToStatusMappingReady + "; " +
        "BorrowedTensorSafetyGateReady=" + BorrowedTensorSafetyGateReady + "; " +
        "AttachVTableSafetyGateReady=" + AttachVTableSafetyGateReady + "; " +
        "NativeAttachNoThrowPreflightReady=" + NativeAttachNoThrowPreflightReady + "; " +
        "NativeOwnerAddressDesignGateReady=" + NativeOwnerAddressDesignGateReady + "; " +
        "NativeNoThrowVTableDesignGateReady=" + NativeNoThrowVTableDesignGateReady + "; " +
        "NativeAttachEntryDesignGateReady=" + NativeAttachEntryDesignGateReady + "; " +
        "NativeDetachBeforeReleaseDesignGateReady=" + NativeDetachBeforeReleaseDesignGateReady + "; " +
        "NativeOwnerLifecycleDryRunReady=" + NativeOwnerLifecycleDryRunReady + "; " +
        "NativeAttachEntryRuntimeScaffoldReady=" + NativeAttachEntryRuntimeScaffoldReady + "; " +
        "NativeOwnerStableIdentityReady=" + NativeOwnerStableIdentityReady + "; " +
        "OwnerIdentityDiagnosticsReady=" + OwnerIdentityDiagnosticsReady + "; " +
        "OwnerIdentityPointerFree=" + OwnerIdentityPointerFree + "; " +
        "NativeOwnerNonCopyableStorageReady=" + NativeOwnerNonCopyableStorageReady + "; " +
        "NativeOwnerCopyBlocked=" + NativeOwnerCopyBlocked + "; " +
        "NativeOwnerMoveBlocked=" + NativeOwnerMoveBlocked + "; " +
        "NativeOwnerAddressExposed=" + NativeOwnerAddressExposed + "; " +
        "NativeOwnerPointerProduced=" + NativeOwnerPointerProduced + "; " +
        "NativeNoThrowDestructorGateReady=" + NativeNoThrowDestructorGateReady + "; " +
        "DestructorNoThrowScaffoldReady=" + DestructorNoThrowScaffoldReady + "; " +
        "DestructorExceptionEscapeBlocked=" + DestructorExceptionEscapeBlocked + "; " +
        "DestructorAddressExposed=" + DestructorAddressExposed + "; " +
        "DestructorPointerProduced=" + DestructorPointerProduced + "; " +
        "NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "ManagedDisposeSnapshotReady=" + ManagedDisposeSnapshotReady + "; " +
        "LifecycleScaffoldReady=" + LifecycleScaffoldReady + "; " +
        "ReleaseHookOrderingGateReady=" + ReleaseHookOrderingGateReady + "; " +
        "DisposeIdempotencyGateReady=" + DisposeIdempotencyGateReady + "; " +
        "InFlightDrainGateReady=" + InFlightDrainGateReady + "; " +
        "CallbackStateUnpinAfterDetachGateReady=" + CallbackStateUnpinAfterDetachGateReady + "; " +
        "DelegateUnpinAfterDetachGateReady=" + DelegateUnpinAfterDetachGateReady + "; " +
        "LifecycleAddressExposed=" + LifecycleAddressExposed + "; " +
        "LifecyclePointerProduced=" + LifecyclePointerProduced + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "AttachBridgeShapeReady=" + AttachBridgeShapeReady + "; " +
        "AttachBridgeNoThrowBoundaryReady=" + AttachBridgeNoThrowBoundaryReady + "; " +
        "AttachBridgeVersionGuardReady=" + AttachBridgeVersionGuardReady + "; " +
        "AttachBridgeOwnershipDiagnosticsReady=" + AttachBridgeOwnershipDiagnosticsReady + "; " +
        "AttachBridgePointerFree=" + AttachBridgePointerFree + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "ExceptionStatusMappingGateReady=" + ExceptionStatusMappingGateReady + "; " +
        "NativeCallbackExceptionCaptureReady=" + NativeCallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingGateReady=" + CallbackStatusMappingGateReady + "; " +
        "ExceptionEscapeBlocked=" + ExceptionEscapeBlocked + "; " +
        "DiagnosticCopyReady=" + DiagnosticCopyReady + "; " +
        "InFlightAccountingGateReady=" + InFlightAccountingGateReady + "; " +
        "CallbackEnterAccountingGateReady=" + CallbackEnterAccountingGateReady + "; " +
        "CallbackLeaveAccountingGateReady=" + CallbackLeaveAccountingGateReady + "; " +
        "CallbackInFlightNeverNegativeReady=" + CallbackInFlightNeverNegativeReady + "; " +
        "ReleaseAfterDrainGateReady=" + ReleaseAfterDrainGateReady + "; " +
        "CallbackStateUnpinAfterDrainGateReady=" + CallbackStateUnpinAfterDrainGateReady + "; " +
        "NativeNoThrowVTableScaffoldGateReady=" + NativeNoThrowVTableScaffoldGateReady + "; " +
        "NoThrowVTableScaffoldReady=" + NoThrowVTableScaffoldReady + "; " +
        "VTableDestructorNoThrowReady=" + VTableDestructorNoThrowReady + "; " +
        "ProcessDebugTensorCallbackStubNoThrowReady=" + ProcessDebugTensorCallbackStubNoThrowReady + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "AttachEntryParameterShapeReady=" + AttachEntryParameterShapeReady + "; " +
        "LineSpecificAttachEntryDesignReady=" + LineSpecificAttachEntryDesignReady + "; " +
        "AttachEntryNoThrowReady=" + AttachEntryNoThrowReady + "; " +
        "AttachEntryNoThrowBoundaryReady=" + AttachEntryNoThrowBoundaryReady + "; " +
        "AttachEntryVersionGuardReady=" + AttachEntryVersionGuardReady + "; " +
        "AttachEntryOwnershipReady=" + AttachEntryOwnershipReady + "; " +
        "AttachEntryOwnershipDiagnosticsReady=" + AttachEntryOwnershipDiagnosticsReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "ReleaseHookOrderingReady=" + ReleaseHookOrderingReady + "; " +
        "DisposeIdempotencyReady=" + DisposeIdempotencyReady + "; " +
        "InFlightDrainBeforeReleaseReady=" + InFlightDrainBeforeReleaseReady + "; " +
        "CallbackStateUnpinAfterDetachReady=" + CallbackStateUnpinAfterDetachReady + "; " +
        "DelegateUnpinAfterDetachReady=" + DelegateUnpinAfterDetachReady + "; " +
        "StableNativeOwnerAddressDesignReady=" + StableNativeOwnerAddressDesignReady + "; " +
        "ManagedCallbackKeepAliveDesignReady=" + ManagedCallbackKeepAliveDesignReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NativeOwnerDisposeOrderReady=" + NativeOwnerDisposeOrderReady + "; " +
        "NativeOwnerReleaseHookReady=" + NativeOwnerReleaseHookReady + "; " +
        "NativeOwnerInFlightDrainReady=" + NativeOwnerInFlightDrainReady + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NoThrowVTableDesignReady=" + NoThrowVTableDesignReady + "; " +
        "ExceptionToStatusMappingDesignReady=" + ExceptionToStatusMappingDesignReady + "; " +
        "NativeVTableTrampolineReady=" + NativeVTableTrampolineReady + "; " +
        "CallbackExceptionCaptureReady=" + CallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingReady=" + CallbackStatusMappingReady + "; " +
        "CallbackInFlightAccountingReady=" + CallbackInFlightAccountingReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:blocked={BlockedPrerequisiteCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
