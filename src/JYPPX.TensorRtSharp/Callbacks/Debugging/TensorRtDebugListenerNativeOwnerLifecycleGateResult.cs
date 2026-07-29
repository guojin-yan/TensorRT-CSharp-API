using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native owner lifecycle gate diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a lifecycle gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real native owner, non-null attach
/// bridge, complete native vtable, and full package consumer callback runtime evidence exist.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeOwnerLifecycleGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerLifecycleGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        long releaseHookCount,
        long inFlightCallbackCount,
        bool callbackStatePinned,
        bool delegatePinned,
        bool disposeRequested,
        string lastDiagnostic,
        string releaseDiagnostic,
        bool nativeNoThrowDestructorGateReady,
        bool nativeOwnerNonCopyableReady,
        bool nativeOwnerCopyBlocked,
        bool nativeOwnerMoveBlocked,
        bool nativeOwnerAddressExposed,
        bool nativeOwnerPointerProduced,
        bool destructorNoThrowScaffoldReady,
        bool destructorExceptionEscapeBlocked,
        bool destructorAddressExposed,
        bool destructorPointerProduced,
        bool managedDisposeSnapshotReady,
        bool lifecycleScaffoldReady,
        bool releaseHookOrderingGateReady,
        bool disposeIdempotencyGateReady,
        bool inFlightDrainGateReady,
        bool callbackStateUnpinAfterDetachGateReady,
        bool delegateUnpinAfterDetachGateReady,
        bool lifecycleAddressExposed,
        bool lifecyclePointerProduced,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool noThrowNativeDestructorReady,
        bool nativeOwnerLifecycleReady,
        bool nativeVTableDesignReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        ReleaseHookCount = releaseHookCount;
        InFlightCallbackCount = inFlightCallbackCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
        NativeNoThrowDestructorGateReady = nativeNoThrowDestructorGateReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NativeOwnerCopyBlocked = nativeOwnerCopyBlocked;
        NativeOwnerMoveBlocked = nativeOwnerMoveBlocked;
        NativeOwnerAddressExposed = nativeOwnerAddressExposed;
        NativeOwnerPointerProduced = nativeOwnerPointerProduced;
        DestructorNoThrowScaffoldReady = destructorNoThrowScaffoldReady;
        DestructorExceptionEscapeBlocked = destructorExceptionEscapeBlocked;
        DestructorAddressExposed = destructorAddressExposed;
        DestructorPointerProduced = destructorPointerProduced;
        ManagedDisposeSnapshotReady = managedDisposeSnapshotReady;
        LifecycleScaffoldReady = lifecycleScaffoldReady;
        ReleaseHookOrderingGateReady = releaseHookOrderingGateReady;
        DisposeIdempotencyGateReady = disposeIdempotencyGateReady;
        InFlightDrainGateReady = inFlightDrainGateReady;
        CallbackStateUnpinAfterDetachGateReady = callbackStateUnpinAfterDetachGateReady;
        DelegateUnpinAfterDetachGateReady = delegateUnpinAfterDetachGateReady;
        LifecycleAddressExposed = lifecycleAddressExposed;
        LifecyclePointerProduced = lifecyclePointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this lifecycle gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-owner-lifecycle-gate";

    /// <summary>Gets the callback kind represented by this lifecycle gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "lifecycle-gate";

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

    /// <summary>Gets the copied release hook count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets the copied in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets whether copied callback state remains pinned. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether copied delegate state remains pinned. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested in the copied snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets the copied last diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether no-throw destructor gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowDestructorGateReady { get; }

    /// <summary>Gets whether native owner storage is non-copyable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether native owner copy construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerCopyBlocked { get; }

    /// <summary>Gets whether native owner move construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerMoveBlocked { get; }

    /// <summary>Gets whether the public surface exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressExposed { get; }

    /// <summary>Gets whether the public surface produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerPointerProduced { get; }

    /// <summary>Gets whether the source-visible destructor scaffold is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorNoThrowScaffoldReady { get; }

    /// <summary>Gets whether destructor exception escape is blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorExceptionEscapeBlocked { get; }

    /// <summary>Gets whether the destructor gate exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorAddressExposed { get; }

    /// <summary>Gets whether the destructor gate produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DestructorPointerProduced { get; }

    /// <summary>Gets whether the managed dispose snapshot is clean enough for lifecycle gate evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ManagedDisposeSnapshotReady { get; }

    /// <summary>Gets whether the source-visible lifecycle scaffold is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
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

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether no-throw destructor evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether release hook ordering is complete beyond scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseHookOrderingReady => false;

    /// <summary>Gets whether dispose/release idempotency is complete beyond scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeIdempotencyReady => false;

    /// <summary>Gets whether in-flight callbacks drain before release beyond scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightDrainBeforeReleaseReady => false;

    /// <summary>Gets whether callback state unpin ordering is complete beyond scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDetachReady => false;

    /// <summary>Gets whether delegate unpin ordering is complete beyond scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegateUnpinAfterDetachReady => false;

    /// <summary>Gets whether native owner lifecycle implementation is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native owner and vtable design are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether copied lifecycle gate evidence is ready for the next native owner lifecycle work. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool LifecycleGateReady =>
        NativeNoThrowDestructorGateReady &&
        NativeOwnerNonCopyableReady &&
        NativeOwnerCopyBlocked &&
        NativeOwnerMoveBlocked &&
        !NativeOwnerAddressExposed &&
        !NativeOwnerPointerProduced &&
        DestructorNoThrowScaffoldReady &&
        DestructorExceptionEscapeBlocked &&
        !DestructorAddressExposed &&
        !DestructorPointerProduced &&
        ManagedDisposeSnapshotReady &&
        LifecycleScaffoldReady &&
        ReleaseHookOrderingGateReady &&
        DisposeIdempotencyGateReady &&
        InFlightDrainGateReady &&
        CallbackStateUnpinAfterDetachGateReady &&
        DelegateUnpinAfterDetachGateReady &&
        !LifecycleAddressExposed &&
        !LifecyclePointerProduced &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved lifecycle blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        LifecycleGateReady &&
        NativeAttachEntryLocated &&
        NativeOwnerLifecycleReady &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the lifecycle gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => LifecycleGateReady ? "lifecycle-gate-ready" : "lifecycle-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-owner-lifecycle-gate; RuntimeEvidenceKind=lifecycle-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; LifecycleGateReady=" + LifecycleGateReady + "; " +
        "NativeNoThrowDestructorGateReady=" + NativeNoThrowDestructorGateReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NativeOwnerCopyBlocked=" + NativeOwnerCopyBlocked + "; " +
        "NativeOwnerMoveBlocked=" + NativeOwnerMoveBlocked + "; " +
        "NativeOwnerAddressExposed=" + NativeOwnerAddressExposed + "; " +
        "NativeOwnerPointerProduced=" + NativeOwnerPointerProduced + "; " +
        "DestructorNoThrowScaffoldReady=" + DestructorNoThrowScaffoldReady + "; " +
        "DestructorExceptionEscapeBlocked=" + DestructorExceptionEscapeBlocked + "; " +
        "DestructorAddressExposed=" + DestructorAddressExposed + "; " +
        "DestructorPointerProduced=" + DestructorPointerProduced + "; " +
        "ManagedDisposeSnapshotReady=" + ManagedDisposeSnapshotReady + "; " +
        "LifecycleScaffoldReady=" + LifecycleScaffoldReady + "; " +
        "ReleaseHookOrderingGateReady=" + ReleaseHookOrderingGateReady + "; " +
        "DisposeIdempotencyGateReady=" + DisposeIdempotencyGateReady + "; " +
        "InFlightDrainGateReady=" + InFlightDrainGateReady + "; " +
        "CallbackStateUnpinAfterDetachGateReady=" + CallbackStateUnpinAfterDetachGateReady + "; " +
        "DelegateUnpinAfterDetachGateReady=" + DelegateUnpinAfterDetachGateReady + "; " +
        "LifecycleAddressExposed=" + LifecycleAddressExposed + "; " +
        "LifecyclePointerProduced=" + LifecyclePointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "ReleaseHookOrderingReady=" + ReleaseHookOrderingReady + "; " +
        "DisposeIdempotencyReady=" + DisposeIdempotencyReady + "; " +
        "InFlightDrainBeforeReleaseReady=" + InFlightDrainBeforeReleaseReady + "; " +
        "CallbackStateUnpinAfterDetachReady=" + CallbackStateUnpinAfterDetachReady + "; " +
        "DelegateUnpinAfterDetachReady=" + DelegateUnpinAfterDetachReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:owner={OwnerId}:proof={IsRealCallbackRuntimeProof}";
    }
}
