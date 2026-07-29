using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native detach-before-release design diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a design gate. <see cref="RealCallbackRuntime"/> and <see cref="IsRealCallbackRuntimeProof"/>
/// remain <see langword="false"/> until detach-before-release ordering, native owner/vtable lifecycle, and full
/// package consumer callback runtime evidence exist.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult(
        TensorRtApiLine line,
        bool nativeAttachEntryDesignGateReady,
        bool nativeNoThrowVTableDesignGateReady,
        bool nativeOwnerAddressDesignGateReady,
        bool nativeDetachEntryLocated,
        bool nativeAttachEntryLocated,
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
        bool nativeOwnerLifecycleReady,
        bool nativeVTableDesignReady,
        bool managedCallbackKeepAliveDesignReady,
        bool borrowedDebugTensorMetadataCopyDesignReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeRuntimeReady,
        bool borrowedDebugTensorDataLifetimeRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        NativeAttachEntryDesignGateReady = nativeAttachEntryDesignGateReady;
        NativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGateReady;
        NativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGateReady;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
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
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ManagedCallbackKeepAliveDesignReady = managedCallbackKeepAliveDesignReady;
        BorrowedDebugTensorMetadataCopyDesignReady = borrowedDebugTensorMetadataCopyDesignReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeRuntimeReady = borrowedDebugTensorLifetimeRuntimeReady;
        BorrowedDebugTensorDataLifetimeRuntimeReady = borrowedDebugTensorDataLifetimeRuntimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this design gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-detach-before-release-design-gate";

    /// <summary>Gets the callback kind represented by this design gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether native attach entry design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryDesignGateReady { get; }

    /// <summary>Gets whether native no-throw vtable design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableDesignGateReady { get; }

    /// <summary>Gets whether native owner address design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressDesignGateReady { get; }

    /// <summary>Gets whether a line-specific native detach/clear entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether a line-specific native non-null DebugListener attach entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether TensorRT 10 and 11 line-specific attach entry design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool LineSpecificAttachEntryDesignReady { get; }

    /// <summary>Gets whether the native attach entry has a no-throw boundary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryNoThrowReady { get; }

    /// <summary>Gets whether the native attach entry is guarded by TensorRT version support. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryVersionGuardReady { get; }

    /// <summary>Gets whether the native attach entry ownership contract is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachEntryOwnershipReady { get; }

    /// <summary>Gets whether native detach-before-release ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether the release hook orders detach before native state release. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseHookOrderingReady { get; }

    /// <summary>Gets whether native dispose/release can run idempotently. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DisposeIdempotencyReady { get; }

    /// <summary>Gets whether in-flight callbacks drain before native release. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InFlightDrainBeforeReleaseReady { get; }

    /// <summary>Gets whether callback state is unpinned after detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether delegate handles are unpinned after detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DelegateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether native owner lifecycle design is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native owner and no-throw vtable design are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether managed callback keep-alive and dispose/drain design evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ManagedCallbackKeepAliveDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata copy design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyDesignReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether copied design evidence is ready for detach-before-release native work. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DesignGateReady =>
        NativeAttachEntryDesignGateReady &&
        NativeNoThrowVTableDesignGateReady &&
        NativeOwnerAddressDesignGateReady &&
        NativeDetachEntryLocated &&
        ManagedCallbackKeepAliveDesignReady &&
        BorrowedDebugTensorMetadataCopyDesignReady &&
        BorrowedDebugTensorPointerEscapeBlocked;

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved detach/release blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        DesignGateReady &&
        NativeAttachEntryLocated &&
        LineSpecificAttachEntryDesignReady &&
        AttachEntryNoThrowReady &&
        AttachEntryVersionGuardReady &&
        AttachEntryOwnershipReady &&
        DetachBeforeReleaseReady &&
        ReleaseHookOrderingReady &&
        DisposeIdempotencyReady &&
        InFlightDrainBeforeReleaseReady &&
        CallbackStateUnpinAfterDetachReady &&
        DelegateUnpinAfterDetachReady &&
        NativeOwnerLifecycleReady &&
        NativeVTableDesignReady;

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeRuntimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        BorrowedDebugTensorLifetimeRuntimeReady &&
        BorrowedDebugTensorDataLifetimeRuntimeReady &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-detach-before-release-design-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; DesignGateReady=" + DesignGateReady + "; " +
        "NativeAttachEntryDesignGateReady=" + NativeAttachEntryDesignGateReady + "; " +
        "NativeNoThrowVTableDesignGateReady=" + NativeNoThrowVTableDesignGateReady + "; " +
        "NativeOwnerAddressDesignGateReady=" + NativeOwnerAddressDesignGateReady + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "LineSpecificAttachEntryDesignReady=" + LineSpecificAttachEntryDesignReady + "; " +
        "AttachEntryNoThrowReady=" + AttachEntryNoThrowReady + "; " +
        "AttachEntryVersionGuardReady=" + AttachEntryVersionGuardReady + "; " +
        "AttachEntryOwnershipReady=" + AttachEntryOwnershipReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "ReleaseHookOrderingReady=" + ReleaseHookOrderingReady + "; " +
        "DisposeIdempotencyReady=" + DisposeIdempotencyReady + "; " +
        "InFlightDrainBeforeReleaseReady=" + InFlightDrainBeforeReleaseReady + "; " +
        "CallbackStateUnpinAfterDetachReady=" + CallbackStateUnpinAfterDetachReady + "; " +
        "DelegateUnpinAfterDetachReady=" + DelegateUnpinAfterDetachReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "ManagedCallbackKeepAliveDesignReady=" + ManagedCallbackKeepAliveDesignReady + "; " +
        "BorrowedDebugTensorMetadataCopyDesignReady=" + BorrowedDebugTensorMetadataCopyDesignReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:detach={DetachBeforeReleaseReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
