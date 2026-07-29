using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native owner address design diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a design gate. <see cref="RealCallbackRuntime"/> and <see cref="IsRealCallbackRuntimeProof"/>
/// remain <see langword="false"/> until a real no-throw native owner bridge and full package consumer callback runtime
/// evidence exist.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeOwnerAddressDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerAddressDesignGateResult(
        TensorRtApiLine line,
        bool nativeAttachNoThrowPreflightReady,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool stableNativeOwnerAddressReady,
        bool stableNativeOwnerAddressDesignReady,
        bool managedCallbackKeepAliveDesignReady,
        bool nativeOwnerNonCopyableReady,
        bool nativeOwnerDisposeOrderReady,
        bool nativeOwnerReleaseHookReady,
        bool nativeOwnerInFlightDrainReady,
        bool noThrowNativeDestructorReady,
        bool noThrowVTableDesignReady,
        bool exceptionToStatusMappingDesignReady,
        bool borrowedDebugTensorMetadataCopyDesignReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeRuntimeReady,
        bool borrowedDebugTensorDataLifetimeRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        NativeAttachNoThrowPreflightReady = nativeAttachNoThrowPreflightReady;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        StableNativeOwnerAddressDesignReady = stableNativeOwnerAddressDesignReady;
        ManagedCallbackKeepAliveDesignReady = managedCallbackKeepAliveDesignReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NativeOwnerDisposeOrderReady = nativeOwnerDisposeOrderReady;
        NativeOwnerReleaseHookReady = nativeOwnerReleaseHookReady;
        NativeOwnerInFlightDrainReady = nativeOwnerInFlightDrainReady;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NoThrowVTableDesignReady = noThrowVTableDesignReady;
        ExceptionToStatusMappingDesignReady = exceptionToStatusMappingDesignReady;
        BorrowedDebugTensorMetadataCopyDesignReady = borrowedDebugTensorMetadataCopyDesignReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeRuntimeReady = borrowedDebugTensorLifetimeRuntimeReady;
        BorrowedDebugTensorDataLifetimeRuntimeReady = borrowedDebugTensorDataLifetimeRuntimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this design gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-owner-address-design-gate";

    /// <summary>Gets the callback kind represented by this design gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-native-owner-address";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether native attach/no-throw preflight evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachNoThrowPreflightReady { get; }

    /// <summary>Gets whether a line-specific native non-null DebugListener attach entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a line-specific native detach/clear entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether a stable native owner address is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a stable native owner address design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerAddressDesignReady { get; }

    /// <summary>Gets whether managed callback keep-alive and dispose/drain design evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ManagedCallbackKeepAliveDesignReady { get; }

    /// <summary>Gets whether native owner storage has a non-copyable design. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether native owner dispose ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerDisposeOrderReady { get; }

    /// <summary>Gets whether native owner release hooks are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerReleaseHookReady { get; }

    /// <summary>Gets whether native owner in-flight callback drain is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerInFlightDrainReady { get; }

    /// <summary>Gets whether the native owner destructor is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether a no-throw native vtable design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableDesignReady { get; }

    /// <summary>Gets whether native exception-to-status mapping design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionToStatusMappingDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata copy design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyDesignReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether native owner lifecycle design is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady =>
        StableNativeOwnerAddressReady &&
        StableNativeOwnerAddressDesignReady &&
        NativeOwnerNonCopyableReady &&
        NativeOwnerDisposeOrderReady &&
        NativeOwnerReleaseHookReady &&
        NativeOwnerInFlightDrainReady &&
        NoThrowNativeDestructorReady;

    /// <summary>Gets whether native owner and vtable design are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady =>
        NativeOwnerLifecycleReady &&
        NoThrowVTableDesignReady &&
        ExceptionToStatusMappingDesignReady;

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeRuntimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether this design gate has enough copied evidence to guide native owner address work. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DesignGateReady =>
        NativeAttachNoThrowPreflightReady &&
        NativeDetachEntryLocated &&
        ManagedCallbackKeepAliveDesignReady &&
        BorrowedDebugTensorMetadataCopyDesignReady &&
        BorrowedDebugTensorPointerEscapeBlocked;

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved owner address blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        DesignGateReady &&
        NativeAttachEntryLocated &&
        NativeVTableDesignReady;

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
        "debug-listener-native-owner-address-design-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; DesignGateReady=" + DesignGateReady + "; " +
        "NativeAttachNoThrowPreflightReady=" + NativeAttachNoThrowPreflightReady + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "StableNativeOwnerAddressReady=" + StableNativeOwnerAddressReady + "; " +
        "StableNativeOwnerAddressDesignReady=" + StableNativeOwnerAddressDesignReady + "; " +
        "ManagedCallbackKeepAliveDesignReady=" + ManagedCallbackKeepAliveDesignReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NativeOwnerDisposeOrderReady=" + NativeOwnerDisposeOrderReady + "; " +
        "NativeOwnerReleaseHookReady=" + NativeOwnerReleaseHookReady + "; " +
        "NativeOwnerInFlightDrainReady=" + NativeOwnerInFlightDrainReady + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NoThrowVTableDesignReady=" + NoThrowVTableDesignReady + "; " +
        "ExceptionToStatusMappingDesignReady=" + ExceptionToStatusMappingDesignReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:owner={NativeOwnerLifecycleReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
