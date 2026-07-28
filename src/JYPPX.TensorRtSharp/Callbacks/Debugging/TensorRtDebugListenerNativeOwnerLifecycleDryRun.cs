using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates a pointer-free DebugListener native owner lifecycle dry-run before a native callback owner is implemented.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This dry-run copies managed design evidence only. It does not allocate or expose a native owner pointer, does not
/// attach a non-null <c>IDebugListener</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerLifecycleDryRun
{
    /// <summary>
    /// Evaluates native owner lifecycle dry-run readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native owner lifecycle dry-run result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerLifecycleDryRunResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate);
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight);
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate);
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate);
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachGate,
            borrowedTensorGate,
            attachVTableGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate,
            nativeDetachBeforeReleaseDesignGate);
    }

    /// <summary>
    /// Evaluates native owner lifecycle dry-run readiness from copied owner and detach-before-release evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachEntryDesignGate">The copied native attach entry design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeDetachBeforeReleaseDesignGate">The copied native detach-before-release design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native owner lifecycle dry-run result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerLifecycleDryRunResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate,
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate,
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate)
    {
        bool nativeDetachBeforeReleaseDesignGateReady = nativeDetachBeforeReleaseDesignGate.DesignGateReady;
        bool nativeAttachEntryDesignGateReady = nativeAttachEntryDesignGate.DesignGateReady;
        bool nativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGate.DesignGateReady;
        bool nativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGate.DesignGateReady;
        bool managedCallbackKeepAliveDesignReady =
            nativeDetachBeforeReleaseDesignGate.ManagedCallbackKeepAliveDesignReady &&
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool borrowedDebugTensorMetadataCopyDesignReady =
            nativeDetachBeforeReleaseDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            ownerDesignSnapshot.DebugTensorMetadataCopied;
        bool borrowedDebugTensorPointerEscapeBlocked =
            nativeDetachBeforeReleaseDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            attachVTableSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool stableNativeOwnerIdentityReady = false;
        const bool nativeOwnerNonCopyableReady = false;
        const bool nativeOwnerDisposeOrderReady = false;
        const bool nativeOwnerReleaseHookReady = false;
        const bool nativeOwnerInFlightDrainReady = false;
        const bool detachBeforeReleaseReady = false;
        const bool releaseHookOrderingReady = false;
        const bool disposeIdempotencyReady = false;
        const bool inFlightDrainBeforeReleaseReady = false;
        const bool callbackStateUnpinAfterDetachReady = false;
        const bool delegateUnpinAfterDetachReady = false;
        const bool noThrowNativeDestructorReady = false;
        const bool nativeVTableDesignReady = false;
        const bool borrowedDebugTensorLifetimeRuntimeReady = false;
        const bool borrowedDebugTensorDataLifetimeRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachDetachDesignGate.LineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeDetachBeforeReleaseDesignGateReady, "debug-listener-native-detach-before-release-design-gate is not ready for native owner lifecycle dry-run evaluation.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGateReady, "debug-listener-native-attach-entry-design-gate is not ready for native owner lifecycle dry-run evaluation.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableDesignGateReady, "debug-listener-native-nothrow-vtable-design-gate is not ready for native owner lifecycle dry-run evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerAddressDesignGateReady, "debug-listener-native-owner-address-design-gate is not ready for native owner lifecycle dry-run evaluation.");
        AddBlockerIfFalse(blockers, nativeDetachBeforeReleaseDesignGate.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, managedCallbackKeepAliveDesignReady, "managed DebugListener callback keep-alive and dispose/drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyDesignReady, "borrowed debug tensor metadata copy design is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, nativeDetachBeforeReleaseDesignGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerIdentityReady, "native DebugListener stable owner identity dry-run is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage dry-run is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerDisposeOrderReady, "native DebugListener owner dispose ordering dry-run is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerReleaseHookReady, "native DebugListener owner release hook dry-run is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerInFlightDrainReady, "native DebugListener owner in-flight drain dry-run is not implemented.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "native DebugListener detach-before-release dry-run is not implemented.");
        AddBlockerIfFalse(blockers, releaseHookOrderingReady, "native DebugListener release hook ordering dry-run is not implemented.");
        AddBlockerIfFalse(blockers, disposeIdempotencyReady, "native DebugListener dispose/release idempotency dry-run is not implemented.");
        AddBlockerIfFalse(blockers, inFlightDrainBeforeReleaseReady, "native DebugListener in-flight callback drain before release dry-run is not implemented.");
        AddBlockerIfFalse(blockers, callbackStateUnpinAfterDetachReady, "DebugListener callback state post-detach unpin dry-run is not implemented.");
        AddBlockerIfFalse(blockers, delegateUnpinAfterDetachReady, "DebugListener delegate post-detach unpin dry-run is not implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeDestructorReady, "native DebugListener owner no-throw destructor dry-run is not implemented.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable lifecycle design is not implemented.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeRuntimeReady, "borrowed debug tensor pointer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeRuntimeReady, "borrowed debug tensor data buffer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener owner lifecycle evidence.");

        foreach (string blocker in nativeDetachBeforeReleaseDesignGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerLifecycleDryRunResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.ReleaseHookCount,
            ownerDesignSnapshot.InFlightCallbackCount,
            ownerDesignSnapshot.CallbackStatePinned,
            ownerDesignSnapshot.DelegatePinned,
            ownerDesignSnapshot.DisposeRequested,
            ownerDesignSnapshot.LastDiagnostic,
            ownerDesignSnapshot.ReleaseDiagnostic,
            nativeDetachBeforeReleaseDesignGateReady,
            nativeAttachEntryDesignGateReady,
            nativeNoThrowVTableDesignGateReady,
            nativeOwnerAddressDesignGateReady,
            nativeDetachBeforeReleaseDesignGate.NativeDetachEntryLocated,
            nativeDetachBeforeReleaseDesignGate.NativeAttachEntryLocated,
            stableNativeOwnerIdentityReady,
            nativeOwnerNonCopyableReady,
            nativeOwnerDisposeOrderReady,
            nativeOwnerReleaseHookReady,
            nativeOwnerInFlightDrainReady,
            detachBeforeReleaseReady,
            releaseHookOrderingReady,
            disposeIdempotencyReady,
            inFlightDrainBeforeReleaseReady,
            callbackStateUnpinAfterDetachReady,
            delegateUnpinAfterDetachReady,
            noThrowNativeDestructorReady,
            nativeVTableDesignReady,
            managedCallbackKeepAliveDesignReady,
            borrowedDebugTensorMetadataCopyDesignReady,
            borrowedDebugTensorPointerEscapeBlocked,
            borrowedDebugTensorLifetimeRuntimeReady,
            borrowedDebugTensorDataLifetimeRuntimeReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
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
/// Reports copied DebugListener native owner lifecycle dry-run diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a dry-run evidence object. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real TensorRT callback runtime
/// evidence path exists.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeOwnerLifecycleDryRunResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerLifecycleDryRunResult(
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
        bool nativeDetachBeforeReleaseDesignGateReady,
        bool nativeAttachEntryDesignGateReady,
        bool nativeNoThrowVTableDesignGateReady,
        bool nativeOwnerAddressDesignGateReady,
        bool nativeDetachEntryLocated,
        bool nativeAttachEntryLocated,
        bool stableNativeOwnerIdentityReady,
        bool nativeOwnerNonCopyableReady,
        bool nativeOwnerDisposeOrderReady,
        bool nativeOwnerReleaseHookReady,
        bool nativeOwnerInFlightDrainReady,
        bool detachBeforeReleaseReady,
        bool releaseHookOrderingReady,
        bool disposeIdempotencyReady,
        bool inFlightDrainBeforeReleaseReady,
        bool callbackStateUnpinAfterDetachReady,
        bool delegateUnpinAfterDetachReady,
        bool noThrowNativeDestructorReady,
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
        OwnerId = ownerId;
        LastStatus = lastStatus;
        ReleaseHookCount = releaseHookCount;
        InFlightCallbackCount = inFlightCallbackCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
        NativeDetachBeforeReleaseDesignGateReady = nativeDetachBeforeReleaseDesignGateReady;
        NativeAttachEntryDesignGateReady = nativeAttachEntryDesignGateReady;
        NativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGateReady;
        NativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGateReady;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        StableNativeOwnerIdentityReady = stableNativeOwnerIdentityReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NativeOwnerDisposeOrderReady = nativeOwnerDisposeOrderReady;
        NativeOwnerReleaseHookReady = nativeOwnerReleaseHookReady;
        NativeOwnerInFlightDrainReady = nativeOwnerInFlightDrainReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        ReleaseHookOrderingReady = releaseHookOrderingReady;
        DisposeIdempotencyReady = disposeIdempotencyReady;
        InFlightDrainBeforeReleaseReady = inFlightDrainBeforeReleaseReady;
        CallbackStateUnpinAfterDetachReady = callbackStateUnpinAfterDetachReady;
        DelegateUnpinAfterDetachReady = delegateUnpinAfterDetachReady;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
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

    /// <summary>Gets the marker used by readiness to identify this dry-run. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-owner-lifecycle-dry-run";

    /// <summary>Gets the callback kind represented by this dry-run. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "dry-run";

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

    /// <summary>Gets whether managed callback state remains pinned in the copied snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether managed delegate state remains pinned in the copied snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested in the copied snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets the copied last diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether native detach-before-release design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachBeforeReleaseDesignGateReady { get; }

    /// <summary>Gets whether native attach entry design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryDesignGateReady { get; }

    /// <summary>Gets whether native no-throw vtable design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableDesignGateReady { get; }

    /// <summary>Gets whether native owner address design gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressDesignGateReady { get; }

    /// <summary>Gets whether a line-specific native detach/clear entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether a line-specific native non-null DebugListener attach entry exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a stable native owner identity dry-run exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerIdentityReady { get; }

    /// <summary>Gets whether native owner storage is non-copyable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether native owner dispose ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerDisposeOrderReady { get; }

    /// <summary>Gets whether native owner release hooks are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerReleaseHookReady { get; }

    /// <summary>Gets whether native owner in-flight callback drain is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerInFlightDrainReady { get; }

    /// <summary>Gets whether native detach-before-release ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether native release hook ordering is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ReleaseHookOrderingReady { get; }

    /// <summary>Gets whether native dispose/release idempotency is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DisposeIdempotencyReady { get; }

    /// <summary>Gets whether in-flight callbacks drain before release. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightDrainBeforeReleaseReady { get; }

    /// <summary>Gets whether callback state is unpinned only after detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether delegate handles are unpinned only after detach completes. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DelegateUnpinAfterDetachReady { get; }

    /// <summary>Gets whether the native owner destructor is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether native owner lifecycle dry-run is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady =>
        StableNativeOwnerIdentityReady &&
        NativeOwnerNonCopyableReady &&
        NativeOwnerDisposeOrderReady &&
        NativeOwnerReleaseHookReady &&
        NativeOwnerInFlightDrainReady &&
        DetachBeforeReleaseReady &&
        ReleaseHookOrderingReady &&
        DisposeIdempotencyReady &&
        InFlightDrainBeforeReleaseReady &&
        CallbackStateUnpinAfterDetachReady &&
        DelegateUnpinAfterDetachReady &&
        NoThrowNativeDestructorReady;

    /// <summary>Gets whether native owner and vtable design are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether managed callback keep-alive and dispose/drain design evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ManagedCallbackKeepAliveDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata copy design is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyDesignReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether copied dry-run evidence is ready for native lifecycle work. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DryRunReady =>
        NativeDetachBeforeReleaseDesignGateReady &&
        NativeAttachEntryDesignGateReady &&
        NativeNoThrowVTableDesignGateReady &&
        NativeOwnerAddressDesignGateReady &&
        NativeDetachEntryLocated &&
        ManagedCallbackKeepAliveDesignReady &&
        BorrowedDebugTensorMetadataCopyDesignReady &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeRuntimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is proven by runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved owner lifecycle blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        DryRunReady &&
        NativeAttachEntryLocated &&
        NativeOwnerLifecycleReady &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        BorrowedDebugTensorLifetimeRuntimeReady &&
        BorrowedDebugTensorDataLifetimeRuntimeReady &&
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

    /// <summary>Gets the dry-run status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => DryRunReady ? "dry-run-ready" : "dry-run-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-owner-lifecycle-dry-run; RuntimeEvidenceKind=dry-run; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; DryRunReady=" + DryRunReady + "; " +
        "NativeDetachBeforeReleaseDesignGateReady=" + NativeDetachBeforeReleaseDesignGateReady + "; " +
        "NativeAttachEntryDesignGateReady=" + NativeAttachEntryDesignGateReady + "; " +
        "NativeNoThrowVTableDesignGateReady=" + NativeNoThrowVTableDesignGateReady + "; " +
        "NativeOwnerAddressDesignGateReady=" + NativeOwnerAddressDesignGateReady + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "OwnerId=" + OwnerId + "; " +
        "StableNativeOwnerIdentityReady=" + StableNativeOwnerIdentityReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NativeOwnerDisposeOrderReady=" + NativeOwnerDisposeOrderReady + "; " +
        "NativeOwnerReleaseHookReady=" + NativeOwnerReleaseHookReady + "; " +
        "NativeOwnerInFlightDrainReady=" + NativeOwnerInFlightDrainReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "ReleaseHookOrderingReady=" + ReleaseHookOrderingReady + "; " +
        "DisposeIdempotencyReady=" + DisposeIdempotencyReady + "; " +
        "InFlightDrainBeforeReleaseReady=" + InFlightDrainBeforeReleaseReady + "; " +
        "CallbackStateUnpinAfterDetachReady=" + CallbackStateUnpinAfterDetachReady + "; " +
        "DelegateUnpinAfterDetachReady=" + DelegateUnpinAfterDetachReady + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "ReleaseHookCount=" + ReleaseHookCount + "; " +
        "InFlightCallbackCount=" + InFlightCallbackCount + "; " +
        "CallbackStatePinned=" + CallbackStatePinned + "; " +
        "DelegatePinned=" + DelegatePinned + "; " +
        "DisposeRequested=" + DisposeRequested + "; " +
        "ManagedCallbackKeepAliveDesignReady=" + ManagedCallbackKeepAliveDesignReady + "; " +
        "BorrowedDebugTensorMetadataCopyDesignReady=" + BorrowedDebugTensorMetadataCopyDesignReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
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
