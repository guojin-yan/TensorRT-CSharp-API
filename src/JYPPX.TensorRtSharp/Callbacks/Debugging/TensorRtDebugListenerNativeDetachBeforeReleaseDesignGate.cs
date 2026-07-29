using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free DebugListener native detach-before-release design gate before a native callback owner can be released.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate copies managed design evidence only. It does not attach a non-null <c>IDebugListener</c>, does not detach
/// a real native owner, and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate
{
    /// <summary>
    /// Evaluates native detach-before-release design readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native detach-before-release design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult Evaluate(
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
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachGate,
            borrowedTensorGate,
            attachVTableGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate);
    }

    /// <summary>
    /// Evaluates native detach-before-release design readiness from copied owner and attach-entry evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeAttachEntryDesignGate">The copied native attach entry design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native detach-before-release design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate,
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate)
    {
        bool nativeAttachEntryDesignGateReady = nativeAttachEntryDesignGate.DesignGateReady;
        bool nativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGate.DesignGateReady;
        bool nativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGate.DesignGateReady;
        bool nativeDetachEntryLocated =
            nativeAttachEntryDesignGate.NativeDetachEntryLocated &&
            nativeAttachNoThrowPreflight.NativeDetachEntryLocated &&
            nativeOwnerAddressDesignGate.NativeDetachEntryLocated;
        bool managedCallbackKeepAliveDesignReady =
            nativeAttachEntryDesignGate.ManagedCallbackKeepAliveDesignReady &&
            nativeNoThrowVTableDesignGate.ManagedCallbackKeepAliveDesignReady &&
            nativeOwnerAddressDesignGate.ManagedCallbackKeepAliveDesignReady &&
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool borrowedDebugTensorMetadataCopyDesignReady =
            nativeAttachEntryDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeNoThrowVTableDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeOwnerAddressDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            ownerDesignSnapshot.DebugTensorMetadataCopied;
        bool borrowedDebugTensorPointerEscapeBlocked =
            nativeAttachEntryDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            nativeNoThrowVTableDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            nativeOwnerAddressDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            attachVTableSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool detachBeforeReleaseReady = false;
        const bool releaseHookOrderingReady = false;
        const bool disposeIdempotencyReady = false;
        const bool inFlightDrainBeforeReleaseReady = false;
        const bool callbackStateUnpinAfterDetachReady = false;
        const bool delegateUnpinAfterDetachReady = false;
        const bool borrowedDebugTensorLifetimeRuntimeReady = false;
        const bool borrowedDebugTensorDataLifetimeRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachDetachDesignGate.LineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGateReady, "debug-listener-native-attach-entry-design-gate is not ready for detach-before-release design evaluation.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableDesignGateReady, "debug-listener-native-nothrow-vtable-design-gate is not ready for detach-before-release design evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerAddressDesignGateReady, "debug-listener-native-owner-address-design-gate is not ready for detach-before-release design evaluation.");
        AddBlockerIfFalse(blockers, nativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, managedCallbackKeepAliveDesignReady, "managed DebugListener callback keep-alive and dispose/drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyDesignReady, "borrowed debug tensor metadata copy design is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.LineSpecificAttachEntryDesignReady, "line-specific DebugListener native attach entry design has not been documented for TensorRT 10 and 11.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.AttachEntryNoThrowReady, "native DebugListener attach entry no-throw boundary is not implemented.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.AttachEntryVersionGuardReady, "native DebugListener attach entry TensorRT version guard is not implemented.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.AttachEntryOwnershipReady, "native DebugListener attach entry ownership contract is not implemented.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "native DebugListener detach-before-release ordering is not implemented.");
        AddBlockerIfFalse(blockers, releaseHookOrderingReady, "native DebugListener release hook ordering is not implemented.");
        AddBlockerIfFalse(blockers, disposeIdempotencyReady, "native DebugListener dispose/release idempotency is not implemented.");
        AddBlockerIfFalse(blockers, inFlightDrainBeforeReleaseReady, "native DebugListener in-flight callback drain before release is not implemented.");
        AddBlockerIfFalse(blockers, callbackStateUnpinAfterDetachReady, "DebugListener callback state unpin after detach is not implemented.");
        AddBlockerIfFalse(blockers, delegateUnpinAfterDetachReady, "DebugListener delegate unpin after detach is not implemented.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.NativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not implemented.");
        AddBlockerIfFalse(blockers, nativeAttachEntryDesignGate.NativeVTableDesignReady, "native IDebugListener no-throw vtable design is not implemented.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeRuntimeReady, "borrowed debug tensor pointer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeRuntimeReady, "borrowed debug tensor data buffer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener detach-before-release evidence.");

        foreach (string blocker in nativeAttachEntryDesignGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult(
            ownerDesignSnapshot.Line,
            nativeAttachEntryDesignGateReady,
            nativeNoThrowVTableDesignGateReady,
            nativeOwnerAddressDesignGateReady,
            nativeDetachEntryLocated,
            nativeAttachEntryDesignGate.NativeAttachEntryLocated,
            nativeAttachEntryDesignGate.LineSpecificAttachEntryDesignReady,
            nativeAttachEntryDesignGate.AttachEntryNoThrowReady,
            nativeAttachEntryDesignGate.AttachEntryVersionGuardReady,
            nativeAttachEntryDesignGate.AttachEntryOwnershipReady,
            detachBeforeReleaseReady,
            releaseHookOrderingReady,
            disposeIdempotencyReady,
            inFlightDrainBeforeReleaseReady,
            callbackStateUnpinAfterDetachReady,
            delegateUnpinAfterDetachReady,
            nativeAttachEntryDesignGate.NativeOwnerLifecycleReady,
            nativeAttachEntryDesignGate.NativeVTableDesignReady,
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
