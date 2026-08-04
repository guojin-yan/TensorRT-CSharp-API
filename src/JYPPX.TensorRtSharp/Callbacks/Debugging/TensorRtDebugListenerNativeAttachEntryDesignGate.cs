using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free DebugListener native attach entry design gate before non-null attach can be implemented.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate copies managed design evidence only. It does not create a native <c>IDebugListener</c> owner, does not
/// call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachEntryDesignGate
{
    /// <summary>
    /// Evaluates native attach entry design readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryDesignGateResult Evaluate(
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
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachGate,
            borrowedTensorGate,
            attachVTableGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate);
    }

    /// <summary>
    /// Evaluates native attach entry design readiness from copied owner, preflight, owner, and vtable evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate)
    {
        bool nativeNoThrowVTableDesignGateReady = nativeNoThrowVTableDesignGate.DesignGateReady;
        bool nativeOwnerAddressDesignGateReady = nativeOwnerAddressDesignGate.DesignGateReady;
        bool nativeAttachNoThrowPreflightReady = nativeAttachNoThrowPreflight.PreflightReady;
        bool nativeDetachEntryLocated =
            nativeAttachNoThrowPreflight.NativeDetachEntryLocated &&
            nativeOwnerAddressDesignGate.NativeDetachEntryLocated;
        bool managedCallbackKeepAliveDesignReady =
            nativeNoThrowVTableDesignGate.ManagedCallbackKeepAliveDesignReady &&
            nativeOwnerAddressDesignGate.ManagedCallbackKeepAliveDesignReady &&
            nativeAttachNoThrowPreflight.ManagedCallbackKeepAliveDesignReady &&
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool borrowedDebugTensorMetadataCopyDesignReady =
            nativeNoThrowVTableDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeOwnerAddressDesignGate.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeAttachNoThrowPreflight.BorrowedDebugTensorMetadataCopyDesignReady &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            ownerDesignSnapshot.DebugTensorMetadataCopied;
        bool borrowedDebugTensorPointerEscapeBlocked =
            nativeNoThrowVTableDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            nativeOwnerAddressDesignGate.BorrowedDebugTensorPointerEscapeBlocked &&
            nativeAttachNoThrowPreflight.BorrowedDebugTensorPointerEscapeBlocked &&
            attachVTableSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool nativeAttachEntryLocated = false;
        const bool lineSpecificAttachEntryDesignReady = false;
        const bool attachEntryNoThrowReady = false;
        const bool attachEntryVersionGuardReady = false;
        const bool attachEntryOwnershipReady = false;
        const bool detachBeforeReleaseReady = false;
        const bool borrowedDebugTensorLifetimeRuntimeReady = false;
        const bool borrowedDebugTensorDataLifetimeRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachDetachDesignGate.LineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeAttachNoThrowPreflightReady, "debug-listener-native-attach-nothrow-preflight is not ready for native attach entry design evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerAddressDesignGateReady, "debug-listener-native-owner-address-design-gate is not ready for native attach entry design evaluation.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableDesignGateReady, "debug-listener-native-nothrow-vtable-design-gate is not ready for native attach entry design evaluation.");
        AddBlockerIfFalse(blockers, nativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, managedCallbackKeepAliveDesignReady, "managed DebugListener callback keep-alive and dispose/drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyDesignReady, "borrowed debug tensor metadata copy design is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, lineSpecificAttachEntryDesignReady, "line-specific DebugListener native attach entry design has not been documented for TensorRT 10 and 11.");
        AddBlockerIfFalse(blockers, attachEntryNoThrowReady, "native DebugListener attach entry no-throw boundary is not implemented.");
        AddBlockerIfFalse(blockers, attachEntryVersionGuardReady, "native DebugListener attach entry TensorRT version guard is not implemented.");
        AddBlockerIfFalse(blockers, attachEntryOwnershipReady, "native DebugListener attach entry ownership contract is not implemented.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "native DebugListener detach-before-release ordering is not implemented.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableDesignGate.NativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not implemented.");
        AddBlockerIfFalse(blockers, nativeNoThrowVTableDesignGate.NativeVTableDesignReady, "native IDebugListener no-throw vtable design is not implemented.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeRuntimeReady, "borrowed debug tensor pointer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeRuntimeReady, "borrowed debug tensor data buffer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener attach entry evidence.");

        foreach (string blocker in nativeNoThrowVTableDesignGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachEntryDesignGateResult(
            ownerDesignSnapshot.Line,
            nativeNoThrowVTableDesignGateReady,
            nativeOwnerAddressDesignGateReady,
            nativeAttachNoThrowPreflightReady,
            nativeDetachEntryLocated,
            nativeAttachEntryLocated,
            lineSpecificAttachEntryDesignReady,
            attachEntryNoThrowReady,
            attachEntryVersionGuardReady,
            attachEntryOwnershipReady,
            detachBeforeReleaseReady,
            nativeNoThrowVTableDesignGate.NativeOwnerLifecycleReady,
            nativeNoThrowVTableDesignGate.NativeVTableDesignReady,
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
