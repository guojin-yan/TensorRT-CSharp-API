using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free DebugListener native owner address design gate before a native callback bridge is enabled.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate copies managed design evidence only. It does not create, return, or expose a native owner address, does
/// not attach a non-null <c>IDebugListener</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerAddressDesignGate
{
    /// <summary>
    /// Evaluates native owner address design readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner address design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerAddressDesignGateResult Evaluate(
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
        return Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight);
    }

    /// <summary>
    /// Evaluates native owner address design readiness from copied owner and attach/detach evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner address design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerAddressDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate)
    {
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorGate,
                attachVTableGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight);
    }

    /// <summary>
    /// Evaluates native owner address design readiness from copied owner, attach/detach, and borrowed tensor evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner address design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerAddressDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate)
    {
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorSafetyGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorSafetyGate, attachVTableGate, nativeAttachNoThrowPreflight);
    }

    /// <summary>
    /// Evaluates native owner address design readiness from copied owner, gate, and preflight evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner address design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerAddressDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate)
    {
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorSafetyGate, attachVTableSafetyGate, nativeAttachNoThrowPreflight);
    }

    /// <summary>
    /// Evaluates native owner address design readiness from copied owner, gate, and native attach/no-throw preflight evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner address design result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerAddressDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight)
    {
        bool nativeAttachNoThrowPreflightReady = nativeAttachNoThrowPreflight.PreflightReady;
        bool nativeDetachEntryLocated = nativeAttachNoThrowPreflight.NativeDetachEntryLocated;
        bool managedCallbackKeepAliveDesignReady =
            nativeAttachNoThrowPreflight.ManagedCallbackKeepAliveDesignReady &&
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool borrowedDebugTensorMetadataCopyDesignReady =
            nativeAttachNoThrowPreflight.BorrowedDebugTensorMetadataCopyDesignReady &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            ownerDesignSnapshot.DebugTensorMetadataCopied;
        bool borrowedDebugTensorPointerEscapeBlocked =
            nativeAttachNoThrowPreflight.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            attachVTableSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool nativeAttachEntryLocated = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool stableNativeOwnerAddressDesignReady = false;
        const bool nativeOwnerNonCopyableReady = false;
        const bool nativeOwnerDisposeOrderReady = false;
        const bool nativeOwnerReleaseHookReady = false;
        const bool nativeOwnerInFlightDrainReady = false;
        const bool noThrowNativeDestructorReady = false;
        const bool noThrowVTableDesignReady = false;
        const bool exceptionToStatusMappingDesignReady = false;
        const bool borrowedDebugTensorLifetimeRuntimeReady = false;
        const bool borrowedDebugTensorDataLifetimeRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachDetachDesignGate.LineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeAttachNoThrowPreflightReady, "debug-listener-native-attach-nothrow-preflight is not ready for native owner address design evaluation.");
        AddBlockerIfFalse(blockers, nativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, managedCallbackKeepAliveDesignReady, "managed DebugListener callback keep-alive and dispose/drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyDesignReady, "borrowed debug tensor metadata copy design is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerAddressReady, "stable native DebugListener owner address is not implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerAddressDesignReady, "stable native DebugListener owner address design is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage design is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerDisposeOrderReady, "native DebugListener owner dispose ordering is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerReleaseHookReady, "native DebugListener owner release hook is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerInFlightDrainReady, "native DebugListener owner in-flight drain is not implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeDestructorReady, "native DebugListener owner no-throw destructor is not implemented.");
        AddBlockerIfFalse(blockers, noThrowVTableDesignReady, "native IDebugListener no-throw vtable design is not implemented.");
        AddBlockerIfFalse(blockers, exceptionToStatusMappingDesignReady, "native DebugListener exception-to-status mapping design is not implemented.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeRuntimeReady, "borrowed debug tensor pointer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeRuntimeReady, "borrowed debug tensor data buffer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener native owner address evidence.");

        foreach (string blocker in nativeAttachNoThrowPreflight.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerAddressDesignGateResult(
            ownerDesignSnapshot.Line,
            nativeAttachNoThrowPreflightReady,
            nativeAttachEntryLocated,
            nativeDetachEntryLocated,
            stableNativeOwnerAddressReady,
            stableNativeOwnerAddressDesignReady,
            managedCallbackKeepAliveDesignReady,
            nativeOwnerNonCopyableReady,
            nativeOwnerDisposeOrderReady,
            nativeOwnerReleaseHookReady,
            nativeOwnerInFlightDrainReady,
            noThrowNativeDestructorReady,
            noThrowVTableDesignReady,
            exceptionToStatusMappingDesignReady,
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
