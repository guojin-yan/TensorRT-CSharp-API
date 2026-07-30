using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates DebugListener native attach/no-throw prerequisites before a native callback bridge is enabled.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This preflight is pointer-free. It does not call <c>setDebugListener(non-null)</c>, does not expose a native
/// listener owner address, and does not prove that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachNoThrowPreflight
{
    /// <summary>
    /// Evaluates native attach/no-throw preflight from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach/no-throw preflight result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachNoThrowPreflightResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
        return Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate, attachVTableGate);
    }

    /// <summary>
    /// Evaluates native attach/no-throw preflight from copied owner, attach/detach, borrowed tensor, and attach/vtable evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach/no-throw preflight result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachNoThrowPreflightResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate)
    {
        bool attachVTableSafetyGateReady = attachVTableSafetyGate.SafetyGateReady;
        bool nativeDetachEntryLocated = attachVTableSafetyGate.DetachClearControlAvailable;
        bool managedCallbackKeepAliveDesignReady =
            attachVTableSafetyGate.ManagedOwnerStateMachineReady &&
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool borrowedDebugTensorMetadataCopyDesignReady =
            attachVTableSafetyGate.DebugTensorMetadataCopied &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool borrowedDebugTensorPointerEscapeBlocked =
            attachVTableSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool nativeAttachEntryLocated = false;
        const bool stableNativeOwnerAddressDesignReady = false;
        const bool noThrowVTableDesignReady = false;
        const bool exceptionToStatusMappingDesignReady = false;
        const bool borrowedDebugTensorLifetimeRuntimeReady = false;
        const bool borrowedDebugTensorDataLifetimeRuntimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachDetachDesignGate.LineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, attachVTableSafetyGateReady, "debug-listener-attach-vtable-safety-gate is not ready for native attach/no-throw preflight.");
        AddBlockerIfFalse(blockers, nativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, managedCallbackKeepAliveDesignReady, "managed DebugListener callback keep-alive and dispose/drain design evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyDesignReady, "borrowed debug tensor metadata copy design is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerAddressDesignReady, "stable native DebugListener owner address design is not implemented.");
        AddBlockerIfFalse(blockers, noThrowVTableDesignReady, "native IDebugListener no-throw vtable design is not implemented.");
        AddBlockerIfFalse(blockers, exceptionToStatusMappingDesignReady, "native DebugListener exception-to-status mapping design is not implemented.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeRuntimeReady, "borrowed debug tensor pointer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeRuntimeReady, "borrowed debug tensor data buffer lifetime has not been proven by real TensorRT callback runtime.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener native attach/no-throw evidence.");

        foreach (string blocker in attachVTableSafetyGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachNoThrowPreflightResult(
            ownerDesignSnapshot.Line,
            attachVTableSafetyGateReady,
            nativeAttachEntryLocated,
            nativeDetachEntryLocated,
            stableNativeOwnerAddressDesignReady,
            managedCallbackKeepAliveDesignReady,
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
