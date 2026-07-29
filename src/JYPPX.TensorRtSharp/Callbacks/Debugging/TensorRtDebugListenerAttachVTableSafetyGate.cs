using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates copied DebugListener attach/vtable prerequisites before a native no-throw callback bridge exists.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This safety gate is pointer-free. It does not attach a non-null <c>IDebugListener</c>, does not expose a native
/// listener owner address, and does not prove that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerAttachVTableSafetyGate
{
    /// <summary>
    /// Evaluates attach/vtable safety from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        return Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
    }

    /// <summary>
    /// Evaluates attach/vtable safety from copied owner and attach/detach design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate)
    {
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorGate);
    }

    /// <summary>
    /// Evaluates attach/vtable safety from copied owner, attach/detach, and borrowed tensor safety evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate)
    {
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "debug-listener-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0 &&
            ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool debugTensorMetadataCopied =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            attachDetachDesignGate.DebugTensorMetadataCopied &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool pointerFreeSurfaceReady =
            attachDetachDesignGate.PointerFreeSurfaceReady &&
            borrowedTensorSafetyGate.PointerFreeSurfaceReady &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        bool lineSupportsDebugListener = attachDetachDesignGate.LineSupportsDebugListener;
        bool attachDetachDesignGateReady = attachDetachDesignGate.DesignGateReady;
        bool borrowedTensorSafetyGateReady = borrowedTensorSafetyGate.SafetyGateReady;
        bool managedOwnerStateMachineReady = attachDetachDesignGate.ManagedOwnerStateMachineReady;
        bool borrowedDebugTensorPointerEscapeBlocked = borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked;
        bool detachClearControlAvailable = attachDetachDesignGate.DetachClearControlAvailable;

        const bool attachControlAvailable = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool noThrowNativeVTableReady = false;
        const bool exceptionToStatusMappingReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, ownerDesignReady, "debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        AddBlockerIfFalse(blockers, attachDetachDesignGateReady, "debug-listener-attach-detach-design-gate is not ready for attach/vtable safety evaluation.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGateReady, "debug-listener-borrowed-tensor-safety-gate is not ready for attach/vtable safety evaluation.");
        AddBlockerIfFalse(blockers, managedOwnerStateMachineReady, "managed DebugListener owner state machine has not shown dispose, release hook, unpin, and in-flight drain evidence.");
        AddBlockerIfFalse(blockers, debugTensorMetadataCopied, "debug tensor copied metadata is incomplete.");
        AddBlockerIfFalse(blockers, pointerFreeSurfaceReady, "public DebugListener attach/vtable surface still exposes, produces, or leaks a debug tensor pointer.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, detachClearControlAvailable, "line-specific setDebugListener(nullptr) detach/clear control is not available.");
        AddBlockerIfFalse(blockers, attachControlAvailable, "line-specific setDebugListener(non-null) attach bridge is not implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerAddressReady, "stable native DebugListener owner address is not implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeVTableReady, "native IDebugListener no-throw vtable trampoline is not implemented.");
        AddBlockerIfFalse(blockers, exceptionToStatusMappingReady, "exception-to-status mapping for the native DebugListener vtable bridge is not proven.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGate.BorrowedDebugTensorLifetimeReady, "borrowed debug tensor pointer lifetime has not been proven against a real TensorRT callback.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGate.BorrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data buffer lifetime has not been proven against a real TensorRT callback.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener attach/vtable evidence.");

        foreach (string blocker in attachDetachDesignGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in borrowedTensorSafetyGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerAttachVTableSafetyGateResult(
            ownerDesignSnapshot.Line,
            lineSupportsDebugListener,
            ownerDesignReady,
            attachDetachDesignGateReady,
            borrowedTensorSafetyGateReady,
            managedOwnerStateMachineReady,
            debugTensorMetadataCopied,
            pointerFreeSurfaceReady,
            borrowedDebugTensorPointerEscapeBlocked,
            detachClearControlAvailable,
            attachControlAvailable,
            stableNativeOwnerAddressReady,
            noThrowNativeVTableReady,
            exceptionToStatusMappingReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorLifetimeReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorDataLifetimeReady,
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
