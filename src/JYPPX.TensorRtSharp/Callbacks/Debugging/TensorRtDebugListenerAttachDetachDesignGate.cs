using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free DebugListener attach/detach design gate before any real TensorRT callback bridge exists.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate describes lifecycle readiness only. It does not call <c>setDebugListener</c> with a non-null listener,
/// does not expose a native listener pointer, and is not proof that <c>IDebugListener::processDebugTensor</c> ran.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerAttachDetachDesignGate
{
    /// <summary>
    /// Evaluates the attach/detach design gate from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach/detach design result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerAttachDetachDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "debug-listener-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool managedOwnerStateMachineReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool metadataCopyReady =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        bool detachClearControlAvailable = lineSupportsDebugListener;
        const bool attachControlAvailable = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool noThrowNativeVTableReady = false;
        const bool borrowedDebugTensorLifetimeReady = false;

        List<string> blockers = new List<string>();
        if (!lineSupportsDebugListener)
        {
            blockers.Add("TensorRT 10 or 11 IDebugListener line support has not been selected.");
        }

        if (!ownerDesignReady)
        {
            blockers.Add("debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!managedOwnerStateMachineReady)
        {
            blockers.Add("managed DebugListener owner state machine has not shown dispose, release hook, unpin, and in-flight drain evidence.");
        }

        if (!metadataCopyReady)
        {
            blockers.Add("debug tensor metadata copy-out is incomplete.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public DebugListener attach/detach surface still exposes or produces a borrowed debug tensor pointer.");
        }

        if (!detachClearControlAvailable)
        {
            blockers.Add("line-specific setDebugListener(nullptr) detach/clear control is not available.");
        }

        if (!attachControlAvailable)
        {
            blockers.Add("line-specific setDebugListener(non-null) attach bridge is not implemented.");
        }

        if (!stableNativeOwnerAddressReady)
        {
            blockers.Add("native DebugListener owner stable address is not implemented.");
        }

        if (!noThrowNativeVTableReady)
        {
            blockers.Add("native IDebugListener no-throw vtable trampoline is not implemented.");
        }

        if (!borrowedDebugTensorLifetimeReady)
        {
            blockers.Add("borrowed debug tensor pointer and data buffer lifetime rules are not implemented.");
        }

        return new TensorRtDebugListenerAttachDetachDesignGateResult(
            ownerDesignSnapshot.Line,
            lineSupportsDebugListener,
            ownerDesignReady,
            managedOwnerStateMachineReady,
            metadataCopyReady,
            pointerFreeSurfaceReady,
            detachClearControlAvailable,
            attachControlAvailable,
            stableNativeOwnerAddressReady,
            noThrowNativeVTableReady,
            borrowedDebugTensorLifetimeReady,
            blockers.ToArray());
    }
}
