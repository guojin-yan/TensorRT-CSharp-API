using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach bridge shape evidence before non-null attach is enabled.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate reports source-visible attach bridge parameter shape, version guard, no-throw boundary, and ownership
/// diagnostics only. It does not call <c>setDebugListener(non-null)</c>, does not create a native owner, and is not
/// proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachBridgeShapeGate
{
    /// <summary>
    /// Evaluates attach bridge shape evidence from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach bridge shape gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachBridgeShapeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleGate);
    }

    /// <summary>
    /// Evaluates attach bridge shape evidence from copied owner and lifecycle gate evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="nativeOwnerLifecycleGate">The copied native owner lifecycle gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach bridge shape gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachBridgeShapeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool nativeOwnerLifecycleGateReady = nativeOwnerLifecycleGate.LifecycleGateReady;
        bool attachBridgeShapeReady =
            nativeOwnerLifecycleGateReady &&
            lineSupportsDebugListener &&
            nativeOwnerLifecycleGate.NativeDetachEntryLocated &&
            nativeOwnerLifecycleGate.ManagedDisposeSnapshotReady &&
            nativeOwnerLifecycleGate.BecausePointerFree();
        bool attachBridgeNoThrowBoundaryReady = attachBridgeShapeReady;
        bool attachBridgeVersionGuardReady = nativeOwnerLifecycleGateReady && lineSupportsDebugListener;
        bool attachBridgeOwnershipDiagnosticsReady =
            attachBridgeShapeReady &&
            nativeOwnerLifecycleGate.NativeOwnerNonCopyableReady &&
            nativeOwnerLifecycleGate.NoThrowNativeDestructorReady;
        const bool attachBridgePointerFree = true;
        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeAttachEntryLocated = false;
        const bool nativeVTableDesignReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for attach bridge shape evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGate.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, attachBridgeShapeReady, "native DebugListener attach bridge shape scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeNoThrowBoundaryReady, "native DebugListener attach bridge no-throw boundary scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeVersionGuardReady, "native DebugListener attach bridge TensorRT version guard scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgeOwnershipDiagnosticsReady, "native DebugListener attach bridge ownership diagnostics scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachBridgePointerFree, "native DebugListener attach bridge shape gate exposes a native pointer.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled before native attach bridge proof.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleGate.NativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not complete beyond source-visible scaffold evidence.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable implementation is not complete.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime attach bridge evidence.");

        foreach (string blocker in nativeOwnerLifecycleGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachBridgeShapeGateResult(
            ownerDesignSnapshot.Line,
            nativeOwnerLifecycleGate.OwnerId,
            nativeOwnerLifecycleGate.LastStatus,
            nativeOwnerLifecycleGateReady,
            attachBridgeShapeReady,
            attachBridgeNoThrowBoundaryReady,
            attachBridgeVersionGuardReady,
            attachBridgeOwnershipDiagnosticsReady,
            attachBridgePointerFree,
            setDebugListenerNonNullEnabled,
            nativeAttachEntryLocated,
            nativeOwnerLifecycleGate.NativeDetachEntryLocated,
            nativeOwnerLifecycleGate.NativeOwnerLifecycleReady,
            nativeVTableDesignReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }

    private static bool BecausePointerFree(this TensorRtDebugListenerNativeOwnerLifecycleGateResult result)
    {
        return !result.NativeOwnerAddressExposed &&
            !result.NativeOwnerPointerProduced &&
            !result.DestructorAddressExposed &&
            !result.DestructorPointerProduced &&
            !result.LifecycleAddressExposed &&
            !result.LifecyclePointerProduced;
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
