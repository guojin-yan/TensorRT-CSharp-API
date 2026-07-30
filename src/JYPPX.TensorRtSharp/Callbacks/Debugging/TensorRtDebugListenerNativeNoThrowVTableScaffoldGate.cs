using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native no-throw vtable scaffold gate evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate reports source-visible vtable scaffold, exception/status mapping, and in-flight accounting evidence. It
/// does not install a native <c>IDebugListener</c> into TensorRT and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeNoThrowVTableScaffoldGate
{
    /// <summary>
    /// Evaluates no-throw vtable scaffold evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free no-throw vtable scaffold gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot, attachBridgeShapeGate);
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
        return Evaluate(ownerDesignSnapshot, attachBridgeShapeGate, exceptionStatusMappingGate, inFlightAccountingGate);
    }

    /// <summary>
    /// Evaluates no-throw vtable scaffold evidence from copied owner, attach bridge, mapping, and accounting gates.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachBridgeShapeGate">The copied attach bridge shape gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="exceptionStatusMappingGate">The copied exception/status mapping gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="inFlightAccountingGate">The copied in-flight accounting gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free no-throw vtable scaffold gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate,
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate,
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate)
    {
        bool nativeAttachBridgeShapeGateReady = attachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool exceptionStatusMappingGateReady = exceptionStatusMappingGate.ExceptionStatusMappingGateReady;
        bool inFlightAccountingGateReady = inFlightAccountingGate.InFlightAccountingGateReady;
        bool noThrowVTableScaffoldReady =
            nativeAttachBridgeShapeGateReady &&
            exceptionStatusMappingGateReady &&
            inFlightAccountingGateReady;
        bool vTableDestructorNoThrowReady = noThrowVTableScaffoldReady;
        bool processDebugTensorCallbackStubNoThrowReady = noThrowVTableScaffoldReady;
        bool exceptionEscapeBlocked =
            exceptionStatusMappingGate.ExceptionEscapeBlocked &&
            processDebugTensorCallbackStubNoThrowReady;
        bool callbackExceptionCaptureGateReady = exceptionStatusMappingGate.NativeCallbackExceptionCaptureReady;
        bool callbackStatusMappingGateReady = exceptionStatusMappingGate.CallbackStatusMappingGateReady;
        bool callbackInFlightAccountingGateReady = inFlightAccountingGate.InFlightAccountingGateReady;
        const bool borrowedDebugTensorPointerEscapeBlocked = true;
        const bool vtableAddressExposed = false;
        const bool vtablePointerProduced = false;
        const bool nativeVTableDesignReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeAttachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, exceptionStatusMappingGateReady, "debug-listener-exception-status-mapping-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, inFlightAccountingGateReady, "debug-listener-inflight-accounting-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, noThrowVTableScaffoldReady, "native IDebugListener no-throw vtable scaffold is incomplete.");
        AddBlockerIfFalse(blockers, vTableDestructorNoThrowReady, "native IDebugListener vtable destructor no-throw scaffold is incomplete.");
        AddBlockerIfFalse(blockers, processDebugTensorCallbackStubNoThrowReady, "native IDebugListener processDebugTensor no-throw callback stub scaffold is incomplete.");
        AddBlockerIfFalse(blockers, exceptionEscapeBlocked, "native IDebugListener callback exception escape is not blocked.");
        AddBlockerIfFalse(blockers, callbackExceptionCaptureGateReady, "native IDebugListener callback exception capture gate is incomplete.");
        AddBlockerIfFalse(blockers, callbackStatusMappingGateReady, "native IDebugListener callback status mapping gate is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightAccountingGateReady, "native IDebugListener callback in-flight accounting gate is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, !vtableAddressExposed, "native IDebugListener vtable scaffold exposes a native address.");
        AddBlockerIfFalse(blockers, !vtablePointerProduced, "native IDebugListener vtable scaffold produces a native pointer.");
        AddBlockerIfFalse(blockers, attachBridgeShapeGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable implementation is not complete beyond scaffold evidence.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime vtable scaffold evidence.");

        foreach (string blocker in attachBridgeShapeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in exceptionStatusMappingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in inFlightAccountingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult(
            ownerDesignSnapshot.Line,
            attachBridgeShapeGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            nativeAttachBridgeShapeGateReady,
            exceptionStatusMappingGateReady,
            inFlightAccountingGateReady,
            noThrowVTableScaffoldReady,
            vTableDestructorNoThrowReady,
            processDebugTensorCallbackStubNoThrowReady,
            exceptionEscapeBlocked,
            callbackExceptionCaptureGateReady,
            callbackStatusMappingGateReady,
            callbackInFlightAccountingGateReady,
            borrowedDebugTensorPointerEscapeBlocked,
            vtableAddressExposed,
            vtablePointerProduced,
            attachBridgeShapeGate.NativeAttachEntryLocated,
            nativeVTableDesignReady,
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
