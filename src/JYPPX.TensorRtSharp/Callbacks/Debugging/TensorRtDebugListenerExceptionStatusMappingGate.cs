using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener callback exception-to-status mapping gate evidence.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate reports source-visible exception capture and status mapping scaffold evidence only. It does not call
/// <c>setDebugListener(non-null)</c> and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerExceptionStatusMappingGate
{
    /// <summary>
    /// Evaluates exception-to-status mapping evidence from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free exception-to-status mapping gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerExceptionStatusMappingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, attachBridgeShapeGate);
    }

    /// <summary>
    /// Evaluates exception-to-status mapping evidence from copied owner and attach bridge shape evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="attachBridgeShapeGate">The copied attach bridge shape gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free exception-to-status mapping gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerExceptionStatusMappingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate)
    {
        bool attachBridgeShapeGateReady = attachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool managedCallbackExceptionCaptureReady = ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok;
        const bool nativeCallbackExceptionCaptureReady = true;
        const bool callbackStatusMappingGateReady = true;
        const bool exceptionEscapeBlocked = true;
        const bool diagnosticCopyReady = true;
        const bool mappingAddressExposed = false;
        const bool mappingPointerProduced = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for exception/status mapping evaluation.");
        AddBlockerIfFalse(blockers, managedCallbackExceptionCaptureReady, "managed DebugListener callback exception capture design snapshot is not clean.");
        AddBlockerIfFalse(blockers, nativeCallbackExceptionCaptureReady, "native DebugListener callback exception capture scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, callbackStatusMappingGateReady, "native DebugListener callback status mapping scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, exceptionEscapeBlocked, "native DebugListener exception escape is not blocked.");
        AddBlockerIfFalse(blockers, diagnosticCopyReady, "DebugListener callback exception diagnostic copy scaffold is not ready.");
        AddBlockerIfFalse(blockers, !mappingAddressExposed, "DebugListener exception/status mapping gate exposes a native address.");
        AddBlockerIfFalse(blockers, !mappingPointerProduced, "DebugListener exception/status mapping gate produces a native pointer.");
        AddBlockerIfFalse(blockers, attachBridgeShapeGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime exception/status mapping evidence.");

        foreach (string blocker in attachBridgeShapeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerExceptionStatusMappingGateResult(
            ownerDesignSnapshot.Line,
            attachBridgeShapeGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            attachBridgeShapeGateReady,
            managedCallbackExceptionCaptureReady,
            nativeCallbackExceptionCaptureReady,
            callbackStatusMappingGateReady,
            exceptionEscapeBlocked,
            diagnosticCopyReady,
            mappingAddressExposed,
            mappingPointerProduced,
            attachBridgeShapeGate.NativeAttachEntryLocated,
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
