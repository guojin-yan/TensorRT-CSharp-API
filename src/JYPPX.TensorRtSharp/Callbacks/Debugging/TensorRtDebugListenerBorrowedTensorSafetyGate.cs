using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates copied DebugListener borrowed tensor/data lifetime rules before any real TensorRT callback runtime proof exists.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate models borrowed tensor safety only. It does not consume or return a TensorRT tensor pointer, a debug tensor
/// data pointer, or any native callback owner pointer.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerBorrowedTensorSafetyGate
{
    /// <summary>
    /// Evaluates borrowed tensor safety from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free borrowed tensor safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerBorrowedTensorSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, attachDetachGate);
    }

    /// <summary>
    /// Evaluates borrowed tensor safety from copied owner and attach/detach design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free borrowed tensor safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerBorrowedTensorSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate)
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
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        const bool borrowedDebugTensorPointerEscapeBlocked = true;
        const bool borrowedDebugTensorLifetimeReady = false;
        const bool borrowedDebugTensorDataLifetimeReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        if (!ownerDesignReady)
        {
            blockers.Add("debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!attachDetachDesignGate.DesignGateReady)
        {
            blockers.Add("debug-listener-attach-detach-design-gate is not ready for borrowed tensor safety evaluation.");
        }

        if (!debugTensorMetadataCopied)
        {
            blockers.Add("debug tensor copied metadata is incomplete.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public DebugListener borrowed tensor surface still exposes, produces, or leaks a debug tensor pointer.");
        }

        if (!borrowedDebugTensorLifetimeReady)
        {
            blockers.Add("borrowed debug tensor pointer lifetime has not been proven against a real TensorRT callback.");
        }

        if (!borrowedDebugTensorDataLifetimeReady)
        {
            blockers.Add("borrowed debug tensor data buffer lifetime has not been proven against a real TensorRT callback.");
        }

        if (!processDebugTensorRuntimeReady)
        {
            blockers.Add("IDebugListener::processDebugTensor runtime callback has not been implemented.");
        }

        if (!fullPackageConsumerRuntimeEvidenceReady)
        {
            blockers.Add("full package consumer smoke has not emitted real-callback-runtime debug listener borrowed tensor evidence.");
        }

        return new TensorRtDebugListenerBorrowedTensorSafetyGateResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.TensorName,
            ownerDesignSnapshot.DataType,
            ownerDesignSnapshot.Location,
            ownerDesignSnapshot.ShapeRank,
            ownerDesignSnapshot.ShapeSummary,
            ownerDesignSnapshot.IsInput,
            ownerDesignSnapshot.IsOutput,
            ownerDesignSnapshot.IsShapeTensor,
            ownerDesignSnapshot.IsExecutionTensor,
            ownerDesignSnapshot.ProcessDebugTensorCount,
            attachDetachDesignGate.DesignGateReady,
            ownerDesignReady,
            debugTensorMetadataCopied,
            pointerFreeSurfaceReady,
            borrowedDebugTensorPointerEscapeBlocked,
            borrowedDebugTensorLifetimeReady,
            borrowedDebugTensorDataLifetimeReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }
}
