using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener callback in-flight accounting gate evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate reports source-visible enter/leave accounting and release-after-drain scaffold evidence only. It does not
/// attach a listener to TensorRT and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerInFlightAccountingGate
{
    /// <summary>
    /// Evaluates in-flight accounting evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free in-flight accounting gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerInFlightAccountingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
    }

    /// <summary>
    /// Evaluates in-flight accounting evidence from copied owner and exception/status mapping evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="exceptionStatusMappingGate">The copied exception/status mapping gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free in-flight accounting gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerInFlightAccountingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate)
    {
        bool exceptionStatusMappingGateReady = exceptionStatusMappingGate.ExceptionStatusMappingGateReady;
        bool callbackEnterAccountingGateReady = ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool callbackLeaveAccountingGateReady = ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool callbackInFlightNeverNegativeReady = ownerDesignSnapshot.InFlightCallbackCount >= 0;
        bool releaseAfterDrainGateReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool callbackStateUnpinAfterDrainGateReady =
            releaseAfterDrainGateReady &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned;
        const bool accountingAddressExposed = false;
        const bool accountingPointerProduced = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, exceptionStatusMappingGateReady, "debug-listener-exception-status-mapping-gate is not ready for in-flight accounting evaluation.");
        AddBlockerIfFalse(blockers, callbackEnterAccountingGateReady, "DebugListener callback enter accounting scaffold has not observed copied diagnostic metadata.");
        AddBlockerIfFalse(blockers, callbackLeaveAccountingGateReady, "DebugListener callback leave accounting did not drain to zero.");
        AddBlockerIfFalse(blockers, callbackInFlightNeverNegativeReady, "DebugListener in-flight callback count is invalid.");
        AddBlockerIfFalse(blockers, releaseAfterDrainGateReady, "DebugListener release-after-drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackStateUnpinAfterDrainGateReady, "DebugListener callback state/delegate unpin after drain evidence is incomplete.");
        AddBlockerIfFalse(blockers, !accountingAddressExposed, "DebugListener in-flight accounting gate exposes a native address.");
        AddBlockerIfFalse(blockers, !accountingPointerProduced, "DebugListener in-flight accounting gate produces a native pointer.");
        AddBlockerIfFalse(blockers, exceptionStatusMappingGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime in-flight accounting evidence.");

        foreach (string blocker in exceptionStatusMappingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerInFlightAccountingGateResult(
            ownerDesignSnapshot.Line,
            exceptionStatusMappingGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.ProcessDebugTensorCount,
            ownerDesignSnapshot.InFlightCallbackCount,
            ownerDesignSnapshot.MaxInFlightCallbackCount,
            ownerDesignSnapshot.ReleaseHookCount,
            ownerDesignSnapshot.CallbackStatePinned,
            ownerDesignSnapshot.DelegatePinned,
            ownerDesignSnapshot.DisposeRequested,
            exceptionStatusMappingGateReady,
            callbackEnterAccountingGateReady,
            callbackLeaveAccountingGateReady,
            callbackInFlightNeverNegativeReady,
            releaseAfterDrainGateReady,
            callbackStateUnpinAfterDrainGateReady,
            accountingAddressExposed,
            accountingPointerProduced,
            exceptionStatusMappingGate.NativeAttachEntryLocated,
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
