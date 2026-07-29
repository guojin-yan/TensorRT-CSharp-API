using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the final, pointer-free DebugListener real callback runtime proof promotion gate.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate does not install a native <c>IDebugListener</c> vtable and does not synthesize callback invocations.
/// It promotes evidence only when a full package consumer report proves that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerRealCallbackRuntimeProof
{
    /// <summary>
    /// Evaluates the proof promotion gate from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimePackageKey">The runtime package key, when known. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="optInEnabled">Whether the caller explicitly enabled the runtime proof attempt. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="fullPackageConsumerReport">Whether this report came from a full package consumer smoke. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free real callback runtime proof gate report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRealCallbackRuntimeProofResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        string runtimePackageKey = "",
        bool optInEnabled = false,
        bool fullPackageConsumerReport = false)
    {
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke =
            TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
                ownerDesignSnapshot,
                runtimePackageKey,
                optInEnabled,
                fullPackageConsumerReport);
        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult trampoline =
            TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(
                ownerDesignSnapshot,
                runtimePackageKey,
                optInEnabled,
                fullPackageConsumerReport);
        return Evaluate(runtimeSmoke, trampoline);
    }

    /// <summary>
    /// Evaluates the proof promotion gate from copied lower-level reports.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="runtimeSmoke">The copied real non-null attach runtime smoke report. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="callbackTrampoline">The copied processDebugTensor callback trampoline report. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free real callback runtime proof gate report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRealCallbackRuntimeProofResult Evaluate(
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke,
        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult callbackTrampoline)
    {
        bool optInEnabled = runtimeSmoke.OptInEnabled || callbackTrampoline.OptInEnabled;
        bool fullPackageConsumerReport =
            runtimeSmoke.FullPackageConsumerReport &&
            callbackTrampoline.FullPackageConsumerReport;
        bool attachAttempted =
            runtimeSmoke.AttachAttempted ||
            callbackTrampoline.AttachAttempted;
        bool attachSucceeded =
            runtimeSmoke.AttachSucceeded &&
            callbackTrampoline.AttachSucceeded;
        bool detachAttempted = runtimeSmoke.DetachAttempted;
        bool detachSucceeded = runtimeSmoke.DetachSucceeded || !attachSucceeded;
        bool rollbackAttempted = runtimeSmoke.RollbackAttempted;
        bool rollbackSucceeded = runtimeSmoke.RollbackSucceeded || !rollbackAttempted;
        bool nativeVTableInstalled =
            runtimeSmoke.NativeVTableInstalled &&
            callbackTrampoline.NativeVTableInstalled;
        bool processDebugTensorInvoked =
            runtimeSmoke.ProcessDebugTensorInvoked &&
            callbackTrampoline.ProcessDebugTensorInvoked;
        int invocationCount = Math.Min(runtimeSmoke.InvocationCount, callbackTrampoline.InvocationCount);
        int failureCount = runtimeSmoke.FailureCount + callbackTrampoline.FailureCount;
        int inFlightCallbackCount = Math.Max(runtimeSmoke.InFlightCallbackCount, callbackTrampoline.InFlightCallbackCount);
        bool borrowedDebugTensorMetadataCopied =
            callbackTrampoline.BorrowedDebugTensorMetadataCopyReady &&
            callbackTrampoline.Metadata.MetadataCopied;
        bool pointerFreeSurfaceReady =
            runtimeSmoke.ReportPointerFree &&
            callbackTrampoline.PointerFreeSurfaceReady &&
            !callbackTrampoline.BorrowedDebugTensorPointerExposed &&
            !callbackTrampoline.BorrowedDebugTensorDataPointerExposed;
        bool trampolineShapeReady = callbackTrampoline.TrampolineShapeReady;
        bool runtimeSmokeReady = runtimeSmoke.AttachGuardReady && runtimeSmoke.NativeVTableReady && runtimeSmoke.CallbackInvocationReady;
        bool processDebugTensorRuntimeReady =
            runtimeSmoke.ProcessDebugTensorInvoked &&
            callbackTrampoline.ProcessDebugTensorRuntimeReady;
        bool attemptedNoInvocation =
            optInEnabled &&
            attachAttempted &&
            !processDebugTensorInvoked &&
            invocationCount == 0;

        bool canPromoteRealCallbackRuntime =
            fullPackageConsumerReport &&
            runtimeSmoke.IsRealCallbackRuntimeProof &&
            callbackTrampoline.IsRealCallbackRuntimeProof &&
            attachSucceeded &&
            detachSucceeded &&
            rollbackSucceeded &&
            nativeVTableInstalled &&
            processDebugTensorInvoked &&
            invocationCount > 0 &&
            failureCount == 0 &&
            inFlightCallbackCount == 0 &&
            borrowedDebugTensorMetadataCopied &&
            pointerFreeSurfaceReady &&
            trampolineShapeReady &&
            processDebugTensorRuntimeReady;

        string runtimeEvidenceKind = canPromoteRealCallbackRuntime
            ? "real-callback-runtime"
            : attemptedNoInvocation
                ? "attempted-no-invocation"
                : optInEnabled
                    ? "real-callback-runtime-blocked"
                    : "runtime-smoke-skipped";
        BridgeStatusCode lastStatus = canPromoteRealCallbackRuntime
            ? BridgeStatusCode.Ok
            : optInEnabled
                ? BridgeStatusCode.NotImplemented
                : BridgeStatusCode.NotReady;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, optInEnabled, "debug listener real callback runtime proof was not explicitly enabled.");
        AddBlockerIfFalse(blockers, fullPackageConsumerReport, "full package consumer report is required before real-callback-runtime promotion.");
        AddBlockerIfFalse(blockers, runtimeSmokeReady, "real non-null attach runtime smoke prerequisites are not ready.");
        AddBlockerIfFalse(blockers, trampolineShapeReady, "processDebugTensor callback trampoline shape is not ready.");
        AddBlockerIfFalse(blockers, attachAttempted, "setDebugListener(non-null) attach was not attempted.");
        AddBlockerIfFalse(blockers, attachSucceeded, "setDebugListener(non-null) attach did not succeed.");
        AddBlockerIfFalse(blockers, detachAttempted || !attachSucceeded, "detach was not attempted after attach.");
        AddBlockerIfFalse(blockers, detachSucceeded, "detach-before-release did not succeed.");
        AddBlockerIfFalse(blockers, rollbackSucceeded, "rollback did not succeed after a failed attempt.");
        AddBlockerIfFalse(blockers, nativeVTableInstalled, "native IDebugListener vtable was not installed.");
        AddBlockerIfFalse(blockers, processDebugTensorInvoked, "TensorRT did not invoke IDebugListener::processDebugTensor.");
        AddBlockerIfFalse(blockers, invocationCount > 0, "runtime invocation count is zero.");
        AddBlockerIfFalse(blockers, failureCount == 0, "runtime proof attempt has failure count.");
        AddBlockerIfFalse(blockers, inFlightCallbackCount == 0, "callback in-flight count is not zero.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopied, "borrowed debug tensor metadata was not copied.");
        AddBlockerIfFalse(blockers, pointerFreeSurfaceReady, "real callback runtime proof report is not pointer-free.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "processDebugTensor runtime execution is not ready.");
        AddBlockerIfFalse(blockers, canPromoteRealCallbackRuntime, "real-callback-runtime proof markers are incomplete or not backed by full package consumer invocation evidence.");
        foreach (string blocker in runtimeSmoke.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in callbackTrampoline.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        string diagnostic = BuildDiagnostic(
            optInEnabled,
            fullPackageConsumerReport,
            attemptedNoInvocation,
            canPromoteRealCallbackRuntime,
            invocationCount,
            runtimeSmoke.LastDiagnostic,
            callbackTrampoline.LastDiagnostic);

        return new TensorRtDebugListenerRealCallbackRuntimeProofResult(
            callbackTrampoline.Line,
            runtimeSmoke.RuntimePackageKey,
            runtimeEvidenceKind,
            optInEnabled,
            fullPackageConsumerReport,
            runtimeSmokeReady,
            trampolineShapeReady,
            attachAttempted,
            attachSucceeded,
            detachAttempted,
            detachSucceeded,
            rollbackAttempted,
            rollbackSucceeded,
            nativeVTableInstalled,
            processDebugTensorInvoked,
            invocationCount,
            failureCount,
            inFlightCallbackCount,
            borrowedDebugTensorMetadataCopied,
            pointerFreeSurfaceReady,
            processDebugTensorRuntimeReady,
            attemptedNoInvocation,
            lastStatus,
            diagnostic,
            canPromoteRealCallbackRuntime,
            blockers.ToArray());
    }

    private static string BuildDiagnostic(
        bool optInEnabled,
        bool fullPackageConsumerReport,
        bool attemptedNoInvocation,
        bool canPromoteRealCallbackRuntime,
        int invocationCount,
        string runtimeSmokeDiagnostic,
        string callbackTrampolineDiagnostic)
    {
        if (canPromoteRealCallbackRuntime)
        {
            return "debug listener real callback runtime proof was promoted from full package consumer invocation evidence.";
        }

        if (!optInEnabled)
        {
            return "debug listener real callback runtime proof skipped because opt-in is disabled.";
        }

        if (attemptedNoInvocation)
        {
            return "debug listener real callback runtime proof was attempted but TensorRT did not invoke processDebugTensor; InvocationCount=" + invocationCount + ".";
        }

        return "debug listener real callback runtime proof is blocked; FullPackageConsumerReport=" + fullPackageConsumerReport +
            "; InvocationCount=" + invocationCount +
            "; runtime smoke: " + (runtimeSmokeDiagnostic ?? string.Empty) +
            "; callback trampoline: " + (callbackTrampolineDiagnostic ?? string.Empty);
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
