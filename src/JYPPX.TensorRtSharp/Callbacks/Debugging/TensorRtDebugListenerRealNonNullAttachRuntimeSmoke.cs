using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Builds a pointer-free report for a disabled-by-default DebugListener non-null attach runtime smoke attempt.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// The report is an opt-in smoke attempt contract. It does not expose native pointers and does not prove
/// <c>IDebugListener::processDebugTensor</c> runtime execution unless full package consumer evidence reports a real
/// TensorRT callback invocation.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerRealNonNullAttachRuntimeSmoke
{
    /// <summary>
    /// Evaluates the runtime smoke attempt report from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimePackageKey">The runtime package key, when known. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="optInEnabled">Whether the caller explicitly enabled the runtime smoke attempt. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="fullPackageConsumerReport">Whether this report came from a full package consumer smoke. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free runtime smoke attempt report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        string runtimePackageKey = "",
        bool optInEnabled = false,
        bool fullPackageConsumerReport = false)
    {
        return Evaluate(
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(ownerDesignSnapshot),
            runtimePackageKey,
            optInEnabled,
            fullPackageConsumerReport);
    }

    /// <summary>
    /// Evaluates the runtime smoke attempt report from proof-attempt preflight evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="attemptPreflight">The copied proof-attempt preflight evidence. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimePackageKey">The runtime package key, when known. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="optInEnabled">Whether the caller explicitly enabled the runtime smoke attempt. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="fullPackageConsumerReport">Whether this report came from a full package consumer smoke. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free runtime smoke attempt report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult Evaluate(
        TensorRtDebugListenerRuntimeProofAttemptPreflightResult attemptPreflight,
        string runtimePackageKey = "",
        bool optInEnabled = false,
        bool fullPackageConsumerReport = false)
    {
        bool attachGuardReady =
            attemptPreflight.CanEnableSetDebugListenerNonNull &&
            !attemptPreflight.NonNullAttachStillDisabled &&
            attemptPreflight.NativeAttachEntryLocated &&
            attemptPreflight.NativeOwnerLifecycleReady &&
            attemptPreflight.CanImplementNativeAttach;
        bool nativeVTableReady =
            attemptPreflight.CanInstallNativeVTable &&
            attemptPreflight.NativeVTableReady &&
            attemptPreflight.NoThrowVTableDesignReady &&
            attemptPreflight.NativeVTableTrampolineReady &&
            attemptPreflight.CallbackExceptionCaptureReady &&
            attemptPreflight.CallbackStatusMappingReady &&
            attemptPreflight.CallbackInFlightAccountingReady &&
            !attemptPreflight.VTableAddressExposed &&
            !attemptPreflight.VTablePointerProduced;
        bool borrowedDebugTensorRuntimeReady =
            attemptPreflight.BorrowedDebugTensorPointerEscapeBlocked &&
            attemptPreflight.BorrowedDebugTensorLifetimeReady &&
            attemptPreflight.BorrowedDebugTensorDataLifetimeReady;
        bool callbackInvocationReady =
            attemptPreflight.CanCallProcessDebugTensorRuntime &&
            attemptPreflight.ProcessDebugTensorRuntimeReady &&
            borrowedDebugTensorRuntimeReady;

        bool attachAttempted = optInEnabled && attachGuardReady;
        bool attachSucceeded = false;
        bool detachAttempted = attachAttempted;
        bool detachSucceeded = false;
        bool rollbackAttempted = optInEnabled && !attachSucceeded;
        bool rollbackSucceeded = rollbackAttempted;
        bool nativeVTableInstalled = false;
        bool processDebugTensorInvoked = false;
        int invocationCount = 0;
        int failureCount = optInEnabled && !attachSucceeded ? 1 : 0;
        int inFlightCallbackCount = 0;
        bool reportPointerFree =
            !attemptPreflight.VTableAddressExposed &&
            !attemptPreflight.VTablePointerProduced &&
            attemptPreflight.BorrowedDebugTensorPointerEscapeBlocked;
        bool canPromoteRealCallbackRuntime =
            fullPackageConsumerReport &&
            attachSucceeded &&
            nativeVTableInstalled &&
            processDebugTensorInvoked &&
            invocationCount > 0 &&
            failureCount == 0 &&
            inFlightCallbackCount == 0 &&
            callbackInvocationReady &&
            attemptPreflight.CanPromoteRealCallbackRuntime;

        string runtimeEvidenceKind = !optInEnabled
            ? "runtime-smoke-skipped"
            : canPromoteRealCallbackRuntime
                ? "real-callback-runtime"
                : attachAttempted
                    ? "runtime-smoke-attempted"
                    : "runtime-smoke-blocked";
        BridgeStatusCode lastStatus = canPromoteRealCallbackRuntime
            ? BridgeStatusCode.Ok
            : optInEnabled
                ? BridgeStatusCode.NotImplemented
                : BridgeStatusCode.NotReady;
        string lastDiagnostic = BuildDiagnostic(
            optInEnabled,
            attachAttempted,
            attachGuardReady,
            nativeVTableReady,
            callbackInvocationReady,
            fullPackageConsumerReport,
            attemptPreflight.ReasonNonNullAttachStillBlocked,
            attemptPreflight.ReasonNativeVTableStillBlocked,
            attemptPreflight.ReasonRuntimeProofStillBlocked);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, optInEnabled, "debug listener real non-null attach runtime smoke was not explicitly enabled.");
        AddBlockerIfFalse(blockers, attachGuardReady, attemptPreflight.ReasonNonNullAttachStillBlocked);
        AddBlockerIfFalse(blockers, nativeVTableReady, attemptPreflight.ReasonNativeVTableStillBlocked);
        AddBlockerIfFalse(blockers, callbackInvocationReady, attemptPreflight.ReasonRuntimeProofStillBlocked);
        AddBlockerIfFalse(blockers, fullPackageConsumerReport, "full package consumer report is required before real-callback-runtime promotion.");
        AddBlockerIfFalse(blockers, attachSucceeded, "setDebugListener(non-null) attach did not succeed.");
        AddBlockerIfFalse(blockers, detachSucceeded, "detach-before-release did not run after a successful attach.");
        AddBlockerIfFalse(blockers, nativeVTableInstalled, "native IDebugListener vtable is not installed.");
        AddBlockerIfFalse(blockers, processDebugTensorInvoked, "IDebugListener::processDebugTensor was not invoked by TensorRT.");
        AddBlockerIfFalse(blockers, invocationCount > 0, "runtime invocation count is zero.");
        AddBlockerIfFalse(blockers, failureCount == 0, "runtime smoke attempt has failure count.");
        AddBlockerIfFalse(blockers, reportPointerFree, "runtime smoke attempt report is not pointer-free.");
        AddBlockerIfFalse(blockers, canPromoteRealCallbackRuntime, "debug listener real non-null attach runtime smoke is not promotable proof.");
        foreach (string blocker in attemptPreflight.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult(
            attemptPreflight.Line,
            string.IsNullOrWhiteSpace(runtimePackageKey) ? string.Empty : runtimePackageKey,
            runtimeEvidenceKind,
            optInEnabled,
            fullPackageConsumerReport,
            attachGuardReady,
            nativeVTableReady,
            borrowedDebugTensorRuntimeReady,
            callbackInvocationReady,
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
            lastStatus,
            lastDiagnostic,
            reportPointerFree,
            canPromoteRealCallbackRuntime,
            blockers.ToArray());
    }

    private static string BuildDiagnostic(
        bool optInEnabled,
        bool attachAttempted,
        bool attachGuardReady,
        bool nativeVTableReady,
        bool callbackInvocationReady,
        bool fullPackageConsumerReport,
        string nonNullReason,
        string nativeVTableReason,
        string runtimeReason)
    {
        if (!optInEnabled)
        {
            return "debug listener real non-null attach runtime smoke skipped because opt-in is disabled.";
        }

        if (!attachAttempted)
        {
            return "debug listener real non-null attach runtime smoke blocked before attach. " +
                "AttachGuardReady=" + attachGuardReady + "; NativeVTableReady=" + nativeVTableReady + "; " +
                "CallbackInvocationReady=" + callbackInvocationReady + "; FullPackageConsumerReport=" + fullPackageConsumerReport + "; " +
                nonNullReason + " " + nativeVTableReason + " " + runtimeReason;
        }

        return "debug listener real non-null attach runtime smoke attempted but cannot be promoted without a successful " +
            "full package consumer real-callback-runtime report.";
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
