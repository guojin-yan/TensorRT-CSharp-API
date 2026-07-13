using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

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

/// <summary>
/// Reports a pointer-free DebugListener non-null attach runtime smoke attempt.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult(
        TensorRtApiLine line,
        string runtimePackageKey,
        string runtimeEvidenceKind,
        bool optInEnabled,
        bool fullPackageConsumerReport,
        bool attachGuardReady,
        bool nativeVTableReady,
        bool borrowedDebugTensorRuntimeReady,
        bool callbackInvocationReady,
        bool attachAttempted,
        bool attachSucceeded,
        bool detachAttempted,
        bool detachSucceeded,
        bool rollbackAttempted,
        bool rollbackSucceeded,
        bool nativeVTableInstalled,
        bool processDebugTensorInvoked,
        int invocationCount,
        int failureCount,
        int inFlightCallbackCount,
        BridgeStatusCode lastStatus,
        string lastDiagnostic,
        bool reportPointerFree,
        bool canPromoteRealCallbackRuntime,
        string[] blockedPrerequisites)
    {
        Line = line;
        RuntimePackageKey = runtimePackageKey ?? string.Empty;
        RuntimeEvidenceKind = runtimeEvidenceKind ?? "runtime-smoke-blocked";
        OptInEnabled = optInEnabled;
        FullPackageConsumerReport = fullPackageConsumerReport;
        AttachGuardReady = attachGuardReady;
        NativeVTableReady = nativeVTableReady;
        BorrowedDebugTensorRuntimeReady = borrowedDebugTensorRuntimeReady;
        CallbackInvocationReady = callbackInvocationReady;
        AttachAttempted = attachAttempted;
        AttachSucceeded = attachSucceeded;
        DetachAttempted = detachAttempted;
        DetachSucceeded = detachSucceeded;
        RollbackAttempted = rollbackAttempted;
        RollbackSucceeded = rollbackSucceeded;
        NativeVTableInstalled = nativeVTableInstalled;
        ProcessDebugTensorInvoked = processDebugTensorInvoked;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        LastStatus = lastStatus;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReportPointerFree = reportPointerFree;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this runtime smoke attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => CanPromoteRealCallbackRuntime ? "real-callback-runtime" : "debug-listener-real-non-null-attach-runtime-smoke";

    /// <summary>Gets the callback kind represented by this smoke report. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind { get; }

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets the TensorRT API line represented by this report. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the TensorRT API line as an integer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int TensorRtLine => (int)Line;

    /// <summary>Gets the runtime package key, when known. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimePackageKey { get; }

    /// <summary>Gets whether the caller explicitly enabled the runtime smoke attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool OptInEnabled { get; }

    /// <summary>Gets whether this report came from a full package consumer smoke. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerReport { get; }

    /// <summary>Gets whether non-null attach guards are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachGuardReady { get; }

    /// <summary>Gets whether native vtable prerequisites are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableReady { get; }

    /// <summary>Gets whether borrowed debug tensor runtime lifetime prerequisites are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether callback invocation prerequisites are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInvocationReady { get; }

    /// <summary>Gets whether non-null attach was attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachAttempted { get; }

    /// <summary>Gets whether non-null attach succeeded. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachSucceeded { get; }

    /// <summary>Gets whether detach was attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachAttempted { get; }

    /// <summary>Gets whether detach succeeded. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachSucceeded { get; }

    /// <summary>Gets whether rollback was attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RollbackAttempted { get; }

    /// <summary>Gets whether rollback succeeded. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RollbackSucceeded { get; }

    /// <summary>Gets whether a native IDebugListener vtable was installed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether TensorRT invoked processDebugTensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorInvoked { get; }

    /// <summary>Gets the copied callback invocation count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InvocationCount { get; }

    /// <summary>Gets the copied allocation count. DebugListener does not allocate through this report. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int AllocationCount => 0;

    /// <summary>Gets the copied release count. DebugListener does not release allocations through this report. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int ReleaseCount => DetachSucceeded ? 1 : 0;

    /// <summary>Gets the copied failure count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InFlightCallbackCount { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied diagnostic text. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets whether this report keeps raw native pointers out of public API. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ReportPointerFree { get; }

    /// <summary>Gets whether this result can be promoted to real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => OptInEnabled && AttachGuardReady && NativeVTableReady && CallbackInvocationReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets why runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonRuntimeProofStillBlocked => LastDiagnostic;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the smoke attempt status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime
        ? "ready"
        : !OptInEnabled
            ? "skipped"
            : AttachAttempted
                ? "attempted"
                : "blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-real-non-null-attach-runtime-smoke; RuntimeEvidenceKind=" + RuntimeEvidenceKind + "; " +
        "RealCallbackRuntime=" + RealCallbackRuntime + "; IsRealCallbackRuntimeProof=" + IsRealCallbackRuntimeProof + "; " +
        "CallbackKind=" + CallbackKind + "; TensorRtLine=" + TensorRtLine + "; RuntimePackageKey=" + RuntimePackageKey + "; " +
        "OptInEnabled=" + OptInEnabled + "; FullPackageConsumerReport=" + FullPackageConsumerReport + "; " +
        "AttachGuardReady=" + AttachGuardReady + "; NativeVTableReady=" + NativeVTableReady + "; " +
        "BorrowedDebugTensorRuntimeReady=" + BorrowedDebugTensorRuntimeReady + "; CallbackInvocationReady=" + CallbackInvocationReady + "; " +
        "AttachAttempted=" + AttachAttempted + "; AttachSucceeded=" + AttachSucceeded + "; " +
        "DetachAttempted=" + DetachAttempted + "; DetachSucceeded=" + DetachSucceeded + "; " +
        "RollbackAttempted=" + RollbackAttempted + "; RollbackSucceeded=" + RollbackSucceeded + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; ProcessDebugTensorInvoked=" + ProcessDebugTensorInvoked + "; " +
        "InvocationCount=" + InvocationCount + "; AllocationCount=" + AllocationCount + "; ReleaseCount=" + ReleaseCount + "; " +
        "FailureCount=" + FailureCount + "; InFlightCallbackCount=" + InFlightCallbackCount + "; " +
        "LastStatus=" + LastStatus + "; LastDiagnostic=" + LastDiagnostic + "; " +
        "ReportPointerFree=" + ReportPointerFree + "; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={TensorRtLine}:status={Status}:optIn={OptInEnabled}:proof={IsRealCallbackRuntimeProof}";
    }
}
