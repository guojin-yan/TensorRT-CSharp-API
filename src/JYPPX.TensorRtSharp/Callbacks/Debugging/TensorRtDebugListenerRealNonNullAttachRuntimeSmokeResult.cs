using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

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
