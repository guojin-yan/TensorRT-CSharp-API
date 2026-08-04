using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports the final pointer-free DebugListener real callback runtime proof promotion gate.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerRealCallbackRuntimeProofResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerRealCallbackRuntimeProofResult(
        TensorRtApiLine line,
        string runtimePackageKey,
        string runtimeEvidenceKind,
        bool optInEnabled,
        bool fullPackageConsumerReport,
        bool runtimeSmokeReady,
        bool trampolineShapeReady,
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
        bool borrowedDebugTensorMetadataCopied,
        bool pointerFreeSurfaceReady,
        bool processDebugTensorRuntimeReady,
        bool attemptedNoInvocation,
        BridgeStatusCode lastStatus,
        string lastDiagnostic,
        bool canPromoteRealCallbackRuntime,
        string[] blockedPrerequisites)
    {
        Line = line;
        RuntimePackageKey = runtimePackageKey ?? string.Empty;
        RuntimeEvidenceKind = runtimeEvidenceKind ?? "real-callback-runtime-blocked";
        OptInEnabled = optInEnabled;
        FullPackageConsumerReport = fullPackageConsumerReport;
        RuntimeSmokeReady = runtimeSmokeReady;
        TrampolineShapeReady = trampolineShapeReady;
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
        BorrowedDebugTensorMetadataCopied = borrowedDebugTensorMetadataCopied;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        AttemptedNoInvocation = attemptedNoInvocation;
        LastStatus = lastStatus;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this proof gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => CanPromoteRealCallbackRuntime
        ? "real-callback-runtime"
        : "debug-listener-real-callback-runtime-proof";

    /// <summary>Gets the callback kind represented by this proof gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
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

    /// <summary>Gets whether the caller explicitly enabled the runtime proof attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool OptInEnabled { get; }

    /// <summary>Gets whether this report came from a full package consumer smoke. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerReport { get; }

    /// <summary>Gets whether runtime smoke prerequisites are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeSmokeReady { get; }

    /// <summary>Gets whether callback trampoline shape prerequisites are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool TrampolineShapeReady { get; }

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

    /// <summary>Gets the runtime callback invocation count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InvocationCount { get; }

    /// <summary>Gets the copied failure count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InFlightCallbackCount { get; }

    /// <summary>Gets whether borrowed debug tensor metadata was copied into a pointer-free snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopied { get; }

    /// <summary>Gets whether this proof gate remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether a proof attempt ran but observed no callback invocation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttemptedNoInvocation { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied diagnostic text. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets whether this result can be promoted to real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets whether all local prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => OptInEnabled && RuntimeSmokeReady && TrampolineShapeReady && PointerFreeSurfaceReady;

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

    /// <summary>Gets the proof gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime
        ? "ready"
        : AttemptedNoInvocation
            ? "attempted-no-invocation"
            : !OptInEnabled
                ? "skipped"
                : "blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-real-callback-runtime-proof; RuntimeEvidenceKind=" + RuntimeEvidenceKind + "; " +
        "RealCallbackRuntime=" + RealCallbackRuntime + "; IsRealCallbackRuntimeProof=" + IsRealCallbackRuntimeProof + "; " +
        "CallbackKind=" + CallbackKind + "; TensorRtLine=" + TensorRtLine + "; RuntimePackageKey=" + RuntimePackageKey + "; " +
        "OptInEnabled=" + OptInEnabled + "; FullPackageConsumerReport=" + FullPackageConsumerReport + "; " +
        "RuntimeSmokeReady=" + RuntimeSmokeReady + "; TrampolineShapeReady=" + TrampolineShapeReady + "; " +
        "AttachAttempted=" + AttachAttempted + "; AttachSucceeded=" + AttachSucceeded + "; " +
        "DetachAttempted=" + DetachAttempted + "; DetachSucceeded=" + DetachSucceeded + "; " +
        "RollbackAttempted=" + RollbackAttempted + "; RollbackSucceeded=" + RollbackSucceeded + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; ProcessDebugTensorInvoked=" + ProcessDebugTensorInvoked + "; " +
        "InvocationCount=" + InvocationCount + "; FailureCount=" + FailureCount + "; " +
        "InFlightCallbackCount=" + InFlightCallbackCount + "; BorrowedDebugTensorMetadataCopied=" + BorrowedDebugTensorMetadataCopied + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "AttemptedNoInvocation=" + AttemptedNoInvocation + "; LastStatus=" + LastStatus + "; " +
        "LastDiagnostic=" + LastDiagnostic + "; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={TensorRtLine}:status={Status}:invocations={InvocationCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
