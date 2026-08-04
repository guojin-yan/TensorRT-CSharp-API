using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports pointer-free DebugListener <c>processDebugTensor</c> callback trampoline diagnostics.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult(
        TensorRtApiLine line,
        string runtimePackageKey,
        string runtimeEvidenceKind,
        TensorRtDebugTensorMetadataSnapshot metadata,
        bool trampolineShapeReady,
        bool nativeCallbackEntryLocated,
        bool noThrowCallbackEntryReady,
        bool exceptionCaptureReady,
        bool callbackStatusMappingReady,
        bool inFlightAccountingReady,
        bool detachBeforeReleaseReady,
        bool borrowedDebugTensorMetadataCopyReady,
        bool borrowedDebugTensorPointerExposed,
        bool borrowedDebugTensorDataPointerExposed,
        bool pointerFreeSurfaceReady,
        bool processDebugTensorRuntimeReady,
        bool optInEnabled,
        bool fullPackageConsumerReport,
        bool attachAttempted,
        bool attachSucceeded,
        bool nativeVTableInstalled,
        bool processDebugTensorInvoked,
        int invocationCount,
        long callbackStubEntryCount,
        long callbackStubLeaveCount,
        int failureCount,
        int inFlightCallbackCount,
        BridgeStatusCode lastStatus,
        string lastDiagnostic,
        bool canPromoteRealCallbackRuntime,
        string[] blockedPrerequisites)
    {
        Line = line;
        RuntimePackageKey = runtimePackageKey ?? string.Empty;
        RuntimeEvidenceKind = runtimeEvidenceKind ?? "callback-trampoline-shape";
        Metadata = metadata;
        TrampolineShapeReady = trampolineShapeReady;
        NativeCallbackEntryLocated = nativeCallbackEntryLocated;
        NoThrowCallbackEntryReady = noThrowCallbackEntryReady;
        ExceptionCaptureReady = exceptionCaptureReady;
        CallbackStatusMappingReady = callbackStatusMappingReady;
        InFlightAccountingReady = inFlightAccountingReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        BorrowedDebugTensorMetadataCopyReady = borrowedDebugTensorMetadataCopyReady;
        BorrowedDebugTensorPointerExposed = borrowedDebugTensorPointerExposed;
        BorrowedDebugTensorDataPointerExposed = borrowedDebugTensorDataPointerExposed;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        OptInEnabled = optInEnabled;
        FullPackageConsumerReport = fullPackageConsumerReport;
        AttachAttempted = attachAttempted;
        AttachSucceeded = attachSucceeded;
        NativeVTableInstalled = nativeVTableInstalled;
        ProcessDebugTensorInvoked = processDebugTensorInvoked;
        InvocationCount = invocationCount;
        CallbackStubEntryCount = callbackStubEntryCount;
        CallbackStubLeaveCount = callbackStubLeaveCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        LastStatus = lastStatus;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this callback trampoline. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => CanPromoteRealCallbackRuntime
        ? "real-callback-runtime"
        : "debug-listener-process-debug-tensor-callback-trampoline";

    /// <summary>Gets the callback kind represented by this trampoline report. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind { get; }

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets the TensorRT API line represented by the copied evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the TensorRT API line as an integer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int TensorRtLine => (int)Line;

    /// <summary>Gets the runtime package key, when known. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimePackageKey { get; }

    /// <summary>Gets copied pointer-free debug tensor metadata. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtDebugTensorMetadataSnapshot Metadata { get; }

    /// <summary>Gets whether private/internal trampoline shape evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TrampolineShapeReady { get; }

    /// <summary>Gets whether a native callback entry shape is source-visible. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeCallbackEntryLocated { get; }

    /// <summary>Gets whether the callback entry is no-throw ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowCallbackEntryReady { get; }

    /// <summary>Gets whether callback exception capture evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingReady { get; }

    /// <summary>Gets whether callback in-flight accounting evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingReady { get; }

    /// <summary>Gets whether detach-before-release sequencing is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether borrowed debug tensor metadata copy evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyReady { get; }

    /// <summary>Gets whether a borrowed debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerExposed { get; }

    /// <summary>Gets whether a borrowed debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether public trampoline evidence remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether the caller explicitly enabled runtime smoke. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool OptInEnabled { get; }

    /// <summary>Gets whether this report came from a full package consumer smoke. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerReport { get; }

    /// <summary>Gets whether non-null attach was attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachAttempted { get; }

    /// <summary>Gets whether non-null attach succeeded. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachSucceeded { get; }

    /// <summary>Gets whether a native IDebugListener vtable was installed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether TensorRT invoked processDebugTensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorInvoked { get; }

    /// <summary>Gets the runtime callback invocation count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int InvocationCount { get; }

    /// <summary>Gets the copied source-visible callback stub entry count. This is not runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long CallbackStubEntryCount { get; }

    /// <summary>Gets the copied source-visible callback stub leave count. This is not runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long CallbackStubLeaveCount { get; }

    /// <summary>Gets the runtime failure count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int FailureCount { get; }

    /// <summary>Gets the runtime in-flight callback count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int InFlightCallbackCount { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied diagnostic text. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets whether this result can be promoted to real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => OptInEnabled && TrampolineShapeReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets why runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReasonRuntimeProofStillBlocked => LastDiagnostic;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the callback trampoline status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime
        ? "ready"
        : TrampolineShapeReady
            ? "callback-trampoline-shape-ready"
            : "callback-trampoline-shape-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-process-debug-tensor-callback-trampoline; RuntimeEvidenceKind=" + RuntimeEvidenceKind + "; " +
        "RealCallbackRuntime=" + RealCallbackRuntime + "; IsRealCallbackRuntimeProof=" + IsRealCallbackRuntimeProof + "; " +
        "CallbackKind=" + CallbackKind + "; TensorRtLine=" + TensorRtLine + "; RuntimePackageKey=" + RuntimePackageKey + "; " +
        "TrampolineShapeReady=" + TrampolineShapeReady + "; NativeCallbackEntryLocated=" + NativeCallbackEntryLocated + "; " +
        "NoThrowCallbackEntryReady=" + NoThrowCallbackEntryReady + "; ExceptionCaptureReady=" + ExceptionCaptureReady + "; " +
        "CallbackStatusMappingReady=" + CallbackStatusMappingReady + "; InFlightAccountingReady=" + InFlightAccountingReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "BorrowedDebugTensorMetadataCopyReady=" + BorrowedDebugTensorMetadataCopyReady + "; " +
        "BorrowedDebugTensorPointerExposed=" + BorrowedDebugTensorPointerExposed + "; " +
        "BorrowedDebugTensorDataPointerExposed=" + BorrowedDebugTensorDataPointerExposed + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "OptInEnabled=" + OptInEnabled + "; FullPackageConsumerReport=" + FullPackageConsumerReport + "; " +
        "AttachAttempted=" + AttachAttempted + "; AttachSucceeded=" + AttachSucceeded + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; ProcessDebugTensorInvoked=" + ProcessDebugTensorInvoked + "; " +
        "InvocationCount=" + InvocationCount + "; CallbackStubEntryCount=" + CallbackStubEntryCount + "; " +
        "CallbackStubLeaveCount=" + CallbackStubLeaveCount + "; FailureCount=" + FailureCount + "; " +
        "InFlightCallbackCount=" + InFlightCallbackCount + "; LastStatus=" + LastStatus + "; " +
        "LastDiagnostic=" + LastDiagnostic + "; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={TensorRtLine}:status={Status}:trampoline={TrampolineShapeReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
