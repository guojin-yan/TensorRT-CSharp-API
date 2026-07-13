using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener <c>processDebugTensor</c> callback trampoline shape evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This report joins the no-throw vtable callback stub, borrowed debug tensor metadata copy gate, and real non-null
/// attach runtime smoke report. It describes the private/internal callback trampoline boundary; it does not prove
/// TensorRT invoked <c>IDebugListener::processDebugTensor</c> unless full package consumer runtime evidence reports a
/// real callback invocation.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerProcessDebugTensorCallbackTrampoline
{
    /// <summary>
    /// Evaluates callback trampoline shape evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="runtimePackageKey">The runtime package key, when known. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="optInEnabled">Whether the caller explicitly enabled the runtime smoke attempt. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="fullPackageConsumerReport">Whether this report came from a full package consumer smoke. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free callback trampoline report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        string runtimePackageKey = "",
        bool optInEnabled = false,
        bool fullPackageConsumerReport = false)
    {
        TensorRtDebugListenerNoThrowVTableCallbackStubResult callbackStub =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult metadataGate =
            TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke =
            TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
                ownerDesignSnapshot,
                runtimePackageKey,
                optInEnabled,
                fullPackageConsumerReport);
        return Evaluate(callbackStub, metadataGate, runtimeSmoke);
    }

    /// <summary>
    /// Evaluates callback trampoline shape evidence from copied lower-level reports.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="callbackStub">The copied no-throw vtable callback stub evidence. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="metadataGate">The copied borrowed debug tensor metadata gate evidence. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="runtimeSmoke">The copied real non-null attach runtime smoke report. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free callback trampoline report. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult Evaluate(
        TensorRtDebugListenerNoThrowVTableCallbackStubResult callbackStub,
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult metadataGate,
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke)
    {
        bool lineSupportsDebugListener =
            callbackStub.Line == TensorRtApiLine.TensorRt10 ||
            callbackStub.Line == TensorRtApiLine.TensorRt11;
        bool nativeCallbackEntryLocated =
            lineSupportsDebugListener &&
            callbackStub.CallbackStubShapeReady;
        bool noThrowCallbackEntryReady =
            nativeCallbackEntryLocated &&
            callbackStub.CallbackStubNoThrowReady;
        bool exceptionCaptureReady =
            callbackStub.CallbackExceptionCaptureReady;
        bool callbackStatusMappingReady =
            callbackStub.CallbackStatusMappingReady &&
            exceptionCaptureReady;
        bool inFlightAccountingReady =
            callbackStub.CallbackInFlightEnterReady &&
            callbackStub.CallbackInFlightLeaveReady &&
            callbackStub.CallbackInFlightPairingReady &&
            callbackStub.CallbackInFlightNeverNegativeReady;
        bool detachBeforeReleaseReady =
            !runtimeSmoke.AttachSucceeded ||
            runtimeSmoke.DetachSucceeded;
        bool borrowedDebugTensorMetadataCopyReady =
            metadataGate.MetadataGateReady &&
            metadataGate.BorrowedDebugTensorMetadataCopyReady &&
            metadataGate.TensorNameCopied &&
            metadataGate.TensorTypeCopied &&
            metadataGate.TensorLocationCopied &&
            metadataGate.TensorShapeCopied &&
            metadataGate.TensorFlagsCopied;
        bool borrowedDebugTensorPointerExposed =
            callbackStub.DebugTensorPointerExposed ||
            metadataGate.DebugTensorPointerExposed ||
            !metadataGate.BorrowedDebugTensorPointerEscapeBlocked;
        bool borrowedDebugTensorDataPointerExposed =
            callbackStub.DebugTensorDataPointerExposed ||
            metadataGate.DebugTensorDataPointerExposed ||
            !metadataGate.BorrowedDebugTensorDataPointerEscapeBlocked;
        bool pointerFreeSurfaceReady =
            runtimeSmoke.ReportPointerFree &&
            !borrowedDebugTensorPointerExposed &&
            !borrowedDebugTensorDataPointerExposed;
        bool trampolineShapeReady =
            nativeCallbackEntryLocated &&
            noThrowCallbackEntryReady &&
            exceptionCaptureReady &&
            callbackStatusMappingReady &&
            inFlightAccountingReady &&
            detachBeforeReleaseReady &&
            borrowedDebugTensorMetadataCopyReady &&
            pointerFreeSurfaceReady;
        bool processDebugTensorRuntimeReady =
            runtimeSmoke.ProcessDebugTensorInvoked &&
            metadataGate.ProcessDebugTensorRuntimeReady &&
            callbackStub.ProcessDebugTensorRuntimeReady;
        bool canPromoteRealCallbackRuntime =
            runtimeSmoke.FullPackageConsumerReport &&
            runtimeSmoke.IsRealCallbackRuntimeProof &&
            runtimeSmoke.InvocationCount > 0 &&
            runtimeSmoke.FailureCount == 0 &&
            runtimeSmoke.InFlightCallbackCount == 0 &&
            processDebugTensorRuntimeReady &&
            pointerFreeSurfaceReady;
        string runtimeEvidenceKind = canPromoteRealCallbackRuntime
            ? "real-callback-runtime"
            : "callback-trampoline-shape";
        BridgeStatusCode lastStatus = canPromoteRealCallbackRuntime
            ? BridgeStatusCode.Ok
            : runtimeSmoke.LastStatus;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeCallbackEntryLocated, "native processDebugTensor callback entry shape is not located.");
        AddBlockerIfFalse(blockers, noThrowCallbackEntryReady, "native processDebugTensor callback entry is not no-throw ready.");
        AddBlockerIfFalse(blockers, exceptionCaptureReady, "callback exception capture is not ready.");
        AddBlockerIfFalse(blockers, callbackStatusMappingReady, "callback status mapping is not ready.");
        AddBlockerIfFalse(blockers, inFlightAccountingReady, "callback in-flight enter/leave accounting is not ready.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "detach-before-release sequencing is not ready.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyReady, "borrowed debug tensor metadata copy is not ready.");
        AddBlockerIfFalse(blockers, pointerFreeSurfaceReady, "callback trampoline public surface is not pointer-free.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not ready.");
        AddBlockerIfFalse(blockers, runtimeSmoke.FullPackageConsumerReport, "full package consumer report is required before real-callback-runtime promotion.");
        AddBlockerIfFalse(blockers, runtimeSmoke.InvocationCount > 0, "runtime invocation count is zero.");
        AddBlockerIfFalse(blockers, canPromoteRealCallbackRuntime, "callback-trampoline-shape is non-proof evidence and must not be promoted to real-callback-runtime.");
        foreach (string blocker in callbackStub.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in metadataGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in runtimeSmoke.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        TensorRtDebugTensorMetadataSnapshot metadata = new TensorRtDebugTensorMetadataSnapshot(
            metadataGate.TensorName,
            metadataGate.TensorNameLength,
            metadataGate.DataType,
            metadataGate.Location,
            metadataGate.TensorShapeRank,
            metadataGate.ShapeSummary,
            metadataGate.IsInput,
            metadataGate.IsOutput,
            metadataGate.IsShapeTensor,
            metadataGate.IsExecutionTensor,
            borrowedDebugTensorMetadataCopyReady);

        string diagnostic = BuildDiagnostic(
            trampolineShapeReady,
            processDebugTensorRuntimeReady,
            canPromoteRealCallbackRuntime,
            runtimeSmoke.LastDiagnostic);

        return new TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult(
            callbackStub.Line,
            runtimeSmoke.RuntimePackageKey,
            runtimeEvidenceKind,
            metadata,
            trampolineShapeReady,
            nativeCallbackEntryLocated,
            noThrowCallbackEntryReady,
            exceptionCaptureReady,
            callbackStatusMappingReady,
            inFlightAccountingReady,
            detachBeforeReleaseReady,
            borrowedDebugTensorMetadataCopyReady,
            borrowedDebugTensorPointerExposed,
            borrowedDebugTensorDataPointerExposed,
            pointerFreeSurfaceReady,
            processDebugTensorRuntimeReady,
            runtimeSmoke.OptInEnabled,
            runtimeSmoke.FullPackageConsumerReport,
            runtimeSmoke.AttachAttempted,
            runtimeSmoke.AttachSucceeded,
            runtimeSmoke.NativeVTableInstalled,
            runtimeSmoke.ProcessDebugTensorInvoked,
            runtimeSmoke.InvocationCount,
            callbackStub.CallbackEntryCount,
            callbackStub.CallbackLeaveCount,
            runtimeSmoke.FailureCount,
            runtimeSmoke.InFlightCallbackCount,
            lastStatus,
            diagnostic,
            canPromoteRealCallbackRuntime,
            blockers.ToArray());
    }

    private static string BuildDiagnostic(
        bool trampolineShapeReady,
        bool processDebugTensorRuntimeReady,
        bool canPromoteRealCallbackRuntime,
        string runtimeSmokeDiagnostic)
    {
        if (canPromoteRealCallbackRuntime)
        {
            return "debug listener processDebugTensor callback trampoline produced full package consumer real-callback-runtime proof.";
        }

        return "debug listener processDebugTensor callback trampoline shape is " +
            (trampolineShapeReady ? "ready" : "blocked") +
            "; ProcessDebugTensorRuntimeReady=" + processDebugTensorRuntimeReady +
            "; callback-trampoline-shape is not real-callback-runtime proof. " +
            (runtimeSmokeDiagnostic ?? string.Empty);
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
/// Reports copied, pointer-free debug tensor metadata used by DebugListener trampoline diagnostics.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugTensorMetadataSnapshot
{
    internal TensorRtDebugTensorMetadataSnapshot(
        string tensorName,
        int tensorNameLength,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int tensorShapeRank,
        string shapeSummary,
        bool isInput,
        bool isOutput,
        bool isShapeTensor,
        bool isExecutionTensor,
        bool metadataCopied)
    {
        TensorName = tensorName ?? string.Empty;
        TensorNameLength = tensorNameLength;
        DataType = dataType;
        Location = location;
        TensorShapeRank = tensorShapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        MetadataCopied = metadataCopied;
    }

    /// <summary>Gets the copied debug tensor name. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor name length. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int TensorNameLength { get; }

    /// <summary>Gets the copied debug tensor data type. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int TensorShapeRank { get; }

    /// <summary>Gets the copied debug tensor shape summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether copied metadata describes an input tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether copied metadata describes an output tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether copied metadata describes a shape tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether copied metadata describes an execution tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets whether metadata was copied into this pointer-free snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool MetadataCopied { get; }
}

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
