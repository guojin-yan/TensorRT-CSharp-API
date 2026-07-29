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
