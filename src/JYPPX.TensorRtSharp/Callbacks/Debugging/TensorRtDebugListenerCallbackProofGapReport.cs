using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Builds a pointer-free DebugListener callback proof gap report.
/// 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This report consumes existing copied proof-attempt, smoke, trampoline, and proof-gate evidence. It does not enable
/// <c>setDebugListener(non-null)</c>, install a native vtable, call TensorRT, or expose native pointers.
/// 该报告只聚合已复制的安全门禁和 smoke 诊断，不启用 callback、不安装 vtable、不调用 TensorRT，也不暴露裸指针。
/// </remarks>
public static class TensorRtDebugListenerCallbackProofGapReport
{
    /// <summary>
    /// Aggregates copied DebugListener proof evidence into a single gap report.
    /// 将复制出的 DebugListener proof 证据聚合为单一缺口报告。
    /// </summary>
    /// <param name="attemptPreflight">The proof-attempt preflight evidence. 该参数传入已复制的 DebugListener proof evidence。</param>
    /// <param name="runtimeSmoke">The disabled-by-default runtime smoke report. 该参数传入已复制的 DebugListener proof evidence。</param>
    /// <param name="callbackTrampoline">The processDebugTensor callback trampoline report. 该参数传入已复制的 DebugListener proof evidence。</param>
    /// <param name="runtimeProof">The final proof promotion gate report. 该参数传入已复制的 DebugListener proof evidence。</param>
    /// <returns>A pointer-free gap report. 返回不暴露 native 指针的 pointer-free 缺口报告。</returns>
    public static TensorRtDebugListenerCallbackProofGapReportResult Evaluate(
        TensorRtDebugListenerRuntimeProofAttemptPreflightResult attemptPreflight,
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke,
        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult callbackTrampoline,
        TensorRtDebugListenerRealCallbackRuntimeProofResult runtimeProof)
    {
        bool nonNullAttachStillDisabled =
            attemptPreflight.NonNullAttachStillDisabled ||
            !attemptPreflight.CanEnableSetDebugListenerNonNull ||
            !runtimeSmoke.AttachGuardReady;
        bool nativeAttachEntryReady =
            attemptPreflight.NativeAttachEntryLocated &&
            attemptPreflight.CanImplementNativeAttach &&
            !attemptPreflight.NonNullAttachStillDisabled;
        bool nativeVTableInstallBlocked =
            !attemptPreflight.CanInstallNativeVTable ||
            !runtimeSmoke.NativeVTableReady ||
            !runtimeSmoke.NativeVTableInstalled ||
            !runtimeProof.NativeVTableInstalled;
        bool noThrowCallbackEntryReady =
            callbackTrampoline.NativeCallbackEntryLocated &&
            callbackTrampoline.NoThrowCallbackEntryReady &&
            callbackTrampoline.TrampolineShapeReady;
        bool exceptionStatusMappingReady =
            callbackTrampoline.ExceptionCaptureReady &&
            callbackTrampoline.CallbackStatusMappingReady;
        bool inFlightAccountingReady =
            callbackTrampoline.InFlightAccountingReady &&
            runtimeProof.InFlightCallbackCount == 0;
        bool borrowedMetadataCopyReady =
            callbackTrampoline.BorrowedDebugTensorMetadataCopyReady &&
            callbackTrampoline.Metadata.MetadataCopied &&
            callbackTrampoline.PointerFreeSurfaceReady &&
            !callbackTrampoline.BorrowedDebugTensorPointerExposed &&
            !callbackTrampoline.BorrowedDebugTensorDataPointerExposed;
        bool detachRollbackReady =
            (!runtimeSmoke.AttachSucceeded || runtimeSmoke.DetachSucceeded) &&
            (!runtimeSmoke.RollbackAttempted || runtimeSmoke.RollbackSucceeded) &&
            (!runtimeProof.AttachSucceeded || runtimeProof.DetachSucceeded) &&
            (!runtimeProof.RollbackAttempted || runtimeProof.RollbackSucceeded);
        bool processDebugTensorRuntimeInvoked =
            runtimeSmoke.ProcessDebugTensorInvoked &&
            callbackTrampoline.ProcessDebugTensorInvoked &&
            runtimeProof.ProcessDebugTensorInvoked &&
            runtimeProof.InvocationCount > 0;
        bool fullPackageConsumerRuntimeProofReady =
            runtimeSmoke.FullPackageConsumerReport &&
            callbackTrampoline.FullPackageConsumerReport &&
            runtimeProof.FullPackageConsumerReport &&
            runtimeProof.IsRealCallbackRuntimeProof;
        bool pointerFreeSurfaceReady =
            runtimeSmoke.ReportPointerFree &&
            callbackTrampoline.PointerFreeSurfaceReady &&
            runtimeProof.PointerFreeSurfaceReady;
        bool canPromoteRealCallbackRuntime =
            !nonNullAttachStillDisabled &&
            nativeAttachEntryReady &&
            !nativeVTableInstallBlocked &&
            noThrowCallbackEntryReady &&
            exceptionStatusMappingReady &&
            inFlightAccountingReady &&
            borrowedMetadataCopyReady &&
            detachRollbackReady &&
            processDebugTensorRuntimeInvoked &&
            fullPackageConsumerRuntimeProofReady &&
            pointerFreeSurfaceReady &&
            runtimeProof.CanPromoteRealCallbackRuntime;

        List<string> gaps = new List<string>();
        AddGapIfFalse(gaps, !nonNullAttachStillDisabled, "non-null DebugListener attach remains disabled or unsafe.");
        AddGapIfFalse(gaps, nativeAttachEntryReady, "native setDebugListener(non-null) attach entry is not ready under the TensorRT version guard.");
        AddGapIfFalse(gaps, !nativeVTableInstallBlocked, "native IDebugListener vtable install remains disabled or blocked.");
        AddGapIfFalse(gaps, noThrowCallbackEntryReady, "no-throw processDebugTensor callback entry is not ready.");
        AddGapIfFalse(gaps, exceptionStatusMappingReady, "exception capture and status mapping are not complete.");
        AddGapIfFalse(gaps, inFlightAccountingReady, "callback in-flight enter/leave accounting is not complete.");
        AddGapIfFalse(gaps, borrowedMetadataCopyReady, "borrowed debug tensor metadata is not fully copied into a pointer-free snapshot.");
        AddGapIfFalse(gaps, detachRollbackReady, "detach-before-release, rollback, or dispose-idempotency proof is incomplete.");
        AddGapIfFalse(gaps, processDebugTensorRuntimeInvoked, "TensorRT has not invoked IDebugListener::processDebugTensor.");
        AddGapIfFalse(gaps, fullPackageConsumerRuntimeProofReady, "full package consumer real-callback-runtime proof is not ready.");
        AddGapIfFalse(gaps, pointerFreeSurfaceReady, "DebugListener proof public surface is not pointer-free.");
        AddGapIfFalse(gaps, canPromoteRealCallbackRuntime, "DebugListener real-callback-runtime promotion remains blocked.");
        AddGaps(gaps, attemptPreflight.BlockedPrerequisites);
        AddGaps(gaps, runtimeSmoke.BlockedPrerequisites);
        AddGaps(gaps, callbackTrampoline.BlockedPrerequisites);
        AddGaps(gaps, runtimeProof.BlockedPrerequisites);

        return new TensorRtDebugListenerCallbackProofGapReportResult(
            runtimeProof.Line,
            runtimeProof.RuntimePackageKey,
            nonNullAttachStillDisabled,
            nativeAttachEntryReady,
            nativeVTableInstallBlocked,
            noThrowCallbackEntryReady,
            exceptionStatusMappingReady,
            inFlightAccountingReady,
            borrowedMetadataCopyReady,
            detachRollbackReady,
            processDebugTensorRuntimeInvoked,
            fullPackageConsumerRuntimeProofReady,
            pointerFreeSurfaceReady,
            runtimeProof.AttemptedNoInvocation,
            runtimeProof.InvocationCount,
            runtimeProof.FailureCount,
            runtimeProof.InFlightCallbackCount,
            canPromoteRealCallbackRuntime,
            gaps.ToArray());
    }

    private static void AddGapIfFalse(List<string> gaps, bool condition, string gap)
    {
        if (!condition)
        {
            AddGap(gaps, gap);
        }
    }

    private static void AddGaps(List<string> gaps, IEnumerable<string> values)
    {
        foreach (string value in values)
        {
            AddGap(gaps, value);
        }
    }

    private static void AddGap(List<string> gaps, string gap)
    {
        if (!string.IsNullOrWhiteSpace(gap) && !gaps.Contains(gap))
        {
            gaps.Add(gap);
        }
    }
}
