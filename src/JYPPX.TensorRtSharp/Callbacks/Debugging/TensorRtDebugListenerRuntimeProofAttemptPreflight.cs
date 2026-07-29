using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates whether DebugListener real callback runtime proof work may be attempted.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This preflight consumes pointer-free runtime proof precheck evidence only. It does not enable
/// <c>setDebugListener(non-null)</c>, does not install a native <c>IDebugListener</c> vtable, and does not invoke
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerRuntimeProofAttemptPreflight
{
    /// <summary>
    /// Evaluates proof-attempt readiness from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free proof-attempt preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofAttemptPreflightResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        return Evaluate(TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(ownerDesignSnapshot));
    }

    /// <summary>
    /// Evaluates proof-attempt readiness from a runtime proof precheck result.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="precheck">The pointer-free runtime proof precheck result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free proof-attempt preflight result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofAttemptPreflightResult Evaluate(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck)
    {
        bool canEnableSetDebugListenerNonNull =
            precheck.NativeAttachBridgeShapeGateReady &&
            precheck.AttachBridgeNoThrowBoundaryReady &&
            precheck.AttachBridgeVersionGuardReady &&
            precheck.AttachBridgeOwnershipDiagnosticsReady &&
            precheck.AttachBridgePointerFree &&
            precheck.NativeAttachEntryLocated &&
            !precheck.NonNullAttachStillDisabled &&
            precheck.NativeOwnerLifecycleReady &&
            precheck.CanImplementNativeAttach;

        bool canInstallNativeVTable =
            precheck.NativeVTableReady &&
            precheck.NoThrowVTableDesignReady &&
            precheck.NativeVTableTrampolineReady &&
            precheck.CallbackExceptionCaptureReady &&
            precheck.CallbackStatusMappingReady &&
            precheck.CallbackInFlightAccountingReady &&
            precheck.NativeNoThrowVTableScaffoldGateReady &&
            precheck.NoThrowVTableScaffoldReady &&
            precheck.VTableDestructorNoThrowReady &&
            precheck.ProcessDebugTensorCallbackStubNoThrowReady &&
            !precheck.VTableAddressExposed &&
            !precheck.VTablePointerProduced;

        bool canCallProcessDebugTensorRuntime =
            canEnableSetDebugListenerNonNull &&
            canInstallNativeVTable &&
            precheck.BorrowedDebugTensorPointerEscapeBlocked &&
            precheck.BorrowedDebugTensorLifetimeReady &&
            precheck.BorrowedDebugTensorDataLifetimeReady &&
            precheck.ProcessDebugTensorRuntimeReady;

        bool canPromoteRealCallbackRuntime =
            canCallProcessDebugTensorRuntime &&
            precheck.FullPackageConsumerRuntimeEvidenceReady &&
            precheck.CanAttemptRuntimeProof &&
            precheck.RealCallbackRuntime &&
            precheck.IsRealCallbackRuntimeProof;

        string nonNullReason = BuildNonNullAttachReason(precheck, canEnableSetDebugListenerNonNull);
        string nativeVTableReason = BuildNativeVTableReason(precheck, canInstallNativeVTable);
        string runtimeReason = BuildRuntimeProofReason(
            precheck,
            canEnableSetDebugListenerNonNull,
            canInstallNativeVTable,
            canCallProcessDebugTensorRuntime,
            canPromoteRealCallbackRuntime);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, canEnableSetDebugListenerNonNull, nonNullReason);
        AddBlockerIfFalse(blockers, canInstallNativeVTable, nativeVTableReason);
        AddBlockerIfFalse(blockers, canCallProcessDebugTensorRuntime, runtimeReason);
        AddBlockerIfFalse(blockers, canPromoteRealCallbackRuntime, "real-callback-runtime promotion remains blocked by attempt preflight.");
        foreach (string blocker in precheck.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerRuntimeProofAttemptPreflightResult(
            precheck.Line,
            precheck.NativeAttachEntryLocated,
            precheck.NonNullAttachStillDisabled,
            precheck.NativeOwnerLifecycleReady,
            precheck.CanImplementNativeAttach,
            precheck.NativeVTableReady,
            precheck.NoThrowVTableDesignReady,
            precheck.NativeVTableTrampolineReady,
            precheck.CallbackExceptionCaptureReady,
            precheck.CallbackStatusMappingReady,
            precheck.CallbackInFlightAccountingReady,
            precheck.VTableAddressExposed,
            precheck.VTablePointerProduced,
            precheck.BorrowedDebugTensorPointerEscapeBlocked,
            precheck.BorrowedDebugTensorLifetimeReady,
            precheck.BorrowedDebugTensorDataLifetimeReady,
            precheck.ProcessDebugTensorRuntimeReady,
            precheck.FullPackageConsumerRuntimeEvidenceReady,
            precheck.CanAttemptRuntimeProof,
            canEnableSetDebugListenerNonNull,
            canInstallNativeVTable,
            canCallProcessDebugTensorRuntime,
            canPromoteRealCallbackRuntime,
            nonNullReason,
            nativeVTableReason,
            runtimeReason,
            blockers.ToArray());
    }

    private static string BuildNonNullAttachReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canEnableSetDebugListenerNonNull)
    {
        if (canEnableSetDebugListenerNonNull)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, precheck.NativeAttachEntryLocated, "native line-specific setDebugListener(non-null) attach entry is not implemented.");
        AddBlockerIfFalse(reasons, !precheck.NonNullAttachStillDisabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, precheck.NativeOwnerLifecycleReady, "native owner lifecycle is not ready for non-null attach.");
        AddBlockerIfFalse(reasons, precheck.CanImplementNativeAttach, "native attach implementation is still blocked.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgePointerFree, "attach bridge diagnostics are not pointer-free.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgeNoThrowBoundaryReady, "attach bridge no-throw boundary is not ready.");
        AddBlockerIfFalse(reasons, precheck.AttachBridgeVersionGuardReady, "attach bridge TensorRT version guard is not ready.");
        return string.Join(" ", reasons);
    }

    private static string BuildNativeVTableReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canInstallNativeVTable)
    {
        if (canInstallNativeVTable)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, precheck.NativeVTableReady, "stable native owner address and no-throw native vtable are not ready.");
        AddBlockerIfFalse(reasons, precheck.NoThrowVTableDesignReady, "no-throw vtable design is not complete.");
        AddBlockerIfFalse(reasons, precheck.NativeVTableTrampolineReady, "native IDebugListener vtable trampoline is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackExceptionCaptureReady, "native callback exception capture is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackStatusMappingReady, "callback status mapping is not implemented.");
        AddBlockerIfFalse(reasons, precheck.CallbackInFlightAccountingReady, "callback in-flight accounting is not implemented.");
        AddBlockerIfFalse(reasons, precheck.NativeNoThrowVTableScaffoldGateReady, "native no-throw vtable scaffold gate is not ready.");
        AddBlockerIfFalse(reasons, precheck.NoThrowVTableScaffoldReady, "no-throw vtable scaffold is not ready.");
        AddBlockerIfFalse(reasons, precheck.VTableDestructorNoThrowReady, "vtable destructor no-throw scaffold is not ready.");
        AddBlockerIfFalse(reasons, precheck.ProcessDebugTensorCallbackStubNoThrowReady, "processDebugTensor callback stub no-throw scaffold is not ready.");
        AddBlockerIfFalse(reasons, !precheck.VTableAddressExposed, "native vtable address would be exposed.");
        AddBlockerIfFalse(reasons, !precheck.VTablePointerProduced, "native vtable pointer would be produced.");
        return string.Join(" ", reasons);
    }

    private static string BuildRuntimeProofReason(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck,
        bool canEnableSetDebugListenerNonNull,
        bool canInstallNativeVTable,
        bool canCallProcessDebugTensorRuntime,
        bool canPromoteRealCallbackRuntime)
    {
        if (canPromoteRealCallbackRuntime)
        {
            return string.Empty;
        }

        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, canEnableSetDebugListenerNonNull, "non-null DebugListener attach is not enabled.");
        AddBlockerIfFalse(reasons, canInstallNativeVTable, "native IDebugListener vtable cannot be installed.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorLifetimeReady, "borrowed debug tensor metadata lifetime is not ready.");
        AddBlockerIfFalse(reasons, precheck.BorrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data lifetime is not ready.");
        AddBlockerIfFalse(reasons, precheck.ProcessDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlockerIfFalse(reasons, canCallProcessDebugTensorRuntime, "runtime callback invocation cannot be attempted.");
        AddBlockerIfFalse(reasons, precheck.FullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not produced real-callback-runtime evidence.");
        AddBlockerIfFalse(reasons, precheck.CanAttemptRuntimeProof, "runtime proof precheck is still blocked.");
        AddBlockerIfFalse(reasons, precheck.RealCallbackRuntime, "gate/precheck evidence is not real callback runtime evidence.");
        AddBlockerIfFalse(reasons, precheck.IsRealCallbackRuntimeProof, "gate/precheck evidence is not promotable runtime proof.");
        return string.Join(" ", reasons);
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
