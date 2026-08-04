using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach entry runtime scaffold readiness.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This scaffold records the intended attach entry parameter shape, version guard, no-throw boundary, and ownership
/// diagnostics. It does not allocate a native owner, does not call <c>setDebugListener(non-null)</c>, and is not proof
/// that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachEntryRuntimeScaffold
{
    /// <summary>
    /// Evaluates native attach entry runtime scaffold readiness from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach entry runtime scaffold result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate);
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight);
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate);
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate);
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate);
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun =
            TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(
                ownerDesignSnapshot,
                attachDetachGate,
                borrowedTensorGate,
                attachVTableGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate,
                nativeDetachBeforeReleaseDesignGate);
        return Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleDryRun);
    }

    /// <summary>
    /// Evaluates native attach entry runtime scaffold readiness from copied owner and native owner lifecycle dry-run evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="nativeOwnerLifecycleDryRun">The copied native owner lifecycle dry-run result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native attach entry runtime scaffold result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun)
    {
        bool nativeOwnerLifecycleDryRunReady = nativeOwnerLifecycleDryRun.DryRunReady;
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool attachEntryParameterShapeReady =
            nativeOwnerLifecycleDryRunReady &&
            lineSupportsDebugListener &&
            nativeOwnerLifecycleDryRun.ManagedCallbackKeepAliveDesignReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorMetadataCopyDesignReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorPointerEscapeBlocked;
        bool attachEntryVersionGuardReady =
            nativeOwnerLifecycleDryRunReady &&
            lineSupportsDebugListener;
        bool attachEntryNoThrowBoundaryReady =
            nativeOwnerLifecycleDryRunReady &&
            attachEntryParameterShapeReady;
        bool attachEntryOwnershipDiagnosticsReady =
            nativeOwnerLifecycleDryRunReady &&
            attachEntryParameterShapeReady &&
            nativeOwnerLifecycleDryRun.BorrowedDebugTensorPointerEscapeBlocked;

        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRunReady, "debug-listener-native-owner-lifecycle-dry-run is not ready for native attach entry runtime scaffold evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, attachEntryParameterShapeReady, "native DebugListener attach entry parameter shape scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryVersionGuardReady, "native DebugListener attach entry TensorRT version guard scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryNoThrowBoundaryReady, "native DebugListener attach entry no-throw/status mapping scaffold is incomplete.");
        AddBlockerIfFalse(blockers, attachEntryOwnershipDiagnosticsReady, "native DebugListener attach entry ownership diagnostics scaffold is incomplete.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.StableNativeOwnerIdentityReady, "native DebugListener stable owner identity is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.NoThrowNativeDestructorReady, "native DebugListener owner no-throw destructor is not implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleDryRun.ProcessDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener attach entry scaffold evidence.");

        foreach (string blocker in nativeOwnerLifecycleDryRun.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult(
            ownerDesignSnapshot.Line,
            nativeOwnerLifecycleDryRun.OwnerId,
            nativeOwnerLifecycleDryRun.LastStatus,
            nativeOwnerLifecycleDryRunReady,
            nativeOwnerLifecycleDryRun.NativeAttachEntryLocated,
            nativeOwnerLifecycleDryRun.NativeDetachEntryLocated,
            attachEntryParameterShapeReady,
            attachEntryVersionGuardReady,
            attachEntryNoThrowBoundaryReady,
            attachEntryOwnershipDiagnosticsReady,
            nativeOwnerLifecycleDryRun.StableNativeOwnerIdentityReady,
            nativeOwnerLifecycleDryRun.NativeOwnerNonCopyableReady,
            nativeOwnerLifecycleDryRun.NoThrowNativeDestructorReady,
            nativeOwnerLifecycleDryRun.NativeOwnerLifecycleReady,
            nativeOwnerLifecycleDryRun.ProcessDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
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
