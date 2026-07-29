using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native attach entry minimal-safety evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This evidence reports that a source-visible, no-throw, version-guarded attach entry shape is ready for TensorRT 10
/// and TensorRT 11 review. It does not enable <c>setDebugListener(non-null)</c>, does not expose a native owner pointer,
/// and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeAttachEntryMinimalSafety
{
    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, runtimeScaffold, lifecycleGate);
    }

    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from copied owner and runtime scaffold evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimeScaffold">The copied native attach entry runtime scaffold result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, runtimeScaffold, lifecycleGate);
    }

    /// <summary>
    /// Evaluates native attach entry minimal-safety evidence from copied owner, runtime scaffold, and lifecycle gate evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="runtimeScaffold">The copied native attach entry runtime scaffold result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="lifecycleGate">The copied native owner lifecycle gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native attach entry minimal-safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult runtimeScaffold,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycleGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool runtimeScaffoldReady = runtimeScaffold.RuntimeScaffoldReady;
        bool lifecycleGateReady = lifecycleGate.LifecycleGateReady;
        bool lifecyclePointerFree =
            !lifecycleGate.NativeOwnerAddressExposed &&
            !lifecycleGate.NativeOwnerPointerProduced &&
            !lifecycleGate.DestructorAddressExposed &&
            !lifecycleGate.DestructorPointerProduced &&
            !lifecycleGate.LifecycleAddressExposed &&
            !lifecycleGate.LifecyclePointerProduced;
        bool attachEntryParameterShapeReady =
            lineSupportsDebugListener &&
            runtimeScaffoldReady &&
            runtimeScaffold.AttachEntryParameterShapeReady;
        bool attachEntryVersionGuardReady = lineSupportsDebugListener;
        bool attachEntryNoThrowReady =
            runtimeScaffoldReady &&
            lifecycleGateReady &&
            runtimeScaffold.AttachEntryNoThrowBoundaryReady;
        bool attachEntryOwnershipDiagnosticsReady =
            runtimeScaffoldReady &&
            lifecycleGateReady &&
            lifecyclePointerFree &&
            runtimeScaffold.AttachEntryOwnershipDiagnosticsReady &&
            lifecycleGate.NativeOwnerNonCopyableReady &&
            lifecycleGate.NoThrowNativeDestructorReady;
        bool nativeAttachEntryLocated =
            attachEntryParameterShapeReady &&
            attachEntryVersionGuardReady &&
            attachEntryNoThrowReady &&
            attachEntryOwnershipDiagnosticsReady;

        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeAttachWouldBeBlocked = true;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, runtimeScaffoldReady, "debug-listener-native-attach-entry-runtime-scaffold is not ready for minimal safety evaluation.");
        AddBlockerIfFalse(blockers, lifecycleGateReady, "debug-listener-native-owner-lifecycle-gate is not ready for minimal safety evaluation.");
        AddBlockerIfFalse(blockers, lifecyclePointerFree, "native DebugListener lifecycle diagnostics are not pointer-free.");
        AddBlockerIfFalse(blockers, attachEntryParameterShapeReady, "native DebugListener attach entry parameter shape is not ready.");
        AddBlockerIfFalse(blockers, attachEntryVersionGuardReady, "native DebugListener attach entry TensorRT 10/11 version guard is not ready.");
        AddBlockerIfFalse(blockers, attachEntryNoThrowReady, "native DebugListener attach entry no-throw boundary is not ready.");
        AddBlockerIfFalse(blockers, attachEntryOwnershipDiagnosticsReady, "native DebugListener attach entry ownership diagnostics are not ready.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "source-visible native DebugListener attach entry minimal-safety shape is not located.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled during minimal-safety evaluation.");
        AddBlockerIfFalse(blockers, nativeAttachWouldBeBlocked, "native attach would not be blocked by the minimal-safety gate.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime attach entry evidence.");

        foreach (string blocker in runtimeScaffold.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in lifecycleGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        string reasonNativeAttachStillBlocked = BuildNativeAttachBlockedReason(
            setDebugListenerNonNullEnabled,
            nativeAttachWouldBeBlocked,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady);

        return new TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult(
            ownerDesignSnapshot.Line,
            runtimeScaffold.OwnerId,
            ownerDesignSnapshot.LastStatus,
            runtimeScaffoldReady,
            lifecycleGateReady,
            lifecyclePointerFree,
            nativeAttachEntryLocated,
            runtimeScaffold.NativeDetachEntryLocated,
            attachEntryParameterShapeReady,
            attachEntryNoThrowReady,
            attachEntryVersionGuardReady,
            attachEntryOwnershipDiagnosticsReady,
            setDebugListenerNonNullEnabled,
            nativeAttachWouldBeBlocked,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            reasonNativeAttachStillBlocked,
            blockers.ToArray());
    }

    private static string BuildNativeAttachBlockedReason(
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachWouldBeBlocked,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, setDebugListenerNonNullEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, !nativeAttachWouldBeBlocked, "native attach is deliberately blocked by minimal-safety evidence.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlockerIfFalse(reasons, fullPackageConsumerRuntimeEvidenceReady, "full package consumer real-callback-runtime evidence is not present.");
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
