using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener no-throw vtable callback stub evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This evidence copies debug tensor metadata and callback accounting diagnostics into a no-throw stub shape. It does
/// not install a native vtable, does not call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerNoThrowVTableCallbackStub
{
    /// <summary>
    /// Evaluates callback-stub evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free callback-stub result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNoThrowVTableCallbackStubResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult minimalSafety =
            TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult noThrowVTableScaffold =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, minimalSafety, noThrowVTableScaffold);
    }

    /// <summary>
    /// Evaluates callback-stub evidence from copied owner, minimal-safety, and vtable scaffold evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="minimalSafety">The copied native attach entry minimal-safety result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="noThrowVTableScaffold">The copied native no-throw vtable scaffold result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free callback-stub result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNoThrowVTableCallbackStubResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult minimalSafety,
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult noThrowVTableScaffold)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool callbackMetadataCopyReady =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool callbackStubShapeReady =
            lineSupportsDebugListener &&
            minimalSafety.MinimalSafetyReady &&
            noThrowVTableScaffold.VTableScaffoldGateReady &&
            callbackMetadataCopyReady;
        bool callbackStubNoThrowReady =
            callbackStubShapeReady &&
            noThrowVTableScaffold.ProcessDebugTensorCallbackStubNoThrowReady &&
            noThrowVTableScaffold.ExceptionEscapeBlocked;
        bool callbackExceptionCaptureReady =
            noThrowVTableScaffold.CallbackExceptionCaptureGateReady &&
            noThrowVTableScaffold.ExceptionEscapeBlocked;
        bool callbackStatusMappingReady =
            noThrowVTableScaffold.CallbackStatusMappingGateReady &&
            callbackExceptionCaptureReady;
        bool callbackInFlightEnterReady =
            noThrowVTableScaffold.CallbackInFlightAccountingGateReady &&
            ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool callbackInFlightLeaveReady =
            noThrowVTableScaffold.CallbackInFlightAccountingGateReady &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool callbackInFlightPairingReady =
            callbackInFlightEnterReady &&
            callbackInFlightLeaveReady &&
            ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool callbackInFlightNeverNegativeReady =
            noThrowVTableScaffold.CallbackInFlightAccountingGateReady &&
            ownerDesignSnapshot.InFlightCallbackCount >= 0;
        bool borrowedDebugTensorPointerEscapeBlocked =
            noThrowVTableScaffold.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;
        bool borrowedDebugTensorMetadataCopyReady =
            callbackMetadataCopyReady &&
            borrowedDebugTensorPointerEscapeBlocked;
        long callbackEntryCount = ownerDesignSnapshot.ProcessDebugTensorCount;
        long callbackLeaveCount = callbackInFlightLeaveReady ? callbackEntryCount : 0L;
        const bool debugTensorPointerExposed = false;
        const bool debugTensorDataPointerExposed = false;
        const bool nativeVTableInstalled = false;
        const bool processDebugTensorRuntimeReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, minimalSafety.MinimalSafetyReady, "debug-listener-native-attach-entry-minimal-safety is not ready for callback-stub evaluation.");
        AddBlockerIfFalse(blockers, noThrowVTableScaffold.VTableScaffoldGateReady, "debug-listener-native-nothrow-vtable-scaffold-gate is not ready for callback-stub evaluation.");
        AddBlockerIfFalse(blockers, callbackStubShapeReady, "DebugListener no-throw vtable callback stub shape is incomplete.");
        AddBlockerIfFalse(blockers, callbackStubNoThrowReady, "DebugListener callback stub no-throw boundary is incomplete.");
        AddBlockerIfFalse(blockers, callbackMetadataCopyReady, "DebugListener callback metadata copy evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackExceptionCaptureReady, "DebugListener callback exception capture evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackStatusMappingReady, "DebugListener callback status mapping evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightEnterReady, "DebugListener callback enter accounting evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightLeaveReady, "DebugListener callback leave accounting evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightPairingReady, "DebugListener callback enter/leave pairing evidence is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightNeverNegativeReady, "DebugListener in-flight callback count can become invalid.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyReady, "borrowed debug tensor metadata copy evidence is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, !debugTensorPointerExposed, "DebugListener callback stub exposes a debug tensor pointer.");
        AddBlockerIfFalse(blockers, !debugTensorDataPointerExposed, "DebugListener callback stub exposes a debug tensor data pointer.");
        AddBlockerIfFalse(blockers, !minimalSafety.SetDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled during callback-stub evaluation.");
        AddBlockerIfFalse(blockers, minimalSafety.NativeAttachWouldBeBlocked, "native attach is not blocked during callback-stub evaluation.");
        AddBlockerIfFalse(blockers, !nativeVTableInstalled, "native IDebugListener vtable is unexpectedly installed during callback-stub evaluation.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlocker(blockers, "callback-stub-gate is non-proof evidence and must not be promoted to real-callback-runtime.");
        AddBlocker(blockers, "full package consumer smoke has not emitted real-callback-runtime callback-stub evidence.");

        foreach (string blocker in minimalSafety.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in noThrowVTableScaffold.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        string reasonCallbackRuntimeStillBlocked = BuildCallbackRuntimeBlockedReason(
            minimalSafety.SetDebugListenerNonNullEnabled,
            minimalSafety.NativeAttachWouldBeBlocked,
            nativeVTableInstalled,
            processDebugTensorRuntimeReady);

        return new TensorRtDebugListenerNoThrowVTableCallbackStubResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.TensorName,
            ownerDesignSnapshot.DataType,
            ownerDesignSnapshot.Location,
            ownerDesignSnapshot.ShapeRank,
            ownerDesignSnapshot.ShapeSummary,
            ownerDesignSnapshot.IsInput,
            ownerDesignSnapshot.IsExecutionTensor,
            callbackEntryCount,
            callbackLeaveCount,
            ownerDesignSnapshot.FailureCount,
            minimalSafety.MinimalSafetyReady,
            noThrowVTableScaffold.VTableScaffoldGateReady,
            noThrowVTableScaffold.NoThrowVTableScaffoldReady,
            callbackStubShapeReady,
            callbackStubNoThrowReady,
            callbackMetadataCopyReady,
            callbackExceptionCaptureReady,
            callbackStatusMappingReady,
            callbackInFlightEnterReady,
            callbackInFlightLeaveReady,
            callbackInFlightPairingReady,
            callbackInFlightNeverNegativeReady,
            borrowedDebugTensorMetadataCopyReady,
            borrowedDebugTensorPointerEscapeBlocked,
            debugTensorPointerExposed,
            debugTensorDataPointerExposed,
            minimalSafety.SetDebugListenerNonNullEnabled,
            minimalSafety.NativeAttachWouldBeBlocked,
            nativeVTableInstalled,
            processDebugTensorRuntimeReady,
            reasonCallbackRuntimeStillBlocked,
            blockers.ToArray());
    }

    private static string BuildCallbackRuntimeBlockedReason(
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachWouldBeBlocked,
        bool nativeVTableInstalled,
        bool processDebugTensorRuntimeReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, setDebugListenerNonNullEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, !nativeAttachWouldBeBlocked, "native attach remains deliberately blocked.");
        AddBlockerIfFalse(reasons, nativeVTableInstalled, "native IDebugListener vtable has not been installed.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlocker(reasons, "callback-stub-gate is not real-callback-runtime proof.");
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
