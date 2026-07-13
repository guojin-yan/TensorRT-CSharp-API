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

/// <summary>
/// Reports copied DebugListener no-throw vtable callback-stub diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNoThrowVTableCallbackStubResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNoThrowVTableCallbackStubResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int shapeRank,
        string shapeSummary,
        bool isInput,
        bool isExecutionTensor,
        long callbackEntryCount,
        long callbackLeaveCount,
        long failureCount,
        bool minimalSafetyReady,
        bool noThrowVTableScaffoldGateReady,
        bool noThrowVTableScaffoldReady,
        bool callbackStubShapeReady,
        bool callbackStubNoThrowReady,
        bool callbackMetadataCopyReady,
        bool callbackExceptionCaptureReady,
        bool callbackStatusMappingReady,
        bool callbackInFlightEnterReady,
        bool callbackInFlightLeaveReady,
        bool callbackInFlightPairingReady,
        bool callbackInFlightNeverNegativeReady,
        bool borrowedDebugTensorMetadataCopyReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachWouldBeBlocked,
        bool nativeVTableInstalled,
        bool processDebugTensorRuntimeReady,
        string reasonCallbackRuntimeStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        TensorName = tensorName ?? string.Empty;
        DataType = dataType;
        Location = location;
        ShapeRank = shapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsExecutionTensor = isExecutionTensor;
        CallbackEntryCount = callbackEntryCount;
        CallbackLeaveCount = callbackLeaveCount;
        FailureCount = failureCount;
        MinimalSafetyReady = minimalSafetyReady;
        NoThrowVTableScaffoldGateReady = noThrowVTableScaffoldGateReady;
        NoThrowVTableScaffoldReady = noThrowVTableScaffoldReady;
        CallbackStubShapeReady = callbackStubShapeReady;
        CallbackStubNoThrowReady = callbackStubNoThrowReady;
        CallbackMetadataCopyReady = callbackMetadataCopyReady;
        CallbackExceptionCaptureReady = callbackExceptionCaptureReady;
        CallbackStatusMappingReady = callbackStatusMappingReady;
        CallbackInFlightEnterReady = callbackInFlightEnterReady;
        CallbackInFlightLeaveReady = callbackInFlightLeaveReady;
        CallbackInFlightPairingReady = callbackInFlightPairingReady;
        CallbackInFlightNeverNegativeReady = callbackInFlightNeverNegativeReady;
        BorrowedDebugTensorMetadataCopyReady = borrowedDebugTensorMetadataCopyReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        NativeAttachWouldBeBlocked = nativeAttachWouldBeBlocked;
        NativeVTableInstalled = nativeVTableInstalled;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        ReasonCallbackRuntimeStillBlocked = reasonCallbackRuntimeStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this callback-stub gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-nothrow-vtable-callback-stub";

    /// <summary>Gets the callback kind represented by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "callback-stub-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied debug tensor name. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets the copied debug tensor shape summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether copied metadata describes an input tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether copied metadata describes an execution tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets the copied callback entry count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long CallbackEntryCount { get; }

    /// <summary>Gets the copied callback leave count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long CallbackLeaveCount { get; }

    /// <summary>Gets the copied callback failure count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long FailureCount { get; }

    /// <summary>Gets whether native attach entry minimal-safety evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool MinimalSafetyReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableScaffoldReady { get; }

    /// <summary>Gets whether callback stub parameter shape evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStubShapeReady { get; }

    /// <summary>Gets whether callback stub no-throw evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStubNoThrowReady { get; }

    /// <summary>Gets whether callback metadata copy evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackMetadataCopyReady { get; }

    /// <summary>Gets whether callback exception capture evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingReady { get; }

    /// <summary>Gets whether callback enter accounting evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightEnterReady { get; }

    /// <summary>Gets whether callback leave accounting evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightLeaveReady { get; }

    /// <summary>Gets whether callback enter/leave pairing evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightPairingReady { get; }

    /// <summary>Gets whether in-flight accounting cannot go negative in copied evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightNeverNegativeReady { get; }

    /// <summary>Gets whether borrowed debug tensor metadata was copied into pointer-free state. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for callback-stub evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether native attach remains deliberately blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachWouldBeBlocked { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for callback-stub evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether callback-stub evidence is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStubGateReady =>
        MinimalSafetyReady &&
        NoThrowVTableScaffoldGateReady &&
        NoThrowVTableScaffoldReady &&
        CallbackStubShapeReady &&
        CallbackStubNoThrowReady &&
        CallbackMetadataCopyReady &&
        CallbackExceptionCaptureReady &&
        CallbackStatusMappingReady &&
        CallbackInFlightEnterReady &&
        CallbackInFlightLeaveReady &&
        CallbackInFlightPairingReady &&
        CallbackInFlightNeverNegativeReady &&
        BorrowedDebugTensorMetadataCopyReady &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        !DebugTensorPointerExposed &&
        !DebugTensorDataPointerExposed &&
        !SetDebugListenerNonNullEnabled &&
        NativeAttachWouldBeBlocked &&
        !NativeVTableInstalled &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether a native vtable can be installed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable => false;

    /// <summary>Gets whether the callback runtime can be called by TensorRT. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why callback runtime remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReasonCallbackRuntimeStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the callback-stub gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => CallbackStubGateReady ? "callback-stub-gate-ready" : "callback-stub-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-nothrow-vtable-callback-stub; RuntimeEvidenceKind=callback-stub-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; CallbackStubGateReady=" + CallbackStubGateReady + "; " +
        "MinimalSafetyReady=" + MinimalSafetyReady + "; " +
        "NoThrowVTableScaffoldGateReady=" + NoThrowVTableScaffoldGateReady + "; " +
        "NoThrowVTableScaffoldReady=" + NoThrowVTableScaffoldReady + "; " +
        "CallbackStubShapeReady=" + CallbackStubShapeReady + "; " +
        "CallbackStubNoThrowReady=" + CallbackStubNoThrowReady + "; " +
        "CallbackMetadataCopyReady=" + CallbackMetadataCopyReady + "; " +
        "CallbackExceptionCaptureReady=" + CallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingReady=" + CallbackStatusMappingReady + "; " +
        "CallbackInFlightEnterReady=" + CallbackInFlightEnterReady + "; " +
        "CallbackInFlightLeaveReady=" + CallbackInFlightLeaveReady + "; " +
        "CallbackInFlightPairingReady=" + CallbackInFlightPairingReady + "; " +
        "CallbackInFlightNeverNegativeReady=" + CallbackInFlightNeverNegativeReady + "; " +
        "BorrowedDebugTensorMetadataCopyReady=" + BorrowedDebugTensorMetadataCopyReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NativeAttachWouldBeBlocked=" + NativeAttachWouldBeBlocked + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonCallbackRuntimeStillBlocked=" + ReasonCallbackRuntimeStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:stub={CallbackStubGateReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
