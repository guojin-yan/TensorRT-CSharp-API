using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

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
