using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native no-throw vtable scaffold diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeAttachBridgeShapeGateReady,
        bool exceptionStatusMappingGateReady,
        bool inFlightAccountingGateReady,
        bool noThrowVTableScaffoldReady,
        bool vTableDestructorNoThrowReady,
        bool processDebugTensorCallbackStubNoThrowReady,
        bool exceptionEscapeBlocked,
        bool callbackExceptionCaptureGateReady,
        bool callbackStatusMappingGateReady,
        bool callbackInFlightAccountingGateReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool vtableAddressExposed,
        bool vtablePointerProduced,
        bool nativeAttachEntryLocated,
        bool nativeVTableDesignReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        ExceptionStatusMappingGateReady = exceptionStatusMappingGateReady;
        InFlightAccountingGateReady = inFlightAccountingGateReady;
        NoThrowVTableScaffoldReady = noThrowVTableScaffoldReady;
        VTableDestructorNoThrowReady = vTableDestructorNoThrowReady;
        ProcessDebugTensorCallbackStubNoThrowReady = processDebugTensorCallbackStubNoThrowReady;
        ExceptionEscapeBlocked = exceptionEscapeBlocked;
        CallbackExceptionCaptureGateReady = callbackExceptionCaptureGateReady;
        CallbackStatusMappingGateReady = callbackStatusMappingGateReady;
        CallbackInFlightAccountingGateReady = callbackInFlightAccountingGateReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        VTableAddressExposed = vtableAddressExposed;
        VTablePointerProduced = vtablePointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-nothrow-vtable-scaffold-gate";

    /// <summary>Gets the callback kind represented by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "vtable-scaffold-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether exception/status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionStatusMappingGateReady { get; }

    /// <summary>Gets whether in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingGateReady { get; }

    /// <summary>Gets whether source-visible no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableScaffoldReady { get; }

    /// <summary>Gets whether vtable destructor no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableDestructorNoThrowReady { get; }

    /// <summary>Gets whether processDebugTensor callback stub no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorCallbackStubNoThrowReady { get; }

    /// <summary>Gets whether callback exceptions are blocked from crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionEscapeBlocked { get; }

    /// <summary>Gets whether callback exception capture gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureGateReady { get; }

    /// <summary>Gets whether callback status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingGateReady { get; }

    /// <summary>Gets whether callback in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightAccountingGateReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether the vtable scaffold exposes a native address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether the vtable scaffold produces a native pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether native IDebugListener vtable implementation is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableScaffoldGateReady =>
        NativeAttachBridgeShapeGateReady &&
        ExceptionStatusMappingGateReady &&
        InFlightAccountingGateReady &&
        NoThrowVTableScaffoldReady &&
        VTableDestructorNoThrowReady &&
        ProcessDebugTensorCallbackStubNoThrowReady &&
        ExceptionEscapeBlocked &&
        CallbackExceptionCaptureGateReady &&
        CallbackStatusMappingGateReady &&
        CallbackInFlightAccountingGateReady &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        !VTableAddressExposed &&
        !VTablePointerProduced &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        VTableScaffoldGateReady &&
        NativeAttachEntryLocated &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the vtable scaffold gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => VTableScaffoldGateReady ? "vtable-scaffold-gate-ready" : "vtable-scaffold-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-nothrow-vtable-scaffold-gate; RuntimeEvidenceKind=vtable-scaffold-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; VTableScaffoldGateReady=" + VTableScaffoldGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "ExceptionStatusMappingGateReady=" + ExceptionStatusMappingGateReady + "; " +
        "InFlightAccountingGateReady=" + InFlightAccountingGateReady + "; " +
        "NoThrowVTableScaffoldReady=" + NoThrowVTableScaffoldReady + "; " +
        "VTableDestructorNoThrowReady=" + VTableDestructorNoThrowReady + "; " +
        "ProcessDebugTensorCallbackStubNoThrowReady=" + ProcessDebugTensorCallbackStubNoThrowReady + "; " +
        "ExceptionEscapeBlocked=" + ExceptionEscapeBlocked + "; " +
        "CallbackExceptionCaptureGateReady=" + CallbackExceptionCaptureGateReady + "; " +
        "CallbackStatusMappingGateReady=" + CallbackStatusMappingGateReady + "; " +
        "CallbackInFlightAccountingGateReady=" + CallbackInFlightAccountingGateReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:vtable={NativeVTableDesignReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
