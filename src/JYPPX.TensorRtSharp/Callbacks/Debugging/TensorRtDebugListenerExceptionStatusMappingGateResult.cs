using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener exception-to-status mapping diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
public readonly struct TensorRtDebugListenerExceptionStatusMappingGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerExceptionStatusMappingGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool attachBridgeShapeGateReady,
        bool managedCallbackExceptionCaptureReady,
        bool nativeCallbackExceptionCaptureReady,
        bool callbackStatusMappingGateReady,
        bool exceptionEscapeBlocked,
        bool diagnosticCopyReady,
        bool mappingAddressExposed,
        bool mappingPointerProduced,
        bool nativeAttachEntryLocated,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        AttachBridgeShapeGateReady = attachBridgeShapeGateReady;
        ManagedCallbackExceptionCaptureReady = managedCallbackExceptionCaptureReady;
        NativeCallbackExceptionCaptureReady = nativeCallbackExceptionCaptureReady;
        CallbackStatusMappingGateReady = callbackStatusMappingGateReady;
        ExceptionEscapeBlocked = exceptionEscapeBlocked;
        DiagnosticCopyReady = diagnosticCopyReady;
        MappingAddressExposed = mappingAddressExposed;
        MappingPointerProduced = mappingPointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-exception-status-mapping-gate";

    /// <summary>Gets the callback kind represented by this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "exception-status-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether attach bridge shape gate evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether managed callback exception capture design evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ManagedCallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether native callback exception capture scaffold evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeCallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping scaffold evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CallbackStatusMappingGateReady { get; }

    /// <summary>Gets whether callback exceptions are blocked from crossing the C ABI. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ExceptionEscapeBlocked { get; }

    /// <summary>Gets whether diagnostics are copied into pointer-free status records. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DiagnosticCopyReady { get; }

    /// <summary>Gets whether the mapping gate exposes a native address. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool MappingAddressExposed { get; }

    /// <summary>Gets whether the mapping gate produces a native pointer. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool MappingPointerProduced { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether exception/status mapping gate evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ExceptionStatusMappingGateReady =>
        AttachBridgeShapeGateReady &&
        ManagedCallbackExceptionCaptureReady &&
        NativeCallbackExceptionCaptureReady &&
        CallbackStatusMappingGateReady &&
        ExceptionEscapeBlocked &&
        DiagnosticCopyReady &&
        !MappingAddressExposed &&
        !MappingPointerProduced &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        NativeAttachEntryLocated &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the exception/status mapping gate status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => ExceptionStatusMappingGateReady ? "exception-status-gate-ready" : "exception-status-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-exception-status-mapping-gate; RuntimeEvidenceKind=exception-status-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; ExceptionStatusMappingGateReady=" + ExceptionStatusMappingGateReady + "; " +
        "AttachBridgeShapeGateReady=" + AttachBridgeShapeGateReady + "; " +
        "ManagedCallbackExceptionCaptureReady=" + ManagedCallbackExceptionCaptureReady + "; " +
        "NativeCallbackExceptionCaptureReady=" + NativeCallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingGateReady=" + CallbackStatusMappingGateReady + "; " +
        "ExceptionEscapeBlocked=" + ExceptionEscapeBlocked + "; " +
        "DiagnosticCopyReady=" + DiagnosticCopyReady + "; " +
        "MappingAddressExposed=" + MappingAddressExposed + "; " +
        "MappingPointerProduced=" + MappingPointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:mapping={ExceptionStatusMappingGateReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
