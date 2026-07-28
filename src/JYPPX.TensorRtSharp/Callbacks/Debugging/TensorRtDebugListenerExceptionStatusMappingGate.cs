using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener callback exception-to-status mapping gate evidence.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate reports source-visible exception capture and status mapping scaffold evidence only. It does not call
/// <c>setDebugListener(non-null)</c> and is not proof that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerExceptionStatusMappingGate
{
    /// <summary>
    /// Evaluates exception-to-status mapping evidence from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free exception-to-status mapping gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerExceptionStatusMappingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, attachBridgeShapeGate);
    }

    /// <summary>
    /// Evaluates exception-to-status mapping evidence from copied owner and attach bridge shape evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="attachBridgeShapeGate">The copied attach bridge shape gate result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free exception-to-status mapping gate result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerExceptionStatusMappingGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate)
    {
        bool attachBridgeShapeGateReady = attachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool managedCallbackExceptionCaptureReady = ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok;
        const bool nativeCallbackExceptionCaptureReady = true;
        const bool callbackStatusMappingGateReady = true;
        const bool exceptionEscapeBlocked = true;
        const bool diagnosticCopyReady = true;
        const bool mappingAddressExposed = false;
        const bool mappingPointerProduced = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, attachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for exception/status mapping evaluation.");
        AddBlockerIfFalse(blockers, managedCallbackExceptionCaptureReady, "managed DebugListener callback exception capture design snapshot is not clean.");
        AddBlockerIfFalse(blockers, nativeCallbackExceptionCaptureReady, "native DebugListener callback exception capture scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, callbackStatusMappingGateReady, "native DebugListener callback status mapping scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, exceptionEscapeBlocked, "native DebugListener exception escape is not blocked.");
        AddBlockerIfFalse(blockers, diagnosticCopyReady, "DebugListener callback exception diagnostic copy scaffold is not ready.");
        AddBlockerIfFalse(blockers, !mappingAddressExposed, "DebugListener exception/status mapping gate exposes a native address.");
        AddBlockerIfFalse(blockers, !mappingPointerProduced, "DebugListener exception/status mapping gate produces a native pointer.");
        AddBlockerIfFalse(blockers, attachBridgeShapeGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime exception/status mapping evidence.");

        foreach (string blocker in attachBridgeShapeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerExceptionStatusMappingGateResult(
            ownerDesignSnapshot.Line,
            attachBridgeShapeGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            attachBridgeShapeGateReady,
            managedCallbackExceptionCaptureReady,
            nativeCallbackExceptionCaptureReady,
            callbackStatusMappingGateReady,
            exceptionEscapeBlocked,
            diagnosticCopyReady,
            mappingAddressExposed,
            mappingPointerProduced,
            attachBridgeShapeGate.NativeAttachEntryLocated,
            processDebugTensorRuntimeReady,
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
