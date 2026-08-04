using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports pointer-free readiness for attempting DebugListener real callback runtime proof work.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerRuntimeProofAttemptPreflightResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerRuntimeProofAttemptPreflightResult(
        TensorRtApiLine line,
        bool nativeAttachEntryLocated,
        bool nonNullAttachStillDisabled,
        bool nativeOwnerLifecycleReady,
        bool canImplementNativeAttach,
        bool nativeVTableReady,
        bool noThrowVTableDesignReady,
        bool nativeVTableTrampolineReady,
        bool callbackExceptionCaptureReady,
        bool callbackStatusMappingReady,
        bool callbackInFlightAccountingReady,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        bool precheckCanAttemptRuntimeProof,
        bool canEnableSetDebugListenerNonNull,
        bool canInstallNativeVTable,
        bool canCallProcessDebugTensorRuntime,
        bool canPromoteRealCallbackRuntime,
        string reasonNonNullAttachStillBlocked,
        string reasonNativeVTableStillBlocked,
        string reasonRuntimeProofStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NonNullAttachStillDisabled = nonNullAttachStillDisabled;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        CanImplementNativeAttach = canImplementNativeAttach;
        NativeVTableReady = nativeVTableReady;
        NoThrowVTableDesignReady = noThrowVTableDesignReady;
        NativeVTableTrampolineReady = nativeVTableTrampolineReady;
        CallbackExceptionCaptureReady = callbackExceptionCaptureReady;
        CallbackStatusMappingReady = callbackStatusMappingReady;
        CallbackInFlightAccountingReady = callbackInFlightAccountingReady;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        PrecheckCanAttemptRuntimeProof = precheckCanAttemptRuntimeProof;
        CanEnableSetDebugListenerNonNull = canEnableSetDebugListenerNonNull;
        CanInstallNativeVTable = canInstallNativeVTable;
        CanCallProcessDebugTensorRuntime = canCallProcessDebugTensorRuntime;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        ReasonNonNullAttachStillBlocked = reasonNonNullAttachStillBlocked ?? string.Empty;
        ReasonNativeVTableStillBlocked = reasonNativeVTableStillBlocked ?? string.Empty;
        ReasonRuntimeProofStillBlocked = reasonRuntimeProofStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-runtime-proof-attempt-preflight";

    /// <summary>Gets the callback kind represented by this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "runtime-proof-attempt-preflight";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether non-null attach remains deliberately disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled { get; }

    /// <summary>Gets whether native owner lifetime and release ordering are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native attach implementation prerequisites are satisfied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach { get; }

    /// <summary>Gets whether the native owner and vtable are ready as a unit. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableReady { get; }

    /// <summary>Gets whether the native vtable no-throw design is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableDesignReady { get; }

    /// <summary>Gets whether a native IDebugListener vtable trampoline exists. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableTrampolineReady { get; }

    /// <summary>Gets whether native callback exception capture is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureReady { get; }

    /// <summary>Gets whether callback status mapping is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingReady { get; }

    /// <summary>Gets whether callback in-flight accounting is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightAccountingReady { get; }

    /// <summary>Gets whether a native vtable address would be exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer would be produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor metadata lifetime is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data lifetime is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is implemented. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the consumed runtime proof precheck allows a runtime proof attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PrecheckCanAttemptRuntimeProof { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull { get; }

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime { get; }

    /// <summary>Gets whether real-callback-runtime evidence can be promoted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets why non-null attach remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNonNullAttachStillBlocked { get; }

    /// <summary>Gets why native vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeVTableStillBlocked { get; }

    /// <summary>Gets why runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonRuntimeProofStillBlocked { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether real runtime proof remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the copied list of prerequisites that still block a proof attempt. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the proof-attempt preflight status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime ? "runtime-proof-attempt-ready" : "runtime-proof-attempt-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-runtime-proof-attempt-preflight; RuntimeEvidenceKind=runtime-proof-attempt-preflight; " +
        "RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "NativeVTableReady=" + NativeVTableReady + "; " +
        "NoThrowVTableDesignReady=" + NoThrowVTableDesignReady + "; " +
        "NativeVTableTrampolineReady=" + NativeVTableTrampolineReady + "; " +
        "CallbackExceptionCaptureReady=" + CallbackExceptionCaptureReady + "; " +
        "CallbackStatusMappingReady=" + CallbackStatusMappingReady + "; " +
        "CallbackInFlightAccountingReady=" + CallbackInFlightAccountingReady + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "FullPackageConsumerRuntimeEvidenceReady=" + FullPackageConsumerRuntimeEvidenceReady + "; " +
        "PrecheckCanAttemptRuntimeProof=" + PrecheckCanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + "; " +
        "ReasonNonNullAttachStillBlocked=" + ReasonNonNullAttachStillBlocked + "; " +
        "ReasonNativeVTableStillBlocked=" + ReasonNativeVTableStillBlocked + "; " +
        "ReasonRuntimeProofStillBlocked=" + ReasonRuntimeProofStillBlocked + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={CanEnableSetDebugListenerNonNull}:proof={IsRealCallbackRuntimeProof}";
    }
}
