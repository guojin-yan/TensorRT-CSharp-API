using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native owner/vtable install experiment diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool nativeAttachBridgeShapeGateReady,
        bool nativeNoThrowVTableScaffoldGateReady,
        bool borrowedDebugTensorMetadataGateReady,
        bool nativeVTableInstallPreflightReady,
        bool experimentShapeReady,
        bool installAttemptGuardReady,
        bool nonNullAttachEnabled,
        bool runtimeProofEnabled,
        bool nativeVTableInstallAttempted,
        bool nativeVTableInstalled,
        bool rollbackReady,
        bool detachBeforeReleaseReady,
        bool failureStatusMappingReady,
        bool pointerFree,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool processDebugTensorRuntimeReady,
        string reasonNativeOwnerVTableInstallStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        NativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGateReady;
        BorrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataGateReady;
        NativeVTableInstallPreflightReady = nativeVTableInstallPreflightReady;
        ExperimentShapeReady = experimentShapeReady;
        InstallAttemptGuardReady = installAttemptGuardReady;
        NonNullAttachEnabled = nonNullAttachEnabled;
        RuntimeProofEnabled = runtimeProofEnabled;
        NativeVTableInstallAttempted = nativeVTableInstallAttempted;
        NativeVTableInstalled = nativeVTableInstalled;
        RollbackReady = rollbackReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        FailureStatusMappingReady = failureStatusMappingReady;
        PointerFree = pointerFree;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        ReasonNativeOwnerVTableInstallStillBlocked = reasonNativeOwnerVTableInstallStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-owner-vtable-install-experiment";

    /// <summary>Gets the callback kind represented by this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "native-owner-vtable-install-experiment";

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

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether native attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether copied borrowed debug tensor metadata gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataGateReady { get; }

    /// <summary>Gets whether native vtable install preflight evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallPreflightReady { get; }

    /// <summary>Gets whether the disabled native owner/vtable install experiment shape is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExperimentShapeReady { get; }

    /// <summary>Gets whether install attempt guards are ready and still disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InstallAttemptGuardReady { get; }

    /// <summary>Gets whether non-null setDebugListener attach is enabled. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachEnabled { get; }

    /// <summary>Gets whether runtime proof is enabled. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofEnabled { get; }

    /// <summary>Gets whether native vtable installation was attempted. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallAttempted { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for this experiment. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether rollback diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RollbackReady { get; }

    /// <summary>Gets whether detach-before-release diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether failure/status mapping diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FailureStatusMappingReady { get; }

    /// <summary>Gets whether all experiment diagnostics remain pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFree { get; }

    /// <summary>Gets whether a native vtable address is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer is produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull => false;

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable => false;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why native owner/vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeOwnerVTableInstallStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the experiment status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => ExperimentShapeReady ? "native-owner-vtable-install-experiment-ready" : "native-owner-vtable-install-experiment-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-owner-vtable-install-experiment; RuntimeEvidenceKind=native-owner-vtable-install-experiment; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "NativeNoThrowVTableScaffoldGateReady=" + NativeNoThrowVTableScaffoldGateReady + "; " +
        "BorrowedDebugTensorMetadataGateReady=" + BorrowedDebugTensorMetadataGateReady + "; " +
        "NativeVTableInstallPreflightReady=" + NativeVTableInstallPreflightReady + "; " +
        "ExperimentShapeReady=" + ExperimentShapeReady + "; " +
        "InstallAttemptGuardReady=" + InstallAttemptGuardReady + "; " +
        "NonNullAttachEnabled=" + NonNullAttachEnabled + "; " +
        "RuntimeProofEnabled=" + RuntimeProofEnabled + "; " +
        "NativeVTableInstallAttempted=" + NativeVTableInstallAttempted + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "RollbackReady=" + RollbackReady + "; " +
        "DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "FailureStatusMappingReady=" + FailureStatusMappingReady + "; " +
        "PointerFree=" + PointerFree + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonNativeOwnerVTableInstallStillBlocked=" + ReasonNativeOwnerVTableInstallStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attempted={NativeVTableInstallAttempted}:proof={IsRealCallbackRuntimeProof}";
    }
}
