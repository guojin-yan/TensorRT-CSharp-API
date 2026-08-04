using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native vtable install preflight diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeVTableInstallPreflightResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeVTableInstallPreflightResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool nativeAttachBridgeShapeGateReady,
        bool nativeNoThrowVTableScaffoldGateReady,
        bool borrowedDebugTensorMetadataGateReady,
        bool vTableInstallShapeReady,
        bool vTableInstallVersionGuardReady,
        bool vTableInstallNoThrowBoundaryReady,
        bool vTableInstallOwnershipDiagnosticsReady,
        bool vTableInstallPointerFree,
        bool attachBridgeSetDebugListenerNonNullEnabled,
        bool setDebugListenerNonNullEnabled,
        bool vTableAddressExposed,
        bool vTablePointerProduced,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorDataPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool nativeVTableInstallPreflightReady,
        bool nativeVTableInstalled,
        bool nativeVTableInstallRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string reasonNativeVTableInstallStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        NativeNoThrowVTableScaffoldGateReady = nativeNoThrowVTableScaffoldGateReady;
        BorrowedDebugTensorMetadataGateReady = borrowedDebugTensorMetadataGateReady;
        VTableInstallShapeReady = vTableInstallShapeReady;
        VTableInstallVersionGuardReady = vTableInstallVersionGuardReady;
        VTableInstallNoThrowBoundaryReady = vTableInstallNoThrowBoundaryReady;
        VTableInstallOwnershipDiagnosticsReady = vTableInstallOwnershipDiagnosticsReady;
        VTableInstallPointerFree = vTableInstallPointerFree;
        AttachBridgeSetDebugListenerNonNullEnabled = attachBridgeSetDebugListenerNonNullEnabled;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        VTableAddressExposed = vTableAddressExposed;
        VTablePointerProduced = vTablePointerProduced;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorDataPointerEscapeBlocked = borrowedDebugTensorDataPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        NativeVTableInstallPreflightReady = nativeVTableInstallPreflightReady;
        NativeVTableInstalled = nativeVTableInstalled;
        NativeVTableInstallRuntimeReady = nativeVTableInstallRuntimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        ReasonNativeVTableInstallStillBlocked = reasonNativeVTableInstallStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-vtable-install-preflight";

    /// <summary>Gets the callback kind represented by this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "native-vtable-install-preflight";

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

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether native attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether native no-throw vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeNoThrowVTableScaffoldGateReady { get; }

    /// <summary>Gets whether copied borrowed debug tensor metadata gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataGateReady { get; }

    /// <summary>Gets whether the install parameter shape preflight is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallShapeReady { get; }

    /// <summary>Gets whether the install path is guarded by TensorRT 10/11 support. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallVersionGuardReady { get; }

    /// <summary>Gets whether the install path has a no-throw boundary preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallNoThrowBoundaryReady { get; }

    /// <summary>Gets whether install ownership diagnostics are ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether all install diagnostics remain pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableInstallPointerFree { get; }

    /// <summary>Gets whether the consumed attach bridge gate has non-null setDebugListener enabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool AttachBridgeSetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null attach remains deliberately disabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled => !SetDebugListenerNonNullEnabled;

    /// <summary>Gets whether a native vtable address is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether a native vtable pointer is produced. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor data pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether copied native vtable install preflight evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallPreflightReady { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for this preflight. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether native vtable install runtime is enabled. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallRuntimeReady { get; }

    /// <summary>Gets whether it is safe to enable setDebugListener(non-null). 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanEnableSetDebugListenerNonNull => false;

    /// <summary>Gets whether it is safe to install the native IDebugListener vtable. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanInstallNativeVTable => false;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime invocation can be attempted. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether all prerequisites are satisfied to attempt real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why native vtable installation remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReasonNativeVTableInstallStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the native vtable install preflight status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => NativeVTableInstallPreflightReady ? "native-vtable-install-preflight-ready" : "native-vtable-install-preflight-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-vtable-install-preflight; RuntimeEvidenceKind=native-vtable-install-preflight; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeVTableInstallPreflightReady=" + NativeVTableInstallPreflightReady + "; " +
        "NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "NativeNoThrowVTableScaffoldGateReady=" + NativeNoThrowVTableScaffoldGateReady + "; " +
        "BorrowedDebugTensorMetadataGateReady=" + BorrowedDebugTensorMetadataGateReady + "; " +
        "VTableInstallShapeReady=" + VTableInstallShapeReady + "; " +
        "VTableInstallVersionGuardReady=" + VTableInstallVersionGuardReady + "; " +
        "VTableInstallNoThrowBoundaryReady=" + VTableInstallNoThrowBoundaryReady + "; " +
        "VTableInstallOwnershipDiagnosticsReady=" + VTableInstallOwnershipDiagnosticsReady + "; " +
        "VTableInstallPointerFree=" + VTableInstallPointerFree + "; " +
        "AttachBridgeSetDebugListenerNonNullEnabled=" + AttachBridgeSetDebugListenerNonNullEnabled + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorDataPointerEscapeBlocked=" + BorrowedDebugTensorDataPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "NativeVTableInstallRuntimeReady=" + NativeVTableInstallRuntimeReady + "; " +
        "CanEnableSetDebugListenerNonNull=" + CanEnableSetDebugListenerNonNull + "; " +
        "CanInstallNativeVTable=" + CanInstallNativeVTable + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "FullPackageConsumerRuntimeEvidenceReady=" + FullPackageConsumerRuntimeEvidenceReady + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonNativeVTableInstallStillBlocked=" + ReasonNativeVTableInstallStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:install={NativeVTableInstalled}:proof={IsRealCallbackRuntimeProof}";
    }
}
