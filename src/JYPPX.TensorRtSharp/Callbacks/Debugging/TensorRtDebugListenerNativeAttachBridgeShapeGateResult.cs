using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;
/// <summary>
/// Reports copied DebugListener native attach bridge shape diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is an attach bridge shape gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real non-null attach bridge,
/// complete native vtable, and full package consumer callback runtime evidence exist.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeAttachBridgeShapeGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeAttachBridgeShapeGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeOwnerLifecycleGateReady,
        bool attachBridgeShapeReady,
        bool attachBridgeNoThrowBoundaryReady,
        bool attachBridgeVersionGuardReady,
        bool attachBridgeOwnershipDiagnosticsReady,
        bool attachBridgePointerFree,
        bool setDebugListenerNonNullEnabled,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool nativeOwnerLifecycleReady,
        bool nativeVTableDesignReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeOwnerLifecycleGateReady = nativeOwnerLifecycleGateReady;
        AttachBridgeShapeReady = attachBridgeShapeReady;
        AttachBridgeNoThrowBoundaryReady = attachBridgeNoThrowBoundaryReady;
        AttachBridgeVersionGuardReady = attachBridgeVersionGuardReady;
        AttachBridgeOwnershipDiagnosticsReady = attachBridgeOwnershipDiagnosticsReady;
        AttachBridgePointerFree = attachBridgePointerFree;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-native-attach-bridge-shape-gate";

    /// <summary>Gets the callback kind represented by this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "attach-bridge-shape-gate";

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

    /// <summary>Gets whether native owner lifecycle gate evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleGateReady { get; }

    /// <summary>Gets whether source-visible attach bridge shape evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeShapeReady { get; }

    /// <summary>Gets whether source-visible attach bridge no-throw boundary evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeNoThrowBoundaryReady { get; }

    /// <summary>Gets whether TensorRT 10/11 attach bridge version guard evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeVersionGuardReady { get; }

    /// <summary>Gets whether attach bridge ownership diagnostics evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeOwnershipDiagnosticsReady { get; }

    /// <summary>Gets whether attach bridge diagnostics remain pointer-free. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgePointerFree { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether non-null attach is still disabled. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NonNullAttachStillDisabled => !SetDebugListenerNonNullEnabled;

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether native owner lifecycle implementation is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether native IDebugListener vtable implementation is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether copied attach bridge shape evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachBridgeShapeGateReady =>
        NativeOwnerLifecycleGateReady &&
        AttachBridgeShapeReady &&
        AttachBridgeNoThrowBoundaryReady &&
        AttachBridgeVersionGuardReady &&
        AttachBridgeOwnershipDiagnosticsReady &&
        AttachBridgePointerFree &&
        NonNullAttachStillDisabled &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved blockers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanImplementNativeAttach =>
        AttachBridgeShapeGateReady &&
        NativeAttachEntryLocated &&
        NativeOwnerLifecycleReady &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
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

    /// <summary>Gets the attach bridge shape gate status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => AttachBridgeShapeGateReady ? "attach-bridge-shape-gate-ready" : "attach-bridge-shape-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-native-attach-bridge-shape-gate; RuntimeEvidenceKind=attach-bridge-shape-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; AttachBridgeShapeGateReady=" + AttachBridgeShapeGateReady + "; " +
        "NativeOwnerLifecycleGateReady=" + NativeOwnerLifecycleGateReady + "; " +
        "AttachBridgeShapeReady=" + AttachBridgeShapeReady + "; " +
        "AttachBridgeNoThrowBoundaryReady=" + AttachBridgeNoThrowBoundaryReady + "; " +
        "AttachBridgeVersionGuardReady=" + AttachBridgeVersionGuardReady + "; " +
        "AttachBridgeOwnershipDiagnosticsReady=" + AttachBridgeOwnershipDiagnosticsReady + "; " +
        "AttachBridgePointerFree=" + AttachBridgePointerFree + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={NativeAttachEntryLocated}:proof={IsRealCallbackRuntimeProof}";
    }
}
