using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied DebugListener native owner stable identity diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is an identity gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real native owner, non-null attach
/// bridge, and full package consumer callback runtime evidence exist.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeOwnerStableIdentityResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeOwnerStableIdentityResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        string lastDiagnostic,
        string releaseDiagnostic,
        bool nativeAttachEntryRuntimeScaffoldReady,
        bool stableNativeOwnerIdentityReady,
        bool nativeOwnerNonCopyableReady,
        bool ownerIdentityDiagnosticsReady,
        bool ownerIdentityPointerFree,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool noThrowNativeDestructorReady,
        bool nativeOwnerLifecycleReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
        NativeAttachEntryRuntimeScaffoldReady = nativeAttachEntryRuntimeScaffoldReady;
        StableNativeOwnerIdentityReady = stableNativeOwnerIdentityReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        OwnerIdentityDiagnosticsReady = ownerIdentityDiagnosticsReady;
        OwnerIdentityPointerFree = ownerIdentityPointerFree;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this identity gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-native-owner-stable-identity";

    /// <summary>Gets the callback kind represented by this identity gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "identity-gate";

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

    /// <summary>Gets the copied last diagnostic. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether native attach entry runtime scaffold evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryRuntimeScaffoldReady { get; }

    /// <summary>Gets whether stable native owner identity diagnostics are ready without exposing pointers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool StableNativeOwnerIdentityReady { get; }

    /// <summary>Gets whether native owner storage is non-copyable. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether copied owner id and diagnostic identity evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool OwnerIdentityDiagnosticsReady { get; }

    /// <summary>Gets whether the public owner identity surface is pointer-free. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool OwnerIdentityPointerFree { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether the native owner destructor is no-throw. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether native owner lifecycle implementation is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved owner blockers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanImplementNativeAttach =>
        NativeAttachEntryRuntimeScaffoldReady &&
        StableNativeOwnerIdentityReady &&
        NativeAttachEntryLocated &&
        NativeOwnerNonCopyableReady &&
        NoThrowNativeDestructorReady &&
        NativeOwnerLifecycleReady;

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

    /// <summary>Gets the identity gate status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => StableNativeOwnerIdentityReady ? "identity-gate-ready" : "identity-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-native-owner-stable-identity; RuntimeEvidenceKind=identity-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeAttachEntryRuntimeScaffoldReady=" + NativeAttachEntryRuntimeScaffoldReady + "; " +
        "StableNativeOwnerIdentityReady=" + StableNativeOwnerIdentityReady + "; " +
        "OwnerIdentityDiagnosticsReady=" + OwnerIdentityDiagnosticsReady + "; " +
        "OwnerIdentityPointerFree=" + OwnerIdentityPointerFree + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:owner={OwnerId}:proof={IsRealCallbackRuntimeProof}";
    }
}
