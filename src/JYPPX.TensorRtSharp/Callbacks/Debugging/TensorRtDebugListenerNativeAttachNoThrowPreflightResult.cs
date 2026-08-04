using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports DebugListener native attach/no-throw preflight diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is a preflight. <see cref="RealCallbackRuntime"/> and <see cref="IsRealCallbackRuntimeProof"/> remain
/// <see langword="false"/> until a native no-throw vtable bridge and full package consumer callback runtime evidence
/// exist.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeAttachNoThrowPreflightResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeAttachNoThrowPreflightResult(
        TensorRtApiLine line,
        bool attachVTableSafetyGateReady,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool stableNativeOwnerAddressDesignReady,
        bool managedCallbackKeepAliveDesignReady,
        bool noThrowVTableDesignReady,
        bool exceptionToStatusMappingDesignReady,
        bool borrowedDebugTensorMetadataCopyDesignReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeRuntimeReady,
        bool borrowedDebugTensorDataLifetimeRuntimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        AttachVTableSafetyGateReady = attachVTableSafetyGateReady;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        StableNativeOwnerAddressDesignReady = stableNativeOwnerAddressDesignReady;
        ManagedCallbackKeepAliveDesignReady = managedCallbackKeepAliveDesignReady;
        NoThrowVTableDesignReady = noThrowVTableDesignReady;
        ExceptionToStatusMappingDesignReady = exceptionToStatusMappingDesignReady;
        BorrowedDebugTensorMetadataCopyDesignReady = borrowedDebugTensorMetadataCopyDesignReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeRuntimeReady = borrowedDebugTensorLifetimeRuntimeReady;
        BorrowedDebugTensorDataLifetimeRuntimeReady = borrowedDebugTensorDataLifetimeRuntimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this preflight. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-native-attach-nothrow-preflight";

    /// <summary>Gets the callback kind represented by this preflight. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "preflight";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether attach/vtable safety evidence is ready for native preflight consumption. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachVTableSafetyGateReady { get; }

    /// <summary>Gets whether a line-specific native non-null DebugListener attach entry exists. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a line-specific native detach/clear entry exists. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether a stable native DebugListener owner address design is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool StableNativeOwnerAddressDesignReady { get; }

    /// <summary>Gets whether managed callback keep-alive and dispose/drain design evidence is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ManagedCallbackKeepAliveDesignReady { get; }

    /// <summary>Gets whether a no-throw native vtable design is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NoThrowVTableDesignReady { get; }

    /// <summary>Gets whether native exception-to-status mapping design is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ExceptionToStatusMappingDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata copy design is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool BorrowedDebugTensorMetadataCopyDesignReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether the native owner and vtable design are ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeVTableDesignReady =>
        StableNativeOwnerAddressDesignReady &&
        NoThrowVTableDesignReady &&
        ExceptionToStatusMappingDesignReady;

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is proven by runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool BorrowedDebugTensorLifetimeRuntimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is proven by runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool BorrowedDebugTensorDataLifetimeRuntimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether copied preflight evidence is complete enough to guide native implementation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool PreflightReady =>
        AttachVTableSafetyGateReady &&
        NativeDetachEntryLocated &&
        ManagedCallbackKeepAliveDesignReady &&
        BorrowedDebugTensorMetadataCopyDesignReady &&
        BorrowedDebugTensorPointerEscapeBlocked;

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved design blockers. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanImplementNativeAttach =>
        PreflightReady &&
        NativeAttachEntryLocated &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        BorrowedDebugTensorLifetimeRuntimeReady &&
        BorrowedDebugTensorDataLifetimeRuntimeReady &&
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

    /// <summary>Gets the preflight status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => PreflightReady ? "preflight-ready" : "preflight-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-native-attach-nothrow-preflight; RuntimeEvidenceKind=preflight; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; PreflightReady=" + PreflightReady + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "StableNativeOwnerAddressDesignReady=" + StableNativeOwnerAddressDesignReady + "; " +
        "ManagedCallbackKeepAliveDesignReady=" + ManagedCallbackKeepAliveDesignReady + "; " +
        "NoThrowVTableDesignReady=" + NoThrowVTableDesignReady + "; " +
        "ExceptionToStatusMappingDesignReady=" + ExceptionToStatusMappingDesignReady + "; " +
        "BorrowedDebugTensorLifetimeRuntimeReady=" + BorrowedDebugTensorLifetimeRuntimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeRuntimeReady=" + BorrowedDebugTensorDataLifetimeRuntimeReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "FullPackageConsumerRuntimeEvidenceReady=" + FullPackageConsumerRuntimeEvidenceReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={NativeAttachEntryLocated}:nothrow={NoThrowVTableDesignReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
