using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports one callback owner family in the closure matrix.
/// 表示 callback owner 闭环矩阵中的一个 family。
/// </summary>
public readonly struct TensorRtCallbackOwnerClosureMatrixRow
{
    private readonly string[] _callbackMethods;
    private readonly string[] _blockedPrerequisites;

    internal TensorRtCallbackOwnerClosureMatrixRow(
        string ownerFamily,
        string callbackKind,
        string[] callbackMethods,
        string supportedLines,
        string evidenceKind,
        string runtimeEvidenceKind,
        bool designGateReady,
        bool managedOwnerStateReady,
        bool safeHandleOrGcHandleKeepAliveReady,
        bool nativeNonCopyableOwnerStorageReady,
        bool nativeCreateDestroySymmetricReady,
        bool attachDetachClearControlReady,
        bool detachBeforeReleaseReady,
        bool noThrowDestructorReady,
        bool noThrowVTableReady,
        bool managedExceptionCaptureReady,
        bool exceptionToStatusMappingReady,
        bool inFlightCallbackAccountingReady,
        bool borrowedPointerEscapeBlocked,
        bool optInRuntimeSmokeReady,
        bool packageConsumerRuntimeProofRequired,
        bool packageConsumerRuntimeProofReady,
        bool closureReady,
        bool canAttemptRuntimeProof,
        bool runtimeProofBlocked,
        bool deferredRowsStillRequired,
        string nextWorkItem,
        string[] blockedPrerequisites)
    {
        OwnerFamily = ownerFamily ?? string.Empty;
        CallbackKind = callbackKind ?? string.Empty;
        _callbackMethods = callbackMethods == null ? Array.Empty<string>() : (string[])callbackMethods.Clone();
        SupportedLines = supportedLines ?? string.Empty;
        EvidenceKind = evidenceKind ?? string.Empty;
        RuntimeEvidenceKind = runtimeEvidenceKind ?? string.Empty;
        DesignGateReady = designGateReady;
        ManagedOwnerStateReady = managedOwnerStateReady;
        SafeHandleOrGcHandleKeepAliveReady = safeHandleOrGcHandleKeepAliveReady;
        NativeNonCopyableOwnerStorageReady = nativeNonCopyableOwnerStorageReady;
        NativeCreateDestroySymmetricReady = nativeCreateDestroySymmetricReady;
        AttachDetachClearControlReady = attachDetachClearControlReady;
        DetachBeforeReleaseReady = detachBeforeReleaseReady;
        NoThrowDestructorReady = noThrowDestructorReady;
        NoThrowVTableReady = noThrowVTableReady;
        ManagedExceptionCaptureReady = managedExceptionCaptureReady;
        ExceptionToStatusMappingReady = exceptionToStatusMappingReady;
        InFlightCallbackAccountingReady = inFlightCallbackAccountingReady;
        BorrowedPointerEscapeBlocked = borrowedPointerEscapeBlocked;
        OptInRuntimeSmokeReady = optInRuntimeSmokeReady;
        PackageConsumerRuntimeProofRequired = packageConsumerRuntimeProofRequired;
        PackageConsumerRuntimeProofReady = packageConsumerRuntimeProofReady;
        ClosureReady = closureReady;
        CanAttemptRuntimeProof = canAttemptRuntimeProof;
        RuntimeProofBlocked = runtimeProofBlocked;
        DeferredRowsStillRequired = deferredRowsStillRequired;
        NextWorkItem = nextWorkItem ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the owner family name. 获取 owner family 名称。</summary>
    public string OwnerFamily { get; }

    /// <summary>Gets the callback kind represented by this row. 获取该行代表的 callback 类型。</summary>
    public string CallbackKind { get; }

    /// <summary>Gets the callback methods covered by this row. 获取该行覆盖的 callback 方法。</summary>
    public ReadOnlyCollection<string> CallbackMethods =>
        Array.AsReadOnly(_callbackMethods ?? Array.Empty<string>());

    /// <summary>Gets the supported TensorRT lines for this family. 获取该 family 支持的 TensorRT line。</summary>
    public string SupportedLines { get; }

    /// <summary>Gets the source evidence marker consumed by this row. 获取该行消费的来源 evidence marker。</summary>
    public string EvidenceKind { get; }

    /// <summary>Gets the source runtime evidence kind consumed by this row. 获取该行消费的 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind { get; }

    /// <summary>Gets whether family design-gate evidence is ready. 获取 family design gate 证据是否就绪。</summary>
    public bool DesignGateReady { get; }

    /// <summary>Gets whether managed owner state is closed. 获取 managed owner state 是否闭合。</summary>
    public bool ManagedOwnerStateReady { get; }

    /// <summary>Gets whether SafeHandle or GCHandle keep-alive is closed. 获取 SafeHandle 或 GCHandle keep-alive 是否闭合。</summary>
    public bool SafeHandleOrGcHandleKeepAliveReady { get; }

    /// <summary>Gets whether native non-copyable owner storage is closed. 获取 native non-copyable owner storage 是否闭合。</summary>
    public bool NativeNonCopyableOwnerStorageReady { get; }

    /// <summary>Gets whether native create/destroy symmetry is closed. 获取 native create/destroy 对称性是否闭合。</summary>
    public bool NativeCreateDestroySymmetricReady { get; }

    /// <summary>Gets whether attach/detach/clear controls are closed. 获取 attach/detach/clear 控制是否闭合。</summary>
    public bool AttachDetachClearControlReady { get; }

    /// <summary>Gets whether detach-before-release ordering is closed. 获取 detach-before-release 顺序是否闭合。</summary>
    public bool DetachBeforeReleaseReady { get; }

    /// <summary>Gets whether no-throw destructor safety is closed. 获取 no-throw destructor 是否闭合。</summary>
    public bool NoThrowDestructorReady { get; }

    /// <summary>Gets whether no-throw callback vtable safety is closed. 获取 no-throw callback vtable 是否闭合。</summary>
    public bool NoThrowVTableReady { get; }

    /// <summary>Gets whether managed exception capture is closed. 获取 managed exception capture 是否闭合。</summary>
    public bool ManagedExceptionCaptureReady { get; }

    /// <summary>Gets whether exception-to-status mapping is closed. 获取 exception-to-status 映射是否闭合。</summary>
    public bool ExceptionToStatusMappingReady { get; }

    /// <summary>Gets whether in-flight callback accounting is closed. 获取 in-flight callback accounting 是否闭合。</summary>
    public bool InFlightCallbackAccountingReady { get; }

    /// <summary>Gets whether borrowed pointer escape is blocked on public surfaces. 获取 public surface 是否阻止 borrowed pointer 逃逸。</summary>
    public bool BorrowedPointerEscapeBlocked { get; }

    /// <summary>Gets whether an opt-in runtime smoke can safely run for this family. 获取该 family 是否可安全运行 opt-in runtime smoke。</summary>
    public bool OptInRuntimeSmokeReady { get; }

    /// <summary>Gets whether package-consumer runtime proof is required before promotion. 获取提升前是否需要 package-consumer runtime proof。</summary>
    public bool PackageConsumerRuntimeProofRequired { get; }

    /// <summary>Gets whether package-consumer runtime proof has been captured. 获取 package-consumer runtime proof 是否已捕获。</summary>
    public bool PackageConsumerRuntimeProofReady { get; }

    /// <summary>Gets whether every closure column is ready. 获取闭环列是否全部就绪。</summary>
    public bool ClosureReady { get; }

    /// <summary>Gets whether this family can attempt real runtime proof. 获取该 family 是否可尝试真实 runtime proof。</summary>
    public bool CanAttemptRuntimeProof { get; }

    /// <summary>Gets whether real runtime proof remains blocked. 获取真实 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked { get; }

    /// <summary>Gets whether direct callback deferred rows still must remain deferred. 获取 direct callback deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired { get; }

    /// <summary>Gets the next concrete work item for this family. 获取该 family 的下一步具体工作。</summary>
    public string NextWorkItem { get; }

    /// <summary>Gets copied blocker details. 获取复制出的阻塞项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets copied blocker count. 获取复制出的阻塞项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets how many closure columns are ready. 获取已就绪闭环列数量。</summary>
    public int ReadyClosureColumnCount
    {
        get
        {
            int count = 0;
            count += DesignGateReady ? 1 : 0;
            count += ManagedOwnerStateReady ? 1 : 0;
            count += SafeHandleOrGcHandleKeepAliveReady ? 1 : 0;
            count += NativeNonCopyableOwnerStorageReady ? 1 : 0;
            count += NativeCreateDestroySymmetricReady ? 1 : 0;
            count += AttachDetachClearControlReady ? 1 : 0;
            count += DetachBeforeReleaseReady ? 1 : 0;
            count += NoThrowDestructorReady ? 1 : 0;
            count += NoThrowVTableReady ? 1 : 0;
            count += ManagedExceptionCaptureReady ? 1 : 0;
            count += ExceptionToStatusMappingReady ? 1 : 0;
            count += InFlightCallbackAccountingReady ? 1 : 0;
            count += BorrowedPointerEscapeBlocked ? 1 : 0;
            count += OptInRuntimeSmokeReady ? 1 : 0;
            count += PackageConsumerRuntimeProofReady ? 1 : 0;
            return count;
        }
    }

    /// <summary>Gets the total number of closure columns. 获取闭环列总数。</summary>
    public int TotalClosureColumnCount => 15;

    /// <summary>Gets this row status. 获取该行状态。</summary>
    public string Status => ClosureReady ? "closure-ready" : "closure-blocked";

    /// <summary>Gets a compact pointer-free diagnostic. 获取紧凑无指针诊断。</summary>
    public string Diagnostic =>
        "callback-owner-closure-matrix-row; OwnerFamily=" + OwnerFamily + "; CallbackKind=" + CallbackKind + "; " +
        "DesignGateReady=" + DesignGateReady + "; ManagedOwnerStateReady=" + ManagedOwnerStateReady + "; " +
        "SafeHandleOrGcHandleKeepAliveReady=" + SafeHandleOrGcHandleKeepAliveReady + "; " +
        "NativeNonCopyableOwnerStorageReady=" + NativeNonCopyableOwnerStorageReady + "; " +
        "NativeCreateDestroySymmetricReady=" + NativeCreateDestroySymmetricReady + "; " +
        "AttachDetachClearControlReady=" + AttachDetachClearControlReady + "; DetachBeforeReleaseReady=" + DetachBeforeReleaseReady + "; " +
        "NoThrowDestructorReady=" + NoThrowDestructorReady + "; NoThrowVTableReady=" + NoThrowVTableReady + "; " +
        "ManagedExceptionCaptureReady=" + ManagedExceptionCaptureReady + "; ExceptionToStatusMappingReady=" + ExceptionToStatusMappingReady + "; " +
        "InFlightCallbackAccountingReady=" + InFlightCallbackAccountingReady + "; BorrowedPointerEscapeBlocked=" + BorrowedPointerEscapeBlocked + "; " +
        "OptInRuntimeSmokeReady=" + OptInRuntimeSmokeReady + "; PackageConsumerRuntimeProofRequired=" + PackageConsumerRuntimeProofRequired + "; " +
        "PackageConsumerRuntimeProofReady=" + PackageConsumerRuntimeProofReady + "; ClosureReady=" + ClosureReady + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "DeferredRowsStillRequired=" + DeferredRowsStillRequired + "; ReadyClosureColumnCount=" + ReadyClosureColumnCount + "/" + TotalClosureColumnCount + ".";
}
