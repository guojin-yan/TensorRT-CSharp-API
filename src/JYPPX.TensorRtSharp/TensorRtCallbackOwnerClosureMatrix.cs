using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Builds a pointer-free callback owner closure matrix across TensorRT callback families.
/// 构建不暴露裸指针的 TensorRT callback owner 闭环矩阵。
/// </summary>
/// <remarks>
/// The matrix consumes copied diagnostics from existing gates. It does not attach callbacks, install native vtables,
/// allocate device memory, invoke stream read/write callbacks, or promote deferred rows to runtime proof.
/// 该矩阵只消费已有 gate 复制出的诊断，不 attach callback、不安装 native vtable、不分配 device memory、
/// 不调用 stream read/write callback，也不把 deferred 行提升为 runtime proof。
/// </remarks>
public static class TensorRtCallbackOwnerClosureMatrix
{
    /// <summary>
    /// Aggregates copied callback owner evidence into a family-level closure matrix.
    /// 将复制出的 callback owner evidence 聚合为按 family 展开的闭环矩阵。
    /// </summary>
    /// <param name="allocatorLedgerSafetyGate">Copied IGpuAllocator ledger gate evidence. 复制出的 IGpuAllocator ledger gate 证据。</param>
    /// <param name="outputAllocatorRuntimeProofPrecheck">Copied OutputAllocator runtime proof precheck evidence. 复制出的 OutputAllocator precheck 证据。</param>
    /// <param name="debugListenerRuntimeProofPrecheck">Copied DebugListener runtime proof precheck evidence. 复制出的 DebugListener precheck 证据。</param>
    /// <param name="streamIoInterfaceInfoDesignGate">Copied stream reader/writer design gate evidence. 复制出的 stream reader/writer design gate 证据。</param>
    /// <returns>A pointer-free closure matrix. 不暴露裸指针的闭环矩阵。</returns>
    public static TensorRtCallbackOwnerClosureMatrixResult Evaluate(
        TensorRtAllocatorLedgerSafetyGateResult allocatorLedgerSafetyGate,
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputAllocatorRuntimeProofPrecheck,
        TensorRtDebugListenerRuntimeProofPrecheckResult debugListenerRuntimeProofPrecheck,
        TensorRtStreamIoInterfaceInfoDesignGateResult streamIoInterfaceInfoDesignGate)
    {
        TensorRtCallbackOwnerClosureMatrixRow[] rows =
        {
            BuildGpuAllocatorRow(allocatorLedgerSafetyGate),
            BuildGpuAsyncAllocatorRow(allocatorLedgerSafetyGate),
            BuildOutputAllocatorRow(outputAllocatorRuntimeProofPrecheck),
            BuildDebugListenerRow(debugListenerRuntimeProofPrecheck),
            BuildStreamReaderWriterRow(streamIoInterfaceInfoDesignGate)
        };

        return new TensorRtCallbackOwnerClosureMatrixResult(rows, BuildMatrixBlockers(rows));
    }

    private static TensorRtCallbackOwnerClosureMatrixRow BuildGpuAllocatorRow(
        TensorRtAllocatorLedgerSafetyGateResult gate)
    {
        bool managedOwnerStateReady = gate.ManagedKeepAliveReady || gate.DisposeReleaseReady;
        bool nativeLedgerReady = gate.NativeLedgerDesignReady;
        return CreateRow(
            ownerFamily: "GpuAllocator",
            callbackKind: "gpu-allocator",
            callbackMethods: new[] { "IGpuAllocator::allocate", "IGpuAllocator::free", "IGpuAllocator::deallocate", "IGpuAllocator::reallocate" },
            supportedLines: "TRT8/TRT10/TRT11",
            evidenceKind: gate.EvidenceKind,
            runtimeEvidenceKind: gate.RuntimeEvidenceKind,
            designGateReady: managedOwnerStateReady && nativeLedgerReady && gate.PointerFreeSurfaceReady,
            managedOwnerStateReady: managedOwnerStateReady,
            safeHandleOrGcHandleKeepAliveReady: managedOwnerStateReady,
            nativeNonCopyableOwnerStorageReady: nativeLedgerReady,
            nativeCreateDestroySymmetricReady: nativeLedgerReady,
            attachDetachClearControlReady: gate.LineSpecificAttachDetachReady,
            detachBeforeReleaseReady: gate.DisposeReleaseReady,
            noThrowDestructorReady: gate.DisposeReleaseReady,
            noThrowVTableReady: false,
            managedExceptionCaptureReady: gate.FailureCount == 0 && gate.LastStatus == BridgeStatusCode.Ok,
            exceptionToStatusMappingReady: false,
            inFlightCallbackAccountingReady: gate.InFlightCallbackCount == 0 && gate.ActivePrototypeCallCount == 0,
            borrowedPointerEscapeBlocked: gate.PointerFreeSurfaceReady,
            optInRuntimeSmokeReady: false,
            packageConsumerRuntimeProofReady: gate.FullPackageConsumerRuntimeEvidenceReady,
            canAttemptRuntimeProof: gate.CanAttemptRuntimeProof,
            deferredRowsStillRequired: gate.DeferredRowsStillRequired,
            sourceBlockedPrerequisites: gate.BlockedPrerequisites,
            nextWorkItem: "Implement line-specific setGpuAllocator attach/detach, native no-throw vtable, exception-to-status mapping, device pointer ledger, and package-consumer runtime invocation proof.");
    }

    private static TensorRtCallbackOwnerClosureMatrixRow BuildGpuAsyncAllocatorRow(
        TensorRtAllocatorLedgerSafetyGateResult gate)
    {
        bool managedOwnerStateReady = gate.ManagedKeepAliveReady || gate.DisposeReleaseReady;
        bool nativeLedgerReady = gate.NativeLedgerDesignReady;
        return CreateRow(
            ownerFamily: "GpuAsyncAllocator",
            callbackKind: "gpu-async-allocator",
            callbackMethods: new[] { "IGpuAsyncAllocator::allocateAsync", "IGpuAsyncAllocator::deallocateAsync" },
            supportedLines: "TRT10/TRT11",
            evidenceKind: gate.EvidenceKind,
            runtimeEvidenceKind: gate.RuntimeEvidenceKind,
            designGateReady: managedOwnerStateReady && nativeLedgerReady && gate.PointerFreeSurfaceReady,
            managedOwnerStateReady: managedOwnerStateReady,
            safeHandleOrGcHandleKeepAliveReady: managedOwnerStateReady,
            nativeNonCopyableOwnerStorageReady: nativeLedgerReady,
            nativeCreateDestroySymmetricReady: nativeLedgerReady,
            attachDetachClearControlReady: gate.LineSpecificAttachDetachReady,
            detachBeforeReleaseReady: gate.DisposeReleaseReady,
            noThrowDestructorReady: gate.DisposeReleaseReady,
            noThrowVTableReady: false,
            managedExceptionCaptureReady: gate.FailureCount == 0 && gate.LastStatus == BridgeStatusCode.Ok,
            exceptionToStatusMappingReady: false,
            inFlightCallbackAccountingReady: gate.InFlightCallbackCount == 0 && gate.ActivePrototypeCallCount == 0,
            borrowedPointerEscapeBlocked: gate.PointerFreeSurfaceReady,
            optInRuntimeSmokeReady: false,
            packageConsumerRuntimeProofReady: gate.FullPackageConsumerRuntimeEvidenceReady,
            canAttemptRuntimeProof: gate.CanAttemptRuntimeProof && gate.StreamLifetimeReady,
            deferredRowsStillRequired: gate.DeferredRowsStillRequired,
            sourceBlockedPrerequisites: gate.BlockedPrerequisites,
            nextWorkItem: "Close CUDA stream lifetime, async allocation ordering, no-throw async vtable, and package-consumer allocateAsync/deallocateAsync runtime proof before enabling this family.");
    }

    private static TensorRtCallbackOwnerClosureMatrixRow BuildOutputAllocatorRow(
        TensorRtOutputAllocatorRuntimeProofPrecheckResult precheck)
    {
        return CreateRow(
            ownerFamily: "OutputAllocator",
            callbackKind: precheck.CallbackKind,
            callbackMethods: new[] { "IOutputAllocator::notifyShape", "IOutputAllocator::reallocateOutput" },
            supportedLines: "TRT8/TRT10/TRT11",
            evidenceKind: precheck.EvidenceKind,
            runtimeEvidenceKind: precheck.RuntimeEvidenceKind,
            designGateReady: precheck.OwnerDesignReady && precheck.AttachDetachDesignGateReady && precheck.PointerFreeSurfaceReady,
            managedOwnerStateReady: precheck.OwnerDesignReady && precheck.ManagedOwnerStateMachineReady && precheck.DisposeReleaseReady,
            safeHandleOrGcHandleKeepAliveReady: precheck.DisposeReleaseReady,
            nativeNonCopyableOwnerStorageReady: precheck.NativeLedgerDesignReady,
            nativeCreateDestroySymmetricReady: precheck.NativeLedgerDesignReady,
            attachDetachClearControlReady: precheck.LineSpecificAttachDetachReady,
            detachBeforeReleaseReady: precheck.DisposeReleaseReady && precheck.DetachClearControlAvailable,
            noThrowDestructorReady: precheck.DisposeReleaseReady,
            noThrowVTableReady: precheck.NoThrowNativeVTableReady,
            managedExceptionCaptureReady: precheck.OwnerDesignReady && precheck.PointerFreeSurfaceReady,
            exceptionToStatusMappingReady: precheck.NoThrowNativeVTableReady,
            inFlightCallbackAccountingReady: precheck.ManagedOwnerStateMachineReady,
            borrowedPointerEscapeBlocked: precheck.BorrowedPointerEscapeBlocked,
            optInRuntimeSmokeReady: precheck.CanAttemptRuntimeProof,
            packageConsumerRuntimeProofReady: precheck.FullPackageConsumerRuntimeEvidenceReady,
            canAttemptRuntimeProof: precheck.CanAttemptRuntimeProof,
            deferredRowsStillRequired: precheck.DeferredRowsStillRequired,
            sourceBlockedPrerequisites: precheck.BlockedPrerequisites,
            nextWorkItem: "Add native OutputAllocator stable owner, no-throw vtable, device pointer ownership ledger, stream lifetime handling, and real notifyShape/reallocateOutput package-consumer proof.");
    }

    private static TensorRtCallbackOwnerClosureMatrixRow BuildDebugListenerRow(
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck)
    {
        return CreateRow(
            ownerFamily: "DebugListener",
            callbackKind: "debug-listener-process-debug-tensor",
            callbackMethods: new[] { "IDebugListener::processDebugTensor" },
            supportedLines: "TRT10/TRT11",
            evidenceKind: precheck.EvidenceKind,
            runtimeEvidenceKind: precheck.RuntimeEvidenceKind,
            designGateReady: precheck.OwnerDesignReady && precheck.PointerFreeSurfaceReady && precheck.NativeOwnerLifecycleGateReady,
            managedOwnerStateReady: precheck.OwnerDesignReady && precheck.ManagedOwnerStateMachineReady && precheck.DisposeReleaseReady,
            safeHandleOrGcHandleKeepAliveReady: precheck.ManagedCallbackKeepAliveDesignReady,
            nativeNonCopyableOwnerStorageReady: precheck.NativeOwnerNonCopyableStorageReady && precheck.NativeOwnerCopyBlocked && precheck.NativeOwnerMoveBlocked,
            nativeCreateDestroySymmetricReady: precheck.NativeOwnerLifecycleGateReady && !precheck.LifecyclePointerProduced,
            attachDetachClearControlReady: precheck.DetachClearControlAvailable && precheck.NativeAttachBridgeShapeGateReady && precheck.AttachBridgeShapeReady && !precheck.NonNullAttachStillDisabled,
            detachBeforeReleaseReady: precheck.DetachBeforeReleaseReady && precheck.ReleaseHookOrderingReady && precheck.InFlightDrainBeforeReleaseReady,
            noThrowDestructorReady: precheck.NoThrowNativeDestructorReady && precheck.DestructorNoThrowScaffoldReady && precheck.DestructorExceptionEscapeBlocked,
            noThrowVTableReady: precheck.NoThrowVTableDesignReady && precheck.NativeNoThrowVTableScaffoldGateReady && precheck.ProcessDebugTensorCallbackStubNoThrowReady,
            managedExceptionCaptureReady: precheck.CallbackExceptionCaptureReady && precheck.NativeCallbackExceptionCaptureReady,
            exceptionToStatusMappingReady: precheck.ExceptionToStatusMappingReady && precheck.CallbackStatusMappingReady && precheck.ExceptionToStatusMappingDesignReady,
            inFlightCallbackAccountingReady: precheck.CallbackInFlightAccountingReady && precheck.InFlightAccountingGateReady,
            borrowedPointerEscapeBlocked: precheck.BorrowedDebugTensorPointerEscapeBlocked && !precheck.NativeOwnerPointerProduced && !precheck.VTablePointerProduced,
            optInRuntimeSmokeReady: precheck.CanAttemptRuntimeProof,
            packageConsumerRuntimeProofReady: precheck.FullPackageConsumerRuntimeEvidenceReady,
            canAttemptRuntimeProof: precheck.CanAttemptRuntimeProof,
            deferredRowsStillRequired: precheck.DeferredRowsStillRequired,
            sourceBlockedPrerequisites: precheck.BlockedPrerequisites,
            nextWorkItem: "Enable non-null attach only after native attach is no-throw, owner lifecycle is closed, vtable install is safe, processDebugTensor is invoked by TensorRT, and full package-consumer proof is captured.");
    }

    private static TensorRtCallbackOwnerClosureMatrixRow BuildStreamReaderWriterRow(
        TensorRtStreamIoInterfaceInfoDesignGateResult gate)
    {
        return CreateRow(
            ownerFamily: "StreamReaderWriter",
            callbackKind: "stream-reader-writer",
            callbackMethods: new[] { "IStreamReader::read", "IStreamReaderV2::read", "IStreamReaderV2::seek", "IStreamWriter::write" },
            supportedLines: gate.LineSupportsStreamWriter ? "TRT10/TRT11" : "TRT10",
            evidenceKind: gate.EvidenceKind,
            runtimeEvidenceKind: gate.RuntimeEvidenceKind,
            designGateReady: gate.DesignGateReady,
            managedOwnerStateReady: gate.ManagedOwnerLifetimeReady,
            safeHandleOrGcHandleKeepAliveReady: gate.ManagedOwnerLifetimeReady,
            nativeNonCopyableOwnerStorageReady: gate.NativeOwnerCreateDestroySymmetric,
            nativeCreateDestroySymmetricReady: gate.NativeOwnerCreateDestroySymmetric,
            attachDetachClearControlReady: gate.DetachBeforeReleaseReady,
            detachBeforeReleaseReady: gate.DetachBeforeReleaseReady,
            noThrowDestructorReady: gate.NativeOwnerCreateDestroySymmetric,
            noThrowVTableReady: gate.NoThrowVTableReady,
            managedExceptionCaptureReady: gate.ExceptionToStatusMappingReady,
            exceptionToStatusMappingReady: gate.ExceptionToStatusMappingReady,
            inFlightCallbackAccountingReady: false,
            borrowedPointerEscapeBlocked: gate.PointerFreeSurfaceReady,
            optInRuntimeSmokeReady: false,
            packageConsumerRuntimeProofReady: false,
            canAttemptRuntimeProof: gate.CanPromoteRuntimeProof,
            deferredRowsStillRequired: gate.DeferredRowsStillRequired,
            sourceBlockedPrerequisites: gate.BlockedPrerequisites,
            nextWorkItem: "Create managed stream owner SafeHandle/GCHandle, native create/destroy dry-run, no-throw read/seek/write vtable, exception-to-status mapping, and owner-scoped metadata copy before callback runtime proof.");
    }

    private static TensorRtCallbackOwnerClosureMatrixRow CreateRow(
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
        bool packageConsumerRuntimeProofReady,
        bool canAttemptRuntimeProof,
        bool deferredRowsStillRequired,
        IEnumerable<string> sourceBlockedPrerequisites,
        string nextWorkItem)
    {
        bool packageConsumerRuntimeProofRequired = true;
        bool closureReady =
            designGateReady &&
            managedOwnerStateReady &&
            safeHandleOrGcHandleKeepAliveReady &&
            nativeNonCopyableOwnerStorageReady &&
            nativeCreateDestroySymmetricReady &&
            attachDetachClearControlReady &&
            detachBeforeReleaseReady &&
            noThrowDestructorReady &&
            noThrowVTableReady &&
            managedExceptionCaptureReady &&
            exceptionToStatusMappingReady &&
            inFlightCallbackAccountingReady &&
            borrowedPointerEscapeBlocked &&
            optInRuntimeSmokeReady &&
            packageConsumerRuntimeProofReady &&
            canAttemptRuntimeProof &&
            !deferredRowsStillRequired;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, designGateReady, "family design gate is not ready.");
        AddBlockerIfFalse(blockers, managedOwnerStateReady, "managed owner state is not closed.");
        AddBlockerIfFalse(blockers, safeHandleOrGcHandleKeepAliveReady, "SafeHandle/GCHandle keep-alive is not closed.");
        AddBlockerIfFalse(blockers, nativeNonCopyableOwnerStorageReady, "native non-copyable owner storage is not closed.");
        AddBlockerIfFalse(blockers, nativeCreateDestroySymmetricReady, "native create/destroy symmetry is not closed.");
        AddBlockerIfFalse(blockers, attachDetachClearControlReady, "attach/detach/clear control is not closed.");
        AddBlockerIfFalse(blockers, detachBeforeReleaseReady, "detach-before-release ordering is not closed.");
        AddBlockerIfFalse(blockers, noThrowDestructorReady, "no-throw destructor is not closed.");
        AddBlockerIfFalse(blockers, noThrowVTableReady, "no-throw callback vtable is not closed.");
        AddBlockerIfFalse(blockers, managedExceptionCaptureReady, "managed exception capture is not closed.");
        AddBlockerIfFalse(blockers, exceptionToStatusMappingReady, "exception-to-status mapping is not closed.");
        AddBlockerIfFalse(blockers, inFlightCallbackAccountingReady, "in-flight callback accounting is not closed.");
        AddBlockerIfFalse(blockers, borrowedPointerEscapeBlocked, "borrowed pointer escape blocker is not closed.");
        AddBlockerIfFalse(blockers, optInRuntimeSmokeReady, "opt-in runtime smoke is not ready.");
        AddBlockerIfFalse(blockers, packageConsumerRuntimeProofReady, "package-consumer real callback runtime proof has not been captured.");
        AddBlockerIfFalse(blockers, canAttemptRuntimeProof, "family cannot attempt real runtime proof.");
        AddBlockerIfFalse(blockers, !deferredRowsStillRequired, "direct callback deferred rows still must remain deferred.");

        foreach (string blocker in sourceBlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtCallbackOwnerClosureMatrixRow(
            ownerFamily,
            callbackKind,
            callbackMethods,
            supportedLines,
            evidenceKind,
            runtimeEvidenceKind,
            designGateReady,
            managedOwnerStateReady,
            safeHandleOrGcHandleKeepAliveReady,
            nativeNonCopyableOwnerStorageReady,
            nativeCreateDestroySymmetricReady,
            attachDetachClearControlReady,
            detachBeforeReleaseReady,
            noThrowDestructorReady,
            noThrowVTableReady,
            managedExceptionCaptureReady,
            exceptionToStatusMappingReady,
            inFlightCallbackAccountingReady,
            borrowedPointerEscapeBlocked,
            optInRuntimeSmokeReady,
            packageConsumerRuntimeProofRequired,
            packageConsumerRuntimeProofReady,
            closureReady,
            canAttemptRuntimeProof,
            !closureReady,
            deferredRowsStillRequired,
            nextWorkItem,
            blockers.ToArray());
    }

    private static string[] BuildMatrixBlockers(IEnumerable<TensorRtCallbackOwnerClosureMatrixRow> rows)
    {
        List<string> blockers = new List<string>();
        foreach (TensorRtCallbackOwnerClosureMatrixRow row in rows)
        {
            foreach (string blocker in row.BlockedPrerequisites)
            {
                AddBlocker(blockers, row.OwnerFamily + ": " + blocker);
            }
        }

        return blockers.ToArray();
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

/// <summary>
/// Reports the aggregated callback owner closure matrix.
/// 报告聚合后的 callback owner 闭环矩阵。
/// </summary>
public sealed class TensorRtCallbackOwnerClosureMatrixResult
{
    private readonly TensorRtCallbackOwnerClosureMatrixRow[] _rows;
    private readonly string[] _blockedPrerequisites;

    internal TensorRtCallbackOwnerClosureMatrixResult(
        TensorRtCallbackOwnerClosureMatrixRow[] rows,
        string[] blockedPrerequisites)
    {
        _rows = rows == null ? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>() : (TensorRtCallbackOwnerClosureMatrixRow[])rows.Clone();
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this matrix. 获取该矩阵的 evidence marker。</summary>
    public string EvidenceKind => "callback-owner-closure-matrix";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "closure-matrix";

    /// <summary>Gets whether this matrix proves real TensorRT callback runtime. 获取该矩阵是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether this matrix is promotable as real callback runtime proof. 获取该矩阵是否可提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets matrix rows. 获取矩阵行。</summary>
    public ReadOnlyCollection<TensorRtCallbackOwnerClosureMatrixRow> Rows =>
        Array.AsReadOnly(_rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>());

    /// <summary>Gets row count. 获取矩阵行数。</summary>
    public int FamilyCount => (_rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>()).Length;

    /// <summary>Gets how many families have design-gate evidence ready. 获取 design gate 证据就绪的 family 数量。</summary>
    public int DesignGateReadyFamilyCount => CountRows(static row => row.DesignGateReady);

    /// <summary>Gets how many families have full owner closure. 获取 owner 闭环完成的 family 数量。</summary>
    public int ClosureReadyFamilyCount => CountRows(static row => row.ClosureReady);

    /// <summary>Gets how many families can attempt runtime proof. 获取可尝试 runtime proof 的 family 数量。</summary>
    public int RuntimeProofAttemptReadyFamilyCount => CountRows(static row => row.CanAttemptRuntimeProof);

    /// <summary>Gets how many families have package-consumer runtime proof. 获取已有 package-consumer runtime proof 的 family 数量。</summary>
    public int PackageConsumerRuntimeProofReadyFamilyCount => CountRows(static row => row.PackageConsumerRuntimeProofReady);

    /// <summary>Gets whether public surfaces remain pointer-free across all rows. 获取所有行 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady => CountRows(static row => row.BorrowedPointerEscapeBlocked) == FamilyCount;

    /// <summary>Gets whether all rows have closure complete. 获取全部 family 是否已闭环。</summary>
    public bool AllFamiliesClosureReady => FamilyCount > 0 && ClosureReadyFamilyCount == FamilyCount;

    /// <summary>Gets whether all rows can attempt real runtime proof. 获取全部 family 是否可尝试真实 runtime proof。</summary>
    public bool CanAttemptRuntimeProof => FamilyCount > 0 && RuntimeProofAttemptReadyFamilyCount == FamilyCount;

    /// <summary>Gets whether real runtime proof remains blocked. 获取真实 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof || PackageConsumerRuntimeProofReadyFamilyCount != FamilyCount;

    /// <summary>Gets whether direct callback deferred rows still must remain deferred. 获取 direct callback deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => CountRows(static row => row.DeferredRowsStillRequired) > 0;

    /// <summary>Gets copied blocker details. 获取复制出的阻塞项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets blocker count. 获取阻塞项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets matrix status. 获取矩阵状态。</summary>
    public string Status => AllFamiliesClosureReady ? "closure-ready" : "closure-blocked";

    /// <summary>Gets a compact matrix summary. 获取紧凑矩阵摘要。</summary>
    public string Summary =>
        "callback-owner-closure-matrix; RuntimeEvidenceKind=closure-matrix; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; FamilyCount=" + FamilyCount + "; DesignGateReadyFamilyCount=" + DesignGateReadyFamilyCount + "; " +
        "ClosureReadyFamilyCount=" + ClosureReadyFamilyCount + "; RuntimeProofAttemptReadyFamilyCount=" + RuntimeProofAttemptReadyFamilyCount + "; " +
        "PackageConsumerRuntimeProofReadyFamilyCount=" + PackageConsumerRuntimeProofReadyFamilyCount + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; DeferredRowsStillRequired=" + DeferredRowsStillRequired + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:families={FamilyCount}:closure={ClosureReadyFamilyCount}:proof={IsRealCallbackRuntimeProof}:blocked={BlockedPrerequisiteCount}";
    }

    private int CountRows(Func<TensorRtCallbackOwnerClosureMatrixRow, bool> predicate)
    {
        int count = 0;
        foreach (TensorRtCallbackOwnerClosureMatrixRow row in _rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>())
        {
            if (predicate(row))
            {
                count++;
            }
        }

        return count;
    }
}
