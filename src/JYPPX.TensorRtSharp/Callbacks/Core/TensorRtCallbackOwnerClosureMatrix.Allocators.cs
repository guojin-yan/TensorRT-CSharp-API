using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtCallbackOwnerClosureMatrix
{
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

}
