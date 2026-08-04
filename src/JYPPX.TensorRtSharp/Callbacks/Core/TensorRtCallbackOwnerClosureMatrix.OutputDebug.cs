using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtCallbackOwnerClosureMatrix
{
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

}
