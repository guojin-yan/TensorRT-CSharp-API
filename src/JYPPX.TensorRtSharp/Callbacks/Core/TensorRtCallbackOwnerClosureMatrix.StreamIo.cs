using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtCallbackOwnerClosureMatrix
{
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

}
