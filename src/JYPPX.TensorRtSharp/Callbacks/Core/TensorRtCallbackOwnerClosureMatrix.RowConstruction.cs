using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtCallbackOwnerClosureMatrix
{
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

}
