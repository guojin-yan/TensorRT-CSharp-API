using System.Text;

namespace JYPPX.ProjectQuality.Tests;

internal static class RepositorySourceReader
{
    private static readonly IReadOnlyDictionary<string, string[]> SourceSets =
        new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["TensorRtDebugListenerRuntimeProofPrecheck.cs"] =
            [
                "TensorRtDebugListenerRuntimeProofPrecheck.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.DesignPrerequisites.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.NativeAttachDesign.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.OwnerLifecycle.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.RuntimeScaffold.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.FinalRuntimeGates.cs",
                "TensorRtDebugListenerRuntimeProofPrecheckResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs",
                "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs"
            ],
            ["TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs"] =
            [
                "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
                "TensorRtDebugTensorMetadataSnapshot.cs",
                "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerLifecycleGate.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
                "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs"
            ],
            ["TensorRtDebugListenerNoThrowVTableCallbackStub.cs"] =
            [
                "TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
                "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs"
            ],
            ["TensorRtDebugListenerNativeVTableInstallPreflight.cs"] =
            [
                "TensorRtDebugListenerNativeVTableInstallPreflight.cs",
                "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs"
            ],
            ["TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs"] =
            [
                "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs",
                "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
                "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerAddressDesignGate.cs",
                "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs"
            ],
            ["TensorRtDebugListenerRealCallbackRuntimeProof.cs"] =
            [
                "TensorRtDebugListenerRealCallbackRuntimeProof.cs",
                "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs"
            ],
            ["TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs"] =
            [
                "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
                "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs"
            ],
            ["TensorRtDebugListenerRuntimeProofAttemptPreflight.cs"] =
            [
                "TensorRtDebugListenerRuntimeProofAttemptPreflight.cs",
                "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs"
            ],
            ["TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs"] =
            [
                "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
                "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs"
            ],
            ["TensorRtDebugListenerCallbackProofGapReport.cs"] =
            [
                "TensorRtDebugListenerCallbackProofGapReport.cs",
                "TensorRtDebugListenerCallbackProofGapReportResult.cs"
            ],
            ["TensorRtDebugListenerNativeAttachEntryDesignGate.cs"] =
            [
                "TensorRtDebugListenerNativeAttachEntryDesignGate.cs",
                "TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"] =
            [
                "TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs",
                "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs"] =
            [
                "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
                "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeNoThrowDestructor.cs"] =
            [
                "TensorRtDebugListenerNativeNoThrowDestructor.cs",
                "TensorRtDebugListenerNativeNoThrowDestructorResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs",
                "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs"
            ],
            ["TensorRtDebugListenerBorrowedTensorSafetyGate.cs"] =
            [
                "TensorRtDebugListenerBorrowedTensorSafetyGate.cs",
                "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs"
            ],
            ["TensorRtDebugListenerAttachDetachDesignGate.cs"] =
            [
                "TensorRtDebugListenerAttachDetachDesignGate.cs",
                "TensorRtDebugListenerAttachDetachDesignGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeAttachBridgeShapeGate.cs"] =
            [
                "TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
                "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs"
            ],
            ["TensorRtDebugListenerExceptionStatusMappingGate.cs"] =
            [
                "TensorRtDebugListenerExceptionStatusMappingGate.cs",
                "TensorRtDebugListenerExceptionStatusMappingGateResult.cs"
            ],
            ["TensorRtDebugListenerInFlightAccountingGate.cs"] =
            [
                "TensorRtDebugListenerInFlightAccountingGate.cs",
                "TensorRtDebugListenerInFlightAccountingGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs"] =
            [
                "TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs",
                "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs"
            ],
            ["TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs"] =
            [
                "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs",
                "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs"
            ],
            ["TensorRtDebugListenerAttachVTableSafetyGate.cs"] =
            [
                "TensorRtDebugListenerAttachVTableSafetyGate.cs",
                "TensorRtDebugListenerAttachVTableSafetyGateResult.cs"
            ],
            ["TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"] =
            [
                "TensorRtDebugListenerNativeAttachNoThrowPreflight.cs",
                "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs"
            ],
            ["TensorRtDebugListenerNativeOwnerStableIdentity.cs"] =
            [
                "TensorRtDebugListenerNativeOwnerStableIdentity.cs",
                "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs"
            ],
            ["TensorRtAllocatorCallbackOwner.cs"] =
            [
                "TensorRtAllocatorCallbackOwner.cs",
                "TensorRtAllocatorCallbackOwner.Lifecycle.cs",
                "TensorRtAllocatorCallbackOwner.ManagedDryRun.cs",
                "TensorRtAllocatorCallbackOwner.NativeDryRun.cs",
                "TensorRtAllocatorCallbackOwner.StateLedger.cs",
                "TensorRtAllocatorCallbackOwner.InternalPrototype.cs",
                "TensorRtAllocatorCallbackOwner.ResultMapping.cs",
                "TensorRtAllocatorDryRunRequest.cs",
                "TensorRtAllocatorDryRunResult.cs",
                "TensorRtAllocatorDryRunHandler.cs",
                "TensorRtAllocatorNativeDryRunResult.cs",
                "TensorRtAllocatorOwnerStateDryRunResult.cs",
                "TensorRtAllocatorCallbackOwnerSnapshot.cs",
                "TensorRtAllocatorInternalRuntimePrototypeResult.cs"
            ],
            ["TensorRtAllocatorLedgerSafetyGate.cs"] =
            [
                "TensorRtAllocatorLedgerSafetyGate.cs",
                "TensorRtAllocatorLedgerSafetyGateResult.cs"
            ],
            ["TensorRtOutputAllocatorRuntimeProofPrecheck.cs"] =
            [
                "TensorRtOutputAllocatorRuntimeProofPrecheck.cs",
                "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs"
            ],
            ["TensorRtOutputBufferOwnershipSafetyGate.cs"] =
            [
                "TensorRtOutputBufferOwnershipSafetyGate.cs",
                "TensorRtOutputBufferOwnershipSafetyGateResult.cs"
            ],
            ["TensorRtOutputAllocatorAttachDetachDesignGate.cs"] =
            [
                "TensorRtOutputAllocatorAttachDetachDesignGate.cs",
                "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs"
            ],
            ["TensorRtAllocatorInterfaceInfoDesignGate.cs"] =
            [
                "TensorRtAllocatorInterfaceInfoDesignGate.cs",
                "TensorRtAllocatorInterfaceInfoDesignGateResult.cs"
            ],
            ["TensorRtCallbackAllocatorReadiness.cs"] =
            [
                "TensorRtCallbackAllocatorReadiness.cs",
                "TensorRtCallbackAllocatorReadinessSnapshot.cs"
            ],
            ["TensorRtDebugListenerCallbackOwner.cs"] =
            [
                "TensorRtDebugListenerCallbackOwner.cs",
                "TensorRtDebugListenerCallbackOwner.DesignDiagnostic.cs",
                "TensorRtDebugListenerCallbackOwner.Snapshots.cs",
                "TensorRtDebugListenerCallbackOwner.Lifecycle.cs",
                "TensorRtDebugListenerCallbackOwner.Trampoline.cs",
                "TensorRtDebugListenerCallbackOwner.ShapeFormatting.cs",
                "TensorRtDebugListenerCallbackRequest.cs",
                "TensorRtDebugListenerCallbackOwnerSnapshot.cs"
            ],
            ["TensorRtCallbackOwnerClosureMatrix.cs"] =
            [
                "TensorRtCallbackOwnerClosureMatrix.cs",
                "TensorRtCallbackOwnerClosureMatrix.Allocators.cs",
                "TensorRtCallbackOwnerClosureMatrix.OutputDebug.cs",
                "TensorRtCallbackOwnerClosureMatrix.StreamIo.cs",
                "TensorRtCallbackOwnerClosureMatrix.RowConstruction.cs",
                "TensorRtCallbackOwnerClosureMatrix.Blockers.cs",
                "TensorRtCallbackOwnerClosureMatrixRow.cs",
                "TensorRtCallbackOwnerClosureMatrixResult.cs"
            ],
            ["TensorRtAlgorithmSnapshotDesignGate.cs"] =
            [
                "TensorRtAlgorithmSnapshotDesignGate.cs",
                "TensorRtAlgorithmSnapshotDesignGateResult.cs"
            ],
            ["TensorRtOutputAllocatorRuntimeGate.cs"] =
            [
                "TensorRtOutputAllocatorRuntimeGate.cs",
                "TensorRtOutputAllocatorRuntimeGate.Entries.cs",
                "TensorRtOutputAllocatorRuntimeGate.Snapshots.cs",
                "TensorRtOutputAllocatorRuntimeGate.Lifecycle.cs",
                "TensorRtOutputAllocatorRuntimeGate.Invocation.cs",
                "TensorRtOutputAllocatorRuntimeGate.Trampoline.cs",
                "TensorRtOutputAllocatorRuntimeGate.Formatting.cs",
                "TensorRtOutputAllocatorRuntimeGateRequest.cs",
                "TensorRtOutputAllocatorRuntimeGateResult.cs"
            ],
            ["TensorRtOutputAllocatorCallbackOwner.cs"] =
            [
                "TensorRtOutputAllocatorCallbackOwner.cs",
                "TensorRtOutputAllocatorCallbackOwner.DesignDiagnostic.cs",
                "TensorRtOutputAllocatorCallbackOwner.Snapshots.cs",
                "TensorRtOutputAllocatorCallbackOwner.Lifecycle.cs",
                "TensorRtOutputAllocatorCallbackRequest.cs",
                "TensorRtOutputAllocatorCallbackOwnerSnapshot.cs"
            ],
            ["TensorRtLogger.cs"] =
            [
                "TensorRtLogger.cs",
                "TensorRtLogger.InterfaceMetadata.cs",
                "TensorRtLogger.Diagnostics.cs",
                "TensorRtLogger.Lifecycle.cs",
                "TensorRtLogger.Trampoline.cs",
                "TensorRtLogSeverity.cs",
                "TensorRtLogHandler.cs"
            ],
            ["TensorRtProfiler.cs"] =
            [
                "TensorRtProfiler.cs",
                "TensorRtProfiler.InterfaceMetadata.cs",
                "TensorRtProfiler.Diagnostics.cs",
                "TensorRtProfiler.Lifecycle.cs",
                "TensorRtProfiler.Trampoline.cs",
                "TensorRtProfilerHandler.cs"
            ],
            ["TensorRtProgressMonitor.cs"] =
            [
                "TensorRtProgressMonitor.cs",
                "TensorRtProgressMonitor.InterfaceMetadata.cs",
                "TensorRtProgressMonitor.Diagnostics.cs",
                "TensorRtProgressMonitor.Lifecycle.cs",
                "TensorRtProgressMonitor.Trampoline.cs",
                "TensorRtProgressMonitorEventKind.cs",
                "TensorRtProgressMonitorEvent.cs",
                "TensorRtProgressMonitorDiagnosticResult.cs",
                "TensorRtProgressMonitorHandler.cs"
            ],
            ["TensorRtErrorRecorderSnapshot.cs"] =
            [
                "TensorRtErrorRecord.cs",
                "TensorRtErrorRecorderSnapshot.cs",
                "TensorRtErrorRecorderSummary.cs"
            ],
            ["TensorRtErrorRecorderDiagnosticsDesignGate.cs"] =
            [
                "TensorRtErrorRecorderDiagnosticsDesignGate.cs",
                "TensorRtErrorRecorderDiagnosticsDesignGateResult.cs"
            ],
            ["TensorRtLoggerFinderMetadataDesignGate.cs"] =
            [
                "TensorRtLoggerFinderMetadataDesignGate.cs",
                "TensorRtLoggerFinderMetadataDesignGateResult.cs"
            ],
            ["TensorRtStreamIoInterfaceInfoDesignGate.cs"] =
            [
                "TensorRtStreamIoInterfaceInfoDesignGate.cs",
                "TensorRtStreamIoInterfaceInfoDesignGateResult.cs"
            ]
        };

    public static string Read(string path)
    {
        string fileName = Path.GetFileName(path);
        if (!SourceSets.TryGetValue(fileName, out string[]? sourceSet))
        {
            return File.ReadAllText(path);
        }

        string directory = Path.GetDirectoryName(path)
            ?? throw new InvalidOperationException($"Source path has no parent directory: {path}");
        StringBuilder source = new();
        foreach (string sourceFileName in sourceSet)
        {
            source.AppendLine(File.ReadAllText(Path.Combine(directory, sourceFileName)));
        }

        return source.ToString();
    }
}
