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
